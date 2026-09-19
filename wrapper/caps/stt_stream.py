# Streaming ASR on one in-process transformers load, serving BOTH offline stt and WebSocket stt_stream.
import os
import json
import asyncio
import logging
import threading
import time
import types

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import Response

from .. import hfgate
from .. import tasks
from ..batch import parse_segments
from .. import cgroup
from .. import grouping
from ..gpu import mount_metrics, memory_fraction
from ..contract import register, EngineArgs
from ..audioio import pcm16_to_float32, resample_linear
from ..runtime import Runtime

log = logging.getLogger("audio-stt-stream")

_runtime = Runtime(asr=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
PORT = _runtime.port

_args = EngineArgs()
# vLLM wants a share of the whole card; the platform hands out a quota, so derive one from it.
GPU_UTIL = _args.number("--gpu-memory-utilization", memory_fraction() or 0.45)
# Holds ONE unit of work; the chart sizes it per machine type, since unified memory needs less.
MAX_MODEL_LEN = _args.count("--max-model-len", 8192)
# Capture is where startup wedges holding the vGPU lock.
ENFORCE_EAGER = _args.switch("--enforce-eager")
# One flag for how much audio a generate() carries, in one of two units: `auto` sizes from what
# this machine measures, `600s` is a padded-second budget, `32` a span count, `0` one span a call.


# 🔴 The units cannot be combined; `--batch-max-spans N` is the old spelling, a span
# ceiling on top of what the measurement picked. ENGINE_ARGS is the only app-editable knob, so no env.


def batch_request(args):
    """(grouping, max_spans, notes) for `--batch-max-spans`, the one batching flag.

    🔴 One flag, because one question is all an operator is better placed to answer than this
    engine: how many spans a call may carry, as a backstop for when the sizing is wrong. How
    much MEMORY a call may take is declared once at install as REQUIRED_GPU_MEMORY, and a
    second place to say the same thing is a second place for it to be wrong. How the spans
    inside that count are grouped -- which is what decides the padding -- needs the grant, the
    container and the neighbours on the card, none of which are visible from outside.

    `1` is one span a call, which is what an engine with no batching flag did before this. It is
    the same flag rather than a separate switch because a count of one already says it.
    """
    notes = []
    max_spans = None
    if args.given("--batch-max-spans"):
        raw = (args.text("--batch-max-spans", "") or "").strip()
        if raw.isdigit() and int(raw) > 0:
            max_spans = int(raw)
        else:
            # 🔴 No ceiling rather than a guessed one. A bound nobody asked for is as wrong as a
            # missing one, and `count()` would answer an unreadable value with its default.
            notes.append("WARN --batch-max-spans=%r is not a positive whole number of spans, "
                         "so no span ceiling is applied and the sizing is left to what this "
                         "machine measures" % raw)
    else:
        notes.append("batching is on and sized from what this machine measures. "
                     "`--batch-max-spans 1` is one span a call, which is what an engine with "
                     "no batching flag used to do")
    # 🔴 The count IS the switch: one span a call is grouping turned off, and a separate flag
    # for it would be a second way to say a thing this one already says.
    return max_spans != 1, max_spans, notes


try:
    GROUPING, MAX_SPANS, _BATCH_NOTES = batch_request(_args)
except Exception as _batch_err:
    # 🔴 This runs at import. An exception here does not degrade the batching policy, it stops
    # the module from loading at all -- the engine never starts, over a flag.
    GROUPING, MAX_SPANS = False, None
    _BATCH_NOTES = ["WARN could not read --batch-max-spans (%s: %s); one span per call"
                    % (type(_batch_err).__name__, str(_batch_err)[:160])]
# End a span that has started repeating rather than folding the loop out afterwards.
REPETITION_FALLBACK_TOKENS_PER_SEC = max(
    0, _args.count("--repetition-fallback-tokens-per-sec", 12))
_FALLBACK_ASKED = _args.given("--repetition-fallback-tokens-per-sec")
TOKENS_FLOOR = 64
REPETITION_DEFAULTS = {"min_pattern_size": 2, "max_pattern_size": 20, "min_count": 20}
# switch() is the bare/boolean form; text() is the JSON override.


def repetition_request(args):
    """(asked for?, JSON override, note) for --repetition-detection. 🔴 Only JSON is an override,
    not "every word outside a list": a list has an outside, and a non-empty string is truthy."""
    raw = (args.text("--repetition-detection", "") or "").strip()
    if raw[:1] in ("{", "["):
        return True, raw, None
    word = raw.lower()
    known = EngineArgs.ON_WORDS + EngineArgs.OFF_WORDS
    if raw and word not in known:
        return False, "", ("WARN --repetition-detection=%r is not a value this engine knows; "
                           "treating it as off. Use the bare flag to turn it on, or a JSON "
                           "object to override a threshold." % raw)
    return bool(args.switch("--repetition-detection")), "", None


REPETITION_ON, REPETITION_OVERRIDE, _REP_NOTE = repetition_request(_args)
# 🔴 Every flag has to be READ before this line: warn_unclaimed reports whatever is not yet
# claimed. Seen once: a flag read below this line was honoured and reported discarded at once.
_args.warn_unclaimed(log)

MAX_NEW_TOKENS = 32
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 2.0
DEFAULT_STEP_MS = 500
# Finalize + re-init this often, so a long session never overflows vLLM's ~8192-token encoder cache.
ROLL_SEC = 240.0
# qwen-asr silence-splits at this window; 540s keeps one call inside the ~600s encoder cache.
OFFLINE_MAX_INPUT_SEC = 540
OFFLINE_MAX_TOKENS = 4096

_state = _runtime.state
# One slot each: detector class vs requested params.
_MISSING = object()
_repdet_class = []
_repdet_cache = []
# Says the repetition story once, on the first transcription.
_repdet_said = []
_repset_said = []
# vLLM's generate is blocking and not concurrency-safe, so all inference shares one lock.
_infer_lock = asyncio.Lock()
# The same engine is also driven by the task worker (offline stt), which lives on another thread.
_gpu = threading.Lock()


def _gated(fn, *a):
    with _gpu:
        return fn(*a)


def _p(msg):
    print("[stream] " + msg, flush=True)


def _patch_max_input(seconds):
    """Hold qwen-asr to `seconds` as its clip split length, naming the modules the cap reached.
    🔴 Setting an attribute a module no longer has creates it silently; each is checked alone."""
    import qwen_asr.inference.qwen3_asr as _qasr_mod
    import qwen_asr.inference.utils as _qasr_utils

    patched, missing = [], []
    for m in (_qasr_utils, _qasr_mod):
        if hasattr(m, "MAX_ASR_INPUT_SECONDS"):
            m.MAX_ASR_INPUT_SECONDS = int(seconds)
            patched.append(m.__name__)
        else:
            missing.append(m.__name__)
    if missing:
        _p("WARN qwen-asr no longer has MAX_ASR_INPUT_SECONDS in %s; the clip cap of %ds is "
           "not enforced there" % (", ".join(missing), int(seconds)))
    if patched:
        _p("patched qwen-asr MAX_ASR_INPUT_SECONDS -> %ds in %s"
           % (int(seconds), ", ".join(patched)))


def _load_blocking():
    import torch
    from qwen_asr import Qwen3ASRModel

    _p("importing qwen_asr ...")
    try:
        _patch_max_input(OFFLINE_MAX_INPUT_SEC)
    except Exception as e:
        _p("WARN could not patch MAX_ASR_INPUT_SECONDS (%s)" % e)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    kw = dict(dtype=dtype, device_map=("cpu" if dev == "cpu" else "cuda:0"),
              max_inference_batch_size=-1, max_new_tokens=OFFLINE_MAX_TOKENS)
    token = os.environ.get("HF_TOKEN") or None
    if token:
        kw["token"] = token
    _p("constructing Qwen3ASRModel.from_pretrained(%s) on %s "
       "(--gpu-memory-utilization=%.2f --max-model-len=%d%s unused on transformers)"
       % (MODEL_REPO, dev, GPU_UTIL, MAX_MODEL_LEN,
          " --enforce-eager" if ENFORCE_EAGER else ""))
    try:
        asr = Qwen3ASRModel.from_pretrained(MODEL_REPO, **kw)
    except TypeError:
        kw.pop("token", None)
        asr = Qwen3ASRModel.from_pretrained(MODEL_REPO, **kw)
    _state["asr"] = asr
    _say_batching()
    _say_repetition_once()
    _warmup()
    _state["ready"] = True
    _p("engine READY: %s" % MODEL_REPO)
    log.info("qwen-asr transformers engine loaded: %s", MODEL_REPO)


def _tf_generate(asr, prompt, wav, max_new_tokens):
    inputs = asr.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    inputs = inputs.to(asr.model.device)
    try:
        inputs = inputs.to(asr.model.dtype)
    except Exception:
        pass
    old = getattr(asr, "max_new_tokens", None)
    asr.max_new_tokens = max_new_tokens
    try:
        out = asr.model.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        if old is not None:
            asr.max_new_tokens = old
    seqs = getattr(out, "sequences", out)
    decoded = asr.processor.batch_decode(
        seqs[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0]


def _stream_init(asr, language=None):
    import numpy as np
    from qwen_asr.inference.utils import SAMPLE_RATE

    # Shared with the offline path so both halves accept the same spellings: "zh" used to be a
    # hard error on the socket and a no-op over there.
    force = resolve_language_or_auto(language, "websocket session")
    n = max(1, int(round(float(CHUNK_SIZE_SEC) * SAMPLE_RATE)))
    return types.SimpleNamespace(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM, unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_samples=n, chunk_id=0,
        buffer=np.zeros((0,), dtype=np.float32),
        audio_accum=np.zeros((0,), dtype=np.float32),
        prompt_raw=asr._build_text_prompt(context="", force_language=force),
        force_language=force, language="", text="", _raw_decoded="")


def _stream_prefix(asr, state):
    if state.chunk_id < state.unfixed_chunk_num:
        return ""
    tok = asr.processor.tokenizer
    ids = tok.encode(state._raw_decoded)
    k = int(state.unfixed_token_num)
    while True:
        end = max(0, len(ids) - k)
        prefix = tok.decode(ids[:end]) if end > 0 else ""
        if "\ufffd" not in prefix:
            return prefix
        if end == 0:
            return ""
        k += 1


def _stream_decode(asr, state):
    from qwen_asr.inference.utils import parse_asr_output

    prefix = _stream_prefix(asr, state)
    gen = _tf_generate(asr, state.prompt_raw + prefix, state.audio_accum, MAX_NEW_TOKENS)
    state._raw_decoded = prefix + gen
    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language, state.text = lang, txt
    state.chunk_id += 1


def _stream_step(asr, pcm16k, state):
    import numpy as np

    x = np.asarray(pcm16k).reshape(-1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)
    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)
    n = state.chunk_size_samples
    while state.buffer.shape[0] >= n:
        chunk, state.buffer = state.buffer[:n], state.buffer[n:]
        state.audio_accum = chunk if state.audio_accum.shape[0] == 0 else np.concatenate(
            [state.audio_accum, chunk], axis=0)
        _stream_decode(asr, state)
    return state


def _stream_finish(asr, state):
    import numpy as np

    if state.buffer is None or state.buffer.shape[0] == 0:
        return state
    tail, state.buffer = state.buffer, np.zeros((0,), dtype=np.float32)
    state.audio_accum = tail if state.audio_accum.shape[0] == 0 else np.concatenate(
        [state.audio_accum, tail], axis=0)
    _stream_decode(asr, state)
    return state


# --- sizing a call on this machine -------------------------------------------------


# 🔴 Everything below measures. The branch this replaces carried four constants (7.42 MiB an
# encoder second, a 500 s knee, a 0.19 tail, 1.42 MiB of KV a second) fitted on a runtime we left.


# ⚠️ The shape, and `_strtoul_base0`/`_hami_limit_bytes` whole, come from the aligner on
# beclab/audio-engines#26, whose figures were fitted against a different model's encoder.


#: The share of readable headroom a batch may spend. 🔴 Policy: half is where a doubling of real
#: cost still fits. Not a flag -- it alone can exceed the grant; a smaller REQUIRED_GPU_MEMORY,
#: which is where that decision already lives, is the way down.
BUDGET_FRACTION = 0.5

#: Bytes a padded second costs the card before this machine is measured, off the only two points
#: ever taken: one 21-span request at 6 and at 12 spans a call, whole-card 718 vs 1772 MiB over
#: idle. ⚠️ Not evidence: one pair of points, whole-card, no repeat.
#: 🔴 Those points gave 6.02 when a padded second was `count x longest`. Since every span also
#: carries a floor the same two points refit to 5.82, at 181 and 362 padded seconds rather than
#: 175 and 350. 6.0 is kept because it is the higher of the two, and higher means smaller calls.
OPENING_BYTES_A_PADDED_SECOND = 6.0 * (2 ** 20)

#: 🔴 What one padded second costs the HOST, as arithmetic: float32 at 16 kHz is 64,000 bytes a
#: second and the mel is 128 x 100 x 4 = 51,200. A FLOOR -- the measured cost is well above it.
HOST_FLOOR_BYTES_A_PADDED_SECOND = 64000 + 51200

#: 🔴 How far above that floor to assume when nothing measured this machine: the real host cost is
#: about 2x the floor on this path and about 6x on the alignment one, neither overshoot with a name.
#: Without it, at a fraction of 0.5 a budget from the bare floor spends all of the container at k=2.
HOST_FLOOR_OVERSHOOT = 8

#: What the calibration measured one padded second costing the host, or None when it could not be
#: taken. 🔴 Assigned by `_warmup`; unwritten for one commit, the `or` below then faked a reading.
_host_bytes_a_padded_second = None

#: Calls the smoothing remembers: long enough that one odd call does not move the budget,
#: short enough to follow a neighbour arriving on the card.
SCALE_MEMORY = 8

#: 🔴 Outside this the model is wrong about SHAPE, not scale. A factor of two each way, because that
#: is the size of its one assumption: that a group is padded up to its longest member. Stop padding
#: and a group costs about `sum`; pad to a fixed window and it climbs. Either way: read the library.
SCALE_BAND = (0.5, 2.0)

#: How measured cost compares with predicted. One scalar, folded on every call.
_scale = 1.0
#: 🔴 Two counters, always reported together: the reading is the process's all-time high, so only
#: a call that raises it teaches anything, and `_scale_seen` is not the sample size it looks like.
_scale_seen = 0
_calls_seen = 0
#: What the process holds with no request in flight, measured once the model is loaded.
_resting_bytes = None
#: Padded seconds a call was refused at, if one ever was. A ceiling, not a budget: the refusal is
#: the one reading this model cannot produce, since a call that died allocated nothing to measure.
_refused_above = None
#: When that refusal happened, on the monotonic clock. 🔴 Monotonic, not the wall clock: a stepped
#: container would hold the ceiling for the length of the step, or drop it at once.
_refused_at = 0.0
_budget_said = False


def _say_batching():
    """Which sizing is in force, and anything wrong with the way it was asked for. 🔴 The note is
    the only trace of a flag given and not honoured; said at load, which every deployment hits."""
    for note in _BATCH_NOTES:
        _p(note)
    if not GROUPING:
        _p("batching: one span a call, because --batch-max-spans is 1. Any larger count, or "
           "none at all, sizes a batch from what this machine measures")
        return
    _p("batching: on -- %g of this container's GPU grant, solved against the cost measured at "
       "warmup and said on the first batched request%s"
       % (BUDGET_FRACTION,
          "" if MAX_SPANS is None else ", and never more than %d spans" % MAX_SPANS))


def _memory_reading():
    """(allocated now, peak so far) in bytes, or None where there is no counter to read."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _reset_peak():
    """Put the all-time high back to what is allocated right now. 🔴 Once at load and nowhere
    else: the calibration would read zero against the model load, and per call it breaks a bench."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ⚠️ Two configurations measured in one process both read the larger of the two; it is not a
# per-request figure.
def _peak_bytes():
    """The all-time high since load, or None when there is no counter. 🔴 The high-water mark, not
    this call's increment, and the only way out -- `/metrics` reports the CARD (`wrapper.gpu`)."""
    reading = _memory_reading()
    return None if reading is None else reading[1]


def _used(before, peak_before, peak_after):
    """What this call added, or zero when it never rose above an older peak. Split out from the
    counter so the judgement is testable without a card, where reading and rule failed together."""
    if peak_after <= peak_before:
        return 0
    return max(0, peak_after - before)


_band_said = False


def _observe(padded_seconds, used_bytes):
    """Fold one call's measured cost into the running correction. 🔴 Only a call that RAISES the
    all-time peak is seen, so calls getting dearer is caught and calls getting cheaper is not."""
    global _scale, _scale_seen, _calls_seen
    _calls_seen += 1
    if padded_seconds <= 0 or used_bytes <= 0:
        return
    # 🔴 No guard on the product: the factor is a positive module constant, so the only way it can
    # be zero is the seconds, which is checked above. A second check read as care and was dead.
    ratio = used_bytes / (padded_seconds * OPENING_BYTES_A_PADDED_SECOND)
    _scale_seen += 1
    weight = 1.0 / min(_scale_seen, SCALE_MEMORY)
    _scale = (1.0 - weight) * _scale + weight * ratio
    global _band_said
    if not SCALE_BAND[0] <= _scale <= SCALE_BAND[1] and not _band_said:
        # 🔴 Once. This runs per call, and a machine genuinely outside the band emits one line per
        # group of a forty minute meeting, burying the one-shot lines the sizing depends on.
        _band_said = True
        _p("WARN batch sizing is off by %.2fx over %d of %d calls. The cost model assumes a "
           "group is padded up to its longest member; if that stopped being true, re-read "
           "the processor rather than trusting this budget" % (_scale, _scale_seen, _calls_seen))


def _bytes_a_padded_second():
    return OPENING_BYTES_A_PADDED_SECOND * max(_scale, 0.05)


#: HAMi's unit letters, as its own parser reads them. 🔴 The readers below copy HAMi's reading of
#: its own variable, not a reasonable one: each difference is a factor in the over-sizing direction.
_HAMI_UNITS = {"g": 2 ** 30, "m": 2 ** 20, "k": 2 ** 10}
_hami_said = False


# ⚠️ No sign handling, deliberately: C's strtoul accepts one, and what HAMi does with `"-5g"` is
# pinned by a check written against the real thing rather than re-derived from the standard.
def _strtoul_base0(text):
    """The leading integer as C's `strtoul(s, end, 0)` reads it, or None. 🔴 Base ZERO: a leading
    `0` is octal there and `0x` is hex, so `"04096m"` is 32 MiB to HAMi, not 4096 MiB. 128x."""
    body = text.lstrip()
    if body[:2].lower() == "0x":
        digits, base = "0123456789abcdef", 16
        body = body[2:]
    elif body[:1] == "0":
        digits, base = "01234567", 8
        # The leading zero is itself the value when nothing octal follows, as strtoul has it.
        body = body[1:] or "0"
    else:
        digits, base = "0123456789", 10
    # ⚠️ Trailing text is no error to strtoul: it stops at the first non-digit, so `"4096m "`
    # reads as 4096. Digits matched by hand -- `str.isdigit` is true for scripts `int()` refuses.
    run = ""
    for ch in body:
        if ch.lower() in digits:
            run += ch
        else:
            break
    if not run:
        # A bare "0" prefix that led nowhere is still the number zero to strtoul.
        return 0 if base == 8 else None
    return int(run, base)


# 🔴 A different number from `REQUIRED_GPU_MEMORY`, and the one that actually refuses: on the gb10
# profile the declared figure is absent and the discrete-GPU value comes through in its place.
def _hami_limit_bytes():
    """HAMi's ENFORCED per-container figure in bytes; 0 when there is no limit. 🔴 0 is a VALUE --
    hami-core treats it as no limit, and one of this project's machines is set exactly that way."""
    global _hami_said
    # 🔴 The per-device variable first, then the UNINDEXED one, which hami-core falls back to for
    # every device without its own; reading only `_0` makes a configured deployment look unenforced.
    name = "CUDA_DEVICE_MEMORY_LIMIT_0"
    raw = os.environ.get(name, "") or ""
    if not raw:
        name = "CUDA_DEVICE_MEMORY_LIMIT"
        raw = os.environ.get(name, "") or ""
    if not raw:
        return 0
    # 🔴 NOT stripped first: HAMi indexes the last character of whatever the environment holds, so
    # `"4096m "` is 4096 BYTES there and would be 4 GiB here.
    scalar = _HAMI_UNITS.get(raw[-1].lower(), 1)
    head = (raw[:-1] if scalar != 1 else raw).lstrip()
    value = _strtoul_base0(head)
    if value is not None:
        return value * scalar
    if not _hami_said:
        _hami_said = True
        _p("WARN %s is set to %r, which HAMi's own parser reads as no number at all, so it "
           "is being treated as no limit. If a limit really is enforced, every batch from "
           "here is sized against a figure nothing will hold it to" % (name, raw))
    return 0


def _held_bytes():
    """What this process holds on the card right now, or None when there is none. 🔴 Live, not
    `_resting_bytes`, which is a load-time snapshot and would over-size by everything since."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


def _cached_bytes():
    """Blocks the allocator has bought and freed: spendable again without asking anyone."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return max(0, int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated()))
    except Exception:
        return 0


def _headroom_bytes():
    """Bytes a batch may be sized against, or None when nothing could be read: the platform's GRANT
    minus what this process holds RIGHT NOW. 🔴 The grant, not the card: a quota is what binds."""
    from ..gpu import quota_mib

    if _resting_bytes is None:
        return None
    held = _held_bytes()
    if held is None:
        return None
    hami = _hami_limit_bytes()
    if hami:
        # 🔴 A published limit does not mean the counters were rewritten to respect it (the
        # interposer's meminfo hook is compile-time). A total at or under it can only be that.
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
            if int(total) <= hami:
                # 🔴 Plus the cache: the interposer charges the allocation, not the use, so blocks
                # torch has already bought and freed are spendable again without asking it at all.
                return int(free) + _cached_bytes()
        except Exception:
            pass
        return max(0, hami - held)
    quota = quota_mib() * (2 ** 20)
    if quota <= 0:
        # 🔴 No grant means no automatic sizing at all, rather than sizing off the card: what is
        # free there is what the neighbours have not claimed YET, on a card that may not be ours.
        return None
    # A declared figure with nothing enforcing it. Still better than the card: a number somebody
    # chose for this container, rather than one that counts a neighbour's memory as free.
    return max(0, quota - held)


# ⚠️ On a card shared in software, what the card cannot give is host-backed and charged here, in
# no process's RSS. It barely moves with batch size: 2.243 GB at six spans a call, 2.250 at twelve.
def _host_budget():
    """Padded seconds the container's own memory account can afford, or None. 🔴 The account that
    KILLS -- OOMKill, not a catchable exception -- so it is priced apart and the smaller wins."""
    per_second = (_host_bytes_a_padded_second
                  or HOST_FLOOR_BYTES_A_PADDED_SECOND * HOST_FLOOR_OVERSHOOT)
    return grouping.budget_from_bytes(cgroup.headroom(cgroup.read()), per_second,
                                      fraction=BUDGET_FRACTION)


#: The headroom, and the two sides of the budget solved from it. 🔴 Kept so the health line quotes
#: the readings its own numbers came from: re-solving inside the line reads the card a second time.
_last_headroom = None
_last_gpu_budget = None
_last_host_budget = None


def _solve_budget():
    """Padded seconds one call may carry, None when nothing could be read, 0 when spent."""
    global _last_headroom
    _last_headroom = _headroom_bytes()
    budget = grouping.budget_from_bytes(_last_headroom, _bytes_a_padded_second(),
                                        fraction=BUDGET_FRACTION)
    if budget is None:
        return None
    ceiling = _refusal_ceiling()
    if ceiling is not None:
        # 🔴 A refusal outranks the arithmetic: it is the one reading this model cannot make for
        # itself, since a call that died allocated nothing to measure.
        budget = min(budget, ceiling)
    return budget


# ⚠️ Called under the same lock as inference, so nothing is mid-allocation when it runs.
def _drop_cache():
    """Hand the allocator's cached blocks back before retrying. 🔴 A fragmented cache would fail
    the retry for a SECOND reason; they are a neighbour's: one request went 8290 -> 14460 MiB."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


#: How long a refusal's ceiling stands before the full budget is tried again. 🔴 Policy, not a
#: measurement; a wrong retry costs ONE refused call, so the bound is one wasted call a minute.
RECOVERY_SECONDS = 60.0
#: How many times the wait may double. 🔴 The wait is what tells the two causes apart: a neighbour
#: is gone by the first retry, a wrong cost model refuses again every time the ceiling lifts.
RECOVERY_DOUBLINGS_MAX = 5
#: Refusals since a call at or above the refused size last succeeded. Not "refusals ever": a full
#: size call fitting is the evidence that the cause was not us, and it puts the wait back.
_refusal_streak = 0
#: The size a later call has to reach to be that evidence. 🔴 Separate from `_refused_above`, and
#: the reason is that the two are never set at the same time: while the ceiling stands the budget
#: is HALF of it, so no call can reach the refused size, and once it lifts `_refused_above` is
#: gone. Reading the streak off the ceiling meant it could only ever grow.
_streak_target = None


def _recovery_seconds():
    """How long the current ceiling stands. 🔴 Reads the streak rather than counting calls: two
    refusals inside one request are one wrong budget, not two."""
    return RECOVERY_SECONDS * (2 ** min(max(_refusal_streak - 1, 0), RECOVERY_DOUBLINGS_MAX))


def _note_refusal(padded_seconds):
    """Remember that a call this big was refused, so later groups stay under it."""
    global _refused_above, _refused_at, _refusal_streak, _streak_target
    if padded_seconds and padded_seconds > 0:
        # 🔴 The streak counts ceilings that did not hold, not refusals: a re-plan inside one
        # request refuses again under the SAME ceiling, and that is one piece of evidence.
        if _refused_above is None:
            _refusal_streak += 1
            _streak_target = padded_seconds
        _refused_above = (padded_seconds if _refused_above is None
                          else min(_refused_above, padded_seconds))
        _refused_at = time.monotonic()


def _note_fit(padded_seconds):
    """A call this big went through. 🔴 Only a call at or above the size that was refused clears
    the streak -- smaller ones fitting is what the ceiling was for, so it proves nothing. That
    size outlives the ceiling on purpose: under the ceiling nothing can reach it."""
    global _refusal_streak, _streak_target
    if _streak_target is not None and padded_seconds >= _streak_target:
        _refusal_streak, _streak_target = 0, None


# ⚠️ The clearing lives in a reader because that is the one place guaranteed to run: the
# budget is re-solved on every request and on every re-plan within one.
def _refusal_ceiling():
    """Half the smallest size the card refused, until that evidence ages out. 🔴 Expiring CLEARS
    it rather than ignoring it: left in the `min`, an old transient outranks a fresh refusal."""
    global _refused_above
    if _refused_above is None:
        return None
    if time.monotonic() - _refused_at >= _recovery_seconds():
        _refused_above = None
        return None
    return _refused_above / 2.0


def _budget_now():
    """The memory bound on one call, in padded seconds. Says nothing, changes nothing.
    🔴 A span ceiling does not turn this off: a count is not a memory bound, so letting one
    REPLACE the measurement is how a flag makes a call bigger."""
    if not GROUPING:
        return None
    global _last_gpu_budget, _last_host_budget
    budget = _last_gpu_budget = _solve_budget()
    host = _last_host_budget = _host_budget()
    # 🔴 The host may only LOWER a budget, never stand in for one: standing in put 58 spans into
    # one generate() on a card with no grant. `is not None`: zero is the container AT its limit.
    if budget is not None and host is not None:
        budget = min(budget, host)
    return budget


def _effective_budget():
    """The budget in force for one group, and one line the first time it is known. 🔴 Both are
    said: "nothing was printed" looks exactly like "nobody went looking"."""
    global _budget_said
    budget = _budget_now()
    if not GROUPING:
        return budget
    if not _budget_said:
        _budget_said = True
        if budget is None:
            # 🔴 Which of the two: no counter is a machine that cannot be measured at all and no
            # flag fixes it; no grant is a declaration somebody can go and add to the chart.
            if _resting_bytes is None or _held_bytes() is None:
                _p("WARN batching has nothing to size from -- this process has no "
                   "per-process memory counter, so nothing about this machine can be "
                   "measured and every call carries one span. 🔴 Not a fault and not "
                   "something a flag fixes: it is a deployment with no CUDA device this "
                   "engine can read, and there is no flag that sets a size by hand -- a "
                   "size this engine cannot check is a size nobody should be able to write")
            else:
                _p("WARN batching has nothing to size from -- this container declares "
                   "no GPU quota (REQUIRED_GPU_MEMORY), so every call carries one span. 🔴 "
                   "This is not a fault: sizing off what the card reports free would be "
                   "spending memory the neighbours have not claimed yet, on a card this "
                   "engine cannot tell apart from one it has to itself. A deployment that "
                   "knows the card is its own says so by declaring REQUIRED_GPU_MEMORY, "
                   "which is also what tells the platform to reserve it")
        elif budget <= 0:
            # 🔴 Which account is at zero: a full card frees when a neighbour lets go, a container
            # at its cgroup limit is the one that OOMKills this process if anything here guesses.
            _p("batching: %s, so one span a call is the answer rather than a bigger "
               "batch. Not a fault in the cost model, and it lifts when whatever holds the "
               "memory lets go"
               % ("the container is at its memory limit" if _last_host_budget == 0
                  else "the grant is readable and already spent"))
        else:
            # 🔴 Every factor of the GPU side so a reader can multiply it back out, and then the
            # container's number BESIDE it: quoting the GPU arithmetic alone overstated by 3.5x.
            host, gpu = _last_host_budget, _last_gpu_budget
            spans = "" if MAX_SPANS is None else " and at most %d spans" % MAX_SPANS
            if gpu is None:
                # 🔴 A ceiling standing on its own. Nothing here was measured, so there is no
                # arithmetic to print, and printing the shape of one would invent it.
                _p("batching: nothing on this machine could be measured, so the %g padded "
                   "audio second ceiling stands on its own%s. That is a number somebody "
                   "chose, not one this engine checked against the card" % (budget, spans))
                return budget
            _p("batching: up to %g padded audio seconds a call%s. The "
               "GPU side solves to %g, from a %.0f MiB grant at %.2f MiB a padded second, "
               "spending %g of it (correction %.2f over %d of %d calls); the container's "
               "account %s"
               % (budget, spans, gpu, (_last_headroom or 0) / (2 ** 20),
                  _bytes_a_padded_second() / (2 ** 20), BUDGET_FRACTION,
                  _scale, _scale_seen, _calls_seen,
                  "could not be read, so it bounds nothing" if host is None
                  else ("solves to %g and is what binds" % host if host < gpu
                        else "solves to %g, which is the looser of the two" % host)))
    return budget


#: The calibration shape: two equal clips of this length in one call. 🔴 Two, not one -- one span
#: exercises no padding; equal so the pad is empty; 30 s is what the diarizer ahead of this emits.
CALIBRATION_SPAN_SEC = 30.0
CALIBRATION_SPANS = 2


def _tone(seconds):
    import numpy as np

    # A voiced-band tone, not silence: the encoder may skip a silent clip and warm nothing.
    n = int(seconds * 16000)
    return (0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000.0)).astype("float32")


def _spent(snapshot):
    """What the container is using that it cannot simply give back, or None."""
    cur = snapshot.get("current")
    if cur is None:
        return None
    return cur - (snapshot.get("reclaimable") or 0)


def _measure_host(before, padded):
    """What one padded second cost the container, on the same quantity `cgroup.headroom` spends.
    🔴 At or below the arithmetic floor is NOT a measurement: returning the floor gave a container
    whose counter barely moved a bound 8x looser. ⚠️ One sample: page cache would ratchet a loop."""
    if padded <= 0:
        return None
    after = cgroup.read()
    lo, hi = _spent(before), _spent(after)
    if lo is None or hi is None:
        return None
    measured = (hi - lo) / float(padded)
    return measured if measured > HOST_FLOOR_BYTES_A_PADDED_SECOND else None


# 🔴 This is also the calibration, and it runs BEFORE ready: 503 keeps every other caller off
# the card, so nothing races it for the memory it is measuring.
def _warmup():
    """One throwaway transcription before we report ready, and the measurement that sizes every
    batch after it. Cold start was 74s against 0.3s once warm; llm-init allows an upstream 60s."""
    # 🔴 FIRST, before anything that could raise: a peak left at the model-load high-water mark
    # makes `_used` answer zero for every later call while the log promises a measurement.
    _reset_peak()
    # ⚠️ Non-fatal in every direction, enforced here: the guard used to cover only the
    # transcription, so an OSError from a reading left `ready` unset for the process's life.
    try:
        _measure_at_load()
    except Exception as e:
        _p("WARN warmup could not finish measuring (%s: %s); the engine is serviceable and "
           "batch sizing keeps the correction it had" % (type(e).__name__, e))


def _measure_at_load():
    """Read, calibrate, observe. Every exit here is a log line, not a dead engine."""
    global _resting_bytes, _host_bytes_a_padded_second

    t0 = time.time()
    reading = _memory_reading()
    host_before = cgroup.read()
    calibrating = reading is not None
    if calibrating:
        # At rest means before the calibration call, not before the model: the weights are
        # what this process holds whether or not anyone is talking to it.
        _resting_bytes = reading[0]
    try:
        if calibrating:
            clips = [_tone(CALIBRATION_SPAN_SEC)] * CALIBRATION_SPANS
            _offline_transcribe_many(clips)
        else:
            # No counter to read, so nothing to calibrate: warm on the cheapest shape there
            # is. This is also the CPU path, where a 60 second shape is minutes of warmup.
            _offline_transcribe(_tone(1.0))
        took = time.time() - t0
    except Exception as e:
        # Serviceable either way. 🔴 But the blocks go back FIRST: a calibration that died of OOM
        # is when the caching allocator holds the most, and holds it until the process exits.
        if calibrating:
            _drop_cache()
        _p("WARN warmup transcription failed after %.0fs: %s" % (time.time() - t0, e))
        return
    if not calibrating:
        _p("warmup transcription took %.0fs; no per-process memory counter, so batch sizing "
           "has nothing measured on this machine and the first request will say so" % took)
        return
    after = _memory_reading()
    used = _used(reading[0], reading[1], after[1]) if after is not None else 0
    padded = CALIBRATION_SPANS * CALIBRATION_SPAN_SEC
    # 🔴 Before the GPU verdict below: the two accounts are measured independently, and taking
    # this after the `used <= 0` return tied "host never measured" to "GPU peak did not rise".
    _host_bytes_a_padded_second = _measure_host(host_before, padded)
    # 🔴 After `_measure_host`: on a software-shared card these blocks are host-backed and charged
    # to this cgroup. Before the `used <= 0` return: that branch means the counter did not SEE it.
    _drop_cache()
    if used <= 0:
        # 🔴 Not "calibrated, and the answer was zero" -- no sample was taken at all, and the
        # correction stays whatever it was. The two look the same in a log that prints a number.
        _p("WARN warmed in %.0fs but the calibration took no measurement: the peak did not "
           "rise above what was already allocated, so nothing was observed and the "
           "correction stays at %.2f, fitted elsewhere. Batch sizing is unchecked on this "
           "machine" % (took, _scale))
        return
    _observe(padded, used)
    _p("calibrated in %.0fs: %d x %.0fs in one call is %.0f padded seconds, the model said "
       "%.0f MB and it took %.0f MB, so the correction is %.2f"
       % (took, CALIBRATION_SPANS, CALIBRATION_SPAN_SEC, padded,
          padded * OPENING_BYTES_A_PADDED_SECOND / 1e6, used / 1e6, _scale))


def _decode_to_16k_mono(raw, filename):
    import tempfile
    import librosa

    suffix = os.path.splitext(filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        y, _sr = librosa.load(path, sr=16000, mono=True)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    return y.astype("float32")


def _detector_class():
    """The detector class this vLLM has, or None. Probed once, and says so once -- asked
    independently of the flags, and not at load, so a vLLM without the class still serves."""
    if not _repdet_class:
        try:
            from vllm.sampling_params import RepetitionDetectionParams

            _repdet_class.append(RepetitionDetectionParams)
        except Exception as e:
            # An older vLLM has no such class; this image serves without the detector.
            _p("WARN repetition detection unavailable in this vLLM (%s)" % e)
            _repdet_class.append(None)
    return _repdet_class[0]


def _repetition_params():
    """The configured detector, or None when it was not asked for or cannot run."""
    if not REPETITION_ON:
        return None
    if not _repdet_cache:
        cls = _detector_class()
        if cls is None:
            _repdet_cache.append(None)
        else:
            try:
                params = dict(REPETITION_DEFAULTS)
                if REPETITION_OVERRIDE:
                    params.update(json.loads(REPETITION_OVERRIDE))
                _repdet_cache.append(cls(**params))
                _p("repetition detection: %s%s"
                   % (params, " (overridden)" if REPETITION_OVERRIDE else ""))
            except Exception as e:
                # A malformed override is the operator's typo, not a reason to stop serving.
                _p("WARN --repetition-detection ignored (%s)" % e)
                _repdet_cache.append(None)
    return _repdet_cache[0]


def _apply_repetition(sp):
    """Sets the detector for one call and returns how to put the old value back, or None.
    None when nothing was set, so a `finally` cannot fail again and replace the transcript."""
    params = _repetition_params()
    if params is None:
        return None
    old = getattr(sp, "repetition_detection", None)
    try:
        sp.repetition_detection = params
    except Exception as e:
        # Once: this runs per generate() call, and a 40 minute meeting is hundreds of them.
        if not _repset_said:
            _repset_said.append(True)
            _p("WARN could not set repetition_detection (%s)" % e)
        return None

    def restore():
        sp.repetition_detection = old

    return restore


def _say_repetition_once():
    """One line saying which of the two is in force, and why the other is not. Without it the three
    states -- detector, fallback cap, nothing at all -- all look like an engine that transcribes."""
    if _repdet_said:
        return
    _repdet_said.append(True)
    # Probe only where the answer is used; _detector_class() warns when the class is missing.
    if _REP_NOTE:
        _p(_REP_NOTE)
    if not REPETITION_ON:
        if _FALLBACK_ASKED:
            _p("--repetition-fallback-tokens-per-sec=%d has no effect: it backs up "
               "--repetition-detection, which was not asked for"
               % REPETITION_FALLBACK_TOKENS_PER_SEC)
        return
    asr = _state.get("asr")
    if asr is not None and getattr(asr, "sampling_params", None) is None:
        _p("WARN this qwen-asr exposes no sampling_params: neither --repetition-detection nor "
           "--repetition-fallback-tokens-per-sec can be applied, whatever they are set to")
        return
    if _detector_class() is not None:
        # Asking for the params, not just the class: a rejected override leaves the detector off.
        if _repetition_params() is None:
            _p("WARN --repetition-detection was asked for and is NOT running: this build has "
               "the detector but rejected the settings (see the line above). The fallback "
               "does not step in either, because a length cap truncates real speech")
            return
        if _FALLBACK_ASKED:
            _p("--repetition-fallback-tokens-per-sec=%d has no effect: this build has the "
               "detector, which ends a span for repeating rather than for being long"
               % REPETITION_FALLBACK_TOKENS_PER_SEC)
        return
    if REPETITION_FALLBACK_TOKENS_PER_SEC <= 0:
        _p("WARN repetition fallback off (--repetition-fallback-tokens-per-sec=0) and this "
           "build has no detector: a repeating span is bounded only by max_tokens=%d, and "
           "with spans batched the rest of its group waits on it" % OFFLINE_MAX_TOKENS)
        return
    _p("repetition fallback: no detector in this build, capping output at %d tokens per "
       "audio second (measured speech is about 3.4; too low truncates, and truncation is "
       "not visible here)%s" % (
           REPETITION_FALLBACK_TOKENS_PER_SEC,
           # One SamplingParams covers a whole generate(); in a batch the cap follows the longest clip.
           ". With spans grouped, the cap follows the LONGEST clip in each group, "
           "so a short span that starts repeating is bounded by that clip's budget, not its own"
           if GROUPING else ""))


def _token_budget(seconds):
    # Replace the stock budget only when there is no detector and fallback tokens/sec > 0.
    _say_repetition_once()
    if not REPETITION_ON or _detector_class() is not None:
        return OFFLINE_MAX_TOKENS
    if REPETITION_FALLBACK_TOKENS_PER_SEC <= 0:
        return OFFLINE_MAX_TOKENS
    return max(TOKENS_FLOOR,
               min(OFFLINE_MAX_TOKENS,
                   int(seconds * REPETITION_FALLBACK_TOKENS_PER_SEC) + TOKENS_FLOOR))


# HTTP speaks ISO-639-1 ("zh"), the library full names ("Chinese"), and normalize_language_name()
# only fixes casing. 🔴 This table is also the whole set served, so it is what "unknown" means.


# Re-check against upstream when the model is bumped (30 entries as of Qwen3-ASR-1.7B):
#   python -c "from qwen_asr.inference.utils import SUPPORTED_LANGUAGES as L; print(L)"
_ISO_TO_LANGUAGE = {
    "zh": "Chinese", "cmn": "Chinese", "yue": "Cantonese", "en": "English",
    "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean",
    "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "fil": "Filipino", "tl": "Filipino", "fa": "Persian",
    "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}
LANGUAGE_NAMES = frozenset(_ISO_TO_LANGUAGE.values())


def resolve_language(language):
    """An ISO code or a full name -> the canonical name the library wants; None = auto. Raises
    ValueError on neither, with no opinion on what a request does. Region subtags are dropped."""
    if language is None or not str(language).strip():
        return None
    raw = str(language).strip()
    key = raw.lower()
    name = (_ISO_TO_LANGUAGE.get(key)
            or _ISO_TO_LANGUAGE.get(key.replace("_", "-").split("-")[0]))
    if name is None:
        # Not a code we map, so read it as a full name; this is the same canonical shape
        # normalize_language_name() produces ("cHINese" -> "Chinese").
        name = raw[:1].upper() + raw[1:].lower()
    if name not in LANGUAGE_NAMES:
        raise ValueError("unsupported language: %s. Supported: %s"
                         % (raw, ", ".join(sorted(LANGUAGE_NAMES))))
    return name


def resolve_language_or_auto(language, where):
    """The same answer, but an unservable language falls back to auto rather than failing: it is a
    hint in this contract, and every reply carries the one it got. Logged: no other trace of it."""
    try:
        return resolve_language(language)
    except ValueError as e:
        log.warning("%s: %s; falling back to auto-detect, and the reply says which "
                    "language the transcript is actually in", where, e)
        return None


def _text_and_language(r):
    """(text, language) off one ASRTranscription, tolerating a plain dict. `language` is what the
    model reported: a forced one comes back unchanged, an auto-detected one comes back filled in."""
    if r is None:
        return "", ""
    t = getattr(r, "text", None)
    lang = getattr(r, "language", None)
    if isinstance(r, dict):
        if t is None:
            t = r.get("text")
        if lang is None:
            lang = r.get("language")
    return (t or "").strip(), (lang or "").strip()


def _offline_transcribe(audio, language=None):
    # Native offline transcription on the same load; max_tokens is raised then restored.
    asr = _state["asr"]
    sp = getattr(asr, "sampling_params", None)
    old = getattr(sp, "max_tokens", _MISSING) if sp is not None else _MISSING
    old_n = getattr(asr, "max_new_tokens", _MISSING)
    restore_rep = None
    budget = _token_budget(len(audio) / 16000.0)
    try:
        if sp is not None:
            sp.max_tokens = budget
            restore_rep = _apply_repetition(sp)
        if old_n is not _MISSING:
            asr.max_new_tokens = budget
        results = asr.transcribe(audio=(audio, 16000), language=language,
                                 return_time_stamps=False)
    finally:
        if old_n is not _MISSING:
            asr.max_new_tokens = old_n
        if sp is not None:
            if old is not _MISSING:
                sp.max_tokens = old
            else:
                # The attribute did not exist and the try created it; delattr instead of restoring.
                try:
                    delattr(sp, "max_tokens")
                except Exception:
                    pass
        if restore_rep is not None:
            restore_rep()
    return _text_and_language(results[0] if results else None)


def _offline_transcribe_many(clips, language=None):
    # qwen-asr's transcribe() takes a list and hands the whole list to the engine in one generate().
    asr = _state["asr"]
    sp = getattr(asr, "sampling_params", None)
    old = getattr(sp, "max_tokens", _MISSING) if sp is not None else _MISSING
    old_n = getattr(asr, "max_new_tokens", _MISSING)
    restore_rep = None
    budget = _token_budget(max(len(c) for c in clips) / 16000.0)
    try:
        if sp is not None:
            # One SamplingParams covers the whole call, so the budget follows the longest clip.
            sp.max_tokens = budget
            restore_rep = _apply_repetition(sp)
        if old_n is not _MISSING:
            asr.max_new_tokens = budget
        results = asr.transcribe(audio=[(c, 16000) for c in clips],
                                 language=language, return_time_stamps=False)
    finally:
        if old_n is not _MISSING:
            asr.max_new_tokens = old_n
        if sp is not None:
            if old is not _MISSING:
                sp.max_tokens = old
            else:
                # The attribute did not exist and the try created it; delattr instead of restoring.
                try:
                    delattr(sp, "max_tokens")
                except Exception:
                    pass
        if restore_rep is not None:
            restore_rep()
    out = [_text_and_language(r) for r in (results or [])]
    if len(out) != len(clips):
        raise RuntimeError("transcribe returned %d results for %d clips"
                           % (len(out), len(clips)))
    return out


_long_span_said = False


def _say_if_a_span_outruns_the_model(spans):
    """Say, once, when a span outruns MAX_ASR_INPUT_SECONDS: qwen-asr splits it and hands ALL the
    pieces to one padded call, so this model UNDER-counts it. Not corrected; `_observe` sees it."""
    global _long_span_said
    if _long_span_said:
        return
    longest = max((s for _i, _c, s in spans), default=0.0)
    if longest <= OFFLINE_MAX_INPUT_SEC:
        return
    _long_span_said = True
    _p("WARN a %.0fs span is past the %ds this model takes in one piece, so qwen-asr will "
       "split it and the pieces are padded together. Batch sizing counts it as one clip "
       "and therefore UNDER-counts what the call costs; the correction measured on real "
       "calls is what absorbs it" % (longest, OFFLINE_MAX_INPUT_SEC))


def _plan_groups(spans):
    """[(index, clip, seconds)] -> the groups, one call each, and the line that says how."""
    _say_if_a_span_outruns_the_model(spans)
    budget = _effective_budget()
    # 🔴 The reading the budget was solved from, not a second one. Calling `_host_budget()` again
    # reads the cgroup a second time, and the line below would then name an account that is not
    # the one the plan was made against -- which is the shape an existing test already catches
    # one level up, in the health line.
    host = _last_host_budget
    if not budget or budget <= 0:
        return ([[one] for one in spans], "one span a call")
    packed = grouping.pack([(k, one[2]) for k, one in enumerate(spans)], budget, MAX_SPANS)
    # 🔴 %g, not %.0f: a 0.6 second budget printed as "1" rounds a number nobody can check into
    # one they can.
    how = "%g padded seconds a call" % budget
    if _scale_seen == 0:
        # 🔴 The groups this produces are NOT evidence about the card: nothing here has confirmed
        # the factory figure, so a small group says the opening guess was small, not the card full.
        how += " (nothing on this machine has been measured yet; the factory figure stands)"
    if host is not None and budget >= host:
        # Which account the number came from: a GPU bound that binds is a slower engine, a host
        # bound that binds is how far this container is from being killed.
        how += " (bounded by the container's memory, not the GPU's)"
    if MAX_SPANS is not None:
        how += ", at most %d spans" % MAX_SPANS
    groups = [[spans[k] for k in g] for g in packed]
    if _make_room_for_an_unsplittable(groups, budget):
        how += ", and the allocator's cache was handed back first"
    return (groups, how)


def _make_room_for_an_unsplittable(groups, budget):
    """Give the cache back before a group that is one span and costs more than the budget.

    🔴 The one shape nothing here can shrink. `group_capacity` answers 1 for a single span
    whatever it costs, so a span longer than the budget goes to the card whole -- measured at
    596 padded seconds against a budget of 33, and the container was OOMKilled twenty-seven
    seconds later, taking every in-flight request. That death is not catchable, which is the
    asymmetry that earns this call.

    The cache is what we hand back because it is held and not used: blocks the allocator bought
    and freed, charged to this cgroup under `kernel`, which the kernel cannot reclaim on its own.
    Measured at 3.3-3.7 GB on two machines; dropping it took that container from 15.07 GB to
    about 11.7, the difference between the span fitting and the container dying.

    ⚠️ The reason above is the container's, but the condition tests the budget in force, the
    SMALLER of the two accounts. Deliberate: where the GPU account binds, handing the cache back
    frees card memory too. It buys something on both sides; it only SAVES a container on one.

    🔴 Not a tuned number -- a group costing more than the budget the planner just solved, both
    numbers already in hand. Only one shape satisfies it: `pack` builds every other group to fit,
    so an over-budget group is always a single span it could not divide. Rare, so `empty_cache()`
    is paid only where the alternative is exit 137.

    ⚠️ An earlier version also tested `len(group) == 1`. It could not go red under mutation
    because it is implied, and a condition that cannot fail is not a condition.

    ⚠️ It does not make the span safe. A half-hour span wants about 11 GB in one block while the
    cache is 3.3-3.7 GB, so THE SPAN still fails -- which is not what this prevents, and the two
    must not be read as one: a failed span is an entry in `results` under a 200, a dead container
    is exit 137. Measured after the fix, on the machine where that death was reproduced: the same
    1800 s span came back as an error on that span alone, 8 seconds, container alive, because the
    cache going back first sent the allocation into the card's limit, which raises, rather than
    the cgroup's, which kills.

    ⚠️ One observation, not a bound: nothing here proves a large enough span cannot reach the
    cgroup first. The fix for THAT is a caller that does not send it.

    ⚠️ The plan is NOT re-solved against the memory this frees. Freeing then planning bigger is
    the opposite of what the freeing was for; the conservative groups already decided are what runs.

    ⚠️ Lock: `_plan_groups` runs inside `_work_batch`, which `tasks.dispatch` gates on `_gpu`, so
    nothing is mid-allocation here -- the condition `_drop_cache` documents. Checked rather than
    assumed, because grid I put four requests in flight at once.
    """
    for group in groups:
        if grouping.padded_seconds([s for _i, _c, s in group]) > budget:
            _drop_cache()
            return True
    return False


def _is_oom(e):
    """Whether the card refused this call, as opposed to the engine disliking the request. 🔴 It
    decides whether the budget moves; a message is the only handle torch keeps across versions."""
    if type(e).__name__.lower().replace("_", "").startswith("outofmemory"):
        return True
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _call_group(group, language=None):
    """One generate() for a whole group, and the measurement that sizes the next one."""
    reading = _memory_reading()
    pairs = _offline_transcribe_many([c for _i, c, _s in group], language=language)
    padded = grouping.padded_seconds([s for _i, _c, s in group])
    # 🔴 Before the measurement and not inside it: a machine with no counter still has to be able
    # to clear a streak, or its ceiling doubles away from a card that is perfectly healthy.
    _note_fit(padded)
    after = _memory_reading() if reading is not None else None
    if after is not None:
        _observe(padded, _used(reading[0], reading[1], after[1]))
    return pairs


# ⚠️ It re-plans everything outstanding, halves left by an earlier bisection included, so a
# request narrowing down a span the engine will not take can have it put back with company.
def _smaller_plan(todo, refused):
    """A re-plan of everything still outstanding, but only if it is actually smaller: (stack, how)
    or None. 🔴 This structural check, not the reasoning elsewhere, is what ends the retry."""
    rest = [one for g in todo for one in g] + list(refused)
    groups, how = _plan_groups(rest)
    # 🔴 The LARGEST group, not the first: `pack` sorts longest-first, so the first group holds the
    # FEWEST spans. It shrank while a later group of short spans grew -- the unpriced axis.
    if not groups or max(len(g) for g in groups) >= len(refused):
        return None
    groups = list(groups)
    groups.reverse()
    return groups, how


def _plan_shape(planned):
    """The plan as `count x longest`, and what it costs. 🔴 Cost, not size: the same 21 spans are
    2 calls whether the long one travels alone or drags eleven short ones; those differ by 5x."""
    shapes, total = [], 0.0
    for group in planned:
        seconds = [one[2] for one in group]
        longest = max(seconds) if seconds else 0.0
        total += grouping.padded_seconds(seconds)
        shapes.append("%dx%gs" % (len(group), longest))
    return " ".join(shapes), total


def _batching_report(how, span_count, planned, calls, splits, refusals, replans,
                     replanned_how=None):
    """The one line that says what the sizing did on this request. 🔴 Planned and made are both
    here -- a failed group is split and retried -- and nothing else anywhere reports either one."""
    shape, padded = _plan_shape(planned)
    line = ("batching: %d spans planned into %d calls (%s) as [%s] = %g padded seconds; "
            "%d calls made" % (span_count, len(planned), how, shape, padded, calls))
    if splits:
        line += ", %d groups split and retried" % splits
    if refusals:
        # %g: a ceiling of 0.4 printed as "0" reads as a process with no budget left. 🔴 Counted
        # apart from splits -- a split searches for a bad span, a re-plan is the request shrinking.
        line += (", the card refused %d call%s and the rest was re-planned %d time%s"
                 % (refusals, "" if refusals == 1 else "s",
                    replans, "" if replans == 1 else "s"))
        # 🔴 The ceiling reaches the budget only in the two modes that solve one; saying "groups
        # now stay under X" for a span count was untrue, and a smoke check had pinned the sentence.
        if GROUPING:
            # 🔴 `_budget_now` is the definition and this reads it. The copy written here went
            # stale, and the line then contradicted the size it quoted four words later.
            budget_now = _budget_now()
            if _refusal_ceiling() is None:
                line += (" -- that refusal's ceiling has already lifted, so the next call "
                         "is sized from the arithmetic again")
            elif budget_now is None:
                line += (" -- nothing on this machine can be sized from, so every call "
                         "carries one span and the ceiling bounds nothing")
            else:
                # 🔴 `_recovery_seconds()`, not the constant. After a second refusal the ceiling
                # stands for twice as long, and quoting 60 here told an operator to come back at a
                # minute for a process that would not be sized from the arithmetic again for two.
                # The backoff exists to be SEEN -- a growing wait is what tells a neighbour's
                # minute apart from a cost model that is wrong about this machine.
                line += (" -- the next call is sized at %g padded seconds, and that ceiling "
                         "lifts %ds after the refusal"
                         % (budget_now, int(_recovery_seconds())))
            # 🔴 And what that came out as: the ceiling above is half the size the card refused,
            # while the budget actually drawn against is the smaller of that and the arithmetic.
            if replanned_how:
                # The size only: everything after a `how`'s first clause is about the state of the
                # cost model, which this line already says once.
                line += ", re-planned at %s" % replanned_how.split(" (")[0]
        else:
            line += (" -- a span count is not re-solved, so nothing here lowers it and the "
                     "next request plans the same size again")
    line += ("; cost model %.2fx over %d of %d calls"
             % (_scale, _scale_seen, _calls_seen))
    peak = _peak_bytes()
    # 🔴 "unmeasured" rather than a zero -- no counter and peaked at nothing are different -- and
    # "since load", because `_reset_peak` runs once: this is the PROCESS's high, not this call's.
    line += ("; peak since load %s"
             % ("unmeasured" if peak is None else "%.0f MiB" % (peak / 2 ** 20)))
    return line


def build_app(supports):
    has_stt = "stt" in supports
    has_stream = "stt_stream" in supports
    app = FastAPI(title="audio-stt-stream (Qwen3-ASR)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="stt_stream", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=has_stt)

    if has_stt:
        @app.post("/v1/audio/transcriptions")
        async def transcriptions(file: UploadFile = File(...),
                                 model: str = Form(None),
                                 language: str = Form(None),
                                 response_format: str = Form("json"),
                                 segments: str = Form(None),
                                 async_: str = Form(None, alias="async")):
            if not _state["ready"]:
                raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
            # Resolved before any audio is read. A language this model cannot serve is not refused;
            # the `language` in the reply is what tells the caller its hint was not honoured.
            forced = resolve_language_or_auto(language, "POST /v1/audio/transcriptions")
            raw = await file.read()
            audio = await asyncio.to_thread(_decode_to_16k_mono, raw, file.filename)
            # BATCH mode (opt-in): `segments` JSON [{start,end}] — slice + transcribe each.
            if segments:
                segs = parse_segments(segments)

                def _work_batch(ctx):
                    started = time.monotonic()
                    ctx.progress(stage="transcribe", done=0, total=len(segs))
                    if GROUPING:
                        # Slice every span first, then hand them over in groups.
                        out = [None] * len(segs)
                        spans = []
                        for i, seg in enumerate(segs):
                            # Guarded per span so a bad start/end cannot fail the rest of the batch.
                            try:
                                lo = max(0, int(float(seg.get("start") or 0) * 16000))
                                hi = min(len(audio), int(float(seg.get("end") or 0) * 16000))
                            except Exception as e:
                                out[i] = {"error": "stt failed: %s" % e}
                                continue
                            if hi <= lo:
                                # `language` too: the serial path answers with it, so
                                # results[i]["language"] must not depend on which path ran.
                                out[i] = {"text": "", "language": ""}
                            else:
                                spans.append((i, audio[lo:hi], (hi - lo) / 16000.0))
                        done = len(segs) - len(spans)
                        # Metered once per span even when its group is retried in halves.
                        _metered = [False] * len(segs)
                        planned, how = _plan_groups(spans)
                        # 🔴 Kept because `how` is reassigned by a re-plan, while the report says
                        # the plan the SIZER chose -- the shape is what a reader takes for what ran.
                        first_how, replanned_how = how, None
                        calls = splits = refusals = replans = 0
                        # Work through the groups as a stack so a failed group is split and retried.
                        todo = list(planned)
                        todo.reverse()
                        while todo:
                            group = todo.pop()
                            # One generate() covers a whole group; checkpoint first so cancel is seen.
                            ctx.checkpoint()
                            # Metered here rather than while slicing: meter() is additive and groups retry.
                            for _i, _clip, _secs in group:
                                if not _metered[_i]:
                                    _metered[_i] = True
                                    ctx.meter(input_seconds=_secs)
                            try:
                                calls += 1
                                pairs = _call_group(group, language=forced)
                                # zip stops at the shorter side, so a short answer would silently drop spans.
                                if len(pairs) != len(group):
                                    raise RuntimeError(
                                        "engine returned %d results for %d spans"
                                        % (len(pairs), len(group)))
                                for (i, _, _s), (t, lang) in zip(group, pairs):
                                    out[i] = {"text": t, "language": lang}
                                done += len(group)
                            except tasks.Cancelled:
                                raise
                            except Exception as e:
                                moved = False
                                if _is_oom(e):
                                    refusals += 1
                                    _drop_cache()
                                    was = _effective_budget()
                                    _note_refusal(grouping.padded_seconds(
                                        [_s for _i, _c, _s in group]))
                                    now = _effective_budget()
                                    moved = (was is not None and now is not None
                                             and now < was)
                                # 🔴 Looks redundant, is not: `_smaller_plan`'s structural check
                                # is what terminates this, and `moved` is what buys the report.
                                smaller = (_smaller_plan(todo, group)
                                           if moved and len(group) > 1 else None)
                                if smaller:
                                    # 🔴 A refusal is about SIZE, so it reaches every group still
                                    # outstanding; gated on the budget MOVING, or a count loops.
                                    replans += 1
                                    todo, how = smaller
                                    replanned_how = how
                                    continue
                                if len(group) > 1:
                                    # Not a size failure: halving here is a search for the
                                    # one span the engine will not take, not a smaller batch.
                                    splits += 1
                                    mid = len(group) // 2
                                    todo.append(group[mid:])
                                    todo.append(group[:mid])
                                    continue
                                out[group[0][0]] = {"error": "stt failed: %s" % e}
                                done += 1
                            ctx.progress(done=done, total=len(segs))
                        _p(_batching_report(first_how, len(spans), planned, calls,
                                            splits, refusals, replans, replanned_how))
                        ctx.progress(done=len(segs), total=len(segs))
                        log.info(
                            "stt multi-span complete model=%s spans=%d speech_seconds=%.3f "
                            "inference_calls=%d split_retries=%d failed=%d "
                            "duration_seconds=%.3f",
                            MODEL_NAME, len(segs), sum(s for _, _, s in spans), calls, splits,
                            sum(1 for item in out if item and item.get("error")),
                            time.monotonic() - started)
                        return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}
                    out = []
                    speech_seconds = 0.0
                    inference_calls = 0
                    for i, seg in enumerate(segs, 1):
                        ctx.checkpoint()
                        try:
                            lo = max(0, int(float(seg.get("start") or 0) * 16000))
                            hi = min(len(audio), int(float(seg.get("end") or 0) * 16000))
                            if hi <= lo:
                                out.append({"text": "", "language": ""})
                            else:
                                seconds = (hi - lo) / 16000.0
                                speech_seconds += seconds
                                ctx.meter(input_seconds=seconds)
                                inference_calls += 1
                                t, lang = _offline_transcribe(audio[lo:hi], language=forced)
                                out.append({"text": t, "language": lang})
                        except tasks.Cancelled:
                            raise
                        except Exception as e:
                            out.append({"error": "stt failed: %s" % e})
                        finally:
                            ctx.progress(done=i, total=len(segs))
                    log.info(
                        "stt multi-span complete model=%s spans=%d speech_seconds=%.3f "
                        "inference_calls=%d split_retries=0 failed=%d max_batch_spans=1 "
                        "duration_seconds=%.3f",
                        MODEL_NAME, len(segs), speech_seconds, inference_calls,
                        sum(1 for item in out if item.get("error")), time.monotonic() - started)
                    return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}

                return await tasks.dispatch(async_, "stt", MODEL_NAME, _work_batch,
                                            fail="transcription failed", gate=_gpu)

            # SINGLE mode.
            def _work(ctx):
                ctx.meter(input_seconds=len(audio) / 16000.0)
                ctx.progress(ratio=0.0, stage="transcribe")
                text, lang = _offline_transcribe(audio, language=forced)
                ctx.progress(ratio=1.0, stage="done")
                if response_format in ("text", "srt", "vtt"):
                    return Response(content=text, media_type="text/plain")
                return {"text": text, "language": lang}

            return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                        fail="transcription failed", gate=_gpu)

    if has_stream:
        @app.websocket("/v1/audio/stream")
        async def stream(ws: WebSocket):
            import numpy as np

            await ws.accept()
            if not _state["ready"]:
                await ws.send_text(json.dumps({"type": "error",
                                               "detail": _state["error"] or "model not ready"}))
                await ws.close()
                return
            asr = _state["asr"]
            sample_rate = 16000
            step_ms = DEFAULT_STEP_MS
            language = None

            def _new_state():
                return _stream_init(asr, language=language)

            # prefix = text finalized by earlier rolls; samples resets on a roll, total never does.
            S = {"st": _new_state(), "prefix": "", "samples": 0, "total": 0}
            roll_samples = max(16000, int(ROLL_SEC * 16000))
            pending = np.zeros((0,), dtype="float32")
            await ws.send_text(json.dumps({"type": "ready"}))

            def _join(a, b):
                if not a:
                    return b
                if not b:
                    return a
                # Space only between two ASCII words (CJK needs none).
                if a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
                    return a + " " + b
                return a + b

            def _full_text():
                return _join(S["prefix"], getattr(S["st"], "text", "") or "")

            async def _emit(kind):
                await ws.send_text(json.dumps({
                    "type": kind,
                    "text": _full_text(),
                    "language": getattr(S["st"], "language", None) or language,
                }))

            async def _roll():
                # Fold the finalized text into prefix and start fresh, resetting encoder-cache use.
                async with _infer_lock:
                    await asyncio.to_thread(_gated, _stream_finish, asr, S["st"])
                S["prefix"] = _join(S["prefix"], getattr(S["st"], "text", "") or "")
                S["st"] = _new_state()
                S["samples"] = 0

            async def _feed(cur):
                # Backstop: if the cache overflows despite the proactive roll, roll and retry once.
                try:
                    async with _infer_lock:
                        await asyncio.to_thread(_gated, _stream_step, asr, cur, S["st"])
                except Exception as e:
                    msg = str(e).lower()
                    if "encoder cache" in msg or "exceeds" in msg or "pre-allocated" in msg:
                        log.warning("encoder-cache overflow; rolling session and retrying: %s", e)
                        await _roll()
                        async with _infer_lock:
                            await asyncio.to_thread(_gated, _stream_step, asr, cur, S["st"])
                    else:
                        raise
                S["samples"] += int(cur.shape[0])
                S["total"] += int(cur.shape[0])

            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    text = msg.get("text")
                    if text is not None:
                        try:
                            obj = json.loads(text)
                        except Exception:
                            obj = {}
                        t = obj.get("type")
                        if t == "start":
                            language = obj.get("language") or None
                            sample_rate = int(obj.get("sample_rate") or 16000)
                            step_ms = int(obj.get("step_ms") or DEFAULT_STEP_MS)
                            continue
                        if t in ("stop", "done", "finish"):
                            break
                        continue
                    data = msg.get("bytes")
                    if not data:
                        continue
                    seg = resample_linear(pcm16_to_float32(data), sample_rate)
                    pending = np.concatenate([pending, seg]) if pending.size else seg
                    step = max(1, int(round(step_ms / 1000.0 * 16000)))
                    while pending.shape[0] >= step:
                        cur, pending = pending[:step], pending[step:]
                        await _feed(cur)
                        await _emit("partial")
                        # Proactive roll at a safe point so we never approach the cap.
                        if S["samples"] >= roll_samples:
                            await _roll()
                            await _emit("partial")
                # flush tail + finalize
                if pending.size:
                    await _feed(pending)
                async with _infer_lock:
                    await asyncio.to_thread(_gated, _stream_finish, asr, S["st"])
                await _emit("final")
                # We consumed the audio, so the closing frame — not the caller — reports its length.
                await ws.send_text(json.dumps({
                    "type": "closed",
                    "audio_seconds": round(S["total"] / 16000.0, 3),
                }))
                await ws.close()
            except WebSocketDisconnect:
                pass
            except Exception as e:
                log.exception("stream error: %s", e)
                try:
                    await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                    await ws.close()
                except Exception:
                    pass

    return app


def run(supports):
    _p("stt_stream starting; model=%s port=%s supports=%s" % (MODEL_REPO, PORT, supports))

    def load():
        try:
            _load_blocking()
        except Exception as e:
            _state["error"] = hfgate.explain(MODEL_REPO, e)
            _p("engine load FAILED: %s" % e)
            log.exception("engine load failed: %s", e)

    def build(served):
        app = build_app(served)
        _p("starting uvicorn on :%s (ready=%s)" % (PORT, _state["ready"]))
        return app

    # No server-initiated WS keepalive: bursty inference lags Pong and drops a healthy session.
    _runtime.serve(
        supports,
        load,
        build,
        "qwen-asr",
        load_on_main=True,
        disable_ws_ping=True,
    )
