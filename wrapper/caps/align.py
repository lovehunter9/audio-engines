import os
import time
import collections
import unicodedata
import tempfile
import logging
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import hfgate
from .. import tasks
from .. import ovutil
from ..batch import parse_segments
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import spill, unlink
from ..limits import duration
from ..runtime import Runtime

log = logging.getLogger("audio-align")

_runtime = Runtime(model=None, device="cpu")
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

DEFAULT_LANGUAGE = "auto"

_args = EngineArgs()
# 5000 output classes on an 80 ms grid: past 400 s the argmax saturates and the
# timestamps are wrong rather than missing. _rebind_span_limit re-derives this from
# whatever checkpoint loads, because a stale constant walks straight into that.
EXPRESSIBLE_SPAN_SEC = 5000 * 0.080

_asked_span_sec = _args.number("--max-span-seconds", 300)
MAX_SPAN_SEC = min(_asked_span_sec, EXPRESSIBLE_SPAN_SEC)
if _asked_span_sec > EXPRESSIBLE_SPAN_SEC:
    log.warning("align: --max-span-seconds %.0f is past the %.0fs this model's output head "
                "can express (5000 classes on an 80 ms grid); using %.0fs. Above it the "
                "timestamps saturate and the answer is wrong rather than missing",
                _asked_span_sec, EXPRESSIBLE_SPAN_SEC, MAX_SPAN_SEC)

# A backstop against a malformed body reaching the decoder, not a memory bound: the
# body is already in memory when this refuses it. The gateway in front is the real cap.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

_state = _runtime.state


# On by default. Output is NOT byte-identical to one-span-a-call: 226 of 17550
# timestamps moved on the corpus this was measured against, no text or count differing.
BATCH = _args.switch("--align-batch", True)


def _kv_configs(model):
    """Every config object on the path the text decoder reads `use_cache` from.

    Walked rather than guessed: `Qwen3ASRConfig` -> `thinker_config` -> `text_config`,
    and the decorator reads whichever `self.config` resolves to.
    """
    out = []
    root = getattr(model, "model", None)
    cfg = getattr(root, "config", None) if root is not None else None
    while cfg is not None and cfg not in out:
        out.append(cfg)
        cfg = getattr(cfg, "thinker_config", None) or getattr(cfg, "text_config", None)
    return out


def _say_config():
    """Every setting that moves a number, in one line, before anything runs."""
    log.info("align: batching %s | KV cache %s | budget %s, %.0f%% of what the grant "
             "leaves | group slack %d positions | longest span %.0fs | upload cap %d MiB",
             "ON, groups sized at run time (the default)" if BATCH
             else "off, one span a call",
             "OFF (--no-kv-cache)" if NO_KV_CACHE else "not turned off here",
             ("pinned at %d positions" % ALIGN_FIXED_BUDGET) if ALIGN_FIXED_BUDGET
             else "computed", 100.0 * BUDGET_HEADROOM, ALIGN_GROUP_SLACK, MAX_SPAN_SEC,
             MAX_UPLOAD_BYTES // (1 << 20))


def _p(msg):
    print("[align] " + msg, flush=True)


def _resolve_hf_dir(repo):
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    kw = {"repo_id": repo, "local_files_only": True}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return snapshot_download(**kw)


def _xml_has_input(path, name, limit=1048576):
    try:
        with open(path, "rb") as f:
            head = f.read(limit)
    except OSError:
        return False
    return name.encode("ascii") in head


def _looks_like_ov_ir(path):
    """One-shot align IR from the 0.6B snapshot: encoder+decoder, no beam_idx."""
    if not path or not os.path.isdir(path):
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    if not (os.path.isfile(enc) and os.path.isfile(dec)):
        return False
    return not _xml_has_input(dec, "beam_idx")


def _ov_export_cmd(src, dest):
    # Same 0.6B snapshot as NVIDIA. One thinker forward, not ASR generate:
    # skip -with-past / stateful decoder. Do not remap to *-hf.
    return [
        "optimum-cli", "export", "openvino",
        "--model", src,
        "--task", "automatic-speech-recognition",
        "--disable-stateful",
        "--disable-convert-tokenizer",
        "--weight-format", "fp16",
        "--trust-remote-code",
        dest,
    ]


def _ensure_ov_ir(src):
    nested = os.path.join(src, "openvino")
    if _looks_like_ov_ir(src):
        return src
    if _looks_like_ov_ir(nested):
        return nested
    dest = nested
    if os.path.isdir(dest) and not _looks_like_ov_ir(dest):
        import shutil
        _p("removing unusable export dir %s" % dest)
        shutil.rmtree(dest)
    _p("no OpenVINO align IR in %s; exporting to %s (first start is slow)" % (src, dest))
    os.makedirs(dest, exist_ok=True)
    import subprocess

    cmd = _ov_export_cmd(src, dest)
    _p("running: %s" % " ".join(cmd))
    subprocess.check_call(cmd)
    if not _looks_like_ov_ir(dest):
        raise RuntimeError(
            "align export finished but %s is not a one-shot align IR "
            "(need encoder+decoder xml without beam_idx)"
            % dest
        )
    return dest


def _register_qwen3_asr():
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRProcessor
    from transformers import AutoConfig, AutoProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def _load_ov():
    from optimum.intel import OVModelForSpeechSeq2Seq
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
    from transformers import AutoProcessor

    src = _resolve_hf_dir(MODEL_REPO)
    model_dir = _ensure_ov_ir(src)
    device = ovutil.device()
    _p("loading OpenVINO forced aligner src=%s ir=%s device=%s" % (src, model_dir, device))
    _register_qwen3_asr()
    kw = dict(device=device)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, **kw)
    processor = AutoProcessor.from_pretrained(src, fix_mistral_regex=True)
    cfg = getattr(model, "config", None)
    ts_id = int(getattr(cfg, "timestamp_token_id", 0) or 0)
    ts_seg = float(getattr(cfg, "timestamp_segment_time", 0) or 0)
    if not ts_id or not ts_seg:
        import json

        with open(os.path.join(src, "config.json")) as f:
            raw = json.load(f)
        ts_id = ts_id or int(raw.get("timestamp_token_id") or 0)
        ts_seg = ts_seg or float(raw.get("timestamp_segment_time") or 0)
    _state.update(
        model=model,
        processor=processor,
        aligner_processor=Qwen3ForceAlignProcessor(),
        timestamp_token_id=ts_id,
        timestamp_segment_time=ts_seg,
        device=device,
        backend="openvino",
        ready=True,
    )
    log.info("Qwen3-ForcedAligner %s loaded (openvino %s)", MODEL_REPO, device)


def _load():
    _say_config()
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        kw = dict(dtype=dtype, device_map=(dev if dev == "cpu" else "cuda:0"))
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        try:
            model = Qwen3ForcedAligner.from_pretrained(MODEL_REPO, **kw)
        except TypeError:
            kw.pop("token", None)
            model = Qwen3ForcedAligner.from_pretrained(MODEL_REPO, **kw)
        seen = []
        for cfg in _kv_configs(model):
            seen.append(getattr(cfg, "use_cache", None))
            if NO_KV_CACHE:
                cfg.use_cache = False
            elif getattr(cfg, "use_cache", None) is None:
                cfg.use_cache = True
        _state.update(model=model, device=dev)
        global _encoder_rate
        _encoder_rate = None
        _rebind_cache_pricing([getattr(c, "use_cache", None) for c in _kv_configs(model)])
        _d = _dims()
        _rebind_span_limit(_d)
        log.info("align: use_cache was %s on %d config objects and is now %s; a position is "
                 "priced at %s", seen, len(seen), CACHE_ON,
                 "%.0f B" % _bytes_a_position(_d) if _d
                 else "nothing -- the dimensions did not read, so there is no cost model")
        _reset_peak()
        _calibrate()
        _state["ready"] = True
        log.info("Qwen3-ForcedAligner %s loaded on %s", MODEL_REPO, dev)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("forced-aligner load failed: %s", e)


def _calibrate():
    """One throwaway call, so the cost model is measured on THIS machine before traffic.

    Non-fatal in every direction: a card too full to warm on will refuse the first
    real call too, and saying so beats refusing to start.
    """
    import time as _time

    t0 = _time.time()
    try:
        import numpy as _np
        import soundfile as _sf

        seconds, sr = 30.0, 16000
        n = int(seconds * sr)
        tone = (0.25 * _np.sin(2 * _np.pi * 220 * _np.arange(n) / sr)).astype("float32")
        text = "\u5b57" * int(seconds * 4)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _sf.write(path, tone, sr, format="WAV", subtype="PCM_16")
            _reset_peak()
            reading = _memory_now()
            _align(path, text, DEFAULT_LANGUAGE)
        finally:
            unlink(path)
        took = _time.time() - t0
        cost = _cost([_Span(0, None, text, DEFAULT_LANGUAGE, seconds,
                            _span_positions(seconds, text, DEFAULT_LANGUAGE))])
        if reading is not None and _used_bytes(*reading) <= 0:
            log.warning("align: warmed in %.1fs but the calibration took no measurement -- the "
                        "peak did not rise above what was already allocated, so nothing was "
                        "observed and the correction stays at %.2f, fitted elsewhere. The cost "
                        "model is unchecked on this machine.", took, _scale)
        elif reading is not None:
            used = _used_bytes(*reading)
            _observe(cost, used)
            log.info("align: calibrated on this machine in %.1fs -- one %.0fs span costs "
                     "%.0f positions, the model said %.0f MB and it took %.0f MB, so the "
                     "correction is %.2f", took, seconds, cost,
                     cost * _bytes_a_position(_dims() or {}) / 1e6 if _dims() else 0.0,
                     used / 1e6, _scale)
        else:
            log.info("align: warmed in %.1fs; no per-process memory counter, so the cost "
                     "model is the one fitted elsewhere and nothing here checked it", took)
    except Exception as e:
        _reset_peak()
        log.warning("align: could not calibrate after %.1fs (%s). The cost model is the one "
                    "fitted on another machine and the first real call is what will test "
                    "it", _time.time() - t0, e)


def _field(u, *names):
    for n in names:
        try:
            if isinstance(u, dict):
                if n in u:
                    return u[n]
            elif hasattr(u, n):
                return getattr(u, n)
        except Exception:
            pass
    return None


def _units(res):
    return [{"text": _field(u, "text", "word", "token"),
             "start": _field(u, "start_time", "start"),
             "end": _field(u, "end_time", "end")} for u in (res[0] if res else [])]


def _ov_logits(model, inputs):
    # SpeechSeq2Seq.forward remaps input_features onto input_ids, then **kwargs
    # still carries the text input_ids → "got multiple values for input_ids".
    # thinker(**inputs) hits that. Split encoder / decoder like the official
    # ForcedAligner.forward, still on the 0.6B ASR-task IR.
    payload = dict(inputs)
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    feats = payload.get("input_features")
    if encoder is None or decoder is None or feats is None:
        raise TypeError("align IR has no encoder/decoder/input_features")
    enc = encoder(
        input_features=feats,
        attention_mask=payload.get("input_features_mask"),
    )
    hidden = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
    dec = decoder(
        input_ids=payload.get("input_ids"),
        encoder_hidden_states=hidden,
        attention_mask=payload.get("attention_mask"),
    )
    return dec.logits


def _align(path, text, language):
    """Every model call goes through here, so the counter it leaves behind stays honest.

    A call that raises leaves the all-time peak where it got to, and `_used` answers
    zero below that, so the correction freezes. The success half stays with the
    caller, which is the only party that knows what the call cost.
    """
    _peak_at_failure[0] = None
    try:
        return _align_raw(path, text, language)
    # 🔴 BaseException: asyncio.CancelledError has not been an Exception since 3.8, and
    # every request here runs under tasks.dispatch, so a client hanging up unwinds through
    # this frame. The peak is read before it is put back -- the caller records it on the
    # refused call's row, and that is the number an operator sets the budget fraction from.
    except BaseException:
        _peak_at_failure[0] = _peak_bytes()
        _reset_peak()
        raise


def _align_ov(path, text, language):
    import librosa

    wav, _sr = librosa.load(path, sr=16000, mono=True)
    lang = ovutil.language(language)
    word_list, aligner_input = _state["aligner_processor"].encode_timestamp(text, lang)
    inputs = _state["processor"](
        text=[aligner_input],
        audio=[wav],
        return_tensors="pt",
        padding=True,
    )
    logits = _ov_logits(_state["model"], inputs)
    if hasattr(logits, "argmax") and hasattr(logits, "detach"):
        output_ids = logits.argmax(dim=-1)
        input_ids = inputs["input_ids"][0]
        output_id = output_ids[0]
        masked = output_id[input_ids == _state["timestamp_token_id"]]
        timestamp_ms = (masked * _state["timestamp_segment_time"]).detach().cpu().numpy()
    else:
        import numpy as np

        output_ids = np.argmax(np.asarray(logits), axis=-1)
        input_ids = np.asarray(inputs["input_ids"][0])
        output_id = output_ids[0]
        masked = output_id[input_ids == _state["timestamp_token_id"]]
        timestamp_ms = masked * _state["timestamp_segment_time"]
    items = _state["aligner_processor"].parse_timestamp(word_list, timestamp_ms)
    for it in items:
        it["start_time"] = round(it["start_time"] / 1000.0, 3)
        it["end_time"] = round(it["end_time"] / 1000.0, 3)
    return [items]


def _align_raw(path, text, language):
    if _state.get("backend") == "openvino":
        return _align_ov(path, text, language)
    try:
        return _state["model"].align(audio=path, text=text, language=language)
    except TypeError:
        return _state["model"].align(path, text, language)


# Measured, not chosen: the densest span in a 10572-span meeting corpus sat 1.8x under.
MAX_UNITS_A_SECOND = 24

NO_KV_CACHE = _args.switch("--no-kv-cache", False)

# (per position, per audio second, encoder fixed) in bytes, fitted on hzydemo01 (RTX 4060 Ti).
# 🔴 The pair moves with the cache: with it on a position costs 138 kB, without it
# 28 kB, and the position COUNT moves the other way, so the two nearly cancel on a
# single long span. Picked by CACHE_ON, which is what the checkpoint ends up with.
KAPPA_NO_CACHE = (0.81, 2.74, 5.0e6)
KAPPA_WITH_CACHE = (4.0, 2.54, 0.0)
KAPPA_LM, KAPPA_ENCODER, ENCODER_FIXED_BYTES = (
    KAPPA_NO_CACHE if NO_KV_CACHE else KAPPA_WITH_CACHE)


ALIGN_FIXED_BUDGET = 0


# 🔴 The flag asked; the checkpoint decides. One that ships use_cache false builds no
# cache with the flag off, and transformers treats None as unset rather than as True,
# so all three states differ. _rebind_cache_pricing settles it once the model is open.
CACHE_ON = not NO_KV_CACHE


def _rebind_cache_pricing(values):
    """Point the cost model at the cache this checkpoint actually ends up with."""
    global CACHE_ON, KAPPA_LM, KAPPA_ENCODER, ENCODER_FIXED_BYTES
    real = [v for v in values if v is not None]
    if not real:
        log.warning("align: no config object on this checkpoint exposes use_cache, so "
                    "nothing was written and it will build whatever it ships. Pricing as if "
                    "a cache is built, which is the direction that costs speed rather than "
                    "the card -- and `--no-kv-cache` cannot have taken effect either")
    elif len(set(real)) > 1:
        log.warning("align: the config objects disagree about use_cache (%s). The decoder "
                    "reads one of them and this cannot tell which, so the cost model is "
                    "priced as if a cache is built, which is the safe direction", values)
    CACHE_ON = any(real) if real else True
    KAPPA_LM, KAPPA_ENCODER, ENCODER_FIXED_BYTES = (
        KAPPA_WITH_CACHE if CACHE_ON else KAPPA_NO_CACHE)


# Padded positions a group may add before the call it saves stops being worth it.
# Swept on two machines: 20/50/100 came in at 1.44/1.40/1.42 s against a 10% noise
# floor, so they are one result on the clock; CPU separates them (3.92/2.94/2.76
# core-seconds). Leaving grouping to memory alone costs 1.29-1.49x. A constant
# because the sweep says the band is flat, in a unit no operator owns.
ALIGN_GROUP_SLACK = 50

BUDGET_HEADROOM = _args.number("--gpu-budget-fraction", 0.5)
if not 0 < BUDGET_HEADROOM <= 1:
    log.warning("align: --gpu-budget-fraction=%s is not a fraction of the grant (it must be "
                "above 0 and at most 1), so it is being ignored and %.2f used instead. A "
                "value outside that range reads downstream as a full card, which is a "
                "different problem entirely", BUDGET_HEADROOM, 0.5)
    BUDGET_HEADROOM = 0.5


# 🔴 Last, after every claim. Run earlier it calls a flag this engine takes unclaimed,
# and drains an unreadable value before anyone warns about it.
_args.warn_unclaimed(log)


def _dims():
    """Dimensions from the loaded checkpoint, or None if they cannot be read.

    Read off the model, not the configuration class: the class defaults disagree with
    this checkpoint. The fields hang off `thinker_config`; both layouts are accepted
    because a directly loaded thinker presents them flat.
    """
    try:
        cfg = _top = _state["model"].model.config
        cfg = getattr(cfg, "thinker_config", None) or cfg
        text, audio = cfg.text_config, cfg.audio_config
        width = 2
        try:
            width = next(_state["model"].model.parameters()).element_size()
        except Exception:
            global _width_guessed
            if not _width_guessed:
                _width_guessed = True
                log.warning("align: could not read the parameter width; pricing positions at "
                            "2 bytes each. If this checkpoint is fp32 every batch is twice "
                            "what it should be")
        grid = None
        for src in (cfg, _top):
            try:
                grid = float(src.timestamp_segment_time)
                break
            except Exception:
                continue
        return {"hidden": int(text.hidden_size), "ffn": int(text.intermediate_size),
                "classes": int(cfg.classify_num), "width": width, "grid_ms": grid,
                "downsample": int(audio.downsample_hidden_size),
                "window": int(audio.n_window)}
    except Exception:
        global _dims_unreadable
        if _state.get("model") is not None and not _dims_unreadable:
            _dims_unreadable = True
            log.warning("align: the checkpoint's dimensions could not be read, so there is "
                        "no cost model: batches are not priced, and the check that compares "
                        "the model against what calls actually cost never runs. Every figure "
                        "it would have produced is absent, not wrong", exc_info=True)
        return None


def _expressible_span_sec(d):
    """The longest span this checkpoint's output head can put a timestamp on.

    Falls back to the constant when the dimensions did not read: a stale limit beats
    no limit.
    """
    if not d or not d.get("classes") or not d.get("grid_ms"):
        return EXPRESSIBLE_SPAN_SEC
    return d["classes"] * d["grid_ms"] / 1000.0


def _rebind_span_limit(d):
    """Re-derive the span ceiling from the checkpoint that just loaded.

    Silent when it agrees with the constant, which the shipped checkpoint does.
    """
    global EXPRESSIBLE_SPAN_SEC, MAX_SPAN_SEC
    found = _expressible_span_sec(d)
    if found == EXPRESSIBLE_SPAN_SEC:
        return
    was, was_max = EXPRESSIBLE_SPAN_SEC, MAX_SPAN_SEC
    EXPRESSIBLE_SPAN_SEC = found
    MAX_SPAN_SEC = min(_asked_span_sec, EXPRESSIBLE_SPAN_SEC)
    log.info("align: this checkpoint's output head expresses %.1fs, not the %.1fs this "
             "build was written against (%d classes on a %.0f ms grid); longest span is "
             "now %.1fs, was %.1fs", found, was, d["classes"], d["grid_ms"],
             MAX_SPAN_SEC, was_max)
    if _asked_span_sec > EXPRESSIBLE_SPAN_SEC:
        log.warning("align: --max-span-seconds %.0f is past what this checkpoint's head "
                    "can express (%.1fs); using %.1fs. Above it the timestamps saturate "
                    "and the answer is wrong rather than missing",
                    _asked_span_sec, EXPRESSIBLE_SPAN_SEC, MAX_SPAN_SEC)


def _bytes_a_position(d):
    return KAPPA_LM * (6 * d["hidden"] + 2 * d["ffn"] + d["classes"]) * d["width"]


def _bytes_a_second(d):
    return KAPPA_ENCODER * d["downsample"] * 64 * d["window"] * d["width"]


_encoder_rate = None
_width_guessed = False
_tokenizer_failed = False
_dims_unreadable = False
_budget_priced = False


def _encoder_positions_a_second():
    """What a second of audio is worth in positions, so one budget prices both terms."""
    global _encoder_rate
    if _encoder_rate is None:
        d = _dims()
        if not d:
            return 30.0
        _encoder_rate = _bytes_a_second(d) / _bytes_a_position(d)
    return _encoder_rate


def _encoder_fixed_positions():
    """`ENCODER_FIXED_BYTES` in the unit the budget counts in."""
    d = _dims()
    return ENCODER_FIXED_BYTES / _bytes_a_position(d) if d else 0.0


_no_grant_logged = False


def _headroom_bytes():
    """What one call may still allocate, in bytes, or 0 when nothing authoritative says.

    Under a quota this is `free + cache`. HAMi rewrites `mem_get_info()[0]` as
    `limit - its own ledger`, which already nets out the context, the modules and this
    container's other processes; and `reserved - allocated` is spendable without asking
    HAMi at all, because torch reuses cached blocks without the allocation call HAMi
    charges for. Leaving that term out was most of why the budget fell 8833 -> 3948
    positions over three rounds while the resident set never moved.

    Returns None, not 0, when no authority can be read: a quota that is spoken for is a
    real zero, and the two need different answers downstream.
    """
    import torch

    from .. import gpu

    global _no_grant_logged
    cache = max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    hami = _hami_limit_bytes()
    if hami:
        free, total = torch.cuda.mem_get_info()
        if int(total) <= hami:
            return int(free) + cache
        if not _no_grant_logged:
            _no_grant_logged = True
            log.warning("align: a GPU limit of %d bytes is published but the device still "
                        "reports %d total, so its memory counters are not being rewritten "
                        "to the limit. Sizing against the limit instead of against what "
                        "the device calls free, which here is the whole card",
                        hami, int(total))
        return max(0, hami - torch.cuda.memory_allocated())
    grant = gpu._quota_bytes()
    if grant:
        if not _no_grant_logged:
            _no_grant_logged = True
            log.info("align: no enforced GPU limit to read, sizing against the declared "
                     "REQUIRED_GPU_MEMORY instead. Nothing refuses at that figure, so a "
                     "batch that overshoots it is not caught here")
        return max(0, grant - torch.cuda.memory_allocated())
    if not _no_grant_logged:
        _no_grant_logged = True
        log.warning("align: neither an enforced GPU limit nor a declared one could be read, "
                    "so spans go one to a call. The device's own free figure is not used: "
                    "where the HAMi hook is loaded without a limit it reports the whole "
                    "card minus this container, which counts a neighbour's memory as ours")
    return None


_HAMI_UNITS = {"g": 2 ** 30, "m": 2 ** 20, "k": 2 ** 10}


def _strtoul_base0(text):
    """The leading integer as C's `strtoul(s, end, 0)` reads it, or None for no number.

    Base 0: a leading `0` is octal, `0x` is hex. HAMi parses its own limit this way,
    so `04096m` really is 32 MiB there and must be here too.
    """
    body = text.lstrip()
    sign = 1
    if body[:2].lower() == "0x":
        digits, base = "0123456789abcdef", 16
        body = body[2:]
    elif body[:1] == "0":
        digits, base = "01234567", 8
        body = body[1:] or "0"
    else:
        digits, base = "0123456789", 10
    run = ""
    for ch in body:
        if ch.lower() in digits:
            run += ch
        else:
            break
    if not run:
        return 0 if base == 8 else None
    return sign * int(run, base)
_hami_unreadable_logged = False


def _hami_limit_bytes():
    """HAMi's enforced per-container figure in bytes; 0 when there is no limit."""
    global _hami_unreadable_logged
    name = "CUDA_DEVICE_MEMORY_LIMIT_0"
    raw = os.environ.get(name, "") or ""
    if not raw:
        name = "CUDA_DEVICE_MEMORY_LIMIT"
        raw = os.environ.get(name, "") or ""
    if not raw:
        return 0
    scalar = _HAMI_UNITS.get(raw[-1].lower(), 1)
    head = (raw[:-1] if scalar != 1 else raw).lstrip()
    value = _strtoul_base0(head)
    if value is not None:
        return value * scalar
    if not _hami_unreadable_logged:
        _hami_unreadable_logged = True
        log.warning("align: %s is set to %r, which HAMi's own parser reads as no number at "
                    "all, so it is being treated as no limit. If a limit really is enforced, "
                    "every batch from here is sized against a figure nothing will hold it "
                    "to", name, raw)
    return 0


def _span_positions(seconds, text, language=None):
    return 2 + int(13 * seconds) + _text_positions(text, language)


def _text_positions(text, language=None):
    """Tokens plus two timestamp slots for every unit the aligner will split `text` into."""
    units = _units_of(text, language)
    tok = _tokenizer()
    if tok is not None:
        try:
            ids = tok(units)["input_ids"]
            return sum(len(i) + 2 for i in ids)
        except Exception:
            global _tokenizer_failed
            if not _tokenizer_failed:
                _tokenizer_failed = True
                log.warning("align: the model's tokenizer refused a span, so its positions "
                            "are estimated rather than counted -- the estimate is low for "
                            "languages whose units are words, so this span is priced "
                            "under what it costs", exc_info=True)
    return sum(3 if len(u) == 1 else 2 + max(1, (len(u) + 3) // 4) for u in units)


def _units_of(text, language=None):
    """What `tokenize_space_lang` splits `text` into: whitespace, clean, then CJK singly.

    A hand copy of the library's function. `scripts/check-units-against-upstream.py`
    compares the two and runs in the deps-image build, which is the only place the
    library is installed and the moment its unpinned version can move.
    """
    if (language or "").strip().lower() in ("japanese", "korean"):
        return [ch for ch in text if _is_kept(ch)]
    out = []
    for word in text.split():
        buf = []
        for ch in word:
            if not _is_kept(ch):
                continue
            if _is_cjk(ch):
                if buf:
                    out.append("".join(buf))
                    buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
    return out


def _is_kept(ch):
    """`is_kept_char`: an apostrophe, or anything Unicode calls a letter or a number."""
    if ch == "'":
        return True
    return unicodedata.category(ch)[:1] in ("L", "N")


def _is_cjk(ch):
    c = ord(ch)
    return (0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or 0x20000 <= c <= 0x2A6DF
            or 0x2A700 <= c <= 0x2B73F or 0x2B740 <= c <= 0x2B81F
            or 0x2B820 <= c <= 0x2CEAF or 0xF900 <= c <= 0xFAFF)


LONG_RUN_CHARACTERS = 40


def _tokenizer():
    m = _state.get("model")
    return getattr(getattr(m, "processor", None), "tokenizer", None)


def _admit(seconds, text, language=None):
    """Why this span cannot be aligned, or None."""
    if seconds > MAX_SPAN_SEC:
        return ("span is %.1fs; this deployment aligns at most %.0fs (--max-span-seconds)"
                % (seconds, MAX_SPAN_SEC))
    return None


def _dense(seconds, text, language=None):
    """Whether this span carries more text than its audio could hold, and by how much.

    Reported, never refused. A span nothing can fit is sent alone, refused by the card
    and written its own error anyway, so refusing here saves one call; a false positive
    costs the words permanently, because the caller records a per-span error as a failed
    turn and does not resend. Over 10572 spans of a real meeting corpus none was
    refused and the worst sat 1.8x under the limit.
    """
    units = sum(max(1, len(u) // LONG_RUN_CHARACTERS) if len(u) > LONG_RUN_CHARACTERS else 1
                for u in _units_of(text, language))
    return units if seconds > 0 and units > MAX_UNITS_A_SECOND * seconds else 0


_budget = 0.0
_billed = set()


_scale = 1.0
_scale_seen = 0
_calls_seen = 0
_ratio_lo = None
_ratio_hi = None

SCALE_MEMORY = 8

SCALE_BAND = (0.4, 2.5)


def _observe(cost, used_bytes):
    """Fold one call's measured cost into the running correction."""
    global _scale, _scale_seen, _calls_seen, _ratio_lo, _ratio_hi
    _calls_seen += 1
    d = _dims()
    if not d or cost <= 0 or used_bytes <= 0:
        return
    predicted = cost * _bytes_a_position(d)
    if predicted <= 0:
        return
    ratio = used_bytes / predicted
    _ratio_lo = ratio if _ratio_lo is None else min(_ratio_lo, ratio)
    _ratio_hi = ratio if _ratio_hi is None else max(_ratio_hi, ratio)
    _scale_seen += 1
    weight = 1.0 / min(_scale_seen, SCALE_MEMORY)
    _scale = (1.0 - weight) * _scale + weight * ratio
    if not SCALE_BAND[0] <= _scale <= SCALE_BAND[1]:
        log.warning("align: the cost model is off by %.1fx over %d calls. The dimensions it "
                    "reads have not changed, so what it assumes about the library has: "
                    "re-derive before trusting the budget", _scale, _scale_seen)


def _used(before, peak_before, peak_after):
    """What this call added, or zero when it never rose above an older peak."""
    if peak_after <= peak_before:
        return 0
    return max(0, peak_after - before)


def _used_bytes(before, peak_before):
    """`_used` against the live counter."""
    try:
        import torch

        return _used(before, peak_before, torch.cuda.max_memory_allocated())
    except Exception:
        return 0


def _solve_budget():
    """Positions a call may carry, or None when the grant cannot be read at all."""
    d = _dims()
    if not d:
        return None
    try:
        headroom = _headroom_bytes()
        if headroom is None:
            return None
        per_position = _bytes_a_position(d) * max(_scale, 0.05)
        return BUDGET_HEADROOM * headroom / per_position
    except Exception:
        return None


def _opening_budget():
    """Where to start, computed from the checkpoint and this process's grant."""
    global _budget_priced
    solved = _solve_budget()
    _budget_priced = solved is not None
    return solved or 0.0


_opening_logged = False


def _effective_budget():
    """The budget in force, computing an opening one on the first call."""
    global _budget, _opening_logged, _budget_priced
    if ALIGN_FIXED_BUDGET > 0:
        _budget_priced = True
        if not _opening_logged:
            _opening_logged = True
            log.info("align: budget fixed at %.0f positions by configuration; it will still "
                     "not be re-solved", ALIGN_FIXED_BUDGET)
        return ALIGN_FIXED_BUDGET
    if _budget <= 0:
        _budget = _opening_budget()
        if _budget > 0 and not _opening_logged:
            _opening_logged = True
            log.info("align: opening budget %.0f positions, from the checkpoint's own "
                     "dimensions and this container's grant", _budget)
        elif _budget <= 0 and not _opening_logged:
            _opening_logged = True
            if _budget_priced:
                log.warning("align: the grant is readable and has nothing left, so one span a "
                            "call is the answer rather than a bigger batch. This is not a "
                            "fault in the cost model -- it is the card being full -- and it "
                            "lifts when whatever holds the memory lets go")
            else:
                log.warning("align: no opening budget -- the checkpoint's dimensions or this "
                            "container's grant could not be read, so batches grow by doubling "
                            "until one fails instead of being priced. Every figure the cost "
                            "model would have produced is absent, not wrong")
    return _budget


_groups = []


TELEMETRY_KEEP = 64
_telemetry = collections.deque(maxlen=TELEMETRY_KEEP)
_peak_at_failure = [None]

_oom_count = [0]
_last_request_ended = None


def _cgroup_bytes(name):
    """One cgroup v2 memory figure, or None. "max" is the literal for no limit."""
    try:
        raw = open("/sys/fs/cgroup/memory." + name).read().strip()
    except Exception:
        return None
    return None if raw == "max" else int(raw)


def _card():
    """What the DEVICE says, and whether that number is the card or a quota."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        try:
            stats = torch.cuda.memory_stats() or {}
        except Exception:
            stats = {}
        limit = _hami_limit_bytes()
        return {"free": int(free), "total": int(total), "reserved": int(reserved),
                "allocated": int(torch.cuda.memory_allocated()),
                "outside": int(free) + int(reserved),
                "alloc_retries_since_start": int(stats.get("num_alloc_retries") or 0),
                "torch_ooms_since_start": int(stats.get("num_ooms") or 0),
                "quota_view": bool(limit and int(total) <= limit)}
    except Exception:
        return None


def _snapshot():
    return {"t": time.time(), "gpu": _card(),
            "host_used": _cgroup_bytes("current"), "host_max": _cgroup_bytes("max")}


def _covered_seconds(segs):
    """Seconds the request's spans cover, and never a reason the request fails."""
    total = 0.0
    for sg in segs:
        try:
            total += max(0.0, float(sg.get("end") or 0) - float(sg.get("start") or 0))
        except Exception:
            pass
    return total


def _telemetry_record(started, spans, audio_seconds, calls=None):
    """One request's before and after, and how long nobody was looking."""
    global _last_request_ended
    ended = _snapshot()
    _telemetry.append({
        "started": started, "ended": ended,
        "idle_before": None if _last_request_ended is None
        else round(started["t"] - _last_request_ended, 3),
        "spans": spans, "audio_seconds": round(audio_seconds, 3),
        "calls": len(_groups) if calls is None else calls,
        "largest_call_positions": max((g["cost"] for g in _groups), default=0),
        "closed_by": sorted({g["closed_by"] for g in _groups}),
        "budget": round(_budget, 1), "scale": round(_scale, 4),
        "ooms": _oom_count[0],
    })
    _last_request_ended = ended["t"]


def _say_if_nothing_aligned(results):
    """Shout when a request produced no aligned span at all."""
    if not results:
        return
    aligned = sum(1 for r in results if isinstance(r, dict) and "units" in r)
    if aligned:
        return
    reasons = collections.Counter(
        "out of memory" if "out of memory" in str(r.get("error", "")).lower() else "other"
        for r in results if isinstance(r, dict) and r.get("error"))
    log.error("align: not one of %d spans aligned -- every span in this request came back an "
              "error (%s). The caller was answered 200 with per-span errors, which is how a "
              "single bad span is reported too, so nothing downstream can tell these apart. "
              "If the reason is memory, the readings are at /v1/audio/align/telemetry and "
              "--gpu-budget-fraction is %.2f",
              len(results), ", ".join("%s=%d" % kv for kv in sorted(reasons.items())),
              BUDGET_HEADROOM)


def _record_group(group, why):
    entry = {"spans": len(group),
             "longest_positions": max(m.positions for m in group),
             "longest_seconds": round(max(m.seconds for m in group), 2),
             "real_seconds": round(sum(m.seconds for m in group), 2),
             "closed_by": why,
             "cost": round(_cost(group), 1),
             "failed": False}
    _groups.append(entry)
    return entry


_Span = collections.namedtuple("_Span", "index clip text language seconds positions")


def _cost(group, extra=None):
    """What one call carrying these spans costs, in positions.

    The larger of two phases, not their sum: the encoder frees its working set before the
    language model starts. Measured on the batch axis -- one span and two of the same
    length cost the same, and from four up the cost tracks the batch exactly.

    The encoder is charged once a call however many spans ride in it (it loops per item:
    "audio encoder do not support batch inference to keep precision"), so a single long
    span is not cheap for being alone. The language model is handed a padded batch, so
    every member costs the longest.
    """
    members = group if extra is None else group + [extra]
    return max(len(members) * max(m.positions for m in members),
               _encoder_positions_a_second() * max(m.seconds for m in members)
               + _encoder_fixed_positions())


def _next_group(prep, order, at):
    """The longest run from `at` that still fits the budget, and where to resume.

    Closed by `memory` (a bigger call would be refused), `padding` (a bigger call
    would fit and be slower) or `ramping` (nothing has been priced yet). The three
    are indistinguishable in a wall clock and have opposite fixes.
    """
    group = [prep[order[at]]]
    at += 1
    budget = _effective_budget()
    longest = group[0].positions
    reason = "no more spans"
    while at < len(order):
        candidate = prep[order[at]]
        if _cost(group, candidate) > budget:
            # A grant read and spoken for solves to 0.0, which IS the card refusing; a grant that
            # could not be read at all solves to None and the ramp takes over. Different answers.
            reason = "memory" if _budget_priced else "ramping"
            break
        if ALIGN_GROUP_SLACK > 0:
            repad = (len(group) * max(0, candidate.positions - longest)
                     + max(0, longest - candidate.positions))
            # Both terms, though the ascending walk makes the second zero on every path a request
            # takes: without it, an unsorted walk read zero at every step and closed nothing --
            # twelve spans went as one group padded to 414% of their real work.
            if repad >= ALIGN_GROUP_SLACK:
                reason = "padding"
                break
        group.append(candidate)
        longest = max(longest, candidate.positions)
        at += 1
    return group, at, reason


def _run_group(ctx, group, sr, out, why):
    """Work this group down until every span has an answer. Returns how many ended.

    Out of memory means this many did not fit together: every member is innocent, so the
    group halves and both halves go back on the stack. Anything else is one member the
    library could not read, so each goes alone and only the bad one comes back an error.
    A group of one that fails is that span's error.

    🔴 Nothing is remembered and no budget is lowered. An earlier version recorded a
    failed call as a ceiling that only ever fell: one call unlucky about a neighbour left
    the process at one span a call for its whole life, at a tenth of the throughput, with
    a complete response and a zero error count. A ceiling coming back needs an answer to
    that.
    """
    global _budget, _budget_priced
    done = 0
    # 🔴 A stack, not recursion. Recursing put each retry inside the failed call's except
    # block, so the handlers nested and each held its dead call's tensors alive: 3.4 GiB of
    # a 6 GiB grant on the first failure, and 96 spans became 191 calls that all failed.
    todo = [(group, why)]
    while todo:
        group, why = todo.pop()
        ctx.checkpoint()
        cost = _cost(group)
        reading = _memory_now()
        entry = _record_group(group, why)
        failure = _attempt(ctx, group, sr, out)
        if failure is None:
            if reading is not None:
                _observe(cost, _used_bytes(*reading))
            solved = _solve_budget()
            _budget_priced = solved is not None
            if solved is not None:
                _budget = solved
            else:
                # The ramp, where neither authority reads. It converges at twice the largest request's
                # cost -- ⚠️ which is not the same as converging BELOW what the card will take: if the
                # card is what limits the group, the steady state sits above it and every request pays
                # one refused call. Never exercised on a real machine; both production hosts have a grant.
                _budget = max(_budget, 2.0 * cost)
            done += len(group)
            continue

        oom, message = failure
        entry["failed"] = True
        # Outside `if oom:`: a real memory wall can surface as CUDNN_STATUS_ALLOC_FAILED, which
        # _is_oom does not match, and that path left the peak at the wall with nothing measurable
        # after it.
        entry["peak_bytes"] = (_peak_at_failure[0] if _peak_at_failure[0] is not None
                               else _peak_bytes())
        _reset_peak()
        if oom:
            _drop_cache()
            _oom_count[0] += 1
        if len(group) == 1:
            out[group[0].index] = {"error": "align failed: %s" % message}
            _bill(ctx, group[0])
            done += 1
            continue
        if oom:
            log.warning("align: out of memory on a call carrying %d spans priced at %.0f "
                        "positions; splitting it and retrying. Nothing is remembered -- the "
                        "budget stands and the next request is sized the same way. Repeating "
                        "means the cost model is wrong about this machine: the readings are "
                        "at /v1/audio/align/telemetry and --gpu-budget-fraction is %.2f",
                        len(group), cost, BUDGET_HEADROOM)
            half = len(group) // 2
            todo.append((group[half:], "split"))
            todo.append((group[:half], "split"))
            continue
        for m in reversed(group):
            if out[m.index] is None:
                todo.append(([m], "isolate"))
            else:
                done += 1
    return done


def _attempt(ctx, group, sr, out):
    """Send one group. None when it worked; `(was_out_of_memory, message)` when not.

    Its own frame on purpose: returning here ends the failed call's exception before
    the next attempt starts, so nothing holds the dead call's tensors alive.
    """
    try:
        _call(ctx, group, sr, out)
    except tasks.Cancelled:
        raise
    except Exception as e:
        oom, message = _is_oom(e), str(e)
        return oom, message
    return None


def _call(ctx, group, sr, out):
    audio = [(m.clip, sr) for m in group]
    res = _align(audio, [m.text for m in group], [m.language for m in group])
    if not isinstance(res, (list, tuple)) or len(res) != len(group):
        raise RuntimeError("the aligner answered %s results for %d spans"
                           % (len(res) if hasattr(res, "__len__") else type(res).__name__,
                              len(group)))
    # The length is checked above because zip stops at the shorter side: a library answering
    # fewer results would leave those spans null and report the group finished.
    for m, r in zip(group, res):
        out[m.index] = {"language": m.language, "units": _units([r])}
        _bill(ctx, m)


def _peak_bytes():
    """The process's all-time high right now, or 0 when there is no card to ask."""
    try:
        import torch

        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return 0


def _reset_peak():
    """Drop the historical peak to what is allocated now. Silent when there is no card."""
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _memory_now():
    """This process's allocated and peak bytes, or None when the counters cannot be read.

    None rather than (0, 0): zero is also what a just-reset counter reads.
    """
    try:
        import torch

        return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    except Exception:
        return None


def _bill(ctx, span):
    """Meter one span, once, and say whether it did."""
    if span.index in _billed:
        return False
    _billed.add(span.index)
    ctx.meter(input_seconds=span.seconds)
    return True


def _is_oom(e):
    """Whether this was the wall rather than a bad span."""
    try:
        import torch

        oom = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom is not None and isinstance(e, oom):
            return True
    except Exception:
        pass
    return "out of memory" in str(e).lower()


def _drop_cache():
    """Hand the allocator's cached blocks back before retrying."""
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _batched(ctx, segs, arr, sr, language):
    """Group spans by what a call costs and send each group in one align()."""
    out = [None] * len(segs)
    _billed.clear()
    del _groups[:]
    dense = 0
    prep = []
    for i, seg in enumerate(segs):
        try:
            stext = (str(seg.get("text") or "")).strip()
            if not stext:
                out[i] = {"units": [], "language": None}
                continue
            lo = max(0, int(float(seg.get("start") or 0) * sr))
            hi = min(len(arr), int(float(seg.get("end") or 0) * sr))
            if hi <= lo:
                out[i] = {"error": "empty segment"}
                continue
            span_sec = (hi - lo) / float(sr)
            lang = ((str(seg.get("language") or language or "")).strip()
                    or DEFAULT_LANGUAGE)
            refused = _admit(span_sec, stext, lang)
            if refused:
                out[i] = {"error": refused}
                continue
            dense += 1 if _dense(span_sec, stext, lang) else 0
            prep.append(_Span(i, arr[lo:hi], stext, lang, span_sec,
                              _span_positions(span_sec, stext, lang)))
        except tasks.Cancelled:
            raise
        except Exception as e:
            out[i] = {"error": "segment could not be read: %s" % e}

    order = list(range(len(prep)))
    if ALIGN_GROUP_SLACK > 0:
        order.sort(key=lambda k: prep[k].positions)

    done = len(segs) - len(prep)
    at = 0
    # 🔴 Grouping waits until something on THIS machine has been weighed -- _calibrate
    # swallows its own failure and readiness is set anyway, so the engine can be serving on a
    # correction fitted elsewhere. `or _dims() is None` because a machine with no cost model
    # can never raise the counter, and gating on it alone would shut grouping off for good.
    measured = _scale_seen > 0 or _dims() is None
    if not measured and len(order) > 1:
        log.warning("align: grouping is held off -- nothing on this machine has been "
                    "weighed yet (the calibration took no sample), so the cost model is the "
                    "one fitted elsewhere. Spans go one a call until the first one measures")
    while at < len(order):
        if measured:
            group, at, why = _next_group(prep, order, at)
        else:
            group, at, why = [prep[order[at]]], at + 1, "unmeasured"
        done += _run_group(ctx, group, sr, out, why)
        if not measured:
            measured = _scale_seen > 0 or _dims() is None
        ctx.progress(done=done, total=len(segs))
    if not order:
        ctx.progress(done=done, total=len(segs))
    if _groups:
        closed = collections.Counter(g["closed_by"] for g in _groups)
        failed = sum(1 for g in _groups if g["failed"])
        log.info("align: %d spans in %d calls%s, padded %.0fs against %.0fs real, closed by %s",
                 len(order), len(_groups),
                 (" (%d failed; their spans were retried one at a time)" % failed)
                 if failed else "",
                 sum(g["spans"] * g["longest_seconds"] for g in _groups),
                 sum(m.seconds for m in prep),
                 ", ".join("%s=%d" % kv for kv in sorted(closed.items())))
        if dense:
            log.warning("align: %d of %d spans carry more than %.0f units an audio second. "
                        "They were aligned anyway -- this is a note about the input, not a "
                        "refusal -- and their timings are worth a look",
                        dense, len(order), MAX_UNITS_A_SECOND)
    _say_if_nothing_aligned(out)
    return out


def build_app(supports):
    app = FastAPI(title="audio-align (Qwen3-ForcedAligner)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="align", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.get("/v1/audio/align/telemetry")
    def telemetry():
        """Everything this engine knows about what it is running on. Read-only."""
        d = _dims()
        return {
            "now": _snapshot(),
            "budget": {"positions": round(_budget, 1), "fraction": BUDGET_HEADROOM,
                       "pinned": ALIGN_FIXED_BUDGET or None, "priced": _budget_priced,
                       "group_slack_positions": ALIGN_GROUP_SLACK},
            "model": {"scale": round(_scale, 4),
                      "scale_seen": _scale_seen, "calls_seen": _calls_seen,
                      "ratio_lo": _ratio_lo, "ratio_hi": _ratio_hi,
                      "bytes_a_position": _bytes_a_position(d) if d else None,
                      "bytes_an_audio_second": _bytes_a_second(d) if d else None,
                      "kv_cache": CACHE_ON},
            "requests": list(collections.deque(_telemetry)),
        }

    @app.post("/v1/audio/align")
    async def align(file: UploadFile = File(...), text: str = Form(default=None),
                    language: str = Form(default=None), segments: str = Form(default=None),
                    async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready",
                                headers={"Retry-After": "10"})
        data = await file.read()
        if MAX_UPLOAD_BYTES > 0 and len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413,
                                detail="upload is %d bytes, over the %d this engine accepts"
                                       % (len(data), int(MAX_UPLOAD_BYTES)))
        if segments:
            segs = parse_segments(segments)

            def _probe():
                import io as _io
                import soundfile as _sf

                # The header only -- it stops at the format block, so an unreadable body is a 400 before
                # anything is decoded.
                _sf.info(_io.BytesIO(data))

            try:
                await asyncio.to_thread(_probe)
            except Exception as e:
                raise HTTPException(status_code=400, detail="could not read audio: %s" % e)

            # Decoded on the task worker, not in the request handler: the queue then holds undecoded
            # bytes, and one worker means one decoded request in memory at a time. Crossing the
            # container's memory limit is an OOMKill that takes every in-flight request with it.
            def _decode_all():
                import io as _io
                import soundfile as _sf

                a, sr = _sf.read(_io.BytesIO(data), dtype="float32", always_2d=True)
                return a.mean(axis=1), int(sr)  # -> mono

            def _work_batch(ctx):
                out = []
                started = _snapshot()
                del _groups[:]
                _oom_count[0] = 0
                try:
                    arr, sr = _decode_all()
                except Exception as e:
                    _telemetry_record(started, len(segs), _covered_seconds(segs))
                    raise HTTPException(status_code=400,
                                        detail="could not decode audio: %s" % e)
                ctx.progress(stage="align", done=0, total=len(segs))
                sent = [0]
                if BATCH:
                    try:
                        results = _batched(ctx, segs, arr, sr, language)
                        return {"model": MODEL_NAME, "mode": "align", "batch": True,
                                "grouped": any(g["spans"] > 1 and not g["failed"]
                                               for g in _groups),
                                "results": results}
                    finally:
                        _telemetry_record(started, len(segs),
                                          _covered_seconds(segs))
                try:
                    for i, seg in enumerate(segs, 1):
                        ctx.checkpoint()
                        try:
                            stext = (str(seg.get("text") or "")).strip()
                            if not stext:
                                out.append({"units": [], "language": None})
                                continue
                            lo = max(0, int(float(seg.get("start") or 0) * sr))
                            hi = min(len(arr), int(float(seg.get("end") or 0) * sr))
                            if hi <= lo:
                                out.append({"error": "empty segment"})
                                continue
                            span_sec = (hi - lo) / float(sr)
                            lang = ((str(seg.get("language") or language or "")).strip()
                                    or DEFAULT_LANGUAGE)
                            refused = _admit(span_sec, stext, lang)
                            if refused:
                                out.append({"error": refused})
                                continue
                            import soundfile as _sf

                            clip = arr[lo:hi]
                            p = None
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".wav",
                                                                 delete=False) as f:
                                    p = f.name
                                _sf.write(p, clip, sr, format="WAV", subtype="PCM_16")
                                sent[0] += 1
                                res = _align(p, stext, lang)
                            finally:
                                if p:
                                    unlink(p)
                                # In the finally, and after the span is read: a failed span still cost the audio, and the
                                # batched path bills those too. Metering outside it made the same request cost four
                                # seconds batched and three serial.
                                ctx.meter(input_seconds=span_sec)
                            out.append({"language": lang, "units": _units(res)})
                        except tasks.Cancelled:
                            raise
                        except Exception as e:
                            out.append({"error": "align failed: %s" % e})
                        finally:
                            ctx.progress(done=i, total=len(segs))
                    _say_if_nothing_aligned(out)
                    return {"model": MODEL_NAME, "mode": "align", "batch": True,
                            "grouped": False, "results": out}
                finally:
                    _telemetry_record(started, len(segs),
                                      _covered_seconds(segs), calls=sent[0])

            return await tasks.dispatch(async_, "align", MODEL_NAME, _work_batch,
                                        fail="alignment failed")
        if not (text or "").strip():
            raise HTTPException(status_code=400, detail="`text` is required for forced alignment")
        path = await asyncio.to_thread(spill, data, file.filename)
        seconds = await asyncio.to_thread(duration, path)
        lang = ((str(language or "")).strip() or DEFAULT_LANGUAGE)
        refused = _admit(seconds, text, lang) if seconds is not None else None
        if refused:
            unlink(path)
            raise HTTPException(status_code=413, detail=refused)
        lang = (language or "").strip() or DEFAULT_LANGUAGE

        def _work(ctx):
            started = _snapshot()
            del _groups[:]
            _oom_count[0] = 0
            try:
                ctx.meter(input_seconds=seconds)
                ctx.progress(ratio=0.0, stage="align")
                # Gated on the switch, not on the counter being readable: gated the other way this ran on
                # every host with a card while the promise that grouping-off leaves this machinery unrun
                # went green on CI, which has none.
                # 🔴 _cost, not the raw position count: a single span is where the encoder dominates, so
                # the count understates the call 2.25x with the cache on and 12.2x without, and _scale is
                # shared with the grouped path.
                reading = _memory_now() if BATCH else None
                res = _align(path, text, lang)   # resets the peak itself if it raises
                if reading is not None:
                    _observe(_cost([_Span(0, None, text, lang, seconds or 0.0,
                                          _span_positions(seconds or 0.0, text, lang))]),
                             _used_bytes(*reading))
                ctx.progress(ratio=1.0, stage="done")
                return {"model": MODEL_NAME, "mode": "align", "device": _state["device"],
                        "language": lang, "units": _units(res)}
            finally:
                _telemetry_record(started, 1, seconds or 0.0, calls=1)

        return await tasks.dispatch(async_, "align", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="alignment failed")

    return app


def run(supports):
    ov = ovutil.requested()
    _runtime.serve(supports, (_load_ov if ov else _load), build_app, "Qwen3-ForcedAligner")
