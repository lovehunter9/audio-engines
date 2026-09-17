# Streaming ASR on one in-process transformers load, serving BOTH offline stt and WebSocket stt_stream.
import os
import json
import asyncio
import logging
import math
import threading
import time
import types

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import Response

from .. import hfgate
from .. import tasks
from ..batch import parse_segments
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


def _is_ov():
    """The ov image bakes AUDIO_BASE=ov. Never infer Intel from visible hardware."""
    return (os.environ.get("AUDIO_BASE") or "").strip() == "ov"


def _gpu_mode():
    return (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()


def _default_max_model_len():
    # Chart used to stuff this into empty ENGINE_ARGS. The user-written flag still wins;
    # this is only what the engine fills in when the flag is absent.
    return 3072 if _gpu_mode() == "nvidia-gb10" else 8192


def _default_enforce_eager():
    # Same story as max-model-len: NVIDIA installs used to always get --enforce-eager.
    # OpenVINO does not read the flag, so the inferred default stays off there.
    return not _is_ov()


# vLLM wants a share of the whole card; the platform hands out a quota, so derive one from it.
GPU_UTIL = _args.number("--gpu-memory-utilization", memory_fraction() or 0.45)
# Holds ONE unit of work; the chart sizes it per machine type, since unified memory needs less.
MAX_MODEL_LEN = max(1, _args.count("--max-model-len", _default_max_model_len()))
# Capture is where startup wedges holding the vGPU lock.
ENFORCE_EAGER = _args.switch("--enforce-eager", _default_enforce_eager())
# How many spans one generate() may carry; default 1.
MAX_BATCH_SPANS = max(1, _args.count("--batch-max-spans", 1))
# End a span that has started repeating rather than folding the loop out afterwards.
REPETITION_FALLBACK_TOKENS_PER_SEC = max(
    0, _args.count("--repetition-fallback-tokens-per-sec", 12))
_FALLBACK_ASKED = _args.given("--repetition-fallback-tokens-per-sec")
TOKENS_FLOOR = 64
REPETITION_DEFAULTS = {"min_pattern_size": 2, "max_pattern_size": 20, "min_count": 20}
# switch() is the bare/boolean form; text() is the JSON override.


def repetition_request(args):
    """(asked for?, JSON override, note) for --repetition-detection. A function so it is testable.

    🔴 The rule is "only JSON is an override", not "these words are not overrides". A word
    list has an outside, and everything outside it used to become an override: `false` was
    the spelling review caught, but `disabled`, `none` and `never` all read as "on" for the
    same reason -- a non-empty string is truthy. Inverting it removes the outside.
    """
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
# OpenVINO base (`AUDIO_BASE=ov`) only. Claimed here so a leftover `--device` is not a silent typo
# on the Intel image; the qwen/vLLM load never reads these.
OV_DEVICE = _args.text("--device", "")
OV_MAX_NEW_TOKENS = _args.count("--max-new-tokens", 256)
_args.warn_unclaimed(log)

MAX_NEW_TOKENS = 32
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 2.0
DEFAULT_STEP_MS = 500
# Finalize + re-init this often, so a long session never overflows vLLM's ~8192-token encoder cache.
ROLL_SEC = 240.0
_CAPTURE_SIZES = (1, 2, 4, 8)
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

# Qwen3-ASR on OpenVINO wants English names; WS/start often sends ISO 639-1.
_OV_LANG = {
    "en": "English", "zh": "Chinese", "yue": "Chinese", "ja": "Japanese",
    "ko": "Korean", "de": "German", "fr": "French",
    "english": "English", "chinese": "Chinese", "japanese": "Japanese",
    "korean": "Korean", "german": "German", "french": "French",
}


def _ov_device():
    if OV_DEVICE:
        return OV_DEVICE
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel"):
        return "GPU"
    gpu_raw = (os.environ.get("REQUIRED_GPU_MEMORY") or "").strip()
    if gpu_raw in ("", "0"):
        return "CPU"
    return "GPU"


def _ov_language(raw):
    if not raw:
        return None
    key = str(raw).strip()
    return _OV_LANG.get(key.lower(), key)


def _gated(fn, *a):
    with _gpu:
        return fn(*a)


def _p(msg):
    print("[stream] " + msg, flush=True)


def _capture_kw():
    if ENFORCE_EAGER:
        how = "flag" if _args.given("--enforce-eager") else "inferred"
        _p("--enforce-eager %s: skipping CUDA graphs entirely" % how)
        return {"enforce_eager": True}
    sizes = list(_CAPTURE_SIZES)
    try:
        from vllm.config import CompilationConfig

        fields = set(getattr(CompilationConfig, "model_fields", None) or {})
    except Exception as e:
        _p("WARN cannot inspect vLLM CompilationConfig (%s); leaving capture sizes alone" % e)
        return {}
    for name in ("cudagraph_capture_sizes", "capture_sizes"):
        if name in fields:
            return {"compilation_config": {name: sizes}}
    _p("WARN CompilationConfig has no capture-size field; leaving capture sizes alone")
    return {}


def _xml_has_input(path, name, limit=1048576):
    """True if an OpenVINO IR lists `name` in the first `limit` bytes (graph inputs sit there)."""
    try:
        with open(path, "rb") as f:
            head = f.read(limit)
    except OSError:
        return False
    return name.encode("ascii") in head


def _looks_like_ov_ir(path):
    """GenAI's Qwen3ASRDecoder compiles openvino_decoder_model.xml and sets beam_idx.

    Any .xml is not enough: `--task automatic-speech-recognition` (no -with-past)
    writes a Whisper-style stateless decoder. ASRPipeline then 500s with
    'Port for tensor name beam_idx was not found.'
    """
    if not path or not os.path.isdir(path):
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    if not (os.path.isfile(enc) and os.path.isfile(dec)):
        return False
    return _xml_has_input(dec, "beam_idx")


def _ov_export_cmd(src, dest):
    # Local snapshots cannot infer the HF pipeline task. Hub `auto` then
    # upgrades ASR to `-with-past` (stateful decoder, beam_idx in OpenVINO
    # state). Passing the task without that suffix skips the upgrade.
    return [
        "optimum-cli", "export", "openvino",
        "--model", src,
        "--task", "automatic-speech-recognition-with-past",
        "--trust-remote-code",
        dest,
    ]


def _resolve_hf_dir(repo):
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, local_files_only=True)


def _ensure_ov_ir(src):
    """ASRPipeline wants a converted OpenVINO directory, not the raw HF checkpoint.

    Prefer an `openvino/` subdir (what the chart can ship) or xml already in src;
    otherwise export once next to the snapshot so the next start skips this.
    A previous start that exported the stateless decoder is treated as missing.
    """
    nested = os.path.join(src, "openvino")
    if _looks_like_ov_ir(src):
        return src
    if _looks_like_ov_ir(nested):
        return nested
    dest = nested
    if os.path.isdir(dest) and not _looks_like_ov_ir(dest):
        import shutil
        _p("removing unusable export dir %s (need encoder+decoder with beam_idx)" % dest)
        shutil.rmtree(dest)
    _p("no OpenVINO IR in %s; exporting to %s (first start is slow)" % (src, dest))
    os.makedirs(dest, exist_ok=True)
    import subprocess

    cmd = _ov_export_cmd(src, dest)
    _p("running: %s" % " ".join(cmd))
    subprocess.check_call(cmd)
    if not _looks_like_ov_ir(dest):
        raise RuntimeError(
            "optimum-cli export finished but %s is not a GenAI ASR IR "
            "(need openvino_encoder_model.xml + openvino_decoder_model.xml with beam_idx)"
            % dest
        )
    return dest


def _load_ov():
    _p("importing openvino_genai (device=%s) ..." % _ov_device())
    import openvino_genai as ov_genai

    src = _resolve_hf_dir(MODEL_REPO)
    model_dir = _ensure_ov_ir(src)
    device = _ov_device()
    # ov20 blobs rewired Unsqueeze to [B,T,H]; short T died on Intel GPU.
    # ov21 flatten+index-shift compiled on iGPU and died CL_OUT on Arc B>1 speech.
    # ov22 is [B,T,H] again plus 16s silence pad. Bust the ov21 compile cache.
    cache = os.path.join(os.environ.get("HF_HOME") or "/tmp", "openvino_cache_ov22")
    os.makedirs(cache, exist_ok=True)
    _p("ASRPipeline(model=%s, device=%s)" % (model_dir, device))
    pipe = ov_genai.ASRPipeline(model_dir, device, CACHE_DIR=cache)
    _ov_assert_batch_generate(pipe)
    _state["asr"] = pipe
    _state["backend"] = "openvino"
    _say_repetition_once()
    _warmup()
    _state["ready"] = True
    _p("engine READY: %s (openvino %s)" % (MODEL_REPO, device))
    log.info("openvino-genai ASR loaded: %s device=%s", MODEL_REPO, device)


def _ov_result_text(result):
    texts = getattr(result, "texts", None)
    if texts:
        return (texts[0] or "").strip()
    t = getattr(result, "text", None)
    if t:
        return str(t).strip()
    return (str(result) if result is not None else "").strip()


def _ov_result_language(result, fallback=None):
    langs = getattr(result, "languages", None)
    if langs:
        return langs[0] or fallback
    return fallback


def _ov_generate(audio, language=None, streamer=None):
    asr = _state["asr"]
    raw = audio.astype("float32").reshape(-1).tolist()
    kw = {"max_new_tokens": _ov_max_new_tokens(len(raw) / 16000.0)}
    lang = _ov_language(language)
    if lang:
        kw["language"] = lang
    if streamer is not None:
        kw["streamer"] = streamer
    return asr.generate(raw, **kw)


def _ov_result_texts(result, n):
    texts = getattr(result, "texts", None)
    if not texts:
        if n == 1:
            return [_ov_result_text(result)]
        raise RuntimeError("openvino generate returned no texts for %d clips" % n)
    out = [(t or "").strip() for t in texts]
    if len(out) != n:
        raise RuntimeError("openvino generate returned %d texts for %d clips"
                           % (len(out), n))
    return out


def _ov_assert_batch_generate(pipe):
    """Fail load if generate() cannot take a list, or if two clips copy one text.

    Silence-only probes used to pass TypeError and still ship the clip-0
    flatten. Two short tones that both come back empty are inconclusive;
    two nonempty identical strings are the ov17/ov19 bug and abort load.
    The decoder constructor rewires the flatten Unsqueeze to [B,T,H].
    CL_OUT_OF_RESOURCES must abort load: ov21's sine probe swallowed it and
    the engine came up READY, then the first real-speech batch poisoned Arc.
    """
    if MAX_BATCH_SPANS <= 1:
        return
    def _tone(freq, n=16000):
        return [math.sin(2.0 * math.pi * freq * i / 16000.0) for i in range(n)]

    try:
        result = pipe.generate([_tone(440.0), _tone(880.0)], max_new_tokens=8)
    except TypeError as e:
        raise RuntimeError(
            "ENGINE_ARGS --batch-max-spans is %d but this OpenVINO build "
            "rejects a list of waveforms (%s). Use the patched audio-qwen-ov image."
            % (MAX_BATCH_SPANS, e)
        ) from e
    except Exception as e:
        msg = str(e)
        if "CL_OUT_OF_RESOURCES" in msg or "clFinish" in msg or "ocl_stream" in msg:
            raise RuntimeError(
                "openvino batch generate died on GPU (%s). "
                "This build cannot run a list of waveforms on this device."
                % e
            ) from e
        _p("batch generate probe accepted a list (%s); continuing" % e)
        return
    texts = [(t or "").strip() for t in (getattr(result, "texts", None) or [])]
    nonempty = [t for t in texts if t]
    if len(nonempty) >= 2 and len(set(nonempty)) == 1:
        raise RuntimeError(
            "openvino batch generate copied one text onto every clip: %r" % texts
        )


def _ov_generate_many(clips, language=None):
    """One OpenVINO generate() for the group. The patched binding takes a list
    of waveforms; infer() encodes per clip then one decoder.generate() on the
    stacked hidden states. Flatten [1,B*T,H] plus unshifted GatherElements
    copies clip 0 (intel-ov17/ov19); ov20's [B,T,H] rewire dies on short T;
    ov21's flatten+index-shift dies CL_OUT on Arc speech. ov22 keeps [B,T,H]
    and pads clips shorter than 16s.
    A TypeError is the unpatched wheel, which cannot take a list at all.
    """
    if len(clips) == 1:
        return [_ov_result_text(_ov_generate(clips[0], language=language))]
    asr = _state["asr"]
    raws = [c.astype("float32").reshape(-1).tolist() for c in clips]
    kw = {"max_new_tokens": _ov_max_new_tokens(max(len(r) for r in raws) / 16000.0)}
    lang = _ov_language(language)
    if lang:
        kw["language"] = lang
    result = asr.generate(raws, **kw)
    return _ov_result_texts(result, len(clips))


def _load_blocking():
    import torch
    from qwen_asr import Qwen3ASRModel

    _p("importing qwen_asr ...")
    try:
        import qwen_asr.inference.qwen3_asr as _qasr_mod
        import qwen_asr.inference.utils as _qasr_utils

        _qasr_utils.MAX_ASR_INPUT_SECONDS = OFFLINE_MAX_INPUT_SEC
        _qasr_mod.MAX_ASR_INPUT_SECONDS = OFFLINE_MAX_INPUT_SEC
        _p("patched qwen-asr MAX_ASR_INPUT_SECONDS -> %ds" % OFFLINE_MAX_INPUT_SEC)
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
       "(--gpu-memory-utilization=%.2f --max-model-len=%d unused on transformers)"
       % (MODEL_REPO, dev, GPU_UTIL, MAX_MODEL_LEN))
    try:
        asr = Qwen3ASRModel.from_pretrained(MODEL_REPO, **kw)
    except TypeError:
        kw.pop("token", None)
        asr = Qwen3ASRModel.from_pretrained(MODEL_REPO, **kw)
    _state["asr"] = asr
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
    from qwen_asr.inference.utils import SAMPLE_RATE, normalize_language_name, validate_language

    force = None
    if language is not None and str(language).strip():
        ln = normalize_language_name(str(language))
        validate_language(ln)
        force = ln
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


def _warmup():
    """One throwaway transcription before we report ready, so no caller pays the cold-start cost.

    vLLM answers as soon as its constructor returns, but the first real inference still compiles
    and captures graphs: measured at 74s on a time-sliced card against 0.3s once warm. llm-init
    allows an upstream 60s to produce response headers, so without this the first offline caller
    reads a 502 where a transcript belongs. Streaming pays the same cost, only spread across an
    already-open socket where nothing times out -- and either path warms the other, so warming
    the simpler one here covers both.
    """
    import numpy as np

    t0 = time.time()
    try:
        # A voiced-band tone, not silence: the encoder may skip a silent clip and warm nothing.
        n = 16000
        tone = (0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000.0)).astype("float32")
        _offline_transcribe(tone)
        _p("warmup transcription took %.0fs" % (time.time() - t0))
    except Exception as e:
        _p("WARN warmup transcription failed after %.0fs: %s" % (time.time() - t0, e))
        if _is_ov():
            # A broken IR still compiles and reports READY; every real request then 500s.
            raise RuntimeError("openvino warmup failed (IR is not usable): %s" % e) from e


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
    """The detector class this vLLM has, or None. Probed once, and says so once.

    Asked independently of the flags, because the backstop below turns on what this build
    CANNOT do rather than on what was requested.

    Not imported at module load: a vLLM without RepetitionDetectionParams must still serve,
    and the failure has to be one log line rather than an engine that will not start.
    """
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

    None when nothing was set, so a build that will not take the attribute does not then
    fail again inside a finally block -- where the failure would replace the transcript.
    """
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
    """One line saying which of the two is in force, and why the other is not.

    Without it the three states are indistinguishable from outside: detector running,
    fallback capping, and nothing at all all look like an engine that transcribes.
    """
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
    # OpenVINO is a different runtime, not a vLLM-without-the-class. It applies the
    # same per-second fallback through max_new_tokens; do not look for sampling_params
    # or import vLLM just to log that they are missing.
    if _is_ov():
        if REPETITION_FALLBACK_TOKENS_PER_SEC <= 0:
            _p("WARN repetition fallback off (--repetition-fallback-tokens-per-sec=0) and "
               "OpenVINO has no vLLM detector: a repeating span is bounded only by "
               "max_new_tokens=%d" % OV_MAX_NEW_TOKENS)
            return
        _p("repetition fallback: OpenVINO has no vLLM detector, capping output at %d tokens "
           "per audio second via max_new_tokens (same rule as a vLLM build without the "
           "detector; too low truncates, and truncation is not visible here)%s" % (
               REPETITION_FALLBACK_TOKENS_PER_SEC,
               ". --batch-max-spans above 1 sends the group in one generate(); "
               "the cap follows the longest clip"
               if MAX_BATCH_SPANS > 1 else ""))
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
           ". With --batch-max-spans above 1 the cap follows the LONGEST clip in each group, "
           "so a short span that starts repeating is bounded by that clip's budget, not its own"
           if MAX_BATCH_SPANS > 1 else ""))


def _fallback_token_budget(seconds, stock):
    """The length cap that stands in when this build has no vLLM detector.

    One formula, two stock ceilings: CUDA keeps OFFLINE_MAX_TOKENS, OpenVINO keeps
    --max-new-tokens. The per-second number is the model's, not the runtime's.
    """
    if not REPETITION_ON or REPETITION_FALLBACK_TOKENS_PER_SEC <= 0:
        return stock
    return max(TOKENS_FLOOR,
               min(OFFLINE_MAX_TOKENS,
                   int(seconds * REPETITION_FALLBACK_TOKENS_PER_SEC) + TOKENS_FLOOR))


def _token_budget(seconds):
    # Replace the stock budget only when there is no detector and fallback tokens/sec > 0.
    _say_repetition_once()
    if not REPETITION_ON or _detector_class() is not None:
        return OFFLINE_MAX_TOKENS
    return _fallback_token_budget(seconds, OFFLINE_MAX_TOKENS)


def _ov_max_new_tokens(seconds):
    """OpenVINO has no RepetitionDetectionParams; use the same fallback the CUDA path uses
    when this vLLM build lacks the class. Do not import vLLM just to prove it is missing.
    """
    _say_repetition_once()
    return _fallback_token_budget(seconds, OV_MAX_NEW_TOKENS)


def _offline_transcribe(audio, language=None):
    # Native offline transcription on the same load; max_tokens is raised then restored.
    if _is_ov():
        return _ov_result_text(_ov_generate(audio, language=language))
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
        results = asr.transcribe(audio=(audio, 16000), language=None, return_time_stamps=False)
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
    r = results[0] if results else None
    t = getattr(r, "text", None) if r is not None else None
    if t is None and isinstance(r, dict):
        t = r.get("text")
    return (t or "").strip()


def _offline_transcribe_many(clips):
    if _is_ov():
        return _ov_generate_many(clips)
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
                                 language=None, return_time_stamps=False)
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
    texts = []
    for r in (results or []):
        t = getattr(r, "text", None)
        if t is None and isinstance(r, dict):
            t = r.get("text")
        texts.append((t or "").strip())
    if len(texts) != len(clips):
        raise RuntimeError("transcribe returned %d results for %d clips"
                           % (len(texts), len(clips)))
    return texts


async def _ws_openvino(ws: WebSocket):
    """OpenVINO output-side streaming: buffer PCM until stop, then token-stream partials.

    Must not emit `partial` while the client is still sending audio — that would be the
    fake live path we refuse. Decoder tokens after the utterance are the advertised stream.
    """
    import numpy as np

    await ws.accept()
    if not _state["ready"]:
        await ws.send_text(json.dumps({"type": "error",
                                       "detail": _state["error"] or "model not ready"}))
        await ws.close()
        return
    sample_rate = 16000
    language = None
    pending = np.zeros((0,), dtype="float32")
    total = 0
    await ws.send_text(json.dumps({"type": "ready"}))
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
                    continue
                if t in ("stop", "done", "finish"):
                    break
                continue
            data = msg.get("bytes")
            if not data:
                continue
            seg = resample_linear(pcm16_to_float32(data), sample_rate)
            pending = np.concatenate([pending, seg]) if pending.size else seg
            total += int(seg.shape[0])
        if pending.size:
            loop = asyncio.get_running_loop()
            acc = [""]
            out_q = asyncio.Queue()

            def streamer(subword):
                piece = subword if isinstance(subword, str) else str(subword)
                acc[0] += piece
                loop.call_soon_threadsafe(out_q.put_nowait, acc[0])
                try:
                    import openvino_genai as ov_genai
                    return ov_genai.StreamingStatus.RUNNING
                except Exception:
                    return False

            def work():
                try:
                    result = _ov_generate(pending, language=language, streamer=streamer)
                    loop.call_soon_threadsafe(out_q.put_nowait, ("done", result))
                except Exception as e:
                    loop.call_soon_threadsafe(out_q.put_nowait, ("error", e))

            async with _infer_lock:
                worker = asyncio.create_task(asyncio.to_thread(_gated, work))
                result = None
                try:
                    while True:
                        item = await out_q.get()
                        if isinstance(item, tuple) and item[0] == "done":
                            result = item[1]
                            break
                        if isinstance(item, tuple) and item[0] == "error":
                            raise item[1]
                        await ws.send_text(json.dumps({
                            "type": "partial",
                            "text": item,
                            "language": _ov_language(language),
                        }))
                finally:
                    await worker
            final_text = acc[0] or _ov_result_text(result)
            lang = _ov_result_language(result, _ov_language(language))
            if not acc[0] and final_text:
                await ws.send_text(json.dumps({
                    "type": "partial", "text": final_text, "language": lang,
                }))
            await ws.send_text(json.dumps({
                "type": "final", "text": final_text, "language": lang,
            }))
        else:
            await ws.send_text(json.dumps({
                "type": "final", "text": "", "language": _ov_language(language),
            }))
        await ws.send_text(json.dumps({
            "type": "closed",
            "audio_seconds": round(total / 16000.0, 3),
        }))
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.exception("openvino stream error: %s", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
            await ws.close()
        except Exception:
            pass


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
            raw = await file.read()
            audio = await asyncio.to_thread(_decode_to_16k_mono, raw, file.filename)
            # BATCH mode (opt-in): `segments` JSON [{start,end}] — slice + transcribe each.
            if segments:
                segs = parse_segments(segments)

                def _work_batch(ctx):
                    ctx.progress(stage="transcribe", done=0, total=len(segs))
                    if MAX_BATCH_SPANS > 1:
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
                                out[i] = {"text": ""}
                            else:
                                spans.append((i, audio[lo:hi], (hi - lo) / 16000.0))
                        done = len(segs) - len(spans)
                        # Metered once per span even when its group is retried in halves.
                        _metered = [False] * len(segs)
                        # Work through the groups as a stack so a failed group is split and retried.
                        todo = [spans[at:at + MAX_BATCH_SPANS]
                                for at in range(0, len(spans), MAX_BATCH_SPANS)]
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
                                texts = _offline_transcribe_many([c for _, c, _s in group])
                                # zip stops at the shorter side, so a short answer would silently drop spans.
                                if len(texts) != len(group):
                                    raise RuntimeError(
                                        "engine returned %d results for %d spans"
                                        % (len(texts), len(group)))
                                for (i, _, _s), t in zip(group, texts):
                                    out[i] = {"text": t}
                                done += len(group)
                            except tasks.Cancelled:
                                raise
                            except Exception as e:
                                if len(group) > 1:
                                    mid = len(group) // 2
                                    todo.append(group[mid:])
                                    todo.append(group[:mid])
                                    continue
                                out[group[0][0]] = {"error": "stt failed: %s" % e}
                                done += 1
                            ctx.progress(done=done, total=len(segs))
                        ctx.progress(done=len(segs), total=len(segs))
                        return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}
                    out = []
                    for i, seg in enumerate(segs, 1):
                        ctx.checkpoint()
                        try:
                            lo = max(0, int(float(seg.get("start") or 0) * 16000))
                            hi = min(len(audio), int(float(seg.get("end") or 0) * 16000))
                            if hi <= lo:
                                out.append({"text": ""})
                            else:
                                ctx.meter(input_seconds=(hi - lo) / 16000.0)
                                out.append({"text": _offline_transcribe(audio[lo:hi], language=language)})
                        except tasks.Cancelled:
                            raise
                        except Exception as e:
                            out.append({"error": "stt failed: %s" % e})
                        finally:
                            ctx.progress(done=i, total=len(segs))
                    return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}

                return await tasks.dispatch(async_, "stt", MODEL_NAME, _work_batch,
                                            fail="transcription failed", gate=_gpu)

            # SINGLE mode.
            def _work(ctx):
                ctx.meter(input_seconds=len(audio) / 16000.0)
                ctx.progress(ratio=0.0, stage="transcribe")
                text = _offline_transcribe(audio, language=language)
                ctx.progress(ratio=1.0, stage="done")
                if response_format in ("text", "srt", "vtt"):
                    return Response(content=text, media_type="text/plain")
                return {"text": text}

            return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                        fail="transcription failed", gate=_gpu)

    if has_stream:
        @app.websocket("/v1/audio/stream")
        async def stream(ws: WebSocket):
            if _is_ov():
                await _ws_openvino(ws)
                return
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
    _p("stt_stream starting; model=%s port=%s supports=%s ov=%s" % (
        MODEL_REPO, PORT, supports, _is_ov()))

    def load():
        try:
            (_load_ov if _is_ov() else _load_blocking)()
        except Exception as e:
            _state["error"] = hfgate.explain(MODEL_REPO, e)
            _p("engine load FAILED: %s" % e)
            log.exception("engine load failed: %s", e)

    def build(served):
        app = build_app(served)
        _p("starting uvicorn on :%s (ready=%s)" % (PORT, _state["ready"]))
        return app

    # No server-initiated WS keepalive: bursty inference lags Pong and drops a healthy session.
    # First OpenVINO start may export IR; allow the same window faster-whisper uses for CT2.
    _runtime.serve(
        supports,
        load,
        build,
        "openvino-genai ASR" if _is_ov() else "qwen-asr",
        load_on_main=True,
        disable_ws_ping=True,
        **({"timeout_s": 5400} if _is_ov() else {}),
    )
