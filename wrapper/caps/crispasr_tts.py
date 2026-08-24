"""Text to speech through CrispASR ggml: Voxtral-4B-TTS in this process, no child engine."""
import asyncio
import base64
import io
import logging
import os
import re
import struct
import threading
import time

import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .. import hfgate
from .. import tasks
from ..audioio import seconds
from ..contract import register, EngineArgs
from ..crispasr_voices import listed_voices as _listed_voices
from ..runtime import Runtime

log = logging.getLogger("audio-crispasr-tts")

_runtime = Runtime("mistralai/Voxtral-4B-TTS-2603", default_repo="cstr/voxtral-4b-tts-GGUF",
                   session=None, speakers=[], out_sr=0, quant="", voice_set="")
MODEL_NAME = _runtime.model_name
# MODEL_SOURCE may carry llm-init flags after the repo (`--include a.gguf`); those are not a hub id.
MODEL_REPO = (_runtime.model_repo.split() or [""])[0]
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
# Naming the backend turns a wrong guess from GGUF metadata into a startup error, not odd audio.
BACKEND = str(_args.text("--backend", "voxtral-tts") or "voxtral-tts")
# A substring, not a filename, so a repo that renames its quantizations needs no chart change.
GGUF_MATCH = str(_args.text("--gguf", "q8_0") or "q8_0")
N_THREADS = _args.count("--n-threads", 4)
# The voice a request does not name; presets are listed at GET /v1/audio/voices.
DEFAULT_VOICE = str(_args.text("--voice", "") or "")
# 0 on any of these four means "leave the backend's own default alone".
TEMPERATURE = _args.number("--temperature", 0)
SEED = _args.count("--seed", 0)
MAX_NEW_TOKENS = _args.count("--max-new-tokens", 0)
TTS_STEPS = _args.count("--tts-steps", 0)

# Reported by /v1/models before the weights are open; the session's real rate replaces it.
OUT_SR = 24000
# Opening the GGUF is the only boot work; CUDA graphs are captured on the first real request.
BOOT_TIMEOUT_S = 1800.0

# What soundfile can write here; anything else is a 400 rather than a corrupt file.
_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),
}

_state = _runtime.state
# One ggml session is one set of compute buffers, and voice selection is sticky state on it.
_gen_lock = threading.Lock()

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|(?<=[。！？；：])\s*|\n+")
# Below this we keep merging sentences: a 4B AR pass costs more than the extra granularity buys.
_STREAM_MIN_CHARS = 60


def _model_path():
    """The pre-downloaded snapshot: llm-init fetches the weights before this process ever starts."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _pick_gguf(root):
    """The GGUF to open: prefer the --gguf substring, else the largest file."""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".gguf"):
                path = os.path.join(dirpath, name)
                try:
                    found.append((os.path.getsize(path), path))
                except OSError:
                    continue
    if not found:
        raise RuntimeError("no .gguf under %s; MODEL_SOURCE must name a GGUF repo such as "
                           "hf://cstr/voxtral-4b-tts-GGUF" % root)
    want = GGUF_MATCH.strip().lower()
    if want:
        hits = [f for f in found if want in os.path.basename(f[1]).lower()]
        if hits:
            return max(hits)[1]
        log.warning("no GGUF matching %r in %s; using the largest of: %s", want, root,
                    ", ".join(sorted(os.path.basename(p) for _s, p in found)))
    return max(found)[1]


def _quant_label(path):
    """The quantization as /v1/models reports it, read off the filename the repo chose."""
    m = re.search(r"(?:^|[-_.])(iq\d[a-z_]*|q\d[a-z0-9_]*|bf16|f16|f32)(?:[-_.]|$)",
                  os.path.basename(path), re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _apply_tuning(sess):
    """The ENGINE_ARGS knobs that are set once for the life of the session."""
    if TEMPERATURE > 0:
        sess.set_temperature(float(TEMPERATURE), int(SEED))
    elif SEED:
        sess.set_tts_seed(int(SEED))
    if MAX_NEW_TOKENS > 0:
        sess.set_max_new_tokens(int(MAX_NEW_TOKENS))
    if TTS_STEPS > 0:
        sess.set_tts_steps(int(TTS_STEPS))


def _load():
    from crispasr import Session

    gguf = _pick_gguf(_model_path())
    quant = _quant_label(gguf)
    log.info("opening %s (backend=%s, quant=%s, threads=%d)", gguf, BACKEND, quant or "?", N_THREADS)
    t0 = time.time()
    sess = Session(gguf, n_threads=N_THREADS, backend=BACKEND)
    log.info("session open in %.1fs, backend reports %r", time.time() - t0, sess.backend)
    _apply_tuning(sess)
    try:
        listed = [str(s) for s in (sess.speakers() or [])]
    except Exception as e:
        log.info("no preset speakers from %s (%s)", sess.backend or BACKEND, e)
        listed = []
    speakers = _listed_voices(listed, sess.backend or BACKEND)
    if speakers and not listed:
        log.info("session listed no preset voices; advertising the %d Voxtral-4B-TTS names",
                 len(speakers))
    try:
        out_sr = int(sess.output_sample_rate())
    except Exception:
        out_sr = OUT_SR
    if DEFAULT_VOICE and speakers and DEFAULT_VOICE not in speakers:
        # Loud but not fatal: every request would otherwise fail on a value only the chart can fix.
        log.error("--voice %r is not a preset of this checkpoint (%s); unnamed requests will use "
                  "the backend default", DEFAULT_VOICE, ", ".join(speakers))
    return sess, speakers, out_sr, quant


def _select_voice(sess, voice):
    """Point the session at a preset: name first, then path-shaped set_voice()."""
    if not voice or _state.get("voice_set") == voice:
        return
    try:
        sess.set_speaker_name(str(voice))
    except Exception as by_name:
        try:
            sess.set_voice(str(voice))
        except Exception as by_path:
            raise RuntimeError("could not select voice %r (by name: %s; as a path: %s)"
                               % (voice, by_name, by_path))
    _state["voice_set"] = voice


def _synthesize_blocking(text, voice):
    """One synthesis under the lock; marked audio, never the Art. 50 unmarked path."""
    sess = _state["session"]
    with _gen_lock:
        _select_voice(sess, voice)
        t0 = time.time()
        audio = sess.synthesize(text)
        sr = int(_state["out_sr"] or OUT_SR)
    seconds = len(audio) / float(sr or 1)
    took = time.time() - t0
    # The figure this base exists to answer: above 1.0 it cannot keep up with a conversation.
    log.info("synthesized %.2fs of audio in %.2fs (RTF %.2f) for %d chars",
             seconds, took, took / seconds if seconds else 0.0, len(text))
    return audio, sr


def _boot():
    try:
        sess, speakers, out_sr, quant = _load()
        _state.update(session=sess, speakers=speakers, out_sr=out_sr, quant=quant)
        _state["ready"] = True
        log.info("ready: %s via %s at %d Hz, %d preset voice(s)", MODEL_NAME, BACKEND, out_sr,
                 len(speakers))
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("load failed: %s", _state["error"])


def _nvml_stats():
    """(used, total, util) for /metrics; device 0 is the one card the container was given."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() <= 0:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu) / 100.0
            except Exception:
                util = 0.0
            return int(mem.used), int(mem.total), util
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _mount_metrics(app):
    """The gauges gpu.py exposes, read through NVML because this image carries no torch."""

    @app.get("/metrics")
    def _metrics():
        stats = _nvml_stats()
        used, total, util = stats or (0, 0, 0.0)
        lines = []
        for name, help_, val in (
            ("gpu_present", "1 if a CUDA device is visible to this engine, else 0", 1 if stats else 0),
            ("gpu_mem_used_bytes", "GPU memory in use on this engine's device slice (bytes)", used),
            ("gpu_mem_total_bytes", "GPU memory CUDA reports to this engine (bytes)", total),
            ("gpu_util_ratio", "GPU compute utilization 0..1 (0 when unavailable)", "%.4f" % util),
        ):
            lines += ["# HELP %s %s" % (name, help_), "# TYPE %s gauge" % name,
                      "%s %s" % (name, val)]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


def _encode(audio, sr, fmt):
    """float32 mono to the requested container, as bytes."""
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if fmt == "pcm":
        return (pcm * 32767.0).astype("<i2").tobytes()
    import soundfile as sf

    subtype = _FORMATS[fmt][1]
    buf = io.BytesIO()
    kwargs = {"format": _FORMATS[fmt][0]}
    if subtype:
        kwargs["subtype"] = subtype
    sf.write(buf, pcm, sr, **kwargs)
    return buf.getvalue()


def _pcm16(audio):
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _wav_stream_header(sr):
    """A WAV header for a body whose length is not known yet, so the sizes stay at their maximum."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def _chunks(text):
    """Text split where audio can be cut, merged up to a floor so a chunk is worth its overhead."""
    parts, buf = [], ""
    for piece in _SENTENCE_END.split(text.strip()):
        if not piece or not piece.strip():
            continue
        buf = ("%s %s" % (buf, piece.strip())).strip() if buf else piece.strip()
        if len(buf) >= _STREAM_MIN_CHARS:
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts or [text.strip()]


def build_app(supports):
    app = FastAPI(title="audio-crispasr-tts (Voxtral-4B-TTS via ggml)")
    _mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="crispasr_tts", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             model_format="gguf", quantization=_state.get("quant") or None,
             sample_rate=_state.get("out_sr") or OUT_SR)

    # No child engine takes the leftovers, so a mistyped flag would otherwise vanish silently.
    _args.warn_unclaimed(log)

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    def _headers(fmt):
        return {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "tts", "X-Audio-Format": fmt,
                "X-Audio-Sample-Rate": str(_state.get("out_sr") or OUT_SR)}

    def _check(payload):
        """Validate the OpenAI body and settle text, format and voice."""
        text = str(payload.get("input") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="input (the text to speak) is required")
        fmt = str(payload.setdefault("response_format", "wav")).strip().lower()
        payload["response_format"] = fmt
        if fmt not in _FORMATS:
            raise HTTPException(status_code=400, detail="response_format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        voice = str(payload.get("voice") or DEFAULT_VOICE or "").strip()
        if voice and _state["speakers"] and voice not in _state["speakers"]:
            raise HTTPException(status_code=400,
                                detail="unknown voice %r; see GET /v1/audio/voices" % voice)
        if payload.get("ref_audio") or payload.get("instructions"):
            # Refused, not ignored: returning a preset would look like cloning that worked badly.
            raise HTTPException(status_code=400,
                                detail="%s speaks its preset voices only: the published weights "
                                       "carry no audio encoder, so reference audio and voice "
                                       "instructions cannot be honoured. Pick a voice from GET "
                                       "/v1/audio/voices." % MODEL_NAME)
        return text, fmt, (voice or None)

    async def _stream_response(text, fmt, voice):
        if fmt not in ("pcm", "wav"):
            raise HTTPException(status_code=400,
                                detail="stream supports response_format pcm or wav; %s only exists "
                                       "as a whole file" % fmt)
        parts = _chunks(text)
        sr = int(_state.get("out_sr") or OUT_SR)

        async def body():
            if fmt == "wav":
                yield _wav_stream_header(sr)
            for part in parts:
                audio, _sr = await asyncio.to_thread(_synthesize_blocking, part, voice)
                yield _pcm16(audio)

        return StreamingResponse(body(), media_type=_FORMATS[fmt][2], headers=_headers(fmt))

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        _require_ready()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON in the OpenAI /v1/audio/speech shape")
        text, fmt, voice = _check(payload)
        if payload.get("stream"):
            return await _stream_response(text, fmt, voice)

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="synthesis")
            audio, sr = _synthesize_blocking(text, voice)
            ctx.meter(output_seconds=seconds(audio, sr))
            out = _encode(audio, sr, fmt)
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(out, _FORMATS[fmt][2], suffix="." + fmt, headers=_headers(fmt))

        # async travels in the query string, so the body stays exactly OpenAI's.
        return await tasks.dispatch(request.query_params.get("async"), "tts", MODEL_NAME, _work,
                                    fail="speech synthesis failed")

    @app.post("/v1/audio/speech/batch")
    async def speech_batch(request: Request):
        # One session, one lock: items run back to back, so the win is one dispatch not parallelism.
        _require_ready()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON for /v1/audio/speech/batch")
        items = payload.get("items")
        if not isinstance(items, list) or not items or len(items) > 32:
            raise HTTPException(status_code=400, detail="items must be a non-empty array (1–32)")
        shared = {k: v for k, v in payload.items() if k != "items"}
        rows = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="items[%d] must be an object" % i)
            row = dict(shared)
            row.update(item)
            row.pop("stream", None)
            rows.append(_check(row))

        def _work(ctx):
            n = len(rows)
            ctx.progress(ratio=0.0, stage="batch", done=0, total=n)
            out = []
            for i, (text, fmt, voice) in enumerate(rows):
                audio, sr = _synthesize_blocking(text, voice)
                ctx.meter(output_seconds=seconds(audio, sr))
                out.append({"index": i, "format": fmt, "sample_rate": sr,
                            "audio": base64.b64encode(_encode(audio, sr, fmt)).decode("ascii")})
                ctx.progress(ratio=(i + 1) / n, stage="batch", done=i + 1, total=n)
            return {"model": MODEL_NAME, "items": out}

        return await tasks.dispatch(request.query_params.get("async"), "tts", MODEL_NAME, _work,
                                    fail="speech batch failed")

    @app.get("/v1/audio/voices")
    def voices_list():
        _require_ready()
        return JSONResponse({"model": MODEL_NAME,
                             "voices": [{"id": s} for s in _state["speakers"]]})

    return app


def run(supports):
    _runtime.serve(supports, _boot, build_app, "crispasr ggml engine",
                   timeout_s=BOOT_TIMEOUT_S)
