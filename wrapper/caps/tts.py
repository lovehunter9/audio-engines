# Text to speech in this process: faster-qwen3-tts holds the model, the wrapper is the only server.
#
# There used to be a child `vllm serve --omni` here. Its two-stage pipeline runs each stage as its
# own engine process with its own CUDA context and KV pool: upstream #2318 measured 22 GB for a
# 0.6B model whose weights are 2.6 GB, and the maintainer confirmed stage 1 never fills the KV pool
# it reserves. Scaled to the 1.7B that is ~13 GB of fixed overhead before a single cache block —
# more than the whole quota. In-process there is one context, one pool, and ~4.4 GB total.
import asyncio
import base64
import io
import json
import os
import logging
import struct
import tempfile
import threading
import time

import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import seconds
from ..runtime import Runtime

log = logging.getLogger("audio-tts")

_runtime = Runtime(model=None, speakers=[], custom_voice=False, voice_design=False)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
# Static-cache length: at 12 codec frames/s, 2048 covers ~2.8 min and pins that much KV.
MAX_SEQ_LEN = int(_args.number("--max-seq-len", 2048))
# 8 steps ≈ 667 ms of audio. Smaller cuts time-to-first-audio and costs a codec decode per chunk.
CHUNK_SIZE = int(_args.number("--chunk-size", 8))
# "Auto" lets the model detect; a caller may still pass `language` per request.
DEFAULT_LANGUAGE = str(_args.text("--language", "Auto") or "Auto")
DEFAULT_VOICE = str(_args.text("--voice", "") or "")
ATTN = str(_args.text("--attn-implementation", "sdpa") or "sdpa")

# The codec decodes at 24 kHz family-wide; a constant because /v1/models predates the weights.
OUT_SR = 24000
# Loading weights + capturing CUDA graphs. Generous, but a hung load must not read as "still loading".
BOOT_TIMEOUT_S = 1800.0
WARMUP_TIMEOUT_S = 300.0

# What soundfile can actually write here. Anything else is a 400 rather than a corrupt file.
_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),  # raw little-endian int16, no container
}
# audioread's ffmpeg backend goes by extension, so an unknown type keeps its own, not .wav.
_REF_SUFFIX = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
               "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
               "audio/x-flac": ".flac", "audio/ogg": ".ogg", "audio/opus": ".opus",
               "audio/aac": ".aac", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
               "audio/webm": ".webm", "video/mp4": ".mp4", "video/webm": ".webm",
               "video/quicktime": ".mov", "video/x-matroska": ".mkv"}

_state = _runtime.state
# CUDA graphs and the static cache are one set of buffers: two generations at once corrupt both.
_gen_lock = threading.Lock()


def _model_path():
    """The pre-downloaded snapshot: llm-init fetches the weights before this process ever starts."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _load():
    from faster_qwen3_tts import FasterQwen3TTS

    path = _model_path()
    log.info("loading %s from %s (max_seq_len=%d, attn=%s)", MODEL_NAME, path, MAX_SEQ_LEN, ATTN)
    m = FasterQwen3TTS.from_pretrained(path, device="cuda", attn_implementation=ATTN,
                                       max_seq_len=MAX_SEQ_LEN, local_files_only=True)
    speakers = []
    for holder in (m, getattr(m, "model", None)):
        getter = getattr(holder, "get_supported_speakers", None)
        if callable(getter):
            speakers = list(getter() or [])
            if speakers:
                break
    # Kind is read off the weights (e.g. custom_voice / voice_design / base), never off MODEL_NAME.
    kind = ""
    try:
        kind = (m.model.model.tts_model_type or "").strip().lower()
    except AttributeError:
        kind = "custom_voice" if speakers else "base"
    return m, speakers, kind == "custom_voice", kind == "voice_design"


def _warmup():
    """Capture the CUDA graphs before we report ready, so no caller pays the capture cost.

    warmup() only captures graphs; the first real synthesis still does lazy per-shape work, so a
    throwaway generation follows it.
    """
    m = _state["model"]
    t0 = time.time()
    m.warmup(prefill_len=100)
    log.info("CUDA graph capture took %.1fs", time.time() - t0)
    t0 = time.time()
    try:
        with _gen_lock:
            if _state["custom_voice"]:
                m.generate_custom_voice(text="Warm up.", speaker=_default_voice(),
                                        language="English")
            elif _state["voice_design"]:
                m.generate_voice_design(text="Warm up.", language="English",
                                        instruct="A calm narrator.")
            else:
                # A clone-only model has no preset voice; it needs a reference clip either way.
                with _temp_ref(_tone_wav(), ".wav") as ref:
                    m.generate_voice_clone(text="Warm up.", language="English",
                                           ref_audio=ref, xvec_only=True)
        log.info("warmup synthesis took %.0fs", time.time() - t0)
    except Exception as e:
        # Serviceable either way; failing here only means the first caller pays after all.
        log.warning("warmup synthesis failed after %.0fs: %s", time.time() - t0, e)


def _boot():
    global OUT_SR
    try:
        m, speakers, custom, design = _load()
        _state.update(model=m, speakers=speakers, custom_voice=custom, voice_design=design)
        rate = int(getattr(m, "sample_rate", OUT_SR) or OUT_SR)
        if rate != OUT_SR:
            log.warning("this checkpoint decodes at %d Hz, not the %d Hz /v1/models already "
                        "advertised; response headers will carry the real rate", rate, OUT_SR)
            OUT_SR = rate
        log.info("%s loaded: custom_voice=%s voice_design=%s speakers=%d sample_rate=%d",
                 MODEL_NAME, custom, design, len(speakers), OUT_SR)
        _warmup()
        _state.update(ready=True, error=None)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("tts engine failed to start: %s", e)


def _default_voice():
    if DEFAULT_VOICE:
        return DEFAULT_VOICE
    return _state["speakers"][0] if _state["speakers"] else "aiden"


def _tone_wav(seconds=1.0, rate=16000, hz=220):
    """A voiced-band tone, to stand in as the reference clip a clone warmup needs."""
    import array
    import math
    import wave

    frames = array.array("h", (int(8000 * math.sin(2 * math.pi * hz * i / rate))
                               for i in range(int(seconds * rate))))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames.tobytes())
    return buf.getvalue()


class _temp_ref:
    """faster-qwen3-tts reads reference audio from a path, so an upload lands on disk first."""

    def __init__(self, data, suffix=".wav"):
        self._data, self._suffix, self._path = data, suffix, None

    def __enter__(self):
        fd, self._path = tempfile.mkstemp(suffix=self._suffix, prefix="tts-ref-")
        with os.fdopen(fd, "wb") as f:
            f.write(self._data)
        return self._path

    def __exit__(self, *exc):
        try:
            os.unlink(self._path)
        except OSError:
            pass
        return False


def _to_mono(audio):
    if isinstance(audio, (list, tuple)):
        audio = audio[0] if len(audio) == 1 else np.concatenate([np.asarray(a).reshape(-1)
                                                                 for a in audio])
    a = np.asarray(audio, dtype="float32").reshape(-1)
    return np.clip(a, -1.0, 1.0)


def _pcm16(audio):
    return (_to_mono(audio) * 32767.0).astype("<i2").tobytes()


def _explain(e):
    """A message for the client even when the exception carries none.

    audioread.NoBackendError is the one that actually reaches users: librosa falls back to
    audioread whenever libsndfile cannot open the reference clip, and that exception is empty,
    so reporting str(e) sent them {"error": ""} to debug.
    """
    name = type(e).__name__
    text = str(e).strip()
    if name == "NoBackendError" or "Format not recognised" in text:
        return ("could not decode the reference audio: no decoder for that container. "
                "Convert it to wav/mp3/flac and retry")
    return text or name


# No length check here: the engine counts tokens exactly and says so, an estimate would not.


def _encode(audio, sr, fmt):
    if fmt == "pcm":
        return _pcm16(audio)
    import soundfile as sf

    container, subtype, _mime = _FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, _to_mono(audio), sr, format=container, subtype=subtype)
    return buf.getvalue()


def _wav_stream_header(sr, channels=1, bits=16):
    """A RIFF header with unknown length: the sizes are only known once generation ends."""
    byte_rate = sr * channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, channels * bits // 8, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF - 36))


def _gen_kwargs(payload):
    """The sampling knobs OpenAI's body does not carry, read off the request when present."""
    out = {}
    for key, cast in (("temperature", float), ("top_k", int), ("top_p", float),
                      ("max_new_tokens", int), ("repetition_penalty", float)):
        if payload.get(key) is not None:
            try:
                out[key] = cast(payload[key])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="%s must be a number" % key)
    return out


def _clone_args(payload):
    """ref_text present means ICL cloning; without it only the speaker embedding is usable."""
    ref_text = str(payload.get("ref_text") or "")
    xvec = payload.get("x_vector_only_mode")
    if xvec is None:
        xvec = not ref_text
    return ref_text, bool(xvec)


def _synthesize_blocking(payload, ref_path):
    """One generation, holding the lock. Returns (audio, sample_rate)."""
    m = _state["model"]
    text = payload["input"]
    language = str(payload.get("language") or DEFAULT_LANGUAGE)
    kw = _gen_kwargs(payload)
    instruct = payload.get("instructions") or None
    with _gen_lock:
        if ref_path:
            ref_text, xvec = _clone_args(payload)
            return m.generate_voice_clone(text=text, language=language, ref_audio=ref_path,
                                          ref_text=ref_text, xvec_only=xvec, instruct=instruct,
                                          **kw)
        if _state["voice_design"]:
            return m.generate_voice_design(text=text, language=language, instruct=instruct, **kw)
        return m.generate_custom_voice(text=text, speaker=str(payload.get("voice")
                                                              or _default_voice()),
                                       language=language, instruct=instruct, **kw)


def _stream_blocking(payload, ref_path, chunk_size):
    """The generator form of _synthesize_blocking; the lock spans the whole stream."""
    m = _state["model"]
    text = payload["input"]
    language = str(payload.get("language") or DEFAULT_LANGUAGE)
    kw = _gen_kwargs(payload)
    instruct = payload.get("instructions") or None
    with _gen_lock:
        if ref_path:
            ref_text, xvec = _clone_args(payload)
            it = m.generate_voice_clone_streaming(text=text, language=language, ref_audio=ref_path,
                                                  ref_text=ref_text, xvec_only=xvec,
                                                  instruct=instruct, chunk_size=chunk_size, **kw)
        elif _state["voice_design"]:
            it = m.generate_voice_design_streaming(text=text, language=language,
                                                   instruct=instruct, chunk_size=chunk_size, **kw)
        else:
            it = m.generate_custom_voice_streaming(text=text,
                                                   speaker=str(payload.get("voice")
                                                               or _default_voice()),
                                                   language=language, instruct=instruct,
                                                   chunk_size=chunk_size, **kw)
        for chunk, sr, _timing in it:
            yield chunk, sr


async def _aiter(make_gen):
    """Drive a blocking generator on a worker thread and yield its items on the event loop."""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=8)
    done = object()

    def pump():
        try:
            for item in make_gen():
                asyncio.run_coroutine_threadsafe(queue.put(("item", item)), loop).result()
        except Exception as e:  # surfaced to the consumer, which decides how to report it
            asyncio.run_coroutine_threadsafe(queue.put(("error", e)), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", done)), loop).result()

    threading.Thread(target=pump, daemon=True).start()
    while True:
        kind, value = await queue.get()
        if kind == "done":
            return
        if kind == "error":
            raise value
        yield value


def build_app(supports):
    app = FastAPI(title="audio-tts (faster-qwen3-tts)")
    mount_metrics(app)

    def _endpoint_available(endpoint):
        if endpoint.get("operation_id") == "voice.list":
            return bool(_state["custom_voice"]), "checkpoint has no preset voice library"
        return True, ""

    register(app, model_name=MODEL_NAME, module="tts", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=OUT_SR, endpoint_available=_endpoint_available)

    # No child engine takes the leftovers here, so a mistyped flag would otherwise vanish silently.
    _args.warn_unclaimed(log)

    has_tts = "tts" in supports
    has_clone = "tts_clone" in supports
    speech_mode = "tts" if has_tts else "tts_clone"

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    def _headers(mode, fmt):
        return {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": mode,
                "X-Audio-Format": fmt, "X-Audio-Sample-Rate": str(OUT_SR)}

    def _check(payload):
        """Validate the OpenAI body and settle the response format. Returns the format."""
        if not str(payload.get("input") or "").strip():
            raise HTTPException(status_code=400, detail="input (the text to speak) is required")
        fmt = str(payload.setdefault("response_format", "wav")).strip().lower()
        payload["response_format"] = fmt
        if fmt not in _FORMATS:
            raise HTTPException(status_code=400, detail="response_format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        bad = _voice_error(payload.get("voice"))
        if bad:
            raise HTTPException(status_code=400, detail=bad)
        return fmt

    def _voice_error(voice):
        """The one check the socket shares with the body; it has no response to raise into."""
        if voice and _state["voice_design"]:
            return ("this checkpoint has no preset voices; describe the voice in instructions")
        if voice and _state["speakers"] and str(voice) not in _state["speakers"]:
            return "unknown voice %r; see GET /v1/audio/voices" % str(voice)
        return None

    def _ref_from_payload(payload):
        """ref_audio arrives as a data: URL; anything else is the caller's own fetch to make.

        One instance holds one checkpoint. A preset-voice checkpoint (e.g. Qwen3-TTS CustomVoice
        etc.) cannot clone; a clone checkpoint (e.g. Qwen3-TTS Base etc.) has no presets; a
        design checkpoint (e.g. Qwen3-TTS VoiceDesign etc.) takes a written description in
        instructions and neither presets nor a reference clip. Refusing here beats letting the
        model fail deep inside generation with something unreadable.
        """
        ref = payload.get("ref_audio")
        if not ref:
            if _state["custom_voice"]:
                return None, None
            if _state["voice_design"]:
                if not str(payload.get("instructions") or "").strip():
                    raise HTTPException(status_code=400,
                                        detail="%s designs a voice from instructions; "
                                               "the description is required" % MODEL_NAME)
                return None, None
            raise HTTPException(status_code=400,
                                detail="%s has no preset voices; supply ref_audio (a data: "
                                       "URL) or POST /v1/audio/speech/clone" % MODEL_NAME)
        if _state["custom_voice"]:
            raise HTTPException(status_code=400,
                                detail="%s speaks its preset voices only; cloning needs an "
                                       "instance of the Base weights" % MODEL_NAME)
        if _state["voice_design"]:
            raise HTTPException(status_code=400,
                                detail="%s cannot clone from a reference clip; describe the "
                                       "voice in instructions" % MODEL_NAME)
        if not isinstance(ref, str) or not ref.startswith("data:"):
            raise HTTPException(status_code=400,
                                detail="ref_audio must be a data: URL (base64 reference audio)")
        head, _, b64 = ref.partition(",")
        mime = head[5:].split(";")[0] or "audio/wav"
        try:
            return base64.b64decode(b64), _REF_SUFFIX.get(mime, ".bin")
        except Exception:
            raise HTTPException(status_code=400, detail="ref_audio is not valid base64")

    async def _stream_response(payload, mode, fmt, ref):
        if fmt not in ("pcm", "wav"):
            raise HTTPException(status_code=400,
                                detail="stream supports response_format pcm or wav; "
                                       "%s only exists as a whole file" % fmt)
        chunk_size = int(payload.get("chunk_size") or CHUNK_SIZE)
        data, suffix = ref

        async def body():
            if fmt == "wav":
                yield _wav_stream_header(OUT_SR)
            ctx = _temp_ref(data, suffix) if data else None
            path = ctx.__enter__() if ctx else None
            try:
                async for chunk, _sr in _aiter(lambda: _stream_blocking(payload, path, chunk_size)):
                    yield _pcm16(chunk)
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)

        return StreamingResponse(body(), media_type=_FORMATS[fmt][2], headers=_headers(mode, fmt))

    async def _synthesize(payload, async_, mode):
        _require_ready()
        fmt = _check(payload)
        payload.setdefault("model", MODEL_NAME)
        ref = _ref_from_payload(payload)
        if payload.get("stream"):
            return await _stream_response(payload, mode, fmt, ref)
        data, suffix = ref

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="synthesis")
            if data:
                with _temp_ref(data, suffix) as path:
                    audio, sr = _synthesize_blocking(payload, path)
            else:
                audio, sr = _synthesize_blocking(payload, None)
            ctx.meter(output_seconds=seconds(audio, sr))
            body = _encode(audio, sr, fmt)
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(body, _FORMATS[fmt][2], suffix="." + fmt,
                                headers=_headers(mode, fmt))

        return await tasks.dispatch(async_, mode, MODEL_NAME, _work, fail="speech synthesis failed")

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON in the OpenAI /v1/audio/speech shape")
        # async travels in the query string here, so the body stays exactly OpenAI's.
        return await _synthesize(payload, request.query_params.get("async"), speech_mode)

    @app.post("/v1/audio/speech/batch")
    async def speech_batch(request: Request):
        # One model, one lock: items run back to back, the win is one dispatch, not parallelism.
        _require_ready()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400, detail="body must be JSON for /v1/audio/speech/batch")
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
            _check(row)
            # Settled here, not in _work: an unusable reference is a bad request like any other,
            # and the single-speech path already refuses it before the task exists.
            rows.append((row, _ref_from_payload(row)))
        async_ = request.query_params.get("async")

        def _work(ctx):
            n = len(rows)
            ctx.progress(ratio=0.0, stage="batch", done=0, total=n)
            out = []
            for i, (row, (data, suffix)) in enumerate(rows):
                if data:
                    with _temp_ref(data, suffix) as path:
                        audio, sr = _synthesize_blocking(row, path)
                else:
                    audio, sr = _synthesize_blocking(row, None)
                fmt = row["response_format"]
                ctx.meter(output_seconds=seconds(audio, sr))
                out.append({"index": i, "format": fmt, "sample_rate": sr,
                            "audio": base64.b64encode(_encode(audio, sr, fmt)).decode("ascii")})
                ctx.progress(ratio=(i + 1) / n, stage="batch", done=i + 1, total=n)
            return {"model": MODEL_NAME, "items": out}

        return await tasks.dispatch(async_, speech_mode, MODEL_NAME, _work,
                                    fail="speech batch failed")

    @app.websocket("/v1/audio/speech/stream")
    async def speech_stream(ws: WebSocket):
        """Incremental text in, audio out — the one thing POST /v1/audio/speech cannot do.

        A caller relaying an LLM's tokens does not have the sentence yet when it wants audio
        started. Text arrives in `input.text` frames and is synthesized a sentence at a time;
        `input.done` flushes whatever is left.
        """
        await ws.accept()
        if not _state["ready"]:
            await ws.send_text(json.dumps({"type": "error",
                                           "message": _state["error"] or "engine not ready"}))
            await ws.close()
            return
        cfg = {"response_format": "pcm", "language": DEFAULT_LANGUAGE, "voice": None,
               "chunk_size": CHUNK_SIZE, "ref_audio": None}
        pending = ""
        # Synthesis gets its own task: one loop doing both deadlocks once text outruns the GPU.
        work = asyncio.Queue(maxsize=256)
        FLUSHED = object()

        async def flush(text):
            text = text.strip()
            if not text:
                return
            payload = dict(cfg)
            payload["input"] = text
            data, suffix = _ref_from_payload(payload)
            ctx = _temp_ref(data, suffix) if data else None
            path = ctx.__enter__() if ctx else None
            try:
                async for chunk, _sr in _aiter(
                        lambda: _stream_blocking(payload, path, int(cfg["chunk_size"]))):
                    await ws.send_bytes(_pcm16(chunk))
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)

        def split_sentences(buf):
            """Emit complete sentences, keep the tail. Latency here is one sentence, not one turn."""
            out, start = [], 0
            for i, ch in enumerate(buf):
                if ch in "。！？!?\n" or (ch == "." and i + 1 < len(buf) and buf[i + 1] == " "):
                    out.append(buf[start:i + 1])
                    start = i + 1
            return out, buf[start:]

        async def speaker():
            """Speak queued sentences in order; a failure on one is reported, not fatal."""
            while True:
                item = await work.get()
                if item is FLUSHED:
                    await ws.send_text(json.dumps({"type": "session.done"}))
                    continue
                try:
                    await flush(item)
                except Exception as e:
                    log.warning("speech/stream sentence failed: %s", e)
                    detail = getattr(e, "detail", None) or _explain(e)
                    await ws.send_text(json.dumps({"type": "error", "message": detail}))

        worker = asyncio.create_task(speaker())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return
                raw = msg.get("text")
                if raw is None:
                    continue
                try:
                    frame = json.loads(raw)
                except Exception:
                    await ws.send_text(json.dumps({"type": "error",
                                                   "message": "frames must be JSON"}))
                    continue
                kind = frame.get("type")
                if kind == "session.config":
                    kept_voice = cfg["voice"]
                    for key in ("response_format", "language", "voice", "chunk_size", "ref_audio",
                                "ref_text", "x_vector_only_mode", "instructions"):
                        if frame.get(key) is not None:
                            cfg[key] = frame[key]
                    bad = _voice_error(cfg["voice"])
                    if bad:
                        await ws.send_text(json.dumps({"type": "error", "message": bad}))
                        cfg["voice"] = kept_voice
                    if str(cfg["response_format"]).lower() != "pcm":
                        await ws.send_text(json.dumps(
                            {"type": "error",
                             "message": "the socket carries raw pcm frames; "
                                        "use POST /v1/audio/speech for a container format"}))
                        cfg["response_format"] = "pcm"
                    await ws.send_text(json.dumps({"type": "session.ready", "model": MODEL_NAME,
                                                   "sample_rate": OUT_SR, "format": "pcm"}))
                elif kind == "input.text":
                    pending += str(frame.get("text") or "")
                    ready, pending = split_sentences(pending)
                    for s in ready:
                        await work.put(s)
                elif kind == "input.done":
                    await work.put(pending)
                    pending = ""
                    await work.put(FLUSHED)
                elif kind == "session.close":
                    await ws.close()
                    return
                else:
                    await ws.send_text(json.dumps({"type": "error",
                                                   "message": "unknown frame type %r" % kind}))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("speech/stream failed: %s", e)
            try:
                await ws.send_text(json.dumps({"type": "error", "message": _explain(e)}))
                await ws.close()
            except Exception:
                pass
        finally:
            worker.cancel()

    if has_tts:
        # Only a preset-voice checkpoint has anything to list; clone/design must not look like they do.
        @app.get("/v1/audio/voices")
        def voices_list():
            _require_ready()
            if not _state["custom_voice"]:
                raise HTTPException(status_code=404,
                                    detail="%s has no preset voices" % MODEL_NAME)
            return JSONResponse({"model": MODEL_NAME,
                                 "voices": [{"id": s} for s in _state["speakers"]]})

    if has_clone:
        @app.post("/v1/audio/speech/clone")
        async def clone(file: UploadFile = File(...),
                        text: str = Form(..., alias="input"),
                        ref_text: str = Form(default=""),
                        language: str = Form(default=""),
                        instructions: str = Form(default=""),
                        fmt: str = Form(default="wav", alias="response_format"),
                        x_vector_only: str = Form(default=None, alias="x_vector_only_mode"),
                        async_: str = Form(default=None, alias="async")):
            _require_ready()
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="file (the reference audio) is empty")
            mime = file.content_type or "audio/wav"
            payload = {"input": text, "response_format": fmt,
                       "ref_audio": "data:%s;base64,%s"
                                    % (mime, base64.b64encode(data).decode("ascii"))}
            # The model treats a missing optional as "unset"; sending "" would override its default.
            if ref_text:
                payload["ref_text"] = ref_text
            if language:
                payload["language"] = language
            if instructions:
                payload["instructions"] = instructions
            if x_vector_only is not None:
                payload["x_vector_only_mode"] = str(x_vector_only).strip().lower() in (
                    "1", "true", "yes", "on")
            return await _synthesize(payload, async_, "tts_clone")

    return app


def run(supports):
    # Ready now means loaded AND warmed, so the deadline has to cover both or it kills a healthy boot.
    _runtime.serve(supports, _boot, build_app, "faster-qwen3-tts engine",
                   timeout_s=BOOT_TIMEOUT_S + WARMUP_TIMEOUT_S)
