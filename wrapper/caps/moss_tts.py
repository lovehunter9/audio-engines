# MOSS-TTS-Nano on official ONNX Runtime. In-process; not Omni, not audio.cpp.
#
# The model's identity is realtime streaming (first-packet audio) plus zero-shot clone, and it
# is small enough to run on CPU. audio.cpp served the GGUF as offline-only and withheld the
# socket, which is how a realtime model ended up looking like a tiny batch TTS.
import asyncio
import base64
import io
import json
import logging
import os
import queue
import struct
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .. import hfgate
from .. import tasks
from ..audioio import seconds
from ..contract import EngineArgs, register
from ..gpu import mount_metrics, quota_mib
from ..runtime import Runtime

log = logging.getLogger("audio-moss-tts")

_runtime = Runtime("OpenMOSS-Team/MOSS-TTS-Nano-100M",
                   default_repo="OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX",
                   model=None, speakers=[])
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
LAYOUT = Path("/tmp/moss-onnx")

# Chart-intrinsic. Filling ENGINE_ARGS in the UI replaces the whole string.
DEFAULT_ENGINE_ARGS = "--execution-provider cuda --cpu-threads 4"
_args = EngineArgs((os.environ.get("ENGINE_ARGS") or "").strip() or DEFAULT_ENGINE_ARGS)
EXEC_PROVIDER = str(_args.text("--execution-provider", "cuda") or "cuda")
CPU_THREADS = int(_args.number("--cpu-threads", 4) or 4)
MAX_NEW_FRAMES = int(_args.number("--max-new-frames", 375) or 375)
SAMPLE_MODE = str(_args.text("--sample-mode", "fixed") or "fixed")
DEFAULT_VOICE = str(_args.text("--voice", "") or "")
CLONE_MAX_TOKENS = int(_args.number("--voice-clone-max-text-tokens", 75) or 75)

# 48 kHz stereo is native; /v1/models advertises 48 kHz and we downmix to mono for the contract.
OUT_SR = 48000
BOOT_TIMEOUT_S = 600.0

_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),
}
_REF_SUFFIX = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
               "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
               "audio/x-flac": ".flac", "audio/ogg": ".ogg", "audio/opus": ".opus",
               "audio/aac": ".aac", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
               "audio/webm": ".webm", "video/mp4": ".mp4", "video/webm": ".webm",
               "video/quicktime": ".mov", "video/x-matroska": ".mkv"}

_state = _runtime.state
_gen_lock = threading.Lock()


def _snapshot(repo):
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, local_files_only=True,
                                  cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN))
    # Hub cache root is blobs/ + refs/ + snapshots/<sha>. Named files live under snapshots.
    if (path / "snapshots").is_dir() and not (path / "tts_browser_onnx_meta.json").exists() \
            and not (path / "codec_browser_onnx_meta.json").exists():
        named = [p for p in path.glob("snapshots/*") if p.is_dir()]
        if not named:
            raise RuntimeError("no snapshot under %s" % path)
        path = max(named, key=lambda p: p.stat().st_mtime)
    return path


def _materialize(src, dst):
    """Copy named files out of a hub snapshot.

    Snapshot entries are symlinks into blobs/<hash>. OnnxTtsRuntime resolve()s the
    manifest and then looks for siblings next to it, so a symlink-to-blobs directory
    makes it read blobs/tts_browser_onnx_meta.json (the original filename), which
    does not exist. Small files are copied; large .data files stay as named symlinks.
    """
    import shutil

    if dst.is_symlink():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name == ".gitattributes":
            continue
        real = item.resolve()
        target = dst / item.name
        # ORT refuses external-data files whose resolved path leaves the model dir,
        # so even the large .data blobs must be real files here, not hub-cache symlinks.
        shutil.copy2(real, target)


def _model_dir():
    """Two snapshots laid out the way OnnxTtsRuntime expects (sibling named folders)."""
    LAYOUT.mkdir(parents=True, exist_ok=True)
    _materialize(_snapshot(TTS_REPO), LAYOUT / "MOSS-TTS-Nano-100M-ONNX")
    _materialize(_snapshot(CODEC_REPO), LAYOUT / "MOSS-Audio-Tokenizer-Nano-ONNX")
    return str(LAYOUT)


def _pick_provider(requested):
    """cuda if the quota and the wheel both allow it; otherwise cpu. Official default is cpu."""
    want = (requested or "cpu").strip().lower()
    if want not in ("cpu", "cuda"):
        raise RuntimeError("ENGINE_ARGS --execution-provider must be cpu or cuda, got %r" % requested)
    if want == "cuda" and quota_mib() <= 0:
        log.info("REQUIRED_GPU_MEMORY is 0; using CPU execution provider")
        return "cpu"
    return want


class MossEngine:
    """Thin face over OnnxTtsRuntime so the HTTP layer and the smoke fake share one seam."""

    def __init__(self, runtime):
        self.runtime = runtime
        cfg = runtime.codec_meta["codec_config"]
        self.sample_rate = int(cfg["sample_rate"])
        self.channels = int(cfg["channels"])
        self.voices = [row["voice"] for row in runtime.list_builtin_voices()]

    def speak(self, text, voice, ref_path):
        result = self.runtime.synthesize(
            text=text,
            voice=voice or None,
            prompt_audio_path=ref_path,
            streaming=True,
            enable_wetext=False,
            enable_normalize_tts_text=True,
            output_audio_path=str(Path("/tmp/moss-speech.wav")),
            max_new_frames=MAX_NEW_FRAMES,
            sample_mode=SAMPLE_MODE,
            voice_clone_max_text_tokens=CLONE_MAX_TOKENS,
        )
        return np.asarray(result["waveform"], dtype=np.float32), int(result["sample_rate"])

    def speak_stream(self, text, voice, ref_path):
        """Native Realtime Streaming Decode: yield (waveform, sample_rate) as frames land."""
        from onnx_tts_runtime import _merge_audio_channels
        from ort_cpu_runtime import _resolve_stream_decode_frame_budget

        rt = self.runtime
        prompt = rt.resolve_prompt_audio_codes(voice=voice or None, prompt_audio_path=ref_path)
        chunks = rt.split_voice_clone_text(str(text or ""), max_tokens=CLONE_MAX_TOKENS)
        if not chunks:
            chunks = [str(text or "").strip()]
        sample_rate = self.sample_rate
        out = queue.Queue(maxsize=128)
        DONE, ERR = object(), object()

        def produce():
            emitted = 0
            first_at = None
            try:
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text:
                        continue
                    ids = rt.encode_text(chunk_text)
                    rows = rt.build_voice_clone_request_rows(prompt, ids)
                    pending = []
                    rt.codec_streaming_session.reset()

                    def _push(waveform, is_pause=False):
                        nonlocal emitted, first_at
                        if first_at is None and not is_pause:
                            first_at = time.perf_counter()
                        emitted += int(waveform.shape[0])
                        out.put(("audio", waveform, sample_rate))

                    def _decode(force):
                        if not pending:
                            return None
                        budget = _resolve_stream_decode_frame_budget(
                            emitted, sample_rate, first_at)
                        if not force and len(pending) < max(1, budget):
                            return None
                        take = len(pending) if force else min(len(pending), max(1, budget))
                        frame_chunk = pending[:take]
                        del pending[:take]
                        decoded = rt.codec_streaming_session.run_frames(frame_chunk)
                        if decoded is None:
                            return None
                        audio, length = decoded
                        if length <= 0:
                            return None
                        return _merge_audio_channels(
                            [audio[0, ch, :length] for ch in range(audio.shape[1])])

                    def _on_frame(_frames, _step, frame):
                        pending.append(list(frame))
                        wave = _decode(False)
                        if wave is not None:
                            _push(wave)

                    try:
                        rt.generate_audio_frames(rows, on_frame=_on_frame)
                        wave = _decode(True)
                        if wave is not None:
                            _push(wave)
                    finally:
                        rt.codec_streaming_session.reset()

                    if i < len(chunks) - 1:
                        pause = rt.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                        n = max(0, int(round(sample_rate * pause)))
                        if n:
                            _push(np.zeros((n, self.channels), dtype=np.float32), True)
                out.put((DONE, None, None))
            except Exception as e:
                out.put((ERR, e, None))

        threading.Thread(target=produce, daemon=True).start()
        while True:
            kind, a, b = out.get()
            if kind is DONE:
                return
            if kind is ERR:
                raise a
            yield a, b


def _load():
    from onnx_tts_runtime import OnnxTtsRuntime

    path = _model_dir()
    requested = _pick_provider(EXEC_PROVIDER)
    log.info("loading ONNX from %s provider=%s threads=%d", path, requested, CPU_THREADS)
    try:
        rt = OnnxTtsRuntime(model_dir=path, thread_count=CPU_THREADS,
                            max_new_frames=MAX_NEW_FRAMES, sample_mode=SAMPLE_MODE,
                            execution_provider=requested)
    except Exception as e:
        if requested != "cuda":
            raise
        log.warning("CUDA execution provider failed (%s); falling back to cpu", e)
        rt = OnnxTtsRuntime(model_dir=path, thread_count=CPU_THREADS,
                            max_new_frames=MAX_NEW_FRAMES, sample_mode=SAMPLE_MODE,
                            execution_provider="cpu")
    return MossEngine(rt)


def _boot():
    global OUT_SR
    try:
        engine = _load()
        _state.update(model=engine, speakers=list(engine.voices))
        if engine.sample_rate != OUT_SR:
            log.info("codec sample rate %d Hz (advertised %d); headers carry the real rate",
                     engine.sample_rate, OUT_SR)
            OUT_SR = engine.sample_rate
        log.info("%s loaded: voices=%s sample_rate=%d provider=%s",
                 MODEL_NAME, engine.voices, OUT_SR, EXEC_PROVIDER)
        _state.update(ready=True, error=None)
    except Exception as e:
        _state["error"] = hfgate.explain(TTS_REPO, e)
        log.exception("moss-tts engine failed to start: %s", e)


def _default_voice():
    if DEFAULT_VOICE:
        return DEFAULT_VOICE
    return _state["speakers"][0] if _state["speakers"] else ""


class _temp_ref:
    def __init__(self, data, suffix=".wav"):
        self._data, self._suffix, self._path = data, suffix, None

    def __enter__(self):
        fd, self._path = tempfile.mkstemp(suffix=self._suffix, prefix="moss-ref-")
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
    a = np.asarray(audio, dtype="float32")
    if a.ndim == 2:
        a = a.mean(axis=1)
    else:
        a = a.reshape(-1)
    return np.clip(a, -1.0, 1.0)


def _pcm16(audio):
    return (_to_mono(audio) * 32767.0).astype("<i2").tobytes()


def _explain(e):
    name = type(e).__name__
    text = str(e).strip()
    if name == "NoBackendError" or "Format not recognised" in text:
        return ("could not decode the reference audio: no decoder for that container. "
                "Convert it to wav/mp3/flac and retry")
    return text or name


def _encode(audio, sr, fmt):
    if fmt == "pcm":
        return _pcm16(audio)
    import soundfile as sf

    container, subtype, _mime = _FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, _to_mono(audio), sr, format=container, subtype=subtype)
    return buf.getvalue()


def _wav_stream_header(sr, channels=1, bits=16):
    byte_rate = sr * channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, channels * bits // 8, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF - 36))


def _synthesize_blocking(payload, ref_path):
    m = _state["model"]
    voice = str(payload.get("voice") or _default_voice() or "") or None
    with _gen_lock:
        return m.speak(payload["input"], voice, ref_path)


def _stream_blocking(payload, ref_path, _chunk_size):
    m = _state["model"]
    voice = str(payload.get("voice") or _default_voice() or "") or None
    with _gen_lock:
        for chunk, sr in m.speak_stream(payload["input"], voice, ref_path):
            yield chunk, sr


async def _aiter(make_gen):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=8)
    done = object()

    def pump():
        try:
            for item in make_gen():
                asyncio.run_coroutine_threadsafe(queue.put(("item", item)), loop).result()
        except Exception as e:
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
    app = FastAPI(title="audio-tts (moss-tts-nano onnx)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="moss_tts", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=lambda: OUT_SR, model_format="onnx")

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
        if voice and _state["speakers"] and str(voice) not in _state["speakers"]:
            return "unknown voice %r; see GET /v1/audio/voices" % str(voice)
        return None

    def _ref_from_payload(payload):
        ref = payload.get("ref_audio")
        if not ref:
            return None, None
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
        data, suffix = ref

        async def body():
            if fmt == "wav":
                yield _wav_stream_header(OUT_SR)
            ctx = _temp_ref(data, suffix) if data else None
            path = ctx.__enter__() if ctx else None
            try:
                async for chunk, _sr in _aiter(lambda: _stream_blocking(payload, path, 0)):
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
        return await _synthesize(payload, request.query_params.get("async"), speech_mode)

    @app.post("/v1/audio/speech/batch")
    async def speech_batch(request: Request):
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
                ctx.meter(output_seconds=seconds(audio, sr))
                fmt = row["response_format"]
                out.append({"index": i, "format": fmt, "sample_rate": sr,
                            "audio": base64.b64encode(_encode(audio, sr, fmt)).decode("ascii")})
                ctx.progress(ratio=(i + 1) / n, stage="batch", done=i + 1, total=n)
            return {"model": MODEL_NAME, "items": out}

        return await tasks.dispatch(async_, speech_mode, MODEL_NAME, _work,
                                    fail="speech batch failed")

    @app.websocket("/v1/audio/speech/stream")
    async def speech_stream(ws: WebSocket):
        await ws.accept()
        if not _state["ready"]:
            await ws.send_text(json.dumps({"type": "error",
                                           "message": _state["error"] or "engine not ready"}))
            await ws.close()
            return
        cfg = {"response_format": "pcm", "language": "", "voice": None, "ref_audio": None}
        pending = ""
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
                async for chunk, _sr in _aiter(lambda: _stream_blocking(payload, path, 0)):
                    await ws.send_bytes(_pcm16(chunk))
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)

        def split_sentences(buf):
            out, start = [], 0
            for i, ch in enumerate(buf):
                if ch in "。！？!?\n" or (ch == "." and i + 1 < len(buf) and buf[i + 1] == " "):
                    out.append(buf[start:i + 1])
                    start = i + 1
            return out, buf[start:]

        async def speaker():
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
                    for key in ("response_format", "language", "voice", "ref_audio", "ref_text"):
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
        @app.get("/v1/audio/voices")
        def voices_list():
            _require_ready()
            return JSONResponse({"model": MODEL_NAME,
                                 "voices": [{"id": s} for s in _state["speakers"]]})

    if has_clone:
        @app.post("/v1/audio/speech/clone")
        async def clone(file: UploadFile = File(...),
                        text: str = Form(..., alias="input"),
                        ref_text: str = Form(default=""),
                        language: str = Form(default=""),
                        fmt: str = Form(default="wav", alias="response_format"),
                        async_: str = Form(default=None, alias="async")):
            _require_ready()
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="file (the reference audio) is empty")
            mime = file.content_type or "audio/wav"
            payload = {"input": text, "response_format": fmt,
                       "ref_audio": "data:%s;base64,%s"
                                    % (mime, base64.b64encode(data).decode("ascii"))}
            if ref_text:
                payload["ref_text"] = ref_text
            if language:
                payload["language"] = language
            return await _synthesize(payload, async_, "tts_clone")

    return app


def run(supports):
    _runtime.serve(supports, _boot, build_app, "moss-tts-nano onnx engine",
                   timeout_s=BOOT_TIMEOUT_S)
