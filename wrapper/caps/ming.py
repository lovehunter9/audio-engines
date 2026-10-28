# OpenAI-shaped HTTP for Ming. The task and the voice card live in tts_voice and ming_backend.
import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .. import hfgate
from .. import tasks
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime
from . import tts_voice
from .ming_backend import MingBackend, load_model
from .ming_catalog import load_ip_catalog

log = logging.getLogger("audio-ming")

_runtime = Runtime(model=None, backend=None, library=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
VOICE_DIR = Path(str(_args.text("--voice-dir", "/data/voices") or "/data/voices"))
_args.warn_unclaimed(log)

_state = _runtime.state
BOOT_TIMEOUT_S = 1800.0

_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),
}
_REF_SUFFIX = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
    "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a", "video/mp4": ".mp4",
}


def pack_root():
    return Path(__file__).resolve().parents[2] / "voices"


def _http(err):
    raise HTTPException(status_code=err.status_code, detail=err.detail)


async def _voice_add_fields(request):
    ctype = request.headers.get("content-type") or ""
    if "multipart/form-data" in ctype:
        form = await request.form()
        upload = form.get("file")
        data = b""
        if upload is not None and hasattr(upload, "read"):
            data = await upload.read()
        return (str(form.get("name") or ""), str(form.get("ref_text") or "").strip(),
                str(form.get("instructions") or "").strip(), data)
    try:
        body = dict(await request.json())
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON or multipart")
    data = b""
    if body.get("ref_audio"):
        data, _suffix = _decode_ref(body.get("ref_audio"))
    return (str(body.get("name") or ""), str(body.get("ref_text") or "").strip(),
            str(body.get("instructions") or "").strip(), data)


def _decode_ref(ref):
    head, _, b64 = str(ref).partition(",")
    mime = head[5:].split(";")[0] or "audio/wav"
    try:
        data = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="ref_audio is not valid base64")
    if not data:
        raise HTTPException(status_code=400, detail="ref_audio is empty")
    return data, _REF_SUFFIX.get(mime, ".bin")


def _pcm_wav(data):
    """Bytes soundfile can open. Containers it cannot (mp4, m4a, aac) go through ffmpeg."""
    if data[:4] == b"RIFF":
        return data
    sound_err = ""
    try:
        import soundfile as sf
        sf.read(io.BytesIO(data), dtype="float32")
        return data
    except Exception as exc:
        sound_err = str(exc).strip() or type(exc).__name__
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=400, detail="could not decode the reference audio: %s" % sound_err)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-f", "wav", "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()[-300:]
        raise HTTPException(status_code=400,
                            detail="could not decode the reference audio: %s" % (detail or sound_err))
    return proc.stdout


def _pcm16(audio):
    return (np.clip(np.asarray(audio, dtype="float32").reshape(-1), -1.0, 1.0)
            * 32767.0).astype("<i2").tobytes()


def _encode(audio, sr, fmt):
    if fmt == "pcm":
        return _pcm16(audio)
    import soundfile as sf

    container, subtype, _mime = _FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype="float32").reshape(-1), sr,
             format=container, subtype=subtype)
    return buf.getvalue()


def _boot(library):
    try:
        if not MODEL_REPO:
            raise RuntimeError("MODEL_SOURCE is empty")
        library.reconcile(load_ip_catalog(), pack_root())
        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL_REPO, local_files_only=True,
                                 cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)
        log.info("loading %s from %s", MODEL_NAME or MODEL_REPO, path)
        backend = MingBackend(load_model(path))
        backend.synthesize(tts_voice.Turn(
            task="ip", text="Hello.", ip="US_James", language_id="en"))
        _state.update(backend=backend, sample_rate=backend.sample_rate, ready=True, error=None)
        log.info("%s ready, %d voices, sample_rate=%d",
                 MODEL_NAME or MODEL_REPO, len(library.list()), backend.sample_rate)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("ming engine failed to start: %s", e)


def build_app(supports, library=None, backend=None):
    app = FastAPI(title="audio-ming")
    mount_metrics(app)
    # Resolve the library on the request. Building the app must not mkdir /data/voices; that path is the container volume.
    def _library():
        if library is not None:
            return library
        return VoiceLibraryHolder.get()

    register(app, model_name=MODEL_NAME, module="ming", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=lambda: _state.get("sample_rate") or None)

    has_tts = "tts" in supports
    has_clone = "tts_clone" in supports
    has_design = "tts_design" in supports

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    def _backend():
        if backend is not None:
            return backend
        return _state.get("backend")

    def _check_format(payload):
        fmt = str(payload.get("response_format") or "wav").strip().lower()
        if fmt not in _FORMATS:
            raise HTTPException(status_code=400, detail="response_format must be one of %s"
                                % ", ".join(sorted(_FORMATS)))
        return fmt

    async def _speak(payload, async_flag):
        _require_ready()
        fmt = _check_format(payload)
        try:
            turn = tts_voice.resolve_openai(payload, _library())
        except tts_voice.VoiceError as e:
            _http(e)
        data = None
        if turn.ref_audio:
            raw, _suffix = _decode_ref(turn.ref_audio)
            data = _pcm_wav(raw)

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="synthesize")
            path = None
            try:
                if data is not None:
                    fd, path = tempfile.mkstemp(prefix="ming-ref-", suffix=".wav")
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    turn.prompt_wav = path
                audio, sr = _backend().synthesize(turn)
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            ctx.progress(ratio=1.0, stage="done")
            body = _encode(audio, sr, fmt)
            headers = {
                "X-Audio-Model": MODEL_NAME,
                "X-Audio-Task": turn.task,
                "X-Audio-Origin": turn.origin,
                "X-Audio-Format": fmt,
                "X-Audio-Sample-Rate": str(sr),
            }
            return tasks.Binary(body, _FORMATS[fmt][2], headers=headers, suffix="." + fmt)

        return await tasks.dispatch(async_flag, "tts", MODEL_NAME or "ming", _work,
                                    fail="speech synthesis failed")

    if has_tts or has_clone or has_design:
        @app.post("/v1/audio/speech")
        async def speech(request: Request):
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400,
                                    detail="body must be JSON in the OpenAI /v1/audio/speech shape")
            return await _speak(payload, request.query_params.get("async"))

    if has_tts:
        @app.get("/v1/audio/voices")
        def voices_list():
            _require_ready()
            rows = _library().list()
            return {"model": MODEL_NAME, "voices": [_library().public(m) for m in rows]}

        @app.delete("/v1/audio/voices/{voice_id}")
        def voices_delete(voice_id: str):
            _require_ready()
            try:
                return _library().delete(voice_id)
            except tts_voice.VoiceError as e:
                _http(e)

        @app.post("/v1/audio/voices")
        async def voices_add(request: Request):
            """Save a user voice. Clone is ref audio plus its transcript; design is the instruction. The id is the name."""
            _require_ready()
            name, ref_text, instructions, data = await _voice_add_fields(request)
            chosen = []
            if data:
                chosen.append("ref_audio")
            if instructions:
                chosen.append("instructions")
            if len(chosen) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="send exactly one of ref_audio (with ref_text) or instructions (got %s)"
                    % (", ".join(chosen) or "none"))
            try:
                if data:
                    if not ref_text:
                        raise tts_voice.VoiceError(
                            400, "ref_text (the exact transcript of the reference) is required")
                    meta = _library().add_clone(name, name, _pcm_wav(data), ref_text)
                else:
                    meta = _library().add_design(name, name, instructions)
            except tts_voice.VoiceError as e:
                _http(e)
            return _library().public(meta)

    if has_clone:
        @app.post("/v1/audio/speech/clone")
        async def clone(file: UploadFile = File(...),
                        text: str = Form(..., alias="input"),
                        ref_text: str = Form(default=""),
                        language: str = Form(default=""),
                        fmt: str = Form(default="wav", alias="response_format"),
                        async_: str = Form(default=None, alias="async")):
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="file (the reference audio) is empty")
            mime = file.content_type or "audio/wav"
            payload = {
                "input": text,
                "response_format": fmt,
                "ref_audio": "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii")),
                "ref_text": ref_text,
            }
            if language:
                payload["language"] = language
            return await _speak(payload, async_)

    return app


class VoiceLibraryHolder:
    """The process-wide library. Tests pass their own into build_app."""

    _library = None

    @classmethod
    def get(cls):
        if cls._library is None:
            cls._library = tts_voice.VoiceLibrary(VOICE_DIR)
        return cls._library


def run(supports):
    library = VoiceLibraryHolder.get()
    _runtime.serve(supports, lambda: _boot(library), build_app, "Ming-3.0-tts",
                   timeout_s=BOOT_TIMEOUT_S)
