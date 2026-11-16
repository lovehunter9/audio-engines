"""S3 file translation through the official CUDA entry point."""
import base64
import io
import logging
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from .. import hfgate, tasks
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-s3-s2st")
_runtime = Runtime(engine=None)
_state = _runtime.state
_args = EngineArgs()
_PRESET = _args.text("--preset", "adaptive")
_KV_GIB = _args.number("--kv-cache-gib", 0.5)
_lock = threading.Lock()


def _language(raw):
    names = {"en": "English", "eng": "English", "english": "English",
             "zh": "Chinese", "cmn": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese"}
    name = names.get(str(raw).strip().lower())
    if name is None:
        raise HTTPException(400, "source_lang and target_lang must be Chinese or English")
    return name


def _boot():
    try:
        from huggingface_hub import snapshot_download
        from .s3_native import OfficialEngine
        repo, include, exclude = _runtime._hub_spec()
        snapshot = snapshot_download(repo, local_files_only=True, allow_patterns=include, ignore_patterns=exclude)
        if _PRESET not in ("adaptive", "fixed-0.1", "fixed-0.5"):
            raise ValueError("--preset must be adaptive, fixed-0.1, or fixed-0.5")
        _state.update(engine=OfficialEngine(snapshot, _PRESET, kv_cache_gib=_KV_GIB), ready=True, error=None)
    except Exception as error:
        _state["error"] = hfgate.explain(_runtime.model_repo, error)
        log.exception("Official S3 runtime failed to initialize")


def build_app_for(runtime, args, supports, module, lock):
    state = runtime.state
    app = FastAPI(title="S3-S2ST official runtime")
    mount_metrics(app)
    register(app, model_name=runtime.model_name, module=module, served=supports,
             repo=runtime.model_repo, is_ready=lambda: state["ready"],
             error=lambda: state["error"], task_api=True, sample_rate=24000)

    @app.post("/v1/audio/translations")
    async def translate(request: Request, file: UploadFile = File(...),
                        source_lang: str = Form("English"), target_lang: str = Form("Chinese"),
                        response_format: str = Form("json"), model: str = Form(None)):
        if not state["ready"]:
            raise HTTPException(503, state["error"] or "engine not ready")
        source, target = _language(source_lang), _language(target_lang)
        if source == target:
            raise HTTPException(400, "source_lang and target_lang must differ")
        fmt = response_format.strip().lower()
        if fmt not in ("json", "wav", "pcm", "text"):
            raise HTTPException(400, "response_format must be json, wav, pcm, or text")
        audio = await file.read()
        if not audio:
            raise HTTPException(400, "file is empty")

        def work(ctx):
            import numpy as np
            import soundfile as sf
            with lock:
                started = time.monotonic()
                row, wav = state["engine"].translate(audio, source, target)
            samples, rate = sf.read(io.BytesIO(wav), dtype="float32")
            ctx.meter(output_seconds=len(samples) / rate)
            text = row.get("text") or row.get("translation") or row.get("prediction") or ""
            if not text and row.get("pieces"):
                text = "".join(piece.get("text", "") for piece in row["pieces"])
            if fmt == "wav":
                return tasks.Binary(wav, "audio/wav", suffix=".wav")
            if fmt == "pcm":
                pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                return tasks.Binary(pcm, "audio/L16", suffix=".pcm")
            if fmt == "text":
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(text)
            return {"text": text, "sample_rate": rate, "audio": base64.b64encode(wav).decode(),
                    "elapsed_seconds": time.monotonic() - started}

        return await tasks.dispatch(request.query_params.get("async"), "audio_s2s",
                                    runtime.model_name, work, fail="official translation failed")
    from .s3_live import mount_stream
    mount_stream(app, runtime, lock, _language)
    return app


def build_app(supports):
    _args.warn_unclaimed(log)
    return build_app_for(_runtime, _args, supports, "s3_s2st", _lock)


def run(supports):
    _runtime.serve(supports, _boot, build_app, "S3 official runtime", timeout_s=3600, disable_ws_ping=True)
