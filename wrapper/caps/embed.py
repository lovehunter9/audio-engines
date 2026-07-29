# Speaker embedding with pyannote.audio: one fixed-length vector for the whole clip.
import asyncio
import os
import logging
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

from .. import tasks
from .. import watchdog
from ..gpu import mount_metrics
from ..contract import register
from ..audioio import decode, spill, unlink

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-embed")

MODEL_NAME = os.environ.get("MODEL_NAME", "pyannote-embedding")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_state = {"ready": False, "error": None, "inference": None, "device": "cpu", "dim": None}


def _load():
    try:
        import torch
        from pyannote.audio import Model, Inference

        # pyannote.audio 4 uses token=; older used use_auth_token=.
        try:
            model = Model.from_pretrained(MODEL_REPO, token=HF_TOKEN)
        except TypeError:
            model = Model.from_pretrained(MODEL_REPO, use_auth_token=HF_TOKEN)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # window="whole" -> a single fixed-length vector for the entire clip.
        try:
            inf = Inference(model, window="whole", device=torch.device(dev))
        except TypeError:
            model.to(torch.device(dev))
            inf = Inference(model, window="whole")
        _state.update(inference=inf, device=dev, ready=True)
        log.info("pyannote embedding %s loaded on %s", MODEL_REPO, dev)
    except Exception as e:
        _state["error"] = str(e)
        log.exception("embedding load failed: %s", e)


def build_app(supports):
    app = FastAPI(title="audio-embedding (pyannote)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="embed", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/embeddings")
    async def embeddings(file: UploadFile = File(...),
                         async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        data = await file.read()
        path = await asyncio.to_thread(spill, data, file.filename)

        def _work(ctx):
            import numpy as np

            ctx.progress(ratio=0.0, stage="decode")
            waveform, sr = decode(path)
            ctx.progress(stage="inference")
            emb = _state["inference"]({"waveform": waveform, "sample_rate": sr})
            vec = np.asarray(emb, dtype="float32").reshape(-1)
            _state["dim"] = int(vec.shape[0])
            ctx.progress(ratio=1.0, stage="done")
            return {"model": MODEL_NAME, "mode": "embed", "device": _state["device"],
                    "dim": int(vec.shape[0]), "embedding": vec.tolist()}

        return await tasks.dispatch(async_, "speaker_embed", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="embedding failed")

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    watchdog.arm(lambda: _state["ready"], lambda: _state["error"], "pyannote embedding")
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
