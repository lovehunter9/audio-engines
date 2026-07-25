# Speaker/audio embedding with pyannote.audio (one whole-file vector). Ported
# from the tested embed.py; deps baked at build time; contract surface via
# wrapper.gpu + wrapper.contract.
import os
import logging
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

from ..gpu import mount_metrics
from ..contract import register
from ..audioio import decode

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

    endpoints = [{"method": "POST", "path": "/v1/audio/embeddings",
                  "description": "Speaker embedding (one vector per clip)"}]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    @app.post("/v1/audio/embeddings")
    async def embeddings(file: UploadFile = File(...)):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        data = await file.read()
        suffix = os.path.splitext(file.filename or "a.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            import numpy as np

            waveform, sr = decode(path)
            emb = _state["inference"]({"waveform": waveform, "sample_rate": sr})
            vec = np.asarray(emb, dtype="float32").reshape(-1)
        except Exception as e:
            raise HTTPException(status_code=500, detail="embedding failed: %s" % e)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        _state["dim"] = int(vec.shape[0])
        return {"model": MODEL_NAME, "mode": "embed", "device": _state["device"],
                "dim": int(vec.shape[0]), "embedding": vec.tolist()}

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
