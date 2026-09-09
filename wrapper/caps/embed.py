# Speaker embedding with pyannote.audio: one fixed-length vector for the whole clip.
import os
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import decode, unlink
from ..limits import Bounds
from ..runtime import Runtime

log = logging.getLogger("audio-embed")

_runtime = Runtime(inference=None, device="cpu", dim=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# window="whole" is one forward pass over the entire clip, with no chunking anywhere to bound it,
# which makes this the least tolerant cap here. The answer is one vector for the whole recording,
# so the long clips the other caps are sized for have no meaning at this endpoint: an enrollment
# sample is seconds, and half an hour is already far past anything a speaker vector says.
_args = EngineArgs()
BOUNDS = Bounds(_args, seconds=1800)
_args.warn_unclaimed(log)

_state = _runtime.state


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
        _state["error"] = hfgate.explain(MODEL_REPO, e)
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
        path, _seconds = await BOUNDS.spill(
            file, "this model embeds the whole clip in one pass")

        def _work(ctx):
            import numpy as np

            ctx.progress(ratio=0.0, stage="decode")
            waveform, sr = decode(path)
            ctx.meter(input_seconds=float(waveform.shape[-1]) / float(sr))
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
    _runtime.serve(supports, _load, build_app, "pyannote embedding")
