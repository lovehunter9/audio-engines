# Speaker diarization with pyannote.audio. Ported from the tested diar.py; deps
# baked at build time; contract surface via wrapper.gpu + wrapper.contract.
import os
import logging
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

from ..gpu import mount_metrics
from ..contract import register
from ..audioio import decode

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-diar")

MODEL_NAME = os.environ.get("MODEL_NAME", "pyannote-community-1")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_state = {"ready": False, "error": None, "pipeline": None, "device": "cpu"}


def _load():
    try:
        import torch
        from pyannote.audio import Pipeline

        # pyannote.audio 4 uses token=; older used use_auth_token=.
        try:
            pipe = Pipeline.from_pretrained(MODEL_REPO, token=HF_TOKEN)
        except TypeError:
            pipe = Pipeline.from_pretrained(MODEL_REPO, use_auth_token=HF_TOKEN)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(torch.device(dev))
        _state["pipeline"], _state["device"], _state["ready"] = pipe, dev, True
        log.info("pyannote pipeline %s loaded on %s", MODEL_REPO, dev)
    except Exception as e:
        _state["error"] = str(e)
        log.exception("pipeline load failed: %s", e)


def build_app(supports):
    app = FastAPI(title="audio-diarization (pyannote)")
    mount_metrics(app)

    endpoints = [{"method": "POST", "path": "/v1/audio/diarization",
                  "description": "Speaker diarization (who spoke when)"}]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    @app.post("/v1/audio/diarization")
    async def diarize(file: UploadFile = File(...), num_speakers: str = Form(default=None),
                      min_speakers: str = Form(default=None),
                      max_speakers: str = Form(default=None)):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "pipeline not ready")
        data = await file.read()
        suffix = os.path.splitext(file.filename or "a.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            kw = {}
            if num_speakers:
                kw["num_speakers"] = int(num_speakers)
            if min_speakers:
                kw["min_speakers"] = int(min_speakers)
            if max_speakers:
                kw["max_speakers"] = int(max_speakers)
            waveform, sr = decode(path)
            out = _state["pipeline"]({"waveform": waveform, "sample_rate": sr}, **kw)
        except Exception as e:
            raise HTTPException(status_code=500, detail="diarization failed: %s" % e)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        # pyannote 4 returns an object whose .speaker_diarization is the Annotation
        # (v3 returned it directly); unwrap to a common itertracks() form.
        ann = getattr(out, "speaker_diarization", out)
        segs = [{"start": round(float(t.start), 3), "end": round(float(t.end), 3),
                 "speaker": str(spk)}
                for t, _, spk in ann.itertracks(yield_label=True)]
        speakers = sorted({s["speaker"] for s in segs})
        return {"model": MODEL_NAME, "mode": "diar", "device": _state["device"],
                "num_speakers": len(speakers), "speakers": speakers,
                "num_segments": len(segs), "segments": segs}

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
