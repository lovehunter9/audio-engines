# Forced alignment via Qwen3-ForcedAligner, a separate model from ASR, hence always its own instance.
import os
import tempfile
import threading
import logging
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

from ..gpu import mount_metrics
from ..contract import register

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-align")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ForcedAligner-0.6B")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# align() REQUIRES language but tolerates an unknown one: "auto" aligns byte-identically to "en".
DEFAULT_LANGUAGE = "auto"

_state = {"ready": False, "error": None, "model": None, "device": "cpu"}
# Serialised and off the event loop: 30~110s of blocking inference under a vGPU would freeze uvicorn.
_align_lock = asyncio.Lock()


def _load():
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        kw = dict(dtype=dtype, device_map=(dev if dev == "cpu" else "cuda:0"))
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        try:
            model = Qwen3ForcedAligner.from_pretrained(MODEL_REPO, **kw)
        except TypeError:
            kw.pop("token", None)
            model = Qwen3ForcedAligner.from_pretrained(MODEL_REPO, **kw)
        _state.update(model=model, device=dev, ready=True)
        log.info("Qwen3-ForcedAligner %s loaded on %s", MODEL_REPO, dev)
    except Exception as e:
        _state["error"] = str(e)
        log.exception("forced-aligner load failed: %s", e)


def _field(u, *names):
    for n in names:
        try:
            if isinstance(u, dict):
                if n in u:
                    return u[n]
            elif hasattr(u, n):
                return getattr(u, n)
        except Exception:
            pass
    return None


def build_app(supports):
    app = FastAPI(title="audio-align (Qwen3-ForcedAligner)")
    mount_metrics(app)

    endpoints = [{"method": "POST", "path": "/v1/audio/align",
                  "description": "Forced alignment (single / batch segments)"}]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    @app.post("/v1/audio/align")
    async def align(file: UploadFile = File(...), text: str = Form(default=None),
                    language: str = Form(default=None), segments: str = Form(default=None)):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        data = await file.read()
        # BATCH mode: `segments` JSON [{start,end,text,[language]}], times slice-relative.
        if segments:
            import json as _json
            import io as _io
            import soundfile as _sf

            try:
                _segs = _json.loads(segments)
            except Exception as e:
                raise HTTPException(status_code=400, detail="invalid `segments` json: %s" % e)
            if not isinstance(_segs, list):
                raise HTTPException(status_code=400, detail="`segments` must be a JSON array")
            try:
                _arr, _sr = _sf.read(_io.BytesIO(data), dtype="float32", always_2d=True)
                _arr = _arr.mean(axis=1)  # -> mono
            except Exception as e:
                raise HTTPException(status_code=400, detail="could not decode audio: %s" % e)
            _out = []
            for _seg in _segs:
                try:
                    _st = (str(_seg.get("text") or "")).strip()
                    if not _st:
                        _out.append({"units": [], "language": None})
                        continue
                    _a = float(_seg.get("start") or 0)
                    _b = float(_seg.get("end") or 0)
                    _lo = max(0, int(_a * _sr))
                    _hi = min(len(_arr), int(_b * _sr))
                    if _hi <= _lo:
                        _out.append({"error": "empty segment"})
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
                        _sf.write(_f.name, _arr[_lo:_hi], _sr, format="WAV", subtype="PCM_16")
                        _sp = _f.name
                    _lg = (str(_seg.get("language") or language or "")).strip() or DEFAULT_LANGUAGE

                    def _do_seg(_p=_sp, _t=_st, _l=_lg):
                        try:
                            return _state["model"].align(audio=_p, text=_t, language=_l)
                        except TypeError:
                            return _state["model"].align(_p, _t, _l)

                    async with _align_lock:
                        _res = await asyncio.to_thread(_do_seg)
                    try:
                        os.unlink(_sp)
                    except Exception:
                        pass
                    _ur = _res[0] if _res else []
                    _out.append({"language": _lg, "units": [
                        {"text": _field(u, "text", "word", "token"),
                         "start": _field(u, "start_time", "start"),
                         "end": _field(u, "end_time", "end")} for u in _ur]})
                except Exception as e:
                    _out.append({"error": "align failed: %s" % e})
            return {"model": MODEL_NAME, "mode": "align", "batch": True, "results": _out}
        # SINGLE mode.
        if not (text or "").strip():
            raise HTTPException(status_code=400, detail="`text` is required for forced alignment")
        suffix = os.path.splitext(file.filename or "a.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            lang = (language or "").strip() or DEFAULT_LANGUAGE

            def _do_align():
                try:
                    return _state["model"].align(audio=path, text=text, language=lang)
                except TypeError:
                    return _state["model"].align(path, text, lang)

            async with _align_lock:
                results = await asyncio.to_thread(_do_align)
            units_raw = results[0] if results else []
            units = []
            for u in units_raw:
                units.append({
                    "text": _field(u, "text", "word", "token"),
                    "start": _field(u, "start_time", "start"),
                    "end": _field(u, "end_time", "end"),
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail="alignment failed: %s" % e)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        return {"model": MODEL_NAME, "mode": "align", "device": _state["device"],
                "language": lang, "units": units}

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
