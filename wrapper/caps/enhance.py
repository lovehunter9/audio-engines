# Speech enhancement / denoise with SpeechBrain (audio in -> 16k mono WAV out).
# Ported from the tested enhance.py; deps baked at build time; contract surface
# via wrapper.gpu + wrapper.contract.
import io
import os
import logging
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import uvicorn

from ..gpu import mount_metrics
from ..contract import register
from ..audioio import decode_mono

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-enhance")

MODEL_NAME = os.environ.get("MODEL_NAME", "mtl-mimic-voicebank")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None
SR = 16000  # SpeechBrain enhancement models operate at 16 kHz mono.

# Long audio handled SERVER-SIDE: past CHUNK_S, slide a fixed window over the
# clip and overlap-add with a linear crossfade (inaudible seams) so peak VRAM is
# bounded by ONE window and the client can always send the whole clip.
CHUNK_S = float(os.environ.get("ENHANCE_CHUNK_S", "120") or 120)   # window length (s)
OVERLAP_S = float(os.environ.get("ENHANCE_OVERLAP_S", "1") or 1)   # crossfade overlap (s)

_state = {"ready": False, "error": None, "model": None, "kind": None, "device": "cpu"}


def _load():
    try:
        import torch
        from huggingface_hub import snapshot_download

        src = snapshot_download(MODEL_REPO, local_files_only=True,
                                cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)
        # speechbrain needs a "<type>:<index>" device string ("cuda" alone errors).
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Different enhancement repos need different inference classes; auto-detect
        # by trying from_hparams in order, handling the speechbrain 1.x vs 0.5.x
        # module path move.
        try:
            from speechbrain.inference.enhancement import (
                WaveformEnhancement, SpectralMaskEnhancement)
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import (
                WaveformEnhancement, SpectralMaskEnhancement, SepformerSeparation)
        candidates = [("waveform", WaveformEnhancement),
                      ("spectralmask", SpectralMaskEnhancement),
                      ("sepformer", SepformerSeparation)]
        last = None
        for kind, cls in candidates:
            try:
                savedir = os.path.join(tempfile.gettempdir(), "sb-enhance-%s" % kind)
                model = cls.from_hparams(source=src, savedir=savedir, run_opts={"device": dev})
                _state.update(model=model, kind=kind, device=dev, ready=True)
                log.info("speechbrain %s loaded as '%s' on %s", MODEL_REPO, kind, dev)
                return
            except Exception as e:
                last = e
                log.info("model is not a %s class (%s)", kind, e)
        raise last or RuntimeError("no compatible speechbrain enhancement class")
    except Exception as e:
        _state["error"] = str(e)
        log.exception("enhance load failed: %s", e)


def _run(noisy):
    # Enhance one (1, time) tensor -> 1-D float32 numpy of the same length.
    import torch

    model = _state["model"]
    if _state["kind"] == "sepformer":
        est = model.separate_batch(noisy)        # (batch, time, n_src)
        enhanced = est[..., 0]
    else:
        lengths = torch.ones(noisy.shape[0])
        enhanced = model.enhance_batch(noisy, lengths=lengths)
    arr = enhanced.detach().cpu().numpy().reshape(-1)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return arr


def build_app(supports):
    app = FastAPI(title="audio-enhance (speechbrain)")
    mount_metrics(app)

    endpoints = [{"method": "POST", "path": "/v1/audio/enhance",
                  "description": "Speech enhancement / denoise (returns 16k mono WAV)"}]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    @app.post("/v1/audio/enhance")
    async def enhance(file: UploadFile = File(...)):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        data = await file.read()
        suffix = os.path.splitext(file.filename or "a.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            import numpy as np
            import soundfile as sf

            wav = decode_mono(path, SR)  # (1, time) @ 16k
            total = int(wav.shape[-1])
            chunk = int(CHUNK_S * SR)
            ov = int(OVERLAP_S * SR)
            if chunk <= 0 or total <= chunk:
                # Short clip: single pass (identical to the pre-chunking behaviour).
                out = _run(wav)
            else:
                # Long clip: sliding window + overlap-add crossfade, one window on
                # the GPU at a time so peak VRAM is bounded regardless of duration.
                hop = max(1, chunk - ov)
                out = np.zeros(total, dtype="float32")
                wsum = np.zeros(total, dtype="float32")
                pos = 0
                while pos < total:
                    end = min(total, pos + chunk)
                    n = end - pos
                    enh = _run(wav[:, pos:end])
                    if enh.shape[0] >= n:
                        enh = enh[:n]
                    else:
                        enh = np.pad(enh, (0, n - enh.shape[0]))
                    w = np.ones(n, dtype="float32")
                    if pos > 0 and ov > 0:
                        r = min(ov, n)
                        w[:r] = np.linspace(0.0, 1.0, r, dtype="float32")  # fade in
                    if end < total and ov > 0:
                        r = min(ov, n)
                        w[-r:] = np.minimum(w[-r:], np.linspace(1.0, 0.0, r, dtype="float32"))
                    out[pos:end] += enh * w
                    wsum[pos:end] += w
                    if end >= total:
                        break
                    pos += hop
                nz = wsum > 1e-6
                out[nz] = out[nz] / wsum[nz]
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > 1.0:
                out = out / peak  # guard against clipping
            buf = io.BytesIO()
            sf.write(buf, out, SR, format="WAV", subtype="PCM_16")
            body = buf.getvalue()
        except Exception as e:
            raise HTTPException(status_code=500, detail="enhance failed: %s" % e)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        return Response(content=body, media_type="audio/wav",
                        headers={"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "enhance"})

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
