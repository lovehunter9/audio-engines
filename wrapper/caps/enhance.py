# Speech enhancement / denoise with SpeechBrain (audio in -> 16k mono, WAV by default).
import asyncio
import contextlib
import io
import os
import logging
import tempfile
import threading
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import uvicorn

from ..gpu import mount_metrics
from ..contract import register
from ..audioio import decode_mono, spill

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-enhance")

MODEL_NAME = os.environ.get("MODEL_NAME", "mtl-mimic-voicebank")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
HF_TOKEN = os.environ.get("HF_TOKEN") or None
SR = 16000  # SpeechBrain enhancement models operate at 16 kHz mono.

# Long clips are windowed with an overlap-add crossfade so peak VRAM is bounded by ONE window.
CHUNK_S = float(os.environ.get("ENHANCE_CHUNK_S", "120") or 120)   # window length (s)
OVERLAP_S = float(os.environ.get("ENHANCE_OVERLAP_S", "1") or 1)   # crossfade overlap (s)
# A window is one big forward pass, so fp16 is the only speed lever — and it can underflow a mask.
AMP = (os.environ.get("ENHANCE_AMP", "").strip().lower() in ("1", "true", "yes", "on"))

# Default stays WAV, but 16k PCM16 is ~2 MB/min and the gateway buffers whole bodies in memory.
_FORMATS = {
    "wav": ("WAV", ("PCM_16",), "audio/wav"),
    "flac": ("FLAC", ("PCM_16",), "audio/flac"),
    "ogg": ("OGG", ("OPUS", "VORBIS"), "audio/ogg"),   # Opus needs libsndfile >= 1.2
}
_FORMAT_ALIAS = {"": "wav", "opus": "ogg", "vorbis": "ogg", "oga": "ogg"}

_state = {"ready": False, "error": None, "model": None, "kind": None, "device": "cpu"}
# Serialised and off the event loop: decode + windowed inference + encode on a long clip
# is minutes of blocking work, which would stop /v1/models answering.
_infer_lock = asyncio.Lock()


def _load():
    try:
        import torch
        from huggingface_hub import snapshot_download

        src = snapshot_download(MODEL_REPO, local_files_only=True,
                                cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)
        # speechbrain needs a "<type>:<index>" device string ("cuda" alone errors).
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Repos need different inference classes; try each in turn (1.x moved the module path).
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
    # SpeechBrain's enhance_batch has no no-grad of its own, and that dead graph dominates VRAM.
    amp = AMP and _state["device"].startswith("cuda")
    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.float16) if amp
                           else contextlib.nullcontext()):
        if _state["kind"] == "sepformer":
            est = model.separate_batch(noisy)    # (batch, time, n_src)
            enhanced = est[..., 0]
        else:
            lengths = torch.ones(noisy.shape[0])
            enhanced = model.enhance_batch(noisy, lengths=lengths)
        arr = enhanced.float().detach().cpu().numpy().reshape(-1)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return arr


def _encode(out, want):
    # Requested container, else lossless FLAC, else WAV; returns the codec that actually ran.
    import soundfile as sf

    order = [want] + [f for f in ("flac", "wav") if f != want]
    errs = []
    for fmt in order:
        container, subtypes, mime = _FORMATS[fmt]
        for sub in subtypes:
            try:
                if not sf.check_format(container, sub):
                    continue
            except Exception:
                pass  # older soundfile without check_format: just try the write
            try:
                buf = io.BytesIO()
                sf.write(buf, out, SR, format=container, subtype=sub)
                if fmt != want:
                    log.warning("%s unavailable in this libsndfile, encoded %s/%s",
                                want, container, sub)
                return buf.getvalue(), mime, "%s/%s" % (fmt, sub.lower())
            except Exception as e:
                errs.append("%s/%s: %s" % (container, sub, e))
    raise RuntimeError("no usable encoder (%s)" % "; ".join(errs))


def build_app(supports):
    app = FastAPI(title="audio-enhance (speechbrain)")
    mount_metrics(app)

    endpoints = [{"method": "POST", "path": "/v1/audio/enhance",
                  "description": "Speech enhancement / denoise "
                                 "(16k mono, format=wav|flac|ogg, default wav)"}]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    @app.post("/v1/audio/enhance")
    async def enhance(file: UploadFile = File(...),
                      fmt: str = Form(default="wav", alias="format")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        want = (fmt or "wav").strip().lower()
        want = _FORMAT_ALIAS.get(want, want)
        if want not in _FORMATS:
            raise HTTPException(status_code=400, detail="format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        data = await file.read()
        path = await asyncio.to_thread(spill, data, file.filename)

        def _work():
            import numpy as np

            wav = decode_mono(path, SR)  # (1, time) @ 16k
            total = int(wav.shape[-1])
            chunk = int(CHUNK_S * SR)
            ov = int(OVERLAP_S * SR)
            if chunk <= 0 or total <= chunk:
                out = _run(wav)      # short clip: single pass
            else:
                # One window on the GPU at a time, so peak VRAM is flat in the clip's duration.
                hop = max(1, chunk - ov)
                nwin = -(-max(1, total - chunk) // hop) + 1
                log.info("enhancing %.1fs in %d windows of %.0fs", total / SR, nwin, CHUNK_S)
                out = np.zeros(total, dtype="float32")
                wsum = np.zeros(total, dtype="float32")
                pos = 0
                idx = 0
                while pos < total:
                    end = min(total, pos + chunk)
                    n = end - pos
                    t0 = time.time()
                    enh = _run(wav[:, pos:end])
                    idx += 1
                    log.info("window %d/%d (%.0f-%.0fs) took %.1fs",
                             idx, nwin, pos / SR, end / SR, time.time() - t0)
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
            return _encode(out, want)

        try:
            async with _infer_lock:
                body, mime, codec = await asyncio.to_thread(_work)
        except Exception as e:
            raise HTTPException(status_code=500, detail="enhance failed: %s" % e)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        return Response(content=body, media_type=mime,
                        headers={"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "enhance",
                                 "X-Audio-Format": codec})

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
