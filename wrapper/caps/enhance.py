# Speech enhancement / denoise with SpeechBrain (audio in -> 16k mono, WAV by default).
import contextlib
import io
import os
import logging
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics, quota_mib
from ..contract import register, EngineArgs
from ..audioio import decode_mono, seconds, unlink
from ..limits import Bounds
from ..runtime import Runtime

log = logging.getLogger("audio-enhance")

_runtime = Runtime(model=None, kind=None, device="cpu")
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None
SR = 16000  # SpeechBrain enhancement models operate at 16 kHz mono.

# Overlap-add windowing bounds peak VRAM by ONE window, so it shrinks with the slice we got.
_mib = quota_mib()
CHUNK_S = 120.0 if _mib >= 4000 else (60.0 if _mib >= 2000 else 30.0)
OVERLAP_S = 1.0   # crossfade overlap (s)

# A window is one big forward pass, so fp16 is the only speed lever — and it can underflow a mask.
_args = EngineArgs()
AMP = _args.switch("--amp")
# Overlap-add keeps VRAM flat in the clip's length, so this bound is about the rest of it: the
# decoded input, the float32 output and the encoded body all sit in memory at once, ~256 KB per
# second between them, and the enhanced audio is then held until the caller collects it.
BOUNDS = Bounds(_args, seconds=14400)
_args.warn_unclaimed(log)

# Default stays WAV, but 16k PCM16 is ~2 MB/min and the gateway buffers whole bodies in memory.
_FORMATS = {
    "wav": ("WAV", ("PCM_16",), "audio/wav"),
    "flac": ("FLAC", ("PCM_16",), "audio/flac"),
    "ogg": ("OGG", ("OPUS", "VORBIS"), "audio/ogg"),   # Opus needs libsndfile >= 1.2
}
_FORMAT_ALIAS = {"": "wav", "opus": "ogg", "vorbis": "ogg", "oga": "ogg"}

_state = _runtime.state


def _speechbrain_device(torch):
    # speechbrain needs a "<type>:<index>" device string ("cuda" alone errors).
    base = (os.environ.get("AUDIO_BASE") or "").strip()
    if base == "enhancexpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("enhancexpu requires an Intel XPU; refusing CPU")
        return "xpu:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load():
    try:
        from huggingface_hub import snapshot_download

        src = snapshot_download(MODEL_REPO, local_files_only=True,
                                cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)
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
        # from_hparams(..., device=xpu) and torch.xpu.get_device_name both
        # SIGSEGV'd (intel2 ~30s, intel4 ~1.5s). Load on CPU first; only then
        # ask XPU anything. Arc still may die on .to — the log line before
        # the move tells us which side.
        want_xpu = (os.environ.get("AUDIO_BASE") or "").strip() == "enhancexpu"
        for kind, cls in candidates:
            try:
                savedir = os.path.join(tempfile.gettempdir(), "sb-enhance-%s" % kind)
                log.info("speechbrain %s from_hparams on cpu", kind)
                model = cls.from_hparams(source=src, savedir=savedir,
                                         run_opts={"device": "cpu"})
                log.info("speechbrain %s cpu load done", kind)
                # intel2/intel5: model.to("xpu:0") SIGSEGV after cpu load
                # (exit 139, not catchable). Stay on CPU. The enhancexpu
                # image still has XPU torch; using it kills the process.
                if want_xpu:
                    log.warning(
                        "enhancexpu: not calling model.to(xpu); "
                        "to('xpu:0') SIGSEGV'd on this GPU after cpu load")
                dev = "cpu"
                _state.update(model=model, kind=kind, device=dev, ready=True)
                log.info("speechbrain %s loaded as '%s' on %s", MODEL_REPO, kind, dev)
                return
            except Exception as e:
                last = e
                log.info("model is not a %s class (%s)", kind, e)
        raise last or RuntimeError("no compatible speechbrain enhancement class")
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("enhance load failed: %s", e)


def _run(noisy):
    # Enhance one (1, time) tensor -> 1-D float32 numpy of the same length.
    import torch

    model = _state["model"]
    # SpeechBrain's enhance_batch has no no-grad of its own, and that dead graph dominates VRAM.
    kind = _state["device"].split(":", 1)[0]
    amp = AMP and kind in ("cuda", "xpu")
    with torch.no_grad(), (torch.autocast(kind, dtype=torch.float16) if amp
                           else contextlib.nullcontext()):
        if _state["kind"] == "sepformer":
            est = model.separate_batch(noisy)    # (batch, time, n_src)
            enhanced = est[..., 0]
        else:
            lengths = torch.ones(noisy.shape[0])
            enhanced = model.enhance_batch(noisy, lengths=lengths)
        arr = enhanced.float().detach().cpu().numpy().reshape(-1)
    try:
        if _state["device"].startswith("xpu"):
            torch.xpu.empty_cache()
        else:
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

    register(app, model_name=MODEL_NAME, module="enhance", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/enhance")
    async def enhance(file: UploadFile = File(...),
                      fmt: str = Form(default="wav", alias="format"),
                      async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        want = (fmt or "wav").strip().lower()
        want = _FORMAT_ALIAS.get(want, want)
        if want not in _FORMATS:
            raise HTTPException(status_code=400, detail="format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        path, _seconds = await BOUNDS.spill(
            file, "this engine holds the clip and its enhanced copy in memory")

        def _work(ctx):
            import numpy as np

            ctx.progress(ratio=0.0, stage="decode")
            wav = decode_mono(path, SR)  # (1, time) @ 16k
            total = int(wav.shape[-1])
            ctx.meter(input_seconds=total / SR)
            chunk = int(CHUNK_S * SR)
            ov = int(OVERLAP_S * SR)
            if chunk <= 0 or total <= chunk:
                ctx.progress(stage="inference", done=0, total=1)
                out = _run(wav)      # short clip: single pass
                ctx.progress(done=1, total=1)
            else:
                # One window on the GPU at a time, so peak VRAM is flat in the clip's duration.
                hop = max(1, chunk - ov)
                nwin = -(-max(1, total - chunk) // hop) + 1
                log.info("enhancing %.1fs in %d windows of %.0fs", total / SR, nwin, CHUNK_S)
                out = np.zeros(total, dtype="float32")
                wsum = np.zeros(total, dtype="float32")
                pos = 0
                idx = 0
                ctx.progress(stage="inference", done=0, total=nwin)
                while pos < total:
                    ctx.checkpoint()
                    end = min(total, pos + chunk)
                    n = end - pos
                    t0 = time.time()
                    enh = _run(wav[:, pos:end])
                    idx += 1
                    ctx.progress(done=idx, total=nwin)
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
            ctx.meter(output_seconds=seconds(out, SR))
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > 1.0:
                out = out / peak  # guard against clipping
            ctx.progress(stage="encode")
            body, mime, codec = _encode(out, want)
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(body, mime, suffix="." + codec.split("/")[0],
                                headers={"X-Audio-Model": MODEL_NAME,
                                         "X-Audio-Mode": "enhance",
                                         "X-Audio-Format": codec})

        return await tasks.dispatch(async_, "enhance", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="enhance failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "speechbrain enhancement")
