# Vocal and accompaniment separation. CUDA runs the network in torch; melbandov compiles it for OpenVINO.
import io
import logging
import os
import subprocess
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .. import hfgate
from .. import tasks
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..limits import Bounds
from ..runtime import Runtime

log = logging.getLogger("audio-separate")

_runtime = Runtime(model=None, compiled=None, device="cpu")
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
SR = 44100
CHUNK = 352800  # 8 seconds at 44100, the width this checkpoint was trained to see
OVERLAP = 4
_STEMS = ("vocals", "instrumental", "both")
_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
}
_CFG = dict(
    dim=384,
    depth=6,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_bands=60,
    dim_head=64,
    heads=8,
    attn_dropout=0.0,
    ff_dropout=0.0,
    flash_attn=False,
    dim_freqs_in=1025,
    sample_rate=SR,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
)

_args = EngineArgs()
# A decoded stereo minute is about 21 MB, and the two stems sit beside it until the response leaves.
BOUNDS = Bounds(_args, seconds=1200, megabytes=512)
_args.warn_unclaimed(log)
_state = _runtime.state
_OV_XML = "melband-net-v1.xml"


def _is_ov():
    return (os.environ.get("AUDIO_BASE") or "").strip() == "melbandov"


def _require_ov_gpu():
    from .. import ovutil

    device = ovutil.device()
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel") and device.upper() != "GPU":
        raise RuntimeError("melbandov on %s must use GPU, got %s" % (mode, device))
    if device.upper() != "GPU":
        raise RuntimeError("melbandov requires OpenVINO GPU, got %s" % device)
    return device


def _checkpoint():
    from huggingface_hub import snapshot_download

    repo, include, exclude = _runtime._hub_spec()
    if not repo:
        raise RuntimeError("MODEL_SOURCE is empty")
    kw = {
        "local_files_only": True,
        "cache_dir": os.environ.get("HF_HUB_CACHE"),
        "token": os.environ.get("HF_TOKEN") or None,
    }
    if include:
        kw["allow_patterns"] = include
    if exclude:
        kw["ignore_patterns"] = exclude
    src = snapshot_download(repo, **kw)
    wanted = [os.path.basename(name) for name in (include or ["MelBandRoformer.ckpt"])]
    found = []
    for root, _dirs, files in os.walk(src):
        for name in files:
            if name in wanted or name.endswith(".ckpt"):
                found.append(os.path.join(root, name))
    for path in found:
        if os.path.basename(path) in wanted:
            return path
    if found:
        return found[0]
    raise RuntimeError("no checkpoint under %s" % src)


def _load_weights(model, path, torch):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    state = blob
    if isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
        if not any(str(key).startswith("band_split") for key in blob):
            state = blob["state_dict"]
    model.load_state_dict(state)


def _compile(model, torch):
    import openvino as ov

    class _Net(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            object.__setattr__(self, "_inner", inner)

        def forward(self, features):
            return self._inner.net_forward(features)

    device = _require_ov_gpu()
    with torch.no_grad():
        _stft, example, _length = model.preprocess(torch.zeros(1, 2, CHUNK))
    log.info("mel-band network features %s", tuple(example.shape))
    cache = os.environ.get("CACHE_DIR") or "/tmp"
    xml = os.path.join(cache, _OV_XML)
    core = ov.Core()
    if os.path.isfile(xml):
        ir = core.read_model(xml)
    else:
        ir = ov.convert_model(_Net(model).eval(), example_input=example)
        ov.save_model(ir, xml)
        log.info("wrote %s", xml)
    return core.compile_model(ir, device)


def _load():
    try:
        import torch

        from .melband.model import MelBandRoformer

        path = _checkpoint()
        log.info("loading %s from %s", MODEL_NAME or "mel-band-roformer", path)
        model = MelBandRoformer(**_CFG)
        _load_weights(model, path, torch)
        model.eval()
        if _is_ov():
            compiled = _compile(model, torch)
            _state["model"] = model
            _state["compiled"] = compiled
            _state["device"] = "openvino"
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("melband requires CUDA")
            model.cuda()
            _state["model"] = model
            _state["device"] = "cuda"
        _state["ready"] = True
        log.info("ready on %s", _state["device"])
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("separate model failed to load: %s", e)


def _decode(path):
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, timeout=600)
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(detail[:400] or "ffmpeg could not decode the upload")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size < 2:
        raise RuntimeError("decoded audio is empty")
    pcm = pcm[:pcm.size // 2 * 2]
    return np.ascontiguousarray(pcm.reshape(-1, 2).T)


def _window(fade):
    import numpy as np

    window = np.ones(CHUNK, dtype=np.float32)
    window[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    window[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return window


def _infer_chunk(part):
    import numpy as np
    import torch

    audio = torch.from_numpy(np.ascontiguousarray(part))
    model = _state["model"]
    with torch.no_grad():
        if _state["device"] == "cuda":
            recon = model(audio.unsqueeze(0).cuda())
        else:
            stft, features, length = model.preprocess(audio.unsqueeze(0))
            compiled = _state["compiled"]
            raw = compiled([features.detach().float().numpy()])[compiled.output(0)]
            masks = torch.from_numpy(np.array(raw, copy=True)).float()
            recon = model.postprocess(stft, masks, length)
    out = recon[0].float().detach().cpu().numpy()
    if out.ndim == 3:
        out = out[0]
    return out


def _vocals(mix, ctx):
    import numpy as np

    step = CHUNK // OVERLAP
    fade = CHUNK // 10
    border = CHUNK - step
    padded = mix.shape[1] > 2 * border and border > 0
    work = np.pad(mix, ((0, 0), (border, border)), mode="reflect") if padded else mix
    window = _window(fade)
    total = work.shape[1]
    nwin = max(1, (total + step - 1) // step)
    result = np.zeros_like(work)
    counter = np.zeros(total, dtype=np.float32)
    pos = 0
    done = 0
    ctx.progress(stage="inference", done=0, total=nwin)
    while pos < total:
        ctx.checkpoint()
        part = work[:, pos:pos + CHUNK]
        length = part.shape[-1]
        if length < CHUNK:
            if length > CHUNK // 2 + 1:
                part = np.pad(part, ((0, 0), (0, CHUNK - length)), mode="reflect")
            else:
                part = np.pad(part, ((0, 0), (0, CHUNK - length)), mode="constant")
        voc = _infer_chunk(part)
        n = min(length, voc.shape[-1])
        weight = window[:n].copy()
        if pos == 0:
            weight[:min(fade, n)] = 1.0
        if pos + CHUNK >= total:
            weight[max(0, n - fade):] = 1.0
        result[:, pos:pos + n] += voc[:, :n] * weight
        counter[pos:pos + n] += weight
        pos += step
        done += 1
        ctx.progress(done=done, total=nwin)
    vocals = np.divide(result, counter, out=np.zeros_like(result), where=counter > 1e-8)
    np.nan_to_num(vocals, copy=False, nan=0.0)
    if padded:
        vocals = vocals[:, border:-border]
    return vocals


def _fit(vocals, mix):
    import numpy as np

    n = min(vocals.shape[-1], mix.shape[-1])
    vocals = vocals[:, :n]
    mix = mix[:, :n]
    if vocals.shape[0] == 1 and mix.shape[0] == 2:
        vocals = np.repeat(vocals, 2, axis=0)
    return vocals, mix


def _guard(audio):
    import numpy as np

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        return audio / peak
    return audio


def _encode(audio, want):
    import soundfile as sf

    container, subtype, mime = _FORMATS[want]
    buf = io.BytesIO()
    sf.write(buf, audio.T, SR, format=container, subtype=subtype)
    return buf.getvalue(), mime


def _stems(mix, stem, ctx):
    vocals, mix = _fit(_vocals(mix, ctx), mix)
    instrumental = mix - vocals
    if stem == "vocals":
        return {"vocals": _guard(vocals)}
    if stem == "instrumental":
        return {"instrumental": _guard(instrumental)}
    peak = max(_peak(vocals), _peak(instrumental))
    scale = peak if peak > 1.0 else 1.0
    return {"vocals": vocals / scale, "instrumental": instrumental / scale}


def _peak(audio):
    import numpy as np

    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _package(stems, want):
    if len(stems) == 1:
        name, audio = next(iter(stems.items()))
        body, mime = _encode(audio, want)
        return body, mime, "." + want, name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, audio in stems.items():
            body, _mime = _encode(audio, want)
            archive.writestr("%s.%s" % (name, want), body)
    return buf.getvalue(), "application/zip", ".zip", "both"


def build_app(supports):
    app = FastAPI(title="audio-separate (mel-band roformer)")
    mount_metrics(app)
    register(app, model_name=MODEL_NAME, module="separate", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/separate")
    async def separate(file: UploadFile = File(...),
                       stem: str = Form(default="vocals"),
                       fmt: str = Form(default="wav", alias="format"),
                       async_: str = Form(default=None, alias="async")):
        from ..audioio import unlink

        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        stem = (stem or "vocals").strip().lower()
        if stem not in _STEMS:
            raise HTTPException(status_code=400, detail="stem must be one of %s" % ", ".join(_STEMS))
        want = (fmt or "wav").strip().lower()
        if want not in _FORMATS:
            raise HTTPException(status_code=400,
                                detail="format must be one of %s" % ", ".join(sorted(_FORMATS)))
        path, _seconds = await BOUNDS.spill(file, "this engine holds the mix and both stems in memory")

        def _work(ctx):
            from ..audioio import seconds

            ctx.progress(ratio=0.0, stage="decode")
            try:
                mix = _decode(path)
            finally:
                unlink(path)
            ctx.meter(input_seconds=mix.shape[-1] / SR)
            stems = _stems(mix, stem, ctx)
            ctx.progress(stage="encode")
            body, mime, suffix, label = _package(stems, want)
            ctx.meter(output_seconds=seconds(next(iter(stems.values())), SR))
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(
                body, mime, suffix=suffix,
                headers={"X-Audio-Model": MODEL_NAME, "X-Audio-Stem": label})

        return await tasks.dispatch(
            async_, "separate", MODEL_NAME, _work,
            cleanup=lambda: unlink(path), fail="separate failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "mel-band roformer", timeout_s=3600)
