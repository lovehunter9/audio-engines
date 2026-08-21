# Sound effects from text: Dasheng-AudioGen in this process, the wrapper is the only server.
#
# A flow-matching diffusion transformer, not an LLM. There is no KV cache, no autoregression and
# no child engine process — one AutoModel with trust_remote_code, plus the two models its config
# names: google/flan-t5-large for text and mispeech/dashengtokenizer for the codec.
#
# Two consequences shape everything below:
#
#   * `generate()` takes a list and denoises the whole list in one pass. Batch here is a real
#     forward-pass win, not just one round trip instead of N like the autoregressive caps.
#   * Every clip is ~10s (248 latent tokens at 25/s) and nothing moves it. There is a
#     DurationPredictor, but it was trained on a corpus that is entirely 10-second clips, so it
#     returns the same length for "a short click" and for "thirty seconds of rain" — measured
#     across captions, caption lengths and a 278-character `asr`, all 9.92s to the token. The
#     paper (arXiv 2605.27838) names the 10s ceiling as a limitation, so do not expose a
#     duration parameter or promise callers that wording controls it.
import base64
import contextlib
import io
import logging
import os
import threading
import time

import numpy as np

from fastapi import FastAPI, HTTPException, Request

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import seconds
from ..runtime import Runtime

log = logging.getLogger("audio-sound-fx")

_runtime = Runtime("mispeech/Dasheng-AudioGen", model=None, dtype=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
# The one deploy-time knob: "auto" resolves against the card, so the manifest default is empty.
DTYPE = str(_args.text("--dtype", "auto") or "auto").strip().lower()

# Upstream's generation defaults, overridable at deploy time; a body field of the same name wins.
DEFAULT_STEPS = _args.count("--num-steps", 25)
DEFAULT_GUIDANCE = _args.number("--guidance-scale", 5.0)
DEFAULT_SWAY = _args.number("--sway-sampling-coef", -1.0)
if not 1 <= DEFAULT_STEPS <= 200:
    # Say it once at boot rather than 400 every call for a value nobody can change per request.
    log.warning("--num-steps %d is outside 1..200; falling back to 25", DEFAULT_STEPS)
    DEFAULT_STEPS = 25

# CFG doubles the batch entering the backbone, so this is derived from the architecture, not asked.
MAX_BATCH = 8

# The codec decodes at 16 kHz here; a constant because /v1/models predates the loaded weights.
OUT_SR = 16000
BOOT_TIMEOUT_S = 1800.0

_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),  # raw little-endian int16, no container
}

# compose_prompt's optional aspects, under the model's own names; `input` carries the caption.
_ASPECTS = ("speech", "asr", "sfx", "music", "env")

_state = _runtime.state
# One model, one set of buffers: generations are serialized, batch is where parallelism lives.
_gen_lock = threading.Lock()


def _model_path():
    """The pre-downloaded snapshot: llm-init fetches the weights before this process ever starts."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _resolve_dtype():
    import torch

    named = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if DTYPE in named:
        return named[DTYPE]
    if DTYPE not in ("", "auto"):
        log.warning("unknown --dtype %r; falling back to auto", DTYPE)
    # fp32 over fp16 without bf16: flow matching runs out of fp16's exponent range first.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def _keep_codec_fp32(m):
    """Undo the half-precision cast for the codec, and make sure its input arrives fp32 too.

    vocos reconstructs the waveform with `torch.istft`, which has no half kernel, and its ISTFTHead
    is decorated `@torch.autocast(enabled=False)` for exactly that reason. Casting it along with the
    rest of the model therefore cannot work, and fails in one of two ways depending on where the
    mismatch surfaces first: `expected scalar type Float but found BFloat16` from istft, or
    `mat1 and mat2 must have the same dtype` from the head's Linear once an autocast region hands it
    fp32 activations. Both were seen on 2026-08-11 before this existed.

    Cheap to leave alone, too: the codec is a rounding error next to the 2.2B-parameter DiT that the
    bf16 budget is actually for.
    """
    import torch

    codec = m.audio_tokenizer
    codec.to(dtype=torch.float32)
    inner = codec.decode

    def decode_fp32(embeddings, *a, **kw):
        # generate() hands the latent over as content.dtype, so the cast belongs at this boundary.
        with torch.autocast("cuda", enabled=False):
            return inner(embeddings.float(), *a, **kw)

    codec.decode = decode_fp32


def _audit_dtypes(m):
    """Log each top-level submodule's dtypes.

    A single fp32 island in an otherwise bf16 model is not a load error — it is a 500 on the first
    request, and the aten message (`expected scalar type Float but found BFloat16`) names neither
    the module nor the tensor. Three of the five submodules here are loaded by upstream's own code
    with no dtype argument, so this is exactly where a seam is likely to be.
    """
    for name, sub in m.named_children():
        seen = {str(p.dtype).replace("torch.", "") for p in sub.parameters()}
        seen |= {str(b.dtype).replace("torch.", "")
                 for b in sub.buffers() if b.is_floating_point()}
        if seen:
            log.info("dtype audit: %-24s %s", name, ", ".join(sorted(seen)))


def _load():
    import torch
    from transformers import AutoModel

    path = _model_path()
    dt = _resolve_dtype()
    log.info("loading %s from %s (dtype=%s)", MODEL_NAME, path, dt)
    # torch_dtype keeps the backbone off host RAM; the cast pulls encoder and codec into it too.
    m = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                  torch_dtype=dt)
    m = m.to(device="cuda", dtype=dt).eval()
    if dt != torch.float32:
        _keep_codec_fp32(m)
    _audit_dtypes(m)
    return m, dt


def _warmup():
    """Spend the first-call cost here: lazy kernel selection, allocator growth, codec setup."""
    t0 = time.time()
    try:
        with _gen_lock:
            # `input`, not `caption`: _compose reads the request shape, and warmup must match it.
            _generate_blocking([{"input": "A short click.", "num_steps": 4,
                                 "guidance_scale": 1.0}])
        log.info("warmup generation took %.0fs", time.time() - t0)
    except Exception as e:
        # Serviceable either way; if it broke for a real reason, this is the cheapest frame to see.
        log.warning("warmup generation failed after %.0fs: %s", time.time() - t0, e, exc_info=True)


def _boot():
    global OUT_SR
    try:
        import torch

        m, dt = _load()
        _state.update(model=m, dtype=dt, half=dt in (torch.bfloat16, torch.float16))
        rate = int(getattr(getattr(m, "config", None), "sample_rate", OUT_SR) or OUT_SR)
        if rate != OUT_SR:
            log.warning("this checkpoint decodes at %d Hz, not the %d Hz /v1/models already "
                        "advertised; response headers will carry the real rate", rate, OUT_SR)
            OUT_SR = rate
        log.info("%s loaded: dtype=%s sample_rate=%d max_batch=%d", MODEL_NAME, dt, OUT_SR, MAX_BATCH)
        _warmup()
        _state.update(ready=True, error=None)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("sound_fx engine failed to start: %s", e)


def _explain(e):
    """Some exceptions carry everything in the type and nothing in str() — say the type too."""
    msg = str(e).strip()
    return msg if msg else type(e).__name__


def _compose(row):
    """Build the tagged prompt from one request body, in the model's own vocabulary."""
    model = _state["model"]
    kwargs = {}
    if row.get("prompt"):
        kwargs["prompt"] = str(row["prompt"])
    else:
        kwargs["caption"] = str(row.get("input") or "")
        for name in _ASPECTS:
            value = row.get(name)
            if value:
                kwargs[name] = str(value)
    try:
        return model.compose_prompt(**kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=_explain(e))


def _gen_kwargs(row):
    def num(name, default):
        try:
            value = row.get(name)
            return default if value is None else float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="%s must be a number" % name)

    steps = int(num("num_steps", DEFAULT_STEPS))
    if not 1 <= steps <= 200:
        raise HTTPException(status_code=400, detail="num_steps must be between 1 and 200")
    return {"num_steps": steps,
            "guidance_scale": num("guidance_scale", DEFAULT_GUIDANCE),
            "sway_sampling_coef": num("sway_sampling_coef", DEFAULT_SWAY)}


def _autocast():
    """Run the generation under autocast when the weights are half precision.

    Casting the modules cannot make this model single-dtype, because upstream's own code is written
    for fp32 and promotes back to it in places `.to()` cannot follow — `content_adapter.forward`
    multiplies the attention output by a mask it has explicitly `.float()`ed, then feeds the fp32
    result to a bf16 LayerNorm. Under autocast each op casts its own operands, so those promotions
    cost a conversion instead of a 500.

    This does not cover the codec: its ISTFT head pins autocast off, which is why _keep_codec_fp32
    exists. The two are complementary, and each alone only moves the failure to the other's half —
    autocast alone died in the codec, the codec fix alone died in content_adapter.

    `half` is settled at load time so this module stays importable without torch; the stubbed tests
    never load real weights.
    """
    if not _state.get("half"):
        return contextlib.nullcontext()
    import torch

    return torch.autocast("cuda", dtype=_state["dtype"])


def _generate_blocking(rows):
    """Denoise every row in one pass. Returns a list of float32 mono arrays."""
    model = _state["model"]
    prompts = [_compose(r) for r in rows]
    # One sampler setting per pass: the first row wins and _check refuses a batch that disagrees.
    kwargs = _gen_kwargs(rows[0])
    with _autocast():
        waveform = model.generate(prompts, **kwargs)
    audio = waveform.float().cpu().numpy()
    if audio.ndim == 1:
        audio = audio[None, :]
    return [np.ascontiguousarray(a) for a in audio]


def _encode(audio, sr, fmt):
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    if fmt == "pcm":
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    container, subtype, _mime = _FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=container, subtype=subtype)
    return buf.getvalue()


def build_app(supports):
    app = FastAPI(title="audio-sound-fx (Dasheng-AudioGen)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="sound_fx", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=OUT_SR)

    # No child engine takes the leftovers here, so a mistyped flag would otherwise vanish silently.
    _args.warn_unclaimed(log)

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    def _headers(fmt):
        return {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "sound_fx",
                "X-Audio-Format": fmt, "X-Audio-Sample-Rate": str(OUT_SR)}

    def _check(row):
        """Validate one body and settle its response format. Returns the format."""
        if not str(row.get("prompt") or row.get("input") or "").strip():
            raise HTTPException(status_code=400,
                                detail="input (a description of the sound) is required; "
                                       "or pass a full tagged prompt as `prompt`")
        fmt = str(row.setdefault("response_format", "wav")).strip().lower()
        row["response_format"] = fmt
        if fmt not in _FORMATS:
            raise HTTPException(status_code=400, detail="response_format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        _gen_kwargs(row)
        # Also composed here so a malformed prompt is a 400 now, not a failed task later.
        _compose(row)
        return fmt

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        """Text description in, audio out. The OpenAI speech shape, with the model's own aspects.

        `input` is the caption; `sfx` / `env` / `music` / `speech` / `asr` are optional and are
        the tags the model itself defines. Descriptive text must be English — the checkpoint's
        text encoder was trained on English captions only, and translating is the caller's job.
        """
        _require_ready()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON in the OpenAI /v1/audio/speech shape")
        fmt = _check(payload)
        payload.setdefault("model", MODEL_NAME)

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="generation")
            with _gen_lock:
                audio = _generate_blocking([payload])[0]
            ctx.meter(output_seconds=seconds(audio, OUT_SR))
            body = _encode(audio, OUT_SR, fmt)
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(body, _FORMATS[fmt][2], suffix="." + fmt, headers=_headers(fmt))

        return await tasks.dispatch(request.query_params.get("async"), "sound_fx", MODEL_NAME,
                                    _work, fail="sound effect generation failed")

    @app.post("/v1/audio/speech/batch")
    async def speech_batch(request: Request):
        """Native batch: every item is denoised in the same pass, not one after another.

        This is the model's own batching, so the ceiling is the card rather than politeness —
        CFG already doubles the batch inside the backbone.
        """
        _require_ready()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON for /v1/audio/speech/batch")
        items = payload.get("items")
        if not isinstance(items, list) or not items or len(items) > MAX_BATCH:
            raise HTTPException(status_code=400,
                                detail="items must be a non-empty array (1–%d)" % MAX_BATCH)
        shared = {k: v for k, v in payload.items() if k != "items"}
        rows = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="items[%d] must be an object" % i)
            row = dict(shared)
            row.update(item)
            _check(row)
            rows.append(row)
        # The sampler schedule is built once for the pass, so the batch has to agree on it.
        first = _gen_kwargs(rows[0])
        for i, row in enumerate(rows[1:], 1):
            if _gen_kwargs(row) != first:
                raise HTTPException(status_code=400,
                                    detail="items[%d] disagrees on num_steps / guidance_scale / "
                                           "sway_sampling_coef; one pass has one schedule, so "
                                           "send differing settings as separate requests" % i)

        def _work(ctx):
            n = len(rows)
            ctx.progress(ratio=0.0, stage="batch", done=0, total=n)
            with _gen_lock:
                audios = _generate_blocking(rows)
            out = []
            for i, (row, audio) in enumerate(zip(rows, audios)):
                fmt = row["response_format"]
                ctx.meter(output_seconds=seconds(audio, OUT_SR))
                out.append({"index": i, "format": fmt, "sample_rate": OUT_SR,
                            "audio": base64.b64encode(_encode(audio, OUT_SR, fmt)).decode("ascii")})
            ctx.progress(ratio=1.0, stage="batch", done=n, total=n)
            return {"model": MODEL_NAME, "items": out}

        return await tasks.dispatch(request.query_params.get("async"), "sound_fx", MODEL_NAME,
                                    _work, fail="sound effect batch failed")

    return app


def run(supports):
    # Ready means loaded AND warmed, so the deadline has to cover both or it kills a healthy boot.
    _runtime.serve(supports, _boot, build_app, "Dasheng-AudioGen engine", timeout_s=BOOT_TIMEOUT_S)
