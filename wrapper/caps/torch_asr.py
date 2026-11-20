# Offline STT for recognisers published as a research repo plus PyTorch weights (e.g.
# FireRedASR2-AED, MiMo-V2.5-ASR etc.), run in this process from the upstream code baked into the
# image.
#
# Which model the weights are is read off the snapshot: FireRed ships model.pth.tar with a
# config.yaml, MiMo a transformers config.json naming MiMoV2ASR. MiMo also needs its audio
# tokenizer, a second repository llm-init downloads as an extra source.
#
# Batch mode decodes the upload once and cuts spans out of memory. FireRed is an
# attention-based encoder-decoder and decodes a padded batch of spans in one call; MiMo
# generates one span at a time.
import importlib.machinery
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import types

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .. import hfgate
from .. import tasks
from ..batch import parse_segments
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-torch-asr")

_runtime = Runtime(model=None, kind=None, device=None, rate=None, max_span=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

KIND_FIRERED = "fireredasr2-aed"
KIND_MIMO = "mimo-v2.5-asr"
# FireRed's encoder was trained on utterances, not long-form audio; MiMo chunks audio itself
# but generates at most 8192 positions per call.
_MAX_SPAN = {KIND_FIRERED: 60.0, KIND_MIMO: 300.0}
_RATE = {KIND_FIRERED: 16000, KIND_MIMO: 24000}

_args = EngineArgs()
BEAM_SIZE = _args.count("--beam-size", 3)
BATCH_SPANS = _args.count("--batch-spans", 8)
_max_span_flag = _args.text("--max-span-seconds", "")
_args.warn_unclaimed(log)
MIMO_TOKENIZER_REPO = os.environ.get("MIMO_TOKENIZER_REPO", "XiaomiMiMo/MiMo-Audio-Tokenizer")
EXTRA_MODEL_PATH_FILE = os.environ.get("EXTRA_MODEL_PATH_FILE", "/run/llm-init/extra_model_path")

_state = _runtime.state
_lock = threading.Lock()


def _snapshot(repo):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def kind_of(path):
    """The model family a weights directory holds, or None."""
    if os.path.isfile(os.path.join(path, "model.pth.tar")) and \
            os.path.isfile(os.path.join(path, "cmvn.ark")):
        return KIND_FIRERED
    try:
        with open(os.path.join(path, "config.json")) as f:
            archs = json.load(f).get("architectures") or []
    except (OSError, ValueError):
        return None
    if any(str(a).startswith("MiMoV2ASR") for a in archs):
        return KIND_MIMO
    return None


def _tokenizer_path():
    try:
        with open(EXTRA_MODEL_PATH_FILE) as f:
            for line in f:
                if line.strip():
                    return line.strip()
    except OSError:
        pass
    return _snapshot(MIMO_TOKENIZER_REPO)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           causal=False, window_size=(-1, -1), **_):
    """flash-attn's packed attention over torch SDPA, for MiMo's audio tokenizer.

    flash-attn has no build for every GPU this image runs on, and the tokenizer only calls this
    one function, with full attention inside each packed sequence.
    """
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("windowed attention")
    import torch.nn.functional as F

    out = q.new_empty(q.shape)
    bounds = cu_seqlens_q.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b] = F.scaled_dot_product_attention(
            q[a:b].transpose(0, 1), k[a:b].transpose(0, 1), v[a:b].transpose(0, 1),
            is_causal=causal).transpose(0, 1)
    return out


def _install_flash_attn_shim():
    if "flash_attn" in sys.modules:
        return
    mod = types.ModuleType("flash_attn")
    # transformers probes optional packages with find_spec, which raises on a spec-less module.
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = mod


def _load():
    try:
        import torch

        path = _snapshot(MODEL_REPO)
        kind = kind_of(path)
        if kind is None:
            raise RuntimeError("%s holds neither FireRedASR2-AED nor MiMo-V2.5-ASR weights" % path)
        cuda = torch.cuda.is_available()
        if kind == KIND_FIRERED:
            from fireredasr2 import FireRedAsr2, FireRedAsr2Config

            model = FireRedAsr2.from_pretrained("aed", path, FireRedAsr2Config(
                use_gpu=cuda, use_half=False, beam_size=BEAM_SIZE, nbest=1, decode_max_len=0,
                softmax_smoothing=1.25, aed_length_penalty=0.6, eos_penalty=1.0))
        else:
            _install_flash_attn_shim()
            from mimo_v25asr.mimo_audio.mimo_audio import MimoAudio

            model = MimoAudio(model_path=path, mimo_audio_tokenizer_path=_tokenizer_path(),
                              device="cuda" if cuda else "cpu")
        max_span = float(_max_span_flag) if _max_span_flag else _MAX_SPAN[kind]
        _state.update(model=model, kind=kind, device="cuda" if cuda else "cpu",
                      rate=_RATE[kind], max_span=max_span, ready=True)
        log.info("%s loaded from %s (max span %.0fs)", kind, path, max_span)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("stt load failed: %s", e)


def _decode(data, filename, rate):
    """The upload as mono float32 samples at `rate`."""
    import numpy as np

    suffix = os.path.splitext(filename or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1",
                            "-ar", str(rate), "-f", "f32le", "pipe:1"], capture_output=True)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        raise HTTPException(status_code=400, detail="cannot decode audio: %s"
                            % (r.stderr or b"")[-300:].decode("utf-8", "replace"))
    return np.frombuffer(r.stdout, dtype=np.float32)


_MIMO_TAG = {"en": "<english>", "english": "<english>",
             "zh": "<chinese>", "chinese": "<chinese>", "mandarin": "<chinese>"}


def plan_batches(lengths, batch_spans, max_samples):
    """Indexes grouped for one padded decode each, longest first so padding stays small."""
    out, cur, cur_max = [], [], 0
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        if cur and (len(cur) >= batch_spans or max(cur_max, lengths[i]) * (len(cur) + 1) > max_samples):
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, lengths[i])
    if cur:
        out.append(cur)
    return out


def _firered(clips):
    import numpy as np

    pcm = [(16000, (np.clip(c, -1, 1) * 32767).astype(np.int16)) for c in clips]
    res = _state["model"].transcribe([str(i) for i in range(len(pcm))], pcm)
    by_id = {r["uttid"]: r.get("text", "") for r in res}
    return [by_id.get(str(i), "") for i in range(len(pcm))]


def _mimo(clip, language):
    import torch

    random.seed(0)
    tag = _MIMO_TAG.get(str(language or "").strip().lower(), "")
    return _state["model"].asr_sft(torch.from_numpy(clip.copy()), audio_tag=tag)


def transcribe_spans(audio, spans, language, ctx=tasks.NULL_CTX):
    """One {text} or {error} per span; a span the model cannot take fails alone."""
    rate, max_span = _state["rate"], _state["max_span"]
    out = [None] * len(spans)
    clips = {}
    for i, s in enumerate(spans):
        try:
            a, b = float(s.get("start") or 0), float(s.get("end") or 0)
        except (TypeError, ValueError, AttributeError):
            out[i] = {"error": "span needs numeric start and end"}
            continue
        if b <= a:
            out[i] = {"text": ""}
        elif b - a > max_span:
            out[i] = {"error": "span is %.1fs; this model takes at most %.0fs" % (b - a, max_span)}
        else:
            clips[i] = audio[int(a * rate):int(b * rate)]
            ctx.meter(input_seconds=b - a)
    todo = list(clips)
    ctx.progress(stage="transcribe", done=0, total=len(spans))
    done = len(spans) - len(todo)
    if _state["kind"] == KIND_FIRERED:
        groups = plan_batches([len(clips[i]) for i in todo], BATCH_SPANS, int(4 * max_span * rate))
        batches = [[todo[j] for j in g] for g in groups]
    else:
        batches = [[i] for i in todo]
    for batch in batches:
        ctx.checkpoint()
        try:
            with _lock:
                if _state["kind"] == KIND_FIRERED:
                    texts = _firered([clips[i] for i in batch])
                else:
                    texts = [_mimo(clips[batch[0]], language)]
            for i, t in zip(batch, texts):
                out[i] = {"text": t.strip()}
        except tasks.Cancelled:
            raise
        except Exception as e:
            log.exception("stt batch failed")
            for i in batch:
                out[i] = {"error": "stt failed: %s" % e}
        done += len(batch)
        ctx.progress(done=done, total=len(spans))
    return out


def _require_ready():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")


def build_app(supports):
    app = FastAPI(title="audio-torch-asr (FireRedASR2-AED / MiMo-V2.5-ASR)")
    mount_metrics(app, nvml_fallback=True)

    register(app, model_name=MODEL_NAME, module="torch_asr", served=supports, repo=MODEL_REPO,
             model_format="pytorch", quantization="bf16",
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form(default=None),
                             language: str = Form(default=None),
                             response_format: str = Form(default="json"),
                             prompt: str = Form(default=None),
                             segments: str = Form(default=None),
                             async_: str = Form(default=None, alias="async")):
        data = await file.read()
        fn = file.filename
        spans = parse_segments(segments) if segments else None

        def _work(ctx):
            _require_ready()
            started = time.monotonic()
            audio = _decode(data, fn, _state["rate"])
            duration = len(audio) / _state["rate"]
            if spans is not None:
                results = transcribe_spans(audio, spans, language, ctx)
                log.info("stt multi-span complete model=%s kind=%s spans=%d failed=%d "
                         "duration_seconds=%.1f", MODEL_NAME, _state["kind"], len(spans),
                         sum(1 for r in results if r.get("error")), time.monotonic() - started)
                return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": results}
            if duration > _state["max_span"]:
                raise HTTPException(status_code=413, detail=(
                    "audio is %.0fs; this model takes at most %.0fs per request, so send segments"
                    % (duration, _state["max_span"])))
            res = transcribe_spans(audio, [{"start": 0, "end": duration}], language, ctx)[0]
            if res.get("error"):
                raise HTTPException(status_code=500, detail=res["error"])
            rf = (response_format or "json").lower()
            if rf == "text":
                return PlainTextResponse(res["text"])
            if rf in ("verbose_json", "verbose"):
                return {"task": "transcribe", "duration": round(duration, 3), "text": res["text"]}
            return {"text": res["text"]}

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work, fail="transcription failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "torch-asr", timeout_s=3600.0)
