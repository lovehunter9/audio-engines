# Whisper STT on OpenVINO GenAI WhisperPipeline (Intel GPU). One snapshot, no second Hub pull.
import logging
import os
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

from .. import ct2_whisper
from .. import hfgate
from .. import ovutil
from .. import tasks
from ..batch import parse_segments
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-whisper-ov")

_runtime = Runtime(model=None, pipeline=None, device=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
_args = EngineArgs()
# Claimed so a leftover --device is not a silent typo. Intel images refuse CPU.
_ = _args.text("--device", "")
BEAM_SIZE = _args.count("--beam-size", 5)
_args.warn_unclaimed(log)
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_LANG = {"english": "en", "chinese": "zh", "mandarin": "zh", "japanese": "ja",
         "korean": "ko", "french": "fr", "german": "de", "spanish": "es",
         "russian": "ru", "italian": "it", "portuguese": "pt", "arabic": "ar"}

_state = _runtime.state


def _require_gpu():
    device = ovutil.device()
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel") and device.upper() != "GPU":
        raise RuntimeError("whisperov on %s must use GPU, got %s" % (mode, device))
    if device.upper() != "GPU":
        raise RuntimeError("whisperov refuses device=%s" % device)
    return device


def _locate():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _prepare_hf(src):
    if ct2_whisper.is_transformers(src) or ct2_whisper.is_whisper_ir(src):
        return src
    if ct2_whisper.is_ct2(src):
        dest = os.path.join(src, "hf-from-ct2")
        return ct2_whisper.to_transformers_dir(src, dest)
    return src


def _ensure_ir(src):
    if ct2_whisper.is_whisper_ir(src):
        return src
    nested = os.path.join(src, "openvino")
    if ct2_whisper.is_whisper_ir(nested):
        return nested
    if os.path.isdir(nested) and not ct2_whisper.is_whisper_ir(nested):
        shutil.rmtree(nested)
    os.makedirs(nested, exist_ok=True)
    cmd = ["optimum-cli", "export", "openvino", "--model", src,
           "--task", "automatic-speech-recognition", nested]
    log.info("exporting Whisper IR: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    if not ct2_whisper.is_whisper_ir(nested):
        raise RuntimeError("optimum export did not write a Whisper IR under %s" % nested)
    return nested


def _load():
    try:
        import openvino_genai as ov_genai

        device = _require_gpu()
        src = _prepare_hf(_locate())
        model_dir = _ensure_ir(src)
        cache = os.path.join(os.environ.get("HF_HOME") or "/tmp", "openvino_cache_whisper")
        os.makedirs(cache, exist_ok=True)
        pipe = ov_genai.WhisperPipeline(model_dir, device, CACHE_DIR=cache)
        _state.update(pipeline=pipe, device=device, ready=True)
        log.info("WhisperPipeline loaded from %s on %s", model_dir, device)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("whisper ov load failed: %s", e)


def _norm_lang(lang):
    if not lang:
        return None
    want = str(lang).strip().lower()
    if not want or want in ("auto", "none", "null", "automatic detection"):
        return None
    if want in _LANG:
        return _LANG[want]
    return want.split("-")[0].split("_")[0]


def _decode(data, filename):
    from ..audioio import decode_mono

    suffix = os.path.splitext(filename or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        wav = decode_mono(path, 16000)
        return wav.reshape(-1)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _generate(audio, task, language):
    pipe = _state["pipeline"]
    kw = {"task": task}
    lang = _norm_lang(language)
    if lang:
        kw["language"] = lang
    if BEAM_SIZE:
        kw["num_beams"] = BEAM_SIZE
    result = pipe.generate(audio, **kw)
    texts = getattr(result, "texts", None)
    if texts:
        return (texts[0] or "").strip()
    t = getattr(result, "text", None)
    if t:
        return str(t).strip()
    return (str(result) if result is not None else "").strip()


def _run(task, data, filename, language, response_format, ctx=tasks.NULL_CTX):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
    audio = _decode(data, filename)
    ctx.meter(input_seconds=len(audio) / 16000.0)
    ctx.progress(ratio=0.0, stage="transcribe")
    text = _generate(audio, task, language)
    ctx.progress(ratio=1.0, stage="done")
    rf = (response_format or "json").lower()
    if rf == "text":
        return PlainTextResponse(text)
    return {"text": text}


def _stt_batch(data, fn, segs, language, ctx=tasks.NULL_CTX):
    audio = _decode(data, fn)
    sr = 16000
    out = []
    ctx.progress(stage="transcribe", done=0, total=len(segs))
    for i, seg in enumerate(segs, 1):
        ctx.checkpoint()
        try:
            a = float(seg.get("start") or 0)
            b = float(seg.get("end") or 0)
            if b <= a:
                out.append({"text": ""})
                continue
            sl = audio[int(a * sr):int(b * sr)]
            ctx.meter(input_seconds=len(sl) / float(sr) if sl.size else (b - a))
            out.append({"text": _generate(sl, "transcribe", language)})
        except tasks.Cancelled:
            raise
        except Exception as e:
            out.append({"error": "stt failed: %s" % e})
        finally:
            ctx.progress(done=i, total=len(segs))
    return out


def build_app(supports):
    app = FastAPI(title="audio-whisper-ov")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="whisper_ov", served=supports, repo=MODEL_REPO,
             model_format="openvino", is_ready=lambda: _state["ready"],
             error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form(default=None),
                             language: str = Form(default=None),
                             response_format: str = Form(default="json"),
                             temperature: str = Form(default=None),
                             prompt: str = Form(default=None),
                             vad_filter: str = Form(default=None),
                             word_timestamps: str = Form(default=None),
                             segments: str = Form(default=None),
                             async_: str = Form(default=None, alias="async")):
        data = await file.read()
        fn = file.filename
        if segments:
            segs = parse_segments(segments)

            def _work_batch(ctx):
                results = _stt_batch(data, fn, segs, language, ctx)
                return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": results}

            return await tasks.dispatch(async_, "stt", MODEL_NAME, _work_batch,
                                        fail="transcription failed")

        def _work(ctx):
            return _run("transcribe", data, fn, language, response_format, ctx)

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                    fail="transcription failed")

    @app.post("/v1/audio/translations")
    async def translations(file: UploadFile = File(...), model: str = Form(default=None),
                           response_format: str = Form(default="json"),
                           temperature: str = Form(default=None),
                           prompt: str = Form(default=None),
                           async_: str = Form(default=None, alias="async")):
        data = await file.read()
        fn = file.filename

        def _work(ctx):
            return _run("translate", data, fn, None, response_format, ctx)

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                    fail="transcription failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "whisper-ov", timeout_s=5400.0)
