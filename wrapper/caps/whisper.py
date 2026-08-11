# Whisper-family STT on CTranslate2: the OpenAI pair, transcriptions + translations (-> English).
import os
import logging
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

from .. import tasks
from ..batch import parse_segments
from ..gpu import mount_metrics
from ..contract import register
from ..runtime import Runtime

log = logging.getLogger("audio-whisper")

_runtime = Runtime(
    "Systran/faster-whisper-large-v3",
    model=None,
    pipeline=None,
    device=None,
    compute=None,
)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
COMPUTE_TYPE = os.environ.get("FW_COMPUTE_TYPE", "float16")
DEVICE = os.environ.get("FW_DEVICE", "auto")
BEAM_SIZE = int(os.environ.get("FW_BEAM_SIZE", "5") or 5)
# WhisperX-style VAD-cut + parallel batched decode; the speedup on a sliced vGPU.
BATCHED = os.environ.get("FW_BATCHED", "1").lower() in ("1", "true", "yes")
BATCH_SIZE = int(os.environ.get("FW_BATCH_SIZE", "16") or 16)
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# So callers can send "English" as well as "en"; anything unknown passes through lower-cased.
_LANG = {"english": "en", "chinese": "zh", "mandarin": "zh", "japanese": "ja",
         "korean": "ko", "french": "fr", "german": "de", "spanish": "es",
         "russian": "ru", "italian": "it", "portuguese": "pt", "arabic": "ar"}

_state = _runtime.state


def _locate():
    # Resolve the model dir from the shared HF cache populated by llm-init.
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _is_ct2(d):
    # CT2 writes model.bin; transformers writes model.safetensors/pytorch_model.bin.
    return os.path.isfile(os.path.join(d, "model.bin"))


# Files faster-whisper reads beside the weights; without them it hits the Hub, offline here.
_CT2_COPY = ("tokenizer.json", "preprocessor_config.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "normalizer.json",
             "vocab.json", "merges.txt")


def _ensure_ct2(src, quantization):
    # CT2 cannot load transformers format (openai/whisper-large-v3), so convert once into the cache.
    if _is_ct2(src):
        return src

    root = os.environ.get("FW_CT2_CACHE") or os.path.join(
        os.environ.get("HF_HUB_CACHE") or "/cache/hf/hub", "ct2-converted")
    out = os.path.join(root, MODEL_REPO.replace("/", "--"))
    if os.path.isfile(os.path.join(out, ".ct2-complete")):
        log.info("using previously converted CT2 model at %s", out)
        return out

    from ctranslate2.converters import TransformersConverter

    log.info("%s is not CTranslate2; converting (this runs once, minutes)", MODEL_REPO)
    # Per-pid staging: two instances of one model can load at once and must not collide.
    tmp = "%s.converting.%d" % (out, os.getpid())
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    copy_files = [f for f in _CT2_COPY if os.path.isfile(os.path.join(src, f))]
    try:
        TransformersConverter(src, copy_files=copy_files,
                              load_as_float16=quantization.startswith("float16")).convert(
            tmp, quantization=quantization, force=True)
    except BaseException:
        # Multiple GB on a shared volume that nothing would reclaim: the next try stages elsewhere.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(out, ignore_errors=True)
    os.rename(tmp, out)
    open(os.path.join(out, ".ct2-complete"), "w").close()
    log.info("converted %s -> %s (%s)", MODEL_REPO, out, quantization)
    return out


def _load():
    try:
        import torch
        from faster_whisper import WhisperModel

        dev = DEVICE
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = COMPUTE_TYPE if dev == "cuda" else "int8"
        path = _ensure_ct2(_locate(), ctype)
        model = WhisperModel(path, device=dev, compute_type=ctype)
        pipeline = None
        if BATCHED:
            try:
                from faster_whisper import BatchedInferencePipeline

                pipeline = BatchedInferencePipeline(model=model)
                log.info("BatchedInferencePipeline enabled (batch_size=%d)", BATCH_SIZE)
            except Exception as e:
                log.warning("BatchedInferencePipeline unavailable (%s); using buffered", e)
        _state.update(model=model, pipeline=pipeline, device=dev, compute=ctype, ready=True)
        log.info("faster-whisper loaded from %s on %s (%s)", path, dev, ctype)
    except Exception as e:
        _state["error"] = str(e)
        log.exception("stt load failed: %s", e)


def _norm_lang(lang):
    if not lang:
        return None
    l = str(lang).strip().lower()
    if not l or l in ("auto", "none", "null", "automatic detection"):
        return None
    if l in _LANG:
        return _LANG[l]
    return l.split("-")[0].split("_")[0]  # en-US/zh_CN -> en/zh


def _parse_temp(raw):
    if raw is None or str(raw).strip() == "":
        return 0.0
    parts = [p for p in str(raw).replace(" ", "").split(",") if p != ""]
    try:
        vals = [float(p) for p in parts]
    except Exception:
        return 0.0
    return vals[0] if len(vals) == 1 else tuple(vals)


def _realize(segments, info, ctx):
    # Iterating the generator is what runs inference, so progress and cancel belong here.
    dur = float(getattr(info, "duration", 0.0) or 0.0)
    out = []
    ctx.progress(ratio=0.0, stage="transcribe")
    for s in segments:
        out.append(s)
        if dur > 0:
            ctx.progress(ratio=min(1.0, float(s.end) / dur))
        ctx.checkpoint()
    return out


def _run(task, data, filename, language, response_format, temperature, prompt,
         vad_filter, word_ts, ctx=tasks.NULL_CTX):
    # Blocking, so callers hand it to the task runner; CTranslate2 needs no lock of its own.
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
    suffix = os.path.splitext(filename or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        kw = {
            "task": task,
            "beam_size": BEAM_SIZE,
            "language": _norm_lang(language),
            "word_timestamps": bool(word_ts),
        }
        if prompt:
            kw["initial_prompt"] = prompt
        temp = _parse_temp(temperature)
        pipe = _state.get("pipeline")
        if pipe is not None:
            # The batched path owns its VAD and needs a scalar temperature.
            bkw = dict(kw)
            bkw["temperature"] = temp[0] if isinstance(temp, tuple) else temp
            bkw["batch_size"] = BATCH_SIZE
            bkw["vad_filter"] = True
            try:
                segments, info = pipe.transcribe(path, **bkw)
                segs = _realize(segments, info, ctx)
            except TypeError as te:
                log.warning("batched rejected kwargs (%s); buffered fallback", te)
                kw["temperature"] = temp
                if vad_filter:
                    kw["vad_filter"] = True
                segments, info = _state["model"].transcribe(path, **kw)
                segs = _realize(segments, info, ctx)
        else:
            kw["temperature"] = temp
            if vad_filter:
                kw["vad_filter"] = True
            segments, info = _state["model"].transcribe(path, **kw)
            segs = _realize(segments, info, ctx)
    except tasks.Cancelled:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="transcription failed: %s" % e)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    full = "".join(s.text for s in segs).strip()
    rf = (response_format or "json").lower()
    if rf == "text":
        return PlainTextResponse(full)
    if rf in ("verbose_json", "verbose"):
        out_segs = []
        for s in segs:
            d = {"id": getattr(s, "id", 0), "seek": getattr(s, "seek", 0),
                 "start": round(float(s.start), 3), "end": round(float(s.end), 3),
                 "text": s.text,
                 "avg_logprob": getattr(s, "avg_logprob", None),
                 "compression_ratio": getattr(s, "compression_ratio", None),
                 "no_speech_prob": getattr(s, "no_speech_prob", None)}
            if word_ts and getattr(s, "words", None):
                d["words"] = [{"word": w.word, "start": round(float(w.start), 3),
                               "end": round(float(w.end), 3),
                               "probability": getattr(w, "probability", None)} for w in s.words]
            out_segs.append(d)
        return {"task": task, "language": getattr(info, "language", None),
                "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
                "text": full, "segments": out_segs}
    return {"text": full}


def _ffmpeg_slice_wav(src, start, dur):
    # To 16k mono WAV bytes on ffmpeg's stdout, since this image has no soundfile.
    r = subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-ss", "%.3f" % start, "-i", src,
         "-t", "%.3f" % dur, "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg slice rc=%d: %s" % (r.returncode, (r.stderr or b"")[-300:]))
    return r.stdout


def _stt_batch(data, fn, segs, language, temperature, prompt, ctx=tasks.NULL_CTX):
    # One {text}|{error} per segment, so a single bad segment cannot fail the batch.
    suffix = os.path.splitext(fn or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as wf:
        wf.write(data)
        whole = wf.name
    out = []
    try:
        ctx.progress(stage="transcribe", done=0, total=len(segs))
        for i, seg in enumerate(segs, 1):
            ctx.checkpoint()
            try:
                a = float(seg.get("start") or 0)
                b = float(seg.get("end") or 0)
                if b <= a:
                    out.append({"text": ""})
                    continue
                sb = _ffmpeg_slice_wav(whole, a, b - a)
                res = _run("transcribe", sb, "seg.wav", language, "json",
                           temperature, prompt, False, False)
                out.append({"text": res.get("text", "") if isinstance(res, dict) else ""})
            except tasks.Cancelled:
                raise
            except Exception as e:
                out.append({"error": "stt failed: %s" % e})
            finally:
                ctx.progress(done=i, total=len(segs))
    finally:
        try:
            os.unlink(whole)
        except Exception:
            pass
    return out


def build_app(supports):
    app = FastAPI(title="audio-whisper (faster-whisper)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="whisper", served=supports, repo=MODEL_REPO,
             model_format="ctranslate2", quantization=COMPUTE_TYPE,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

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
        # BATCH mode (opt-in): `segments` = JSON [{start,end}] -> one {text}|{error} each.
        if segments:
            segs = parse_segments(segments)

            def _work_batch(ctx):
                results = _stt_batch(data, fn, segs, language, temperature, prompt, ctx)
                return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": results}

            return await tasks.dispatch(async_, "stt", MODEL_NAME, _work_batch,
                                        fail="transcription failed")

        def _work(ctx):
            return _run("transcribe", data, fn, language, response_format, temperature, prompt,
                        str(vad_filter).lower() in ("1", "true", "yes"),
                        str(word_timestamps).lower() in ("1", "true", "yes"), ctx)

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                    fail="transcription failed")

    @app.post("/v1/audio/translations")
    async def translations(file: UploadFile = File(...), model: str = Form(default=None),
                           response_format: str = Form(default="json"),
                           temperature: str = Form(default=None),
                           prompt: str = Form(default=None),
                           async_: str = Form(default=None, alias="async")):
        # OpenAI semantics: translations = speech -> English (Whisper "translate" task).
        data = await file.read()
        fn = file.filename

        def _work(ctx):
            return _run("translate", data, fn, None, response_format, temperature, prompt,
                        False, False, ctx)

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                    fail="transcription failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "faster-whisper")
