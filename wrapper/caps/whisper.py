# Whisper-family offline STT on CTranslate2 (faster-whisper). Ported from the
# tested stt_fw.py; deps baked at build time; contract surface via wrapper.gpu +
# wrapper.contract. Serves the OpenAI STT pair: transcriptions (same-language)
# and translations (speech -> English, Whisper's native "translate" task).
# Preferred over vLLM-Whisper on a time-sliced vGPU, where it is ~10x faster.
import os
import asyncio
import logging
import subprocess
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from ..gpu import mount_metrics
from ..contract import register

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-whisper")

MODEL_NAME = os.environ.get("MODEL_NAME", "Systran/faster-whisper-large-v3")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
COMPUTE_TYPE = os.environ.get("FW_COMPUTE_TYPE", "float16")
DEVICE = os.environ.get("FW_DEVICE", "auto")
BEAM_SIZE = int(os.environ.get("FW_BEAM_SIZE", "5") or 5)
# The speedup: WhisperX-style BatchedInferencePipeline (VAD-cut + parallel
# batched decode), far faster than sequential and friendlier to a sliced vGPU.
BATCHED = os.environ.get("FW_BATCHED", "1").lower() in ("1", "true", "yes")
BATCH_SIZE = int(os.environ.get("FW_BATCH_SIZE", "16") or 16)
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Common full-name -> ISO 639-1 so callers can send "English"/"Chinese" and not
# just "en"/"zh"; unknown values pass through lower-cased.
_LANG = {"english": "en", "chinese": "zh", "mandarin": "zh", "japanese": "ja",
         "korean": "ko", "french": "fr", "german": "de", "spanish": "es",
         "russian": "ru", "italian": "it", "portuguese": "pt", "arabic": "ar"}

_state = {"ready": False, "error": None, "model": None, "pipeline": None,
          "device": None, "compute": None}


def _locate():
    # Resolve the CT2 model dir from the shared HF cache populated by llm-init.
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _load():
    try:
        import torch
        from faster_whisper import WhisperModel

        dev = DEVICE
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        ctype = COMPUTE_TYPE if dev == "cuda" else "int8"
        path = _locate()
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


def _run(task, data, filename, language, response_format, temperature, prompt,
         vad_filter, word_ts):
    # Blocking; callers hand it to asyncio.to_thread so the event loop stays free
    # for the concurrent STT fan-out. No lock: CTranslate2 is thread-safe and
    # parallel decode is the whole point.
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
            # Batched fast path (VAD-cut + parallel decode): owns its VAD and
            # needs a scalar temperature; falls back to buffered if this
            # faster_whisper build rejects a kwarg.
            bkw = dict(kw)
            bkw["temperature"] = temp[0] if isinstance(temp, tuple) else temp
            bkw["batch_size"] = BATCH_SIZE
            bkw["vad_filter"] = True
            try:
                segments, info = pipe.transcribe(path, **bkw)
                segs = list(segments)
            except TypeError as te:
                log.warning("batched rejected kwargs (%s); buffered fallback", te)
                kw["temperature"] = temp
                if vad_filter:
                    kw["vad_filter"] = True
                segments, info = _state["model"].transcribe(path, **kw)
                segs = list(segments)
        else:
            kw["temperature"] = temp
            if vad_filter:
                kw["vad_filter"] = True
            segments, info = _state["model"].transcribe(path, **kw)
            segs = list(segments)  # realize the generator (runs inference)
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
    # Cut [start, start+dur] out of `src` to 16k mono WAV bytes via ffmpeg stdout
    # (no soundfile in this image).
    r = subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-ss", "%.3f" % start, "-i", src,
         "-t", "%.3f" % dur, "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg slice rc=%d: %s" % (r.returncode, (r.stderr or b"")[-300:]))
    return r.stdout


def _stt_batch(data, fn, segs, language, temperature, prompt):
    # Write the clip once, ffmpeg-slice each segment, transcribe each via _run
    # (one {text}|{error} per segment, per-item try/except so one bad segment
    # can't fail the batch).
    suffix = os.path.splitext(fn or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as wf:
        wf.write(data)
        whole = wf.name
    out = []
    try:
        for seg in segs:
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
            except Exception as e:
                out.append({"error": "stt failed: %s" % e})
    finally:
        try:
            os.unlink(whole)
        except Exception:
            pass
    return out


def build_app(supports):
    app = FastAPI(title="audio-whisper (faster-whisper)")
    mount_metrics(app)

    endpoints = [
        {"method": "POST", "path": "/v1/audio/transcriptions",
         "description": "Offline transcription (single / batch segments)"},
        {"method": "POST", "path": "/v1/audio/translations",
         "description": "Speech -> English (Whisper translate task)"},
    ]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form(default=None),
                             language: str = Form(default=None),
                             response_format: str = Form(default="json"),
                             temperature: str = Form(default=None),
                             prompt: str = Form(default=None),
                             vad_filter: str = Form(default=None),
                             word_timestamps: str = Form(default=None),
                             segments: str = Form(default=None)):
        data = await file.read()
        fn = file.filename
        # BATCH mode (opt-in): `segments` = JSON [{start,end}] -> one {text}|{error} each.
        if segments:
            import json as _json

            try:
                _segs = _json.loads(segments)
            except Exception as e:
                raise HTTPException(status_code=400, detail="invalid `segments` json: %s" % e)
            if not isinstance(_segs, list):
                raise HTTPException(status_code=400, detail="`segments` must be a JSON array")
            results = await asyncio.to_thread(_stt_batch, data, fn, _segs, language,
                                              temperature, prompt)
            return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": results}
        return await asyncio.to_thread(
            _run, "transcribe", data, fn, language, response_format, temperature, prompt,
            str(vad_filter).lower() in ("1", "true", "yes"),
            str(word_timestamps).lower() in ("1", "true", "yes"))

    @app.post("/v1/audio/translations")
    async def translations(file: UploadFile = File(...), model: str = Form(default=None),
                           response_format: str = Form(default="json"),
                           temperature: str = Form(default=None),
                           prompt: str = Form(default=None)):
        # OpenAI semantics: translations = speech -> English (Whisper "translate" task).
        data = await file.read()
        fn = file.filename
        return await asyncio.to_thread(
            _run, "translate", data, fn, None, response_format, temperature, prompt,
            False, False)

    return app


def run(supports):
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
