# Speaker-attributed transcription in one pass (e.g. MOSS-Transcribe-Diarize, VibeVoice-ASR etc.).
#
# These models hear a whole recording and write who spoke, when and what in one generation, so
# they serve `stt` without `segments`: cutting the audio first would throw away the context the
# speaker labels come from. `response_format=diarized_json` (or verbose_json) returns the parsed
# segments; json and text return the words only. Both run in this process with transformers.
#
# Which model the weights are is read off config.json's model_type, not the chart: the two
# share nothing past the HTTP shape, and a chart naming the wrong one would fail at load anyway.
import json
import logging
import os
import re
import tempfile
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .. import hfgate
from .. import tasks
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-diar-asr")

_runtime = Runtime(model=None, processor=None, kind=None, device=None, dtype=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
# A 15-minute meeting is 6-10k tokens of transcript with timestamps; the checkpoints' own
# defaults (MOSS ships 5120) truncate it in silence.
MAX_NEW_TOKENS = _args.count("--max-new-tokens", 32768)
DTYPE = str(_args.text("--dtype", "auto") or "auto").strip().lower()
_args.warn_unclaimed(log)

KIND_MOSS = "moss"
KIND_VIBEVOICE = "vibevoice"
_KINDS = {"moss_transcribe_diarize": KIND_MOSS, "vibevoice_asr": KIND_VIBEVOICE}

MOSS_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

_state = _runtime.state
_gen_lock = threading.Lock()


def _model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def kind_of(config):
    """The model family a config.json describes, or None."""
    return _KINDS.get(str(config.get("model_type") or "").lower())


def _load():
    try:
        import torch

        path = _model_path()
        with open(os.path.join(path, "config.json")) as f:
            kind = kind_of(json.load(f))
        if kind is None:
            raise RuntimeError("config.json model_type is neither %s" % " nor ".join(_KINDS))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        named = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp16": torch.float16, "float16": torch.float16, "fp32": torch.float32}
        dtype = named.get(DTYPE) or (torch.bfloat16 if device == "cuda" else torch.float32)
        if kind == KIND_MOSS:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa")
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        else:
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

            model = VibeVoiceAsrForConditionalGeneration.from_pretrained(path, dtype=dtype)
            processor = AutoProcessor.from_pretrained(path)
        model = model.to(device).eval()
        _state.update(model=model, processor=processor, kind=kind, device=device,
                      dtype=str(dtype).replace("torch.", ""), ready=True)
        log.info("%s loaded from %s on %s (%s)", kind, path, device, dtype)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("diarized stt load failed: %s", e)


_MOSS_SEGMENT = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.S)


def parse_moss(text):
    """Segments of MOSS's `[start][Sxx]text[end]` transcript; anything unparseable is dropped."""
    out = []
    for m in _MOSS_SEGMENT.finditer(text or ""):
        words = m.group(3).strip()
        if words:
            out.append({"speaker": m.group(2), "start": float(m.group(1)),
                        "end": float(m.group(4)), "text": words})
    return out


_VV_OBJECT = re.compile(r"\{[^{}]*\}", re.S)


def parse_vibevoice(text):
    """Segments of VibeVoice's JSON list of {Start, End, Speaker, Content}.

    The list is generated, not serialized, so a long recording can end mid-object when the token
    budget runs out; every complete object before that point is kept.
    """
    raw = (text or "").strip()
    try:
        items = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
    except ValueError:
        items = []
        for m in _VV_OBJECT.finditer(raw):
            try:
                items.append(json.loads(m.group(0)))
            except ValueError:
                continue
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        words = str(it.get("Content") or "").strip()
        if not words:
            continue
        try:
            start, end = float(it.get("Start")), float(it.get("End"))
        except (TypeError, ValueError):
            continue
        spk = str(it.get("Speaker", ""))
        speaker = "S%02d" % (int(spk) + 1) if spk.isdigit() else spk
        out.append({"speaker": speaker, "start": start, "end": end, "text": words})
    return out


def joined_text(segments):
    """The words of the segments in time order, without speakers."""
    parts = []
    for s in sorted(segments, key=lambda s: s["start"]):
        if parts and not _cjk(parts[-1][-1]) and not _cjk(s["text"][0]):
            parts.append(" ")
        parts.append(s["text"])
    return "".join(parts)


def _cjk(ch):
    return "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af"


def _generate_moss(path, prompt):
    import librosa
    import torch

    model, processor = _state["model"], _state["processor"]
    instruction = MOSS_PROMPT + ("热词提示：" + prompt if prompt else "")
    messages = [{"role": "user", "content": [{"type": "audio", "audio": path},
                                             {"type": "text", "text": instruction}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audio, _ = librosa.load(path, sr=processor.feature_extractor.sampling_rate, mono=True)
    kw = {"device": _state["device"]} if _state["device"] == "cuda" else {}
    inputs = processor(text=rendered, audio=[audio], max_length=131072, audio_kwargs=kw,
                       return_tensors="pt").to(_state["device"])
    prompt_len = int(inputs["attention_mask"][0].sum().item())
    with torch.inference_mode():
        out = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    ids = out[0][prompt_len:]
    raw = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
    return raw, parse_moss(raw), int(ids.numel()), len(audio) / processor.feature_extractor.sampling_rate


def _generate_vibevoice(path, prompt):
    import librosa
    import torch

    model, processor = _state["model"], _state["processor"]
    sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 24000)
    audio, _ = librosa.load(path, sr=sr, mono=True)
    inputs = processor.apply_transcription_request(audio=audio, prompt=prompt or None)
    inputs = inputs.to(_state["device"], model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    ids = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.decode(ids)[0]
    return raw, parse_vibevoice(raw), int(ids.shape[1]), len(audio) / sr


def _run(data, filename, prompt, response_format, ctx=tasks.NULL_CTX):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
    suffix = os.path.splitext(filename or "a.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    started = time.monotonic()
    try:
        ctx.progress(ratio=0.0, stage="transcribe")
        gen = _generate_moss if _state["kind"] == KIND_MOSS else _generate_vibevoice
        with _gen_lock:
            raw, segments, tokens, duration = gen(path, prompt)
    except tasks.Cancelled:
        raise
    except Exception as e:
        log.exception("diarized transcription failed")
        raise HTTPException(status_code=500, detail="transcription failed: %s" % e)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    ctx.meter(input_seconds=duration)
    truncated = tokens >= MAX_NEW_TOKENS
    log.info("diarized stt complete model=%s kind=%s audio_seconds=%.1f tokens=%d segments=%d "
             "speakers=%d truncated=%s duration_seconds=%.1f", MODEL_NAME, _state["kind"], duration,
             tokens, len(segments), len({s["speaker"] for s in segments}), truncated,
             time.monotonic() - started)
    text = joined_text(segments) if segments else raw
    rf = (response_format or "json").lower()
    if rf == "text":
        return PlainTextResponse(text)
    if rf in ("diarized_json", "verbose_json", "verbose"):
        return {"task": "transcribe", "duration": round(duration, 3), "text": text,
                "segments": segments, "generated_tokens": tokens, "truncated": truncated}
    return {"text": text}


def build_app(supports):
    app = FastAPI(title="audio-diar-asr (speaker-attributed transcription)")
    mount_metrics(app, nvml_fallback=True)

    register(app, model_name=MODEL_NAME, module="diar_asr", served=supports, repo=MODEL_REPO,
             model_format="safetensors", quantization=DTYPE,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form(default=None),
                             language: str = Form(default=None),
                             response_format: str = Form(default="json"),
                             prompt: str = Form(default=None),
                             segments: str = Form(default=None),
                             async_: str = Form(default=None, alias="async")):
        if segments:
            raise HTTPException(status_code=400, detail=(
                "this model transcribes whole recordings with speakers; send no segments"))
        data = await file.read()
        fn = file.filename

        def _work(ctx):
            return _run(data, fn, prompt, response_format, ctx)

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work, fail="transcription failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "diarized-asr", timeout_s=3600.0)
