# Forced alignment (e.g. Qwen3-ForcedAligner etc.): its own checkpoint, hence always its own instance.
import os
import tempfile
import logging
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import hfgate
from .. import tasks
from .. import ovutil
from ..batch import parse_segments
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import probe_seconds, spill, unlink
from ..runtime import Runtime

log = logging.getLogger("audio-align")

_runtime = Runtime(model=None, device="cpu")
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# align() REQUIRES language but tolerates an unknown one: "auto" aligns byte-identically to "en".
DEFAULT_LANGUAGE = "auto"

_args = EngineArgs()
_args.warn_unclaimed(log)

_state = _runtime.state


def _p(msg):
    print("[align] " + msg, flush=True)


def _resolve_hf_dir(repo):
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    kw = {"repo_id": repo, "local_files_only": True}
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    return snapshot_download(**kw)


def _xml_has_input(path, name, limit=1048576):
    try:
        with open(path, "rb") as f:
            head = f.read(limit)
    except OSError:
        return False
    return name.encode("ascii") in head


def _looks_like_ov_ir(path):
    """One-shot align IR from the 0.6B snapshot: encoder+decoder, no beam_idx."""
    if not path or not os.path.isdir(path):
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    if not (os.path.isfile(enc) and os.path.isfile(dec)):
        return False
    return not _xml_has_input(dec, "beam_idx")


def _ov_export_cmd(src, dest):
    # Same 0.6B snapshot as NVIDIA. One thinker forward, not ASR generate:
    # skip -with-past / stateful decoder. Do not remap to *-hf.
    return [
        "optimum-cli", "export", "openvino",
        "--model", src,
        "--task", "automatic-speech-recognition",
        "--disable-stateful",
        "--disable-convert-tokenizer",
        "--weight-format", "fp16",
        "--trust-remote-code",
        dest,
    ]


def _ensure_ov_ir(src):
    nested = os.path.join(src, "openvino")
    if _looks_like_ov_ir(src):
        return src
    if _looks_like_ov_ir(nested):
        return nested
    dest = nested
    if os.path.isdir(dest) and not _looks_like_ov_ir(dest):
        import shutil
        _p("removing unusable export dir %s" % dest)
        shutil.rmtree(dest)
    _p("no OpenVINO align IR in %s; exporting to %s (first start is slow)" % (src, dest))
    os.makedirs(dest, exist_ok=True)
    import subprocess

    cmd = _ov_export_cmd(src, dest)
    _p("running: %s" % " ".join(cmd))
    subprocess.check_call(cmd)
    if not _looks_like_ov_ir(dest):
        raise RuntimeError(
            "align export finished but %s is not a one-shot align IR "
            "(need encoder+decoder xml without beam_idx)"
            % dest
        )
    return dest


def _register_qwen3_asr():
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRProcessor
    from transformers import AutoConfig, AutoProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def _load_ov():
    from optimum.intel import OVModelForSpeechSeq2Seq
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
    from transformers import AutoProcessor

    src = _resolve_hf_dir(MODEL_REPO)
    model_dir = _ensure_ov_ir(src)
    device = ovutil.device()
    _p("loading OpenVINO forced aligner src=%s ir=%s device=%s" % (src, model_dir, device))
    _register_qwen3_asr()
    kw = dict(device=device)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, **kw)
    processor = AutoProcessor.from_pretrained(src, fix_mistral_regex=True)
    cfg = getattr(model, "config", None)
    ts_id = int(getattr(cfg, "timestamp_token_id", 0) or 0)
    ts_seg = float(getattr(cfg, "timestamp_segment_time", 0) or 0)
    if not ts_id or not ts_seg:
        import json

        with open(os.path.join(src, "config.json")) as f:
            raw = json.load(f)
        ts_id = ts_id or int(raw.get("timestamp_token_id") or 0)
        ts_seg = ts_seg or float(raw.get("timestamp_segment_time") or 0)
    _state.update(
        model=model,
        processor=processor,
        aligner_processor=Qwen3ForceAlignProcessor(),
        timestamp_token_id=ts_id,
        timestamp_segment_time=ts_seg,
        device=device,
        backend="openvino",
        ready=True,
    )
    log.info("Qwen3-ForcedAligner %s loaded (openvino %s)", MODEL_REPO, device)


def _load():
    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        kw = dict(dtype=dtype, device_map=(dev if dev == "cpu" else "cuda:0"))
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        try:
            model = Qwen3ForcedAligner.from_pretrained(MODEL_REPO, **kw)
        except TypeError:
            kw.pop("token", None)
            model = Qwen3ForcedAligner.from_pretrained(MODEL_REPO, **kw)
        _state.update(model=model, device=dev, backend="torch", ready=True)
        log.info("Qwen3-ForcedAligner %s loaded on %s", MODEL_REPO, dev)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("forced-aligner load failed: %s", e)


def _field(u, *names):
    for n in names:
        try:
            if isinstance(u, dict):
                if n in u:
                    return u[n]
            elif hasattr(u, n):
                return getattr(u, n)
        except Exception:
            pass
    return None


def _units(res):
    return [{"text": _field(u, "text", "word", "token"),
             "start": _field(u, "start_time", "start"),
             "end": _field(u, "end_time", "end")} for u in (res[0] if res else [])]


def _ov_logits(model, inputs):
    # SpeechSeq2Seq.forward remaps input_features onto input_ids, then **kwargs
    # still carries the text input_ids → "got multiple values for input_ids".
    # thinker(**inputs) hits that. Split encoder / decoder like the official
    # ForcedAligner.forward, still on the 0.6B ASR-task IR.
    payload = dict(inputs)
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    feats = payload.get("input_features")
    if encoder is None or decoder is None or feats is None:
        raise TypeError("align IR has no encoder/decoder/input_features")
    enc = encoder(
        input_features=feats,
        attention_mask=payload.get("input_features_mask"),
    )
    hidden = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
    dec = decoder(
        input_ids=payload.get("input_ids"),
        encoder_hidden_states=hidden,
        attention_mask=payload.get("attention_mask"),
    )
    return dec.logits


def _align_ov(path, text, language):
    import librosa

    wav, _sr = librosa.load(path, sr=16000, mono=True)
    lang = ovutil.language(language)
    word_list, aligner_input = _state["aligner_processor"].encode_timestamp(text, lang)
    inputs = _state["processor"](
        text=[aligner_input],
        audio=[wav],
        return_tensors="pt",
        padding=True,
    )
    logits = _ov_logits(_state["model"], inputs)
    if hasattr(logits, "argmax") and hasattr(logits, "detach"):
        output_ids = logits.argmax(dim=-1)
        input_ids = inputs["input_ids"][0]
        output_id = output_ids[0]
        masked = output_id[input_ids == _state["timestamp_token_id"]]
        timestamp_ms = (masked * _state["timestamp_segment_time"]).detach().cpu().numpy()
    else:
        import numpy as np

        output_ids = np.argmax(np.asarray(logits), axis=-1)
        input_ids = np.asarray(inputs["input_ids"][0])
        output_id = output_ids[0]
        masked = output_id[input_ids == _state["timestamp_token_id"]]
        timestamp_ms = masked * _state["timestamp_segment_time"]
    items = _state["aligner_processor"].parse_timestamp(word_list, timestamp_ms)
    for it in items:
        it["start_time"] = round(it["start_time"] / 1000.0, 3)
        it["end_time"] = round(it["end_time"] / 1000.0, 3)
    return [items]


def _align(path, text, language):
    if _state.get("backend") == "openvino":
        return _align_ov(path, text, language)
    # Older builds of the aligner take positional arguments only.
    try:
        return _state["model"].align(audio=path, text=text, language=language)
    except TypeError:
        return _state["model"].align(path, text, language)


def build_app(supports):
    app = FastAPI(title="audio-align (Qwen3-ForcedAligner)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="align", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/align")
    async def align(file: UploadFile = File(...), text: str = Form(default=None),
                    language: str = Form(default=None), segments: str = Form(default=None),
                    async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        data = await file.read()
        # BATCH mode: `segments` JSON [{start,end,text,[language]}], times slice-relative.
        if segments:
            segs = parse_segments(segments)

            def _decode_all():
                import io as _io
                import soundfile as _sf

                a, sr = _sf.read(_io.BytesIO(data), dtype="float32", always_2d=True)
                return a.mean(axis=1), int(sr)  # -> mono

            try:
                arr, sr = await asyncio.to_thread(_decode_all)
            except Exception as e:
                raise HTTPException(status_code=400, detail="could not decode audio: %s" % e)

            def _work_batch(ctx):
                out = []
                ctx.progress(stage="align", done=0, total=len(segs))
                for i, seg in enumerate(segs, 1):
                    ctx.checkpoint()
                    try:
                        stext = (str(seg.get("text") or "")).strip()
                        if not stext:
                            out.append({"units": [], "language": None})
                            continue
                        lo = max(0, int(float(seg.get("start") or 0) * sr))
                        hi = min(len(arr), int(float(seg.get("end") or 0) * sr))
                        if hi <= lo:
                            out.append({"error": "empty segment"})
                            continue
                        ctx.meter(input_seconds=(hi - lo) / float(sr))
                        lang = ((str(seg.get("language") or language or "")).strip()
                                or DEFAULT_LANGUAGE)
                        import soundfile as _sf

                        p = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                p = f.name
                            _sf.write(p, arr[lo:hi], sr, format="WAV", subtype="PCM_16")
                            res = _align(p, stext, lang)
                        finally:
                            if p:
                                unlink(p)
                        out.append({"language": lang, "units": _units(res)})
                    except tasks.Cancelled:
                        raise
                    except Exception as e:
                        out.append({"error": "align failed: %s" % e})
                    finally:
                        ctx.progress(done=i, total=len(segs))
                return {"model": MODEL_NAME, "mode": "align", "batch": True, "results": out}

            return await tasks.dispatch(async_, "align", MODEL_NAME, _work_batch,
                                        fail="alignment failed")
        # SINGLE mode.
        if not (text or "").strip():
            raise HTTPException(status_code=400, detail="`text` is required for forced alignment")
        path = await asyncio.to_thread(spill, data, file.filename)
        lang = (language or "").strip() or DEFAULT_LANGUAGE

        def _work(ctx):
            ctx.meter(input_seconds=probe_seconds(path))
            ctx.progress(ratio=0.0, stage="align")
            res = _align(path, text, lang)
            ctx.progress(ratio=1.0, stage="done")
            return {"model": MODEL_NAME, "mode": "align", "device": _state["device"],
                    "language": lang, "units": _units(res)}

        return await tasks.dispatch(async_, "align", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="alignment failed")

    return app


def run(supports):
    ov = ovutil.is_ov()

    def load():
        try:
            (_load_ov if ov else _load)()
        except Exception as e:
            _state["error"] = hfgate.explain(MODEL_REPO, e)
            _p("engine load FAILED: %s" % e)
            log.exception("forced-aligner load failed: %s", e)

    _runtime.serve(
        supports,
        load,
        build_app,
        "openvino-genai forced aligner" if ov else "Qwen3-ForcedAligner",
        load_on_main=ov,
        **({"timeout_s": 5400} if ov else {}),
    )
