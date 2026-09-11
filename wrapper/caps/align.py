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


def _ov_align_repo(repo=None):
    """OVModelForQwen3ASRForcedAligner loads Qwen3ASRForTokenClassification.

    That is the HuggingFace-native *-hf checkpoint. The qwen_asr package
    snapshot (0.6B, no suffix) is Qwen3ASRForConditionalGeneration; exporting
    it as token-classification dies on Unrecognized configuration class.
    """
    repo = (repo if repo is not None else MODEL_REPO) or ""
    repo = repo.strip()
    if not repo or repo.endswith("-hf"):
        return repo
    return repo + "-hf"


def _resolve_ov_src():
    hub = _ov_align_repo()
    try:
        return _resolve_hf_dir(hub)
    except Exception as e:
        raise RuntimeError(
            "OpenVINO align needs the HuggingFace-native checkpoint %s in the "
            "HF cache (OVModelForQwen3ASRForcedAligner / "
            "Qwen3ASRForTokenClassification). The qwen_asr snapshot %s cannot "
            "be exported as token-classification. Add hf://%s to MODEL_SOURCE "
            "so llm-init downloads it."
            % (hub, MODEL_REPO, hub)
        ) from e


def _xml_has_input(path, name, limit=1048576):
    try:
        with open(path, "rb") as f:
            head = f.read(limit)
    except OSError:
        return False
    return name.encode("ascii") in head


# Written after a token-classification export. The previous ASR-task IR also
# has encoder+decoder xml without beam_idx, so filenames alone would keep it.
_ALIGN_IR_MARKER = "align.task"
_ALIGN_IR_TASK = "token-classification"


def _looks_like_ov_ir(path):
    """Forced-aligner IR: token-classification, encoder+decoder, no beam_idx."""
    if not path or not os.path.isdir(path):
        return False
    try:
        with open(os.path.join(path, _ALIGN_IR_MARKER)) as f:
            task = f.read().strip()
    except OSError:
        return False
    if task != _ALIGN_IR_TASK:
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    single = os.path.join(path, "openvino_model.xml")
    if os.path.isfile(single):
        return True
    if not (os.path.isfile(enc) and os.path.isfile(dec)):
        return False
    return not _xml_has_input(dec, "beam_idx")


def _write_align_ir_marker(dest):
    with open(os.path.join(dest, _ALIGN_IR_MARKER), "w") as f:
        f.write(_ALIGN_IR_TASK + "\n")


def _ov_export_cmd(src, dest):
    # Official OVModelForQwen3ASRForcedAligner._export: token-classification,
    # no KV cache. ASR-task export built a seq2seq decoder; thinker(**inputs)
    # then passed input_ids twice into OVModelForSeq2SeqLM.forward.
    return [
        "optimum-cli", "export", "openvino",
        "--model", src,
        "--task", "token-classification",
        "--disable-stateful",
        "--disable-convert-tokenizer",
        "--weight-format", "fp16",
        "--trust-remote-code",
        dest,
    ]


def _export_align_ir(src, dest):
    from optimum.intel import OVModelForQwen3ASRForcedAligner

    kw = dict(export=True, device="CPU")
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    ov = OVModelForQwen3ASRForcedAligner.from_pretrained(src, **kw)
    ov.save_pretrained(dest)
    return dest


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
    try:
        _export_align_ir(src, dest)
    except Exception as e:
        _p("class export failed (%s); falling back to optimum-cli" % e)
        import subprocess

        cmd = _ov_export_cmd(src, dest)
        _p("running: %s" % " ".join(cmd))
        subprocess.check_call(cmd)
    _write_align_ir_marker(dest)
    if not _looks_like_ov_ir(dest):
        raise RuntimeError(
            "align export finished but %s is not a token-classification IR "
            "(need align.task marker and encoder/decoder or openvino_model.xml)"
            % dest
        )
    return dest


def _load_ov():
    from optimum.intel import OVModelForQwen3ASRForcedAligner
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
    from transformers import AutoProcessor

    # Do not register qwen_asr's Qwen3ASRConfig over transformers' native
    # qwen3_asr: the *-hf IR and AutoProcessor need the HF-native classes.
    src = _resolve_ov_src()
    model_dir = _ensure_ov_ir(src)
    device = ovutil.device()
    _p("loading OpenVINO forced aligner src=%s ir=%s device=%s" % (src, model_dir, device))
    kw = dict(device=device)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    model = OVModelForQwen3ASRForcedAligner.from_pretrained(model_dir, **kw)
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
    # ForcedAligner.forward takes input_ids + input_features. Do not call
    # thinker / SpeechSeq2Seq: that remaps input_features onto input_ids and
    # then **kwargs still carries the text input_ids.
    payload = dict(inputs) if not isinstance(inputs, dict) else inputs
    return model(**payload).logits


def _align_ov(path, text, language):
    import librosa

    wav, _sr = librosa.load(path, sr=16000, mono=True)
    lang = ovutil.language(language)
    processor = _state["processor"]
    model = _state["model"]
    prepare = getattr(processor, "prepare_forced_aligner_inputs", None)
    decode = getattr(processor, "decode_forced_alignment", None)
    if prepare is not None and decode is not None:
        inputs, word_lists = prepare(audio=wav, transcript=text, language=lang)
        outputs = model(**dict(inputs))
        items = decode(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=_state["timestamp_token_id"],
            timestamp_segment_time=_state["timestamp_segment_time"],
        )[0]
        return [items]
    word_list, aligner_input = _state["aligner_processor"].encode_timestamp(text, lang)
    inputs = processor(
        text=[aligner_input],
        audio=[wav],
        return_tensors="pt",
        padding=True,
    )
    logits = _ov_logits(model, inputs)
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
