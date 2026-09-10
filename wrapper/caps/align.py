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


def _ov_hub_id(repo):
    """OpenVINO export uses the HuggingFace-native forced-aligner checkpoint."""
    repo = (repo or "").strip().rstrip("/")
    if repo.endswith("-hf"):
        return repo
    if "ForcedAligner" in repo:
        return repo + "-hf"
    return repo


def _looks_like_ov_aligner_ir(path):
    if not path or not os.path.isdir(path):
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    return os.path.isfile(enc) and os.path.isfile(dec)


def _ensure_ov_aligner(hub_id, device):
    src = _resolve_hf_dir(hub_id)
    ov_dir = os.path.join(src, "openvino")
    if _looks_like_ov_aligner_ir(ov_dir):
        return ov_dir, src

    if os.path.isdir(ov_dir) and not _looks_like_ov_aligner_ir(ov_dir):
        import shutil
        _p("removing unusable export dir %s" % ov_dir)
        shutil.rmtree(ov_dir)

    _p("no OpenVINO IR in %s; exporting forced aligner (first start is slow)" % src)
    os.makedirs(ov_dir, exist_ok=True)
    offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from optimum.intel import OVModelForQwen3ASRForcedAligner

        kw = dict(export=True, device=device)
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        model = OVModelForQwen3ASRForcedAligner.from_pretrained(hub_id, **kw)
        model.save_pretrained(ov_dir)
        del model
    finally:
        if offline is not None:
            os.environ["HF_HUB_OFFLINE"] = offline

    if not _looks_like_ov_aligner_ir(ov_dir):
        raise RuntimeError(
            "optimum export finished but %s is not a forced-aligner OpenVINO IR" % ov_dir
        )
    return ov_dir, src


def _load_ov():
    from optimum.intel import OVModelForQwen3ASRForcedAligner
    from transformers import AutoProcessor

    hub_id = _ov_hub_id(MODEL_REPO)
    device = ovutil.device()
    _p("loading OpenVINO forced aligner hub=%s device=%s" % (hub_id, device))
    ov_dir, proc_dir = _ensure_ov_aligner(hub_id, device)
    kw = dict(device=device)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    model = OVModelForQwen3ASRForcedAligner.from_pretrained(ov_dir, **kw)
    processor = AutoProcessor.from_pretrained(proc_dir)
    _state.update(
        model=model,
        processor=processor,
        device=device,
        backend="openvino",
        ready=True,
    )
    log.info("Qwen3-ForcedAligner %s loaded (openvino %s)", hub_id, device)


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


def _align_ov(path, text, language):
    import librosa

    wav, sr = librosa.load(path, sr=16000, mono=True)
    lang = ovutil.language(language)
    processor = _state["processor"]
    model = _state["model"]
    aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
        audio=wav,
        transcript=text,
        language=lang,
    )
    outputs = model(**aligner_inputs)
    ts = processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=aligner_inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=model.config.timestamp_token_id,
    )
    return [ts[0] if ts else []]


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
