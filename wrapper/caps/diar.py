# Speaker diarization with pyannote.audio.
import asyncio
import os
import logging
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics, quota_mib
from ..contract import register, EngineArgs
from ..audioio import decode, spill, unlink
from ..runtime import Runtime

log = logging.getLogger("audio-diar")

_runtime = Runtime("pyannote-community-1", pipeline=None, device="cpu", batch1=False)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Both pyannote stages default to batch_size=1: ~12 000 launches of 10 s of audio for a 3 h clip.
_args = EngineArgs()
SEG_BATCH = _args.text("--segmentation-batch-size", "auto")  # "auto" = GPU-sized on CUDA, 1 on CPU
EMB_BATCH = _args.text("--embedding-batch-size", "auto")
_args.warn_unclaimed(log)

_state = _runtime.state


def _auto_batch():
    """A batch big enough to be worth a launch, but sized to the slice we were actually given."""
    mib = quota_mib()
    for floor, batch in ((16000, 32), (8000, 16), (4000, 8)):
        if mib >= floor:
            return batch
    return 4


def _batch(raw, cuda):
    raw = (raw or "auto").strip().lower()
    if raw in ("", "auto"):
        return _auto_batch() if cuda else 1
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("ignoring unparsable batch size %r", raw)
        return 1


def _set_batches(pipe, resolve):
    # Probe rather than assume: these properties have been renamed across pyannote versions.
    for attr, raw in (("segmentation_batch_size", SEG_BATCH),
                      ("embedding_batch_size", EMB_BATCH)):
        if not hasattr(pipe, attr):
            log.warning("%s absent on %s — leaving pyannote's default",
                        attr, type(pipe).__name__)
            continue
        want = resolve(raw)
        try:
            setattr(pipe, attr, want)
        except Exception as e:
            log.warning("could not set %s=%s: %s", attr, want, e)
            continue
        log.info("%s = %s", attr, getattr(pipe, attr, "?"))


def _load():
    try:
        import torch
        from pyannote.audio import Pipeline

        # pyannote.audio 4 uses token=; older used use_auth_token=.
        try:
            pipe = Pipeline.from_pretrained(MODEL_REPO, token=HF_TOKEN)
        except TypeError:
            pipe = Pipeline.from_pretrained(MODEL_REPO, use_auth_token=HF_TOKEN)
        cuda = torch.cuda.is_available()
        dev = "cuda" if cuda else "cpu"
        pipe.to(torch.device(dev))
        _set_batches(pipe, lambda raw: _batch(raw, cuda))
        _state["pipeline"], _state["device"], _state["ready"] = pipe, dev, True
        log.info("pyannote pipeline %s loaded on %s", MODEL_REPO, dev)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("pipeline load failed: %s", e)


def _oom(e):
    return isinstance(e, (RuntimeError, MemoryError)) and "out of memory" in str(e).lower()


def _hook(ctx):
    # pyannote reports (step, artifact, file=, total=, completed=) as each stage advances.
    def hook(step, _artifact=None, file=None, total=None, completed=None):
        ctx.checkpoint()
        ctx.progress(stage=step, done=completed, total=total)

    return hook


def _call(pipe, waveform, sr, kw, hook):
    # hook= is not in every pyannote build, and it is only a progress nicety.
    try:
        return pipe({"waveform": waveform, "sample_rate": sr}, hook=hook, **kw)
    except TypeError as e:
        log.warning("this pyannote build takes no hook= (%s); running without progress", e)
        return pipe({"waveform": waveform, "sample_rate": sr}, **kw)


def _infer(waveform, sr, kw, hook):
    pipe = _state["pipeline"]
    try:
        return _call(pipe, waveform, sr, kw, hook)
    except Exception as e:
        # Slow beats a 500; remembered for the process so one oversized clip can't retry forever.
        if not _oom(e) or _state["batch1"]:
            raise
        log.warning("out of memory at the configured batch size, dropping to 1: %s", e)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        _set_batches(pipe, lambda _raw: 1)
        _state["batch1"] = True
        return _call(pipe, waveform, sr, kw, hook)


def build_app(supports):
    app = FastAPI(title="audio-diarization (pyannote)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="diar", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/diarization")
    async def diarize(file: UploadFile = File(...), num_speakers: str = Form(default=None),
                      min_speakers: str = Form(default=None),
                      max_speakers: str = Form(default=None),
                      async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "pipeline not ready")
        data = await file.read()
        path = await asyncio.to_thread(spill, data, file.filename)

        def _work(ctx):
            kw = {}
            if num_speakers:
                kw["num_speakers"] = int(num_speakers)
            if min_speakers:
                kw["min_speakers"] = int(min_speakers)
            if max_speakers:
                kw["max_speakers"] = int(max_speakers)
            ctx.progress(ratio=0.0, stage="decode")
            waveform, sr = decode(path)
            dur = float(waveform.shape[-1]) / float(sr)
            t0 = time.time()
            out = _infer(waveform, sr, kw, _hook(ctx))
            log.info("diarized %.1fs of audio in %.1fs", dur, time.time() - t0)
            # pyannote 4 wraps the Annotation in .speaker_diarization; v3 returned it directly.
            ann = getattr(out, "speaker_diarization", out)
            segs = [{"start": round(float(t.start), 3), "end": round(float(t.end), 3),
                     "speaker": str(spk)}
                    for t, _, spk in ann.itertracks(yield_label=True)]
            speakers = sorted({s["speaker"] for s in segs})
            ctx.progress(ratio=1.0, stage="done")
            return {"model": MODEL_NAME, "mode": "diar", "device": _state["device"],
                    "num_speakers": len(speakers), "speakers": speakers,
                    "num_segments": len(segs), "segments": segs}

        return await tasks.dispatch(async_, "diar", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="diarization failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "pyannote pipeline")
