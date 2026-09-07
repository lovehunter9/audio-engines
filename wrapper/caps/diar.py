# Speaker diarization with pyannote.audio.
import asyncio
import contextlib
import copy
import math
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

_runtime = Runtime(pipeline=None, device="cpu", batch1=False, params={})
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Both pyannote stages default to batch_size=1: ~12 000 launches of 10 s of audio for a 3 h clip.
_args = EngineArgs()
SEG_BATCH = _args.text("--segmentation-batch-size", "auto")  # "auto" = GPU-sized on CUDA, 1 on CPU
EMB_BATCH = _args.text("--embedding-batch-size", "auto")

# The pipeline reads these off itself mid-run rather than taking them in the call, so a caller
# who wants one per request needs it set on the pipeline for the length of that job. Left unset
# here, pyannote's own tuned values stand; a request may still override either for one job.
MIN_DURATION_OFF = _args.text("--min-duration-off")
CLUSTERING_THRESHOLD = _args.text("--clustering-threshold")
EXCLUSIVE = _args.switch("--exclusive", False)
_args.warn_unclaimed(log)

# Wire name -> where it lives in the pipeline's nested parameter dict.
TUNABLES = {"min_duration_off": ("segmentation", "min_duration_off"),
            "clustering_threshold": ("clustering", "threshold")}

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


def _params(pipe):
    """This pipeline's instantiated hyper-parameters, or {} on a build that exposes none."""
    try:
        return copy.deepcopy(pipe.parameters(instantiated=True))
    except Exception as e:
        log.warning("cannot read the hyper-parameters of %s (%s) — min_duration_off and "
                    "clustering_threshold will be refused", type(pipe).__name__, e)
        return {}


def _floats(raw, base):
    """The tunables actually asked for, as numbers, checked against what this pipeline has.

    Raises so a request's typo is a 400 rather than a job that fails minutes later. The load
    path calls this too, where a bad flag is a warning: an unusable knob is not worth
    refusing to serve over.
    """
    out = {}
    for name, value in raw.items():
        if value is None or str(value).strip() == "":
            continue
        section, key = TUNABLES[name]
        if key not in base.get(section, {}):
            raise HTTPException(status_code=400,
                                detail="%s is not a hyper-parameter of %s" % (name, MODEL_REPO))
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400,
                                detail="%s must be a number, got %r" % (name, value))
    return out


def _merged(base, overrides):
    """base with overrides applied: instantiate() wants every parameter, not a patch."""
    out = copy.deepcopy(base)
    for name, value in overrides.items():
        section, key = TUNABLES[name]
        out[section][key] = value
    return out


@contextlib.contextmanager
def _tuned(overrides):
    """Hold the pipeline at these hyper-parameters for one job, then put it back.

    Safe only because the runner runs one job at a time; a second job overlapping this one
    would see the first job's tuning.
    """
    if not overrides:
        yield
        return
    pipe, base = _state["pipeline"], _state["params"]
    pipe.instantiate(_merged(base, overrides))
    try:
        yield
    finally:
        pipe.instantiate(copy.deepcopy(base))


def _effective(overrides):
    """What the run used, including a default the caller never sent."""
    out = {}
    for name, (section, key) in TUNABLES.items():
        if name in overrides:
            out[name] = overrides[name]
        elif key in _state["params"].get(section, {}):
            try:
                out[name] = float(_state["params"][section][key])
            except (TypeError, ValueError):
                pass
    return out


def _seed(pipe):
    """Fold the ENGINE_ARGS hyper-parameters into the pipeline, and record the result.

    What lands in _state["params"] is the baseline every request is measured against and
    restored to, so a flag set here reads back as the default rather than as an override.
    """
    base = _params(pipe)
    try:
        seed = _floats({"min_duration_off": MIN_DURATION_OFF,
                        "clustering_threshold": CLUSTERING_THRESHOLD}, base)
    except HTTPException as e:
        log.warning("ignoring hyper-parameters from ENGINE_ARGS: %s", e.detail)
        seed = {}
    if seed:
        base = _merged(base, seed)
        pipe.instantiate(copy.deepcopy(base))
        log.info("hyper-parameters from ENGINE_ARGS: %s", seed)
    _state["params"] = base


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
        _seed(pipe)
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


def _annotation(out, exclusive):
    """Which of community-1's two diarizations to answer with.

    The exclusive one holds at most one speaker at any instant. A caller that cuts the audio
    along these turns and transcribes each piece wants that: two overlapping turns send the
    same seconds twice and come back with the same words twice. A build that has only the
    overlapping one still gets served, and the response says which it was.
    """
    if exclusive:
        excl = getattr(out, "exclusive_speaker_diarization", None)
        if excl is not None:
            return excl, True
        log.warning("this pyannote build has no exclusive diarization; "
                    "answering with the overlapping one")
    return getattr(out, "speaker_diarization", out), False


def _centroids(out, speakers):
    """The clustering's per-speaker centroid, when its rows line up with the speakers found.

    Row i is cluster i, and pyannote renames clusters to SPEAKER_00.. in sorted order, so the
    two agree only while every cluster produced at least one turn. A cluster that produced
    none shifts every label after it and there is no way to tell from here which one it was,
    so a mismatched row count means no centroids rather than centroids against wrong names.

    These are for looking at. Matching them against vectors from the embed cap is not
    meaningful: a centroid is an average of this recording's windows in this pipeline's own
    embedding space.
    """
    emb = getattr(out, "speaker_embeddings", None)
    if emb is None:
        return None
    rows = []
    try:
        for row in emb:
            vals = [float(x) for x in row]
            if not all(map(math.isfinite, vals)):
                log.info("centroids hold non-finite values — omitting them")
                return None
            rows.append([round(v, 6) for v in vals])
    except TypeError:
        return None
    if len(rows) != len(speakers):
        log.info("%d centroid rows for %d speakers — omitting, they cannot be matched up",
                 len(rows), len(speakers))
        return None
    return dict(zip(speakers, rows))


def build_app(supports):
    app = FastAPI(title="audio-diarization (pyannote)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="diar", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/diarization")
    async def diarize(file: UploadFile = File(...), num_speakers: str = Form(default=None),
                      min_speakers: str = Form(default=None),
                      max_speakers: str = Form(default=None),
                      exclusive: str = Form(default=None),
                      min_duration_off: str = Form(default=None),
                      clustering_threshold: str = Form(default=None),
                      async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "pipeline not ready")
        want_exclusive = EXCLUSIVE if exclusive is None else tasks.truthy(exclusive)
        tuning = _floats({"min_duration_off": min_duration_off,
                          "clustering_threshold": clustering_threshold}, _state["params"])
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
            ctx.meter(input_seconds=dur)
            t0 = time.time()
            with _tuned(tuning):
                out = _infer(waveform, sr, kw, _hook(ctx))
            log.info("diarized %.1fs of audio in %.1fs", dur, time.time() - t0)
            # pyannote 4 wraps the Annotation in .speaker_diarization; v3 returned it directly.
            ann, exclusive_used = _annotation(out, want_exclusive)
            segs = [{"start": round(float(t.start), 3), "end": round(float(t.end), 3),
                     "speaker": str(spk)}
                    for t, _, spk in ann.itertracks(yield_label=True)]
            speakers = sorted({s["speaker"] for s in segs})
            ctx.progress(ratio=1.0, stage="done")
            body = {"model": MODEL_NAME, "mode": "diar", "device": _state["device"],
                    "num_speakers": len(speakers), "speakers": speakers,
                    "num_segments": len(segs), "segments": segs,
                    "exclusive": exclusive_used}
            body.update(_effective(tuning))
            centroids = _centroids(out, speakers)
            if centroids is not None:
                body["speaker_centroids"] = centroids
            return body

        return await tasks.dispatch(async_, "diar", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="diarization failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "pyannote pipeline")
