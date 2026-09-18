# Speaker diarization with speakrs: a Rust reimplementation of the pyannote community-1 pipeline.
#
# The inference lives in a long-lived child process rather than in this one, because speakrs is a
# Rust library with no Python bindings. This wrapper keeps the whole llm-init contract -- model
# list, engine spec, task API, ENGINE_ARGS -- and hands only the inference across.
#
# Why a pipe and a file path, not a socket and a request body: the upload is already spilled to a
# temp file by the time a job runs, and both processes see the same filesystem, so the path moves
# no audio. This removes one copy of a hundreds-of-megabytes clip, not the memory it needs -- the
# engine still holds the whole thing decoded, which is what BOUNDS below caps.
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import tasks
from .. import watchdog
from ..gpu import mount_metrics
from ..contract import register, EngineArgs, cache_dir
from ..audioio import unlink
from ..limits import Bounds
from ..runtime import Runtime

log = logging.getLogger("audio-diar-speakrs")

# The engine binary baked into the image; it speaks the line protocol described in _Child below.
ENGINE_BIN = os.environ.get("SPEAKRS_ENGINE_BIN") or "/usr/local/bin/speakrs-engine"


# llm-init writes the resolved model path here for engines that share its run directory
# (RUN_DIR/model_path, one absolute line, written atomically). The audio engine deployments do
# not mount that directory -- their pod carries the HF cache and nothing else -- so this is a
# preference, not a dependency: when the file is there it is authoritative and no guessing is
# needed, and when it is not, the cache layout below answers.
RUN_DIR = os.environ.get("RUN_DIR") or "/run/llm-init"


# The batched segmentation exports speakrs can be asked to load. Any batch size, because the
# number is upstream's PRIMARY_BATCH_SIZE and the point of matching rather than spelling it is
# that this side does not have to be edited when it changes. Excludes what this function
# writes, or a restart would derive from its own output.
_BATCHED_SEGMENTATION = re.compile(r"segmentation-[\d.]+-b\d+\.onnx")


def _farm_for(models_dir):
    """Where the prepared models for one cache directory go.

    Keyed on the directory it derives from, so two engines reading different caches on one
    machine do not tear down each other's work. The digest only has to separate cache
    directories, not identify them.

    🔴 What this does NOT solve, said here because the first version of this comment claimed it
    did: two replicas of one deployment read the SAME cache, so they land on the same key and
    collide exactly as a fixed name would. Separating those needs a key per process or per pod,
    not per cache. It does not arise under the chart that ships this -- one replica, and a
    Recreate strategy, so no two pods are ever up together -- and that is the reason it is left
    alone rather than an oversight. A chart that scales this out has to revisit the key before
    anything else.
    """
    return "/tmp/speakrs-models-openvino-" + hashlib.sha256(
        os.path.abspath(models_dir).encode()).hexdigest()[:12]


def _openvino_models_dir(models_dir):
    """The directory to hand the engine, with a batched segmentation model OpenVINO can use.

    Batching segmentation is speakrs' own feature and it is on for every other backend. On
    OpenVINO's GPU plugin it is off, because the stock segmentation-3.0-b32 export has a static
    sequence length and that plugin cannot compile an LSTM kernel for that graph -- measured on
    both an Arrow Lake integrated part and an Arc Pro B70.

    🔴 The plugin, not the backend: OpenVINO on the processor compiles the stock export and
    speakrs takes it there. So this runs wider than the consumer that reads its output, and
    deliberately -- deriving a file nothing asks for costs one pass over a 6 MB graph, while
    not deriving one that is asked for turns batching off with nothing in the log. The two
    sides live in different repositories, so the asymmetry is what keeps them safe to drift. The same export with its sample dimension
    made dynamic compiles and runs, and is 15x faster per window than going one at a time --
    529.5 ms against 34.9, one card, 64 windows, best of three, recorded with the rest of the
    conditions in beclab/speakrs-diarization's README.

    That model is derived here rather than baked into the image, because baking it would pin a
    copy of weights the engine resolves separately: a new revision upstream and the two drift
    apart with nothing to notice. Derived at startup, it is always the cache's own file.

    The result is a directory of symlinks plus the one real file, not an edit of the cache.
    The cache is shared with other applications and is huggingface_hub's to manage; adding
    files to a snapshot directory is not ours to do.

    Every failure here returns the original directory. Then speakrs finds no prepared model,
    batching stays off, and the engine runs exactly as it did before this existed -- slower,
    and working. There is no failure mode worth stopping startup for.
    """
    if not EXECUTION_MODE.startswith("openvino"):
        return models_dir
    # 🔴 The same spellings switch() decides by, not the two literal words. `off` is not the
    # only way this flag is turned off: every other boolean in this wrapper answers to
    # 0/false/no/off -- --exclusive, a few lines below where this one is read, is one of them.
    # Reading only the word "off" sends `--openvino-batching false` into the branch below,
    # which keeps deriving, and the flag exists to be turned off for measurement: the number
    # that comes back says "batching off" while batching was on.
    # An unrecognised value still derives, deliberately -- see the note in tests/caps_smoke.py.
    _batching = OPENVINO_BATCHING.strip().lower()
    if _batching in EngineArgs.OFF_WORDS:
        log.info("--openvino-batching %s: segmentation will run one window at a time",
                 _batching)
        return models_dir
    if _batching not in EngineArgs.ON_WORDS:
        log.warning("--openvino-batching %r is neither on nor off; treating it as on",
                    OPENVINO_BATCHING)
    # 🔴 Found by pattern, and the derived name follows the one found, rather than both being
    # written out here. speakrs asks for "<the batched export's name>-dynseq.onnx", building
    # the batch number from its own PRIMARY_BATCH_SIZE constant; spelling 32 on this side made
    # the two agree only as long as nobody changed that constant. Deriving from whatever
    # export is actually in the cache keeps them in step through a change of batch size,
    # because the export upstream ships and the constant it compiles against move together.
    #
    # Every match is derived, not just one, so a cache carrying more than one batched export
    # has a prepared model for whichever the engine turns out to ask for.
    # 🔴 Inside a try, like everything below it. _models_dir hands over a directory that may
    # not exist yet -- the wrapper script starts this process offline when llm-init's sentinel
    # never arrives, and its comment says the engine reports the missing weights by name,
    # which is a better error than one invented here. Listing that directory unguarded turned
    # the engine's clean report into a traceback at import, before the engine was ever run.
    try:
        stock = sorted(
            os.path.join(models_dir, name)
            for name in os.listdir(models_dir)
            if _BATCHED_SEGMENTATION.fullmatch(name)
        )
    except OSError as e:
        log.info("cannot list %s (%s); leaving batching to the engine's own report", models_dir, e)
        return models_dir
    if not stock:
        log.info("no batched segmentation model in the cache; leaving batching off")
        return models_dir
    farm = _farm_for(models_dir)
    try:
        import onnx
        from onnx import shape_inference

        # 🔴 Rebuilt, never reused. /tmp is a mounted volume here and survives a restart, so
        # keeping what is already there would mean symlinks still aimed at the revision that
        # was current when they were made, and a derived model still carrying its weights --
        # against a cache that has moved on. Nothing would report the disagreement. Rebuilding
        # costs a directory of symlinks and one pass over a 6 MB graph.
        shutil.rmtree(farm, ignore_errors=True)
        os.makedirs(farm)
        for name in os.listdir(models_dir):
            os.symlink(os.path.join(models_dir, name), os.path.join(farm, name))

        # 🔴 One export failing does not take the others with it. This loop was inside the
        # single try below, so a cache holding both a b32 and a b64 export lost BOTH when either
        # one would not convert -- the farm was torn down and the engine ran unbatched, with the
        # log naming a file that had nothing wrong with it. Each conversion answers for itself;
        # the farm survives as long as one of them lands, and speakrs picks whichever it asks for.
        derived = 0
        for source in stock:
            prepared = os.path.join(
                farm, os.path.basename(source)[: -len(".onnx")] + "-dynseq.onnx")
            try:
                model = onnx.load(source)
                # The sample count, and only it. The batch dimension is deliberately left static:
                # making that one dynamic instead was measured and does not avoid the failure.
                dims = model.graph.input[0].type.tensor_type.shape.dim
                dims[2].ClearField("dim_value")
                dims[2].dim_param = "samples"
                out = model.graph.output[0].type.tensor_type.shape.dim
                out[1].ClearField("dim_value")
                out[1].dim_param = "frames"
                model = shape_inference.infer_shapes(model, strict_mode=True)
                onnx.checker.check_model(model)
                onnx.save(model, prepared + ".partial")
            except Exception:
                log.warning("could not prepare %s; the others are unaffected", source,
                            exc_info=True)
                with contextlib.suppress(OSError):
                    os.remove(prepared + ".partial")
                continue
            # Renamed into place. Not because a half-written model could otherwise be loaded
            # -- the farm above is deleted and rebuilt on every start, so nothing from a
            # crashed run survives to be found. It is that the engine is handed this directory
            # as soon as the function returns, and a reader arriving between the write and the
            # end of it would see a truncated file under the name speakrs looks for.
            os.replace(prepared + ".partial", prepared)
            derived += 1
            log.info("prepared a batched segmentation model for OpenVINO: %s", prepared)
        if not derived:
            raise RuntimeError("no batched segmentation model could be prepared")
        return farm
    except Exception:
        log.warning("could not prepare the batched segmentation model; "
                    "OpenVINO will run segmentation one window at a time", exc_info=True)
        # Nothing points at the half-built farm once the cache directory is returned, so
        # leaving it would not break anything. It is removed because a directory full of
        # symlinks named like a working farm is the first thing someone debugging "why is
        # batching off" will find, and it says the opposite of what happened.
        shutil.rmtree(farm, ignore_errors=True)
        return models_dir


def _models_dir(repo):
    """The directory holding this repo's weights.

    llm-init downloads through huggingface_hub, which stores a repo as
    models--<owner>--<name>/snapshots/<revision>/ rather than at a path anyone can predict. The
    engine is Rust and has no huggingface_hub to resolve that for it, so the resolution happens
    here, where the rest of this repo's cache knowledge already lives.

    Newest revision wins when several are present: a re-download leaves the old one behind, and
    the one just fetched is the one llm-init is waiting on.
    """
    told = os.path.join(RUN_DIR, "model_path")
    try:
        with open(told) as f:
            path = f.read().strip()
        if path:
            log.info("llm-init resolved the model path for us: %s", path)
            return path
    except OSError:
        pass
    root = cache_dir(repo)
    snapshots = os.path.join(root, "snapshots")
    try:
        revisions = [os.path.join(snapshots, r) for r in os.listdir(snapshots)]
        revisions = [r for r in revisions if os.path.isdir(r)]
        if revisions:
            return max(revisions, key=os.path.getmtime)
    except OSError:
        pass
    # No snapshots directory: either the weights were placed flat (a hand-built image, the
    # benchmark box) or they are not there at all. The engine reports the missing file by name,
    # which is a better error than one invented here from a directory listing.
    log.info("no HF snapshot under %s; passing the directory itself to the engine", root)
    return root

_runtime = Runtime(pipeline=None, device="cpu", params={})
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo

# One segmentation frame, in seconds. speakrs states its duration filters in frames; every caller
# and every other engine in this repo states them in seconds, so the conversion lives here and the
# wire stays in seconds. The value is a property of the segmentation model's output stride and does
# NOT change with the execution mode -- the 1s / 0.96s / 2s figures in speakrs are the sliding
# window's step, which is a different thing.
FRAME_SECONDS = 0.016875


def _frames(seconds):
    """Seconds -> whole frames, or None when the caller did not ask for this knob.

    None and 0 are different answers and must stay different: speakrs' fast modes default
    min_duration_off to 3 frames, so passing 0 for "unset" would silently disable a filter the
    upstream turned on deliberately.
    """
    if seconds is None:
        return None
    return max(0, int(round(float(seconds) / FRAME_SECONDS)))


_args = EngineArgs()
# Upstream's own spelling for the mode; "cuda-fast" trades ~0.3 points of DER for roughly double
# the speed by stepping the segmentation window 2s instead of 1s.
EXECUTION_MODE = _args.text("--execution-mode", "cuda")
# The one switch on the OpenVINO batched-segmentation derivation, and it exists for measuring:
# the 4x on the integrated part was established by turning this off and running the same clip
# again, and the next person with a number to check should not have to break the derivation
# to do that. "on" is the default and the only other value is "off"; anything else is a typo,
# said so in the log, and treated as on -- silently losing 4x is the outcome to avoid.
OPENVINO_BATCHING = _args.text("--openvino-batching", "on")
CLUSTERING_THRESHOLD = _args.text("--clustering-threshold")
MIN_DURATION_OFF = _args.text("--min-duration-off")
MIN_DURATION_ON = _args.text("--min-duration-on")
EXCLUSIVE = _args.switch("--exclusive", False)
# speakrs takes the whole clip as one resident f32 buffer -- run(audio: &[f32]) -- so memory grows
# with duration and nothing streams: an hour is 230 MB, three hours 690 MB, on top of the model.
# Passing the file path instead of the bytes saved one copy, not the buffer itself. Four hours is
# past any real meeting; a deployment that knows its own memory ceiling can move it.
BOUNDS = Bounds(_args, seconds=14400)
_args.warn_unclaimed(log)

# Wire name -> (child field, converter). Kept in one place so the request path, the ENGINE_ARGS
# path and the "what did this run actually use" report cannot drift apart.
TUNABLES = {
    "min_duration_off": ("min_duration_off_frames", _frames),
    "min_duration_on": ("min_duration_on_frames", _frames),
    "clustering_threshold": ("clustering_threshold", float),
}

_state = _runtime.state


class _Child:
    """The speakrs engine process, and the one-job-at-a-time protocol spoken to it.

    Protocol, one JSON object per line in each direction:

      child -> us, once at startup:  {"ready": true, "device": "cuda", "load_seconds": 12.3}
                   or on failure:    {"ready": false, "error": "..."}
      us -> child, per job:          {"id": "...", "path": "/tmp/upload-x.wav", "exclusive": false,
                                      "min_duration_off_frames": 178, ...}
      child -> us, per job:          {"id": "...", "ok": true, "device": "cuda",
                                      "segments": [[0.5, 3.2, "SPEAKER_00"], ...]}
                   or:               {"id": "...", "ok": false, "error": "..."}

    stdout carries the protocol and nothing else; the child logs to stderr. A stray print on the
    child's stdout would desynchronise the stream, so the reader treats an unparseable line as a
    protocol error rather than skipping it -- a silently dropped reply would hang the job instead.

    Serialisation is the task runner's, not ours: it runs one job at a time (see tasks.py), so a
    single child with a strict request/response cycle is enough. A second concurrent caller would
    interleave lines, so _lock enforces what the runner already guarantees rather than trusting it.
    """

    def __init__(self, argv):
        self._argv = argv
        self._proc = None
        self._lock = threading.Lock()
        self._seq = 0

    def start(self):
        log.info("starting engine: %s", " ".join(self._argv))
        self._proc = subprocess.Popen(
            self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,  # inherit: the child's log belongs in the container log, interleaved
            text=True, bufsize=1,
            # Its own process group, so the SIGTERM that ends this pod reaches the child through
            # us rather than beside us. Otherwise a normal shutdown and a crash look identical
            # from the reaper, and it has to treat one of them wrongly.
            start_new_session=True,
        )
        threading.Thread(target=self._reap, daemon=True).start()
        hello = self._readline("startup")
        if not hello.get("ready"):
            raise RuntimeError(hello.get("error") or "engine failed to load the model")
        log.info("engine ready on %s in %.1fs", hello.get("device"), hello.get("load_seconds") or 0)
        return hello

    def _reap(self):
        """A child that dies takes the engine with it, and nothing here can put it back.

        Everything this container serves runs in that process, so once it is gone the pod is an
        endpoint that answers 503 to every request, forever: the load watchdog only fires while a
        model is still loading, and a wrapper that once loaded is "failed with a reason", which
        is deliberately left alone. Recovery took someone noticing and deleting the pod.

        Exiting hands that to the thing that already does it. k8s restarts the container, backs
        off if the crash repeats, and a CrashLoopBackOff says what a permanently unhealthy
        endpoint does not. Reloading in-process was the alternative and is worse in the same
        way the old comment says: minutes of model load behind a port that reports ready.
        """
        rc = self._proc.wait()
        _state["ready"] = False
        _state["error"] = "engine process exited with code %d" % rc
        if _stopping.is_set():
            log.info("engine process exited with code %d during shutdown", rc)
            return
        log.error("engine process exited with code %d; exiting so the container is restarted", rc)
        # Long enough for the job that was in flight to come back as a 500 rather than as a
        # connection the caller has to guess about.
        time.sleep(_EXIT_GRACE_S)
        _exit(watchdog.EXIT_CODE)

    def _readline(self, what):
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("engine closed its output during %s" % what)
        try:
            return json.loads(line)
        except ValueError:
            raise RuntimeError("engine wrote a non-protocol line during %s: %r" % (what, line[:200]))

    def run(self, payload):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError(_state["error"] or "engine process is not running")
            self._seq += 1
            payload = dict(payload, id=str(self._seq))
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            reply = self._readline("job %s" % payload["id"])
        if reply.get("id") != payload["id"]:
            # Answering the wrong job means the stream is off by one and every later reply would be
            # wrong too. There is no recovery that does not risk returning one caller's answer to
            # another, so fail the job and let the mismatch be visible.
            raise RuntimeError("engine replied to job %r while %r was in flight"
                               % (reply.get("id"), payload["id"]))
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "diarization failed in the engine")
        return reply


_child = _Child([ENGINE_BIN, "--mode", EXECUTION_MODE,
                 "--models-dir", _openvino_models_dir(_models_dir(MODEL_REPO))])

# Set once this process is on its way out, so the child dying with us is not read as a crash.
_stopping = threading.Event()

# Indirections, so a test can watch the decision without the test runner being the thing that
# exits, and without waiting out the grace period.
_exit = os._exit
_EXIT_GRACE_S = 2.0


def _seed():
    """The ENGINE_ARGS tunables, as the baseline every request is measured against.

    A flag set here reads back as this engine's default rather than as a per-request override, so
    the response's "what did this run use" fields mean the same thing whoever set them.
    """
    base = {}
    for name, raw in (("min_duration_off", MIN_DURATION_OFF),
                      ("min_duration_on", MIN_DURATION_ON),
                      ("clustering_threshold", CLUSTERING_THRESHOLD)):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            base[name] = float(raw)
        except (TypeError, ValueError):
            log.warning("ignoring %s from ENGINE_ARGS: %r is not a number", name, raw)
    if base:
        log.info("tunables from ENGINE_ARGS: %s", base)
    _state["params"] = base


def _tunables(raw):
    """The per-request tunables as numbers, checked here so a typo is a 400 and not a late failure."""
    out = {}
    for name, value in raw.items():
        if value is None or str(value).strip() == "":
            continue
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400,
                                detail="%s must be a number, got %r" % (name, value))
    return out


def _effective(overrides):
    """What the run used, including a default the caller never sent."""
    out = dict(_state["params"])
    out.update(overrides)
    return out


def _load():
    try:
        _seed()
        hello = _child.start()
        _state["device"] = hello.get("device") or "cpu"
        _state["ready"] = True
    except Exception as e:
        _state["error"] = str(e)
        log.exception("engine load failed: %s", e)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    yield
    # Uvicorn runs this when it starts shutting down, which is the only warning the reaper gets
    # that the exit it is about to see is ours.
    _stopping.set()


def build_app(supports):
    app = FastAPI(title="audio-diarization (speakrs)", lifespan=_lifespan)
    # This image carries no torch, so the gauges come from NVML; see gpu.mount_metrics.
    mount_metrics(app, nvml_fallback=True)

    register(app, model_name=MODEL_NAME, module="diar_speakrs", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             model_format="onnx")

    @app.post("/v1/audio/diarization")
    async def diarize(file: UploadFile = File(...),
                      num_speakers: str = Form(default=None),
                      min_speakers: str = Form(default=None),
                      max_speakers: str = Form(default=None),
                      exclusive: str = Form(default=None),
                      min_duration_off: str = Form(default=None),
                      min_duration_on: str = Form(default=None),
                      clustering_threshold: str = Form(default=None),
                      async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")
        # speakrs derives the speaker count from the clustering; it has no way to be told one.
        # Refusing beats accepting and ignoring: a caller that asked for three speakers and got
        # seven would have no way to tell that its constraint was dropped.
        asked = [n for n, v in (("num_speakers", num_speakers), ("min_speakers", min_speakers),
                                ("max_speakers", max_speakers)) if v not in (None, "")]
        if asked:
            raise HTTPException(status_code=400,
                                detail="%s: this engine derives the speaker count from clustering "
                                       "and cannot be constrained to one" % ", ".join(asked))
        want_exclusive = EXCLUSIVE if exclusive is None else tasks.truthy(exclusive)
        tuning = _tunables({"min_duration_off": min_duration_off,
                            "min_duration_on": min_duration_on,
                            "clustering_threshold": clustering_threshold})
        # Read off the header, not a decode: this cap never holds samples, the child does.
        path, seconds = await BOUNDS.spill(
            file, "this engine holds the whole clip in memory")

        def _work(ctx):
            ctx.meter(input_seconds=seconds)
            # Two stages is all the progress there is. speakrs exposes no callback inside a run,
            # and splitting the clip to fake one would change the answer -- its clustering is over
            # the whole recording, so chunks would not agree on who SPEAKER_00 is.
            ctx.progress(ratio=0.0, stage="inference")
            ctx.checkpoint()
            payload = {"path": path, "exclusive": want_exclusive}
            for name, (field, cast) in TUNABLES.items():
                merged = _effective(tuning)
                payload[field] = cast(merged[name]) if name in merged else None
            t0 = time.time()
            reply = _child.run(payload)
            log.info("diarized %s in %.1fs", path, time.time() - t0)
            ctx.progress(ratio=1.0, stage="done")
            segs = [{"start": round(float(s), 3), "end": round(float(e), 3), "speaker": str(spk)}
                    for s, e, spk in reply.get("segments") or []]
            speakers = sorted({s["speaker"] for s in segs})
            body = {"model": MODEL_NAME, "mode": "diar", "device": reply.get("device") or "cpu",
                    "num_speakers": len(speakers), "speakers": speakers,
                    "num_segments": len(segs), "segments": segs,
                    "exclusive": bool(want_exclusive)}
            body.update(_effective(tuning))
            return body

        return await tasks.dispatch(async_, "diar", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="diarization failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "speakrs engine")
