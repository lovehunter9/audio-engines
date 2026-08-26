# Speaker diarization with speakrs: a Rust reimplementation of the pyannote community-1 pipeline.
#
# The inference lives in a long-lived child process rather than in this one, because speakrs is a
# Rust library with no Python bindings. This wrapper keeps the whole llm-init contract -- model
# list, engine spec, task API, ENGINE_ARGS -- and hands only the inference across.
#
# Why a pipe and a file path, not a socket and a request body: the upload is already spilled to a
# temp file by the time a job runs, and both processes see the same filesystem, so passing the path
# copies nothing. A three-hour clip is hundreds of megabytes; moving those bytes across a socket
# would be the single most expensive thing this engine does, and it would buy nothing.
import json
import logging
import os
import subprocess
import threading
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import tasks
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import probe_seconds, spill, unlink
from ..runtime import Runtime

log = logging.getLogger("audio-diar-speakrs")

# The engine binary baked into the image; it speaks the line protocol described in _Child below.
ENGINE_BIN = os.environ.get("SPEAKRS_ENGINE_BIN") or "/usr/local/bin/speakrs-engine"

_runtime = Runtime("speakrs-community-1", pipeline=None, device="cpu", params={})
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
CLUSTERING_THRESHOLD = _args.text("--clustering-threshold")
MIN_DURATION_OFF = _args.text("--min-duration-off")
MIN_DURATION_ON = _args.text("--min-duration-on")
EXCLUSIVE = _args.switch("--exclusive", False)
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
        )
        threading.Thread(target=self._reap, daemon=True).start()
        hello = self._readline("startup")
        if not hello.get("ready"):
            raise RuntimeError(hello.get("error") or "engine failed to load the model")
        log.info("engine ready on %s in %.1fs", hello.get("device"), hello.get("load_seconds") or 0)
        return hello

    def _reap(self):
        """A child that dies takes the engine with it; say so loudly rather than hanging.

        Restarting it here would be wrong: the model takes a long time to load, so a crash loop
        would answer 503 for minutes while looking alive. The watchdog already handles a process
        that is neither ready nor failed, and k8s rebuilds the container.
        """
        rc = self._proc.wait()
        _state["ready"] = False
        _state["error"] = "engine process exited with code %d" % rc
        log.error("engine process exited with code %d", rc)

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


_child = _Child([ENGINE_BIN, "--mode", EXECUTION_MODE])


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


def build_app(supports):
    app = FastAPI(title="audio-diarization (speakrs)")
    mount_metrics(app)

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
        data = await file.read()
        path = await _to_thread(spill, data, file.filename)

        def _work(ctx):
            # Metered off the file header rather than a decode: this cap never holds samples, the
            # child does. None is a real answer and bills as "not measured".
            ctx.meter(input_seconds=probe_seconds(path))
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


async def _to_thread(fn, *a):
    import asyncio

    return await asyncio.to_thread(fn, *a)


def run(supports):
    _runtime.serve(supports, _load, build_app, "speakrs engine")
