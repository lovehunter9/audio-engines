# Long jobs as tasks: `async=1` on a capability POST answers with a task id instead of the result.
import asyncio
import concurrent.futures as futures
import logging
import os
import queue
import tempfile
import threading
import time
import uuid

log = logging.getLogger("audio-tasks")

# A result is kept this long after the job ends, so a client that polls slowly can still fetch it.
TTL_S = 1800.0
QUEUE_MAX = 32
_GC_EVERY_S = 30.0

# The cross-engine async-tasks contract is defined in llm-init's docs/api/openapi.yaml.
TASKS_PATH = "/v1/tasks"
# What this engine shipped before the contract existed. Same runner, same tasks, kept for old clients.
LEGACY_PATH = "/v1/audio/tasks"

# What this image's tasks report as the contract's `kind`; ocr and image say so from their own.
KIND = "audio"

_ROUTES = [
    ("GET", "", "List tasks (oldest first) with queue capacity; ?status= and ?limit="),
    ("GET", "/{id}", "Task status / progress / JSON result"),
    ("GET", "/{id}/result", "Task result (audio for enhance, else JSON)"),
    ("DELETE", "/{id}", "Cancel a running task, or drop a finished one's result"),
]

# Per-second pricing bills on audio duration, and the engine is the only party that knows it:
# the gateway forwards request and response as streams and never sees a decoded clip. Reported
# on the response headers and on the task document, or not at all — a missing header means
# "not measured", which is not the same as zero.
INPUT_SECONDS_HEADER = "X-Audio-Input-Duration-Seconds"
OUTPUT_SECONDS_HEADER = "X-Audio-Output-Duration-Seconds"

# Advertised by /api/engine-spec next to the capability's own endpoints.
ENDPOINTS = [
    {"method": method, "path": TASKS_PATH + tail, "description": desc}
    for method, tail, desc in _ROUTES
] + [
    {"method": method, "path": LEGACY_PATH + tail, "deprecated": True,
     "description": "%s; alias of %s" % (desc, TASKS_PATH + tail)}
    for method, tail, desc in _ROUTES
]

# Appended to a capability's own description so callers can discover the async mode.
ASYNC_HINT = "async=1 -> 202 + task"

_TERMINAL = ("succeeded", "failed", "canceled")
_STATUSES = ("queued", "running") + _TERMINAL

# GET /v1/tasks bounds, per the contract: without them a list is "every task still inside the TTL".
LIST_LIMIT_DEFAULT = 100
LIST_LIMIT_MAX = 1000


class Cancelled(Exception):
    """Raised inside a job by ctx.checkpoint() once cancellation was requested."""


class Busy(Exception):
    """The queue is full; the caller should retry later."""


class Binary:
    # A result that is not JSON (enhanced audio, plain text): bytes plus how to serve them.
    def __init__(self, data, media_type, headers=None, suffix=""):
        self.data = data
        self.media_type = media_type
        self.headers = dict(headers or {})
        self.suffix = suffix


def truthy(v):
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


class _Ctx:
    # Handed to the job: report progress, and notice a cancel between two units of work.
    def __init__(self, task=None):
        self._task = task

    def progress(self, ratio=None, stage=None, done=None, total=None):
        t = self._task
        if t is None:
            return
        p = dict(t.progress)
        if stage is not None:
            # Each stage counts its own units, so carrying the last one's over reads as "8 of 6".
            if p.get("stage") != str(stage):
                p.pop("done", None)
                p.pop("total", None)
            p["stage"] = str(stage)
        if done is not None:
            p["done"] = int(done)
        if total is not None:
            p["total"] = int(total)
        if ratio is None and p.get("total"):
            ratio = float(p.get("done") or 0) / float(p["total"])
        if ratio is not None:
            p["ratio"] = round(max(0.0, min(1.0, float(ratio))), 4)
        t.progress = p

    def meter(self, input_seconds=None, output_seconds=None):
        """Report metered audio duration. Additive, so a batch reports once per item."""
        t = self._task
        if t is None:
            return
        for attr, value in (("input_seconds", input_seconds),
                            ("output_seconds", output_seconds)):
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
            setattr(t, attr, round((getattr(t, attr) or 0.0) + value, 3))

    def cancelled(self):
        return bool(self._task is not None and self._task.cancel)

    def checkpoint(self):
        if self.cancelled():
            raise Cancelled("canceled by client")


# For blocking helpers that are also called outside a job (batch mode reuses the single-clip path).
NULL_CTX = _Ctx()


class Task:
    def __init__(self, cap, model, work, cleanup=None, keep=True, gate=None):
        self.id = "tsk_" + uuid.uuid4().hex[:24]
        self.cap = cap
        self.model = model
        self.work = work
        self.gate = gate            # a cap whose engine is also driven outside the queue (WS) shares it
        self.keep = keep            # sync callers take the result off the future and want no copy
        self.status = "queued"
        self.created = time.time()
        self.started = None
        self.finished = None
        self.progress = {}
        self.error = None
        self.result = None
        self.result_kind = None
        self.result_path = None     # binary results live on disk, not in the engine's heap
        self.result_bytes = None
        self.content_type = None
        self.headers = {}
        self.input_seconds = None    # None means the job never measured it, not that it was 0
        self.output_seconds = None
        self.cancel = False
        self.future = futures.Future()
        self._cleanup = cleanup
        self._cleaned = threading.Lock()
        self._done_cleanup = False

    def run_cleanup(self):
        # The upload's temp file: dropped as soon as the job is over, not at GC time.
        with self._cleaned:
            if self._done_cleanup or self._cleanup is None:
                return
            self._done_cleanup = True
        try:
            self._cleanup()
        except Exception as e:
            log.warning("task %s cleanup failed: %s", self.id, e)

    def drop_result(self):
        p, self.result_path = self.result_path, None
        self.result = None
        if p:
            try:
                os.unlink(p)
            except Exception:
                pass

    def meter_headers(self):
        out = {}
        if self.input_seconds is not None:
            out[INPUT_SECONDS_HEADER] = "%.3f" % self.input_seconds
        if self.output_seconds is not None:
            out[OUTPUT_SECONDS_HEADER] = "%.3f" % self.output_seconds
        return out

    def doc(self):
        d = {"object": "task", "id": self.id, "kind": KIND,
             "cap": self.cap, "model": self.model, "status": self.status,
             "created": round(self.created, 3),
             "started": round(self.started, 3) if self.started else None,
             "finished": round(self.finished, 3) if self.finished else None,
             "progress": self.progress or None,
             "result_kind": self.result_kind,
             "result": self.result if self.result_kind == "json" else None,
             "error": self.error}
        if self.result_kind == "binary":
            d["content_type"] = self.content_type
            d["result_bytes"] = self.result_bytes
        # Omitted rather than zeroed when unmeasured: a caller billing on these has to be able
        # to tell "no audio" from "this cap does not report it".
        if self.input_seconds is not None:
            d["input_duration_seconds"] = self.input_seconds
        if self.output_seconds is not None:
            d["output_duration_seconds"] = self.output_seconds
        # Always the contract path, even when the caller arrived on the legacy alias.
        d["poll"] = "%s/%s" % (TASKS_PATH, self.id)
        d["result_url"] = "%s/%s/result" % (TASKS_PATH, self.id)
        return d


def _sweep_stale():
    # /tmp is a host volume in the charts, so a crash used to strand uploads and results for good.
    root = tempfile.gettempdir()
    cutoff = time.time() - max(TTL_S, 300.0)
    n = 0
    for name in os.listdir(root):
        if not name.startswith(("task-", "upload-")):
            continue
        p = os.path.join(root, name)
        try:
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.unlink(p)
                n += 1
        except Exception:
            pass
    if n:
        log.info("swept %d file(s) a previous run left behind in %s", n, root)


class _Runner:
    # One worker, because one instance owns one model on one (time-sliced) GPU.
    def __init__(self):
        self._q = queue.Queue()
        self._tasks = {}
        self._lock = threading.Lock()
        self._started = False
        self.running_id = None

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        _sweep_stale()
        threading.Thread(target=self._work, name="audio-task-worker", daemon=True).start()
        threading.Thread(target=self._gc, name="audio-task-gc", daemon=True).start()
        log.info("task runner up (queue_max=%d, ttl=%.0fs)", QUEUE_MAX, TTL_S)

    def submit(self, task):
        self.start()
        with self._lock:
            if sum(1 for t in self._tasks.values() if t.status == "queued") >= QUEUE_MAX:
                raise Busy("queue is full")
            self._tasks[task.id] = task
        self._q.put(task)
        return task

    def get(self, tid):
        return self._tasks.get(tid)

    def all(self):
        return [t for t in list(self._tasks.values()) if t.keep]

    def capacity(self):
        # Counts sync tasks too: they hold a queue slot and are what submit() checks against.
        queued = sum(1 for t in list(self._tasks.values()) if t.status == "queued")
        return {"queued": queued, "running": 1 if self.running_id else 0,
                "limit": QUEUE_MAX, "accepting": queued < QUEUE_MAX}

    def queue_position(self, task):
        if task.status != "queued":
            return None
        n = 0
        for t in list(self._tasks.values()):
            if t is task:
                break
            if t.status == "queued":
                n += 1
        return n

    def cancel(self, task):
        if task.status in _TERMINAL:
            return False
        task.cancel = True
        if task.status == "queued":
            # Not the worker's turn yet, so finish it here rather than hold the upload for hours.
            task.status = "canceled"
            task.finished = time.time()
            task.error = {"code": 499, "message": "canceled before it started"}
            task.run_cleanup()
            if not task.future.done():
                task.future.set_exception(Cancelled("canceled before it started"))
        return True

    def forget(self, task):
        task.drop_result()
        self._tasks.pop(task.id, None)

    def _work(self):
        while True:
            task = self._q.get()
            if task.status == "canceled":
                continue
            task.status = "running"
            task.started = time.time()
            self.running_id = task.id
            try:
                if task.gate is None:
                    payload = task.work(_Ctx(task))
                else:
                    with task.gate:
                        payload = task.work(_Ctx(task))
            except Cancelled as e:
                task.error = {"code": 499, "message": str(e)}
                self._settle(task, "canceled", exc=e)
            except BaseException as e:
                code = getattr(e, "status_code", 500) or 500
                task.error = {"code": code, "message": getattr(e, "detail", None) or str(e)}
                # 5xx is ours and its message is often unplaceable without the frame; 4xx is not.
                log.warning("task %s (%s) failed: %s", task.id, task.cap, e, exc_info=code >= 500)
                self._settle(task, "failed", exc=e)
            else:
                if task.keep:
                    self._store(task, payload)
                self._settle(task, "succeeded", payload=payload)
            finally:
                self.running_id = None

    def _settle(self, task, status, payload=None, exc=None):
        task.finished = time.time()
        task.run_cleanup()
        # Flipped last: a client that sees a terminal status must see a complete document.
        task.status = status
        if not task.future.done():
            if exc is not None:
                task.future.set_exception(exc)
            else:
                task.future.set_result(payload)
        if not task.keep:
            # The sync caller has it off the future; keeping a second copy would only leak.
            self._tasks.pop(task.id, None)
        elif status == "succeeded":
            log.info("task %s (%s) done in %.1fs", task.id, task.cap,
                     task.finished - (task.started or task.finished))

    def _store(self, task, payload):
        if isinstance(payload, (dict, list)):
            task.result, task.result_kind = payload, "json"
            return
        data, media, headers, suffix = _binary_parts(payload)
        task.result_kind = "binary"
        task.content_type = media
        task.result_bytes = len(data)
        headers.update(task.meter_headers())
        task.headers = headers
        with tempfile.NamedTemporaryFile(prefix="task-", suffix=suffix, delete=False) as f:
            f.write(data)
            task.result_path = f.name

    def _gc(self):
        while True:
            time.sleep(_GC_EVERY_S)
            cutoff = time.time() - TTL_S
            for t in list(self._tasks.values()):
                if t.status in _TERMINAL and (t.finished or 0) < cutoff:
                    log.info("task %s expired after %.0fs", t.id, TTL_S)
                    self.forget(t)


_runner = _Runner()


def _binary_parts(payload):
    # Binary results arrive as our Binary, or as a starlette Response (whisper's response_format=text).
    if isinstance(payload, Binary):
        return payload.data, payload.media_type, payload.headers, payload.suffix
    body = getattr(payload, "body", None)
    if body is None:
        raise TypeError("a job must return dict/list, Binary, or a Response, not %s"
                        % type(payload).__name__)
    media = getattr(payload, "media_type", None) or "application/octet-stream"
    suffix = ".txt" if media.startswith("text/") else ""
    return bytes(body), media, {}, suffix


def to_response(payload, headers=None):
    # The sync path answers exactly what the job returned, so it stays byte-identical.
    if isinstance(payload, Binary):
        from fastapi.responses import Response

        merged = dict(payload.headers)
        merged.update(headers or {})
        return Response(content=payload.data, media_type=payload.media_type,
                        headers=merged or None)
    if not headers:
        return payload
    if isinstance(payload, (dict, list)):
        # Wrapped only to carry the headers; the body is what the job returned.
        from fastapi.responses import JSONResponse

        return JSONResponse(content=payload, headers=headers)
    for name, value in headers.items():
        payload.headers[name] = value
    return payload


async def dispatch(async_flag, cap, model, work, *, cleanup=None, fail="job failed", gate=None):
    """Run `work(ctx)` on the engine's worker: awaited when sync, or handed back as a task.

    `work` returns a JSON-able dict/list, a Binary, or a Response. It may call
    ctx.progress(...) and ctx.checkpoint() — both no-ops on the sync path.
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    want = truthy(async_flag)
    task = Task(cap, model, work, cleanup=cleanup, keep=want, gate=gate)
    try:
        _runner.submit(task)
    except Busy:
        if cleanup:
            task.run_cleanup()
        raise HTTPException(status_code=503,
                            detail="engine is busy: %d jobs already queued" % QUEUE_MAX)
    if want:
        return JSONResponse(status_code=202, content={"task": _doc(task)})
    try:
        payload = await asyncio.wrap_future(task.future)
    except HTTPException:
        raise
    except Cancelled as e:
        raise HTTPException(status_code=499, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="%s: %s" % (fail, e))
    return to_response(payload, task.meter_headers())


def _doc(t):
    """The task doc as every route answers it: `queue_position` only while it means something."""
    d = t.doc()
    pos = _runner.queue_position(t)
    if pos is not None:
        d["queue_position"] = pos
    return d


def _list_params(status, limit):
    """The contract's two list parameters, or a 400 naming the one that is wrong."""
    from fastapi import HTTPException

    if status in (None, ""):
        status = None
    elif status not in _STATUSES:
        raise HTTPException(status_code=400, detail="unknown status %r: expected %s" % (
            status, "|".join(_STATUSES)))
    if limit in (None, ""):
        return status, LIST_LIMIT_DEFAULT
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = 0
    if not 1 <= n <= LIST_LIMIT_MAX:
        raise HTTPException(status_code=400, detail="invalid limit %r: expected an integer in 1..%d"
                            % (limit, LIST_LIMIT_MAX))
    return status, n


def _select(tasks, status, limit):
    """Filter, then spend `limit` on finished tasks only, newest kept.

    Queued and running always survive: their count is already bounded by the queue, and a client
    reading its own `queue_position` must not lose the row to jobs that finished while it waited.
    Returns the rows oldest first, plus whether the limit dropped any (the list's `truncated`).
    """
    rows = [t for t in tasks if status is None or t.status == status]
    live = [t for t in rows if t.status not in _TERMINAL]
    done = [t for t in rows if t.status in _TERMINAL]
    room = max(limit - len(live), 0)
    kept = live + done[max(len(done) - room, 0):]
    return sorted(kept, key=lambda t: t.created), len(done) > room


def mount(app):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    # At boot, not on the first request, so a restart reclaims what the last run stranded.
    _runner.start()

    def _need(tid):
        t = _runner.get(tid)
        if t is None or not t.keep:
            raise HTTPException(status_code=404, detail="no such task: %s" % tid)
        return t

    async def list_tasks(status: str = None, limit: str = None):
        # Both parameters arrive as strings so a bad one answers 400 (the contract's code) rather
        # than FastAPI's own 422 for a failed int coercion.
        want, cap = _list_params(status, limit)
        rows, truncated = _select(_runner.all(), want, cap)
        out = {"object": "list", "running": _runner.running_id,
               "capacity": _runner.capacity(),
               "data": [_doc(t) for t in rows]}
        if truncated:
            out["truncated"] = True
        return out

    async def get_task(tid: str):
        return _doc(_need(tid))

    async def get_result(tid: str):
        t = _need(tid)
        if t.status != "succeeded":
            raise HTTPException(status_code=409, detail="task is %s%s" % (
                t.status, ": " + str((t.error or {}).get("message")) if t.error else ""))
        if t.result_kind == "json":
            return JSONResponse(content=t.result, headers=t.meter_headers() or None)
        if not (t.result_path and os.path.isfile(t.result_path)):
            raise HTTPException(status_code=410, detail="the result was already dropped")
        return FileResponse(t.result_path, media_type=t.content_type, headers=t.headers or None)

    def _dropped(t):
        _runner.forget(t)
        return {"id": t.id, "status": t.status, "dropped": True}

    async def del_task(tid: str):
        t = _need(tid)
        if t.status in _TERMINAL:
            return _dropped(t)
        # cancel() says False when the worker settled the task between the check above and the
        # call. It is the terminal case after all: reporting `canceling` would promise a stop
        # that already cannot happen, and would leave the result for the GC to collect.
        if not _runner.cancel(t):
            return _dropped(t)
        return {"id": tid, "status": t.status, "canceling": True}

    # One handler per route, reachable under both bases, so the alias can never drift.
    for base in (TASKS_PATH, LEGACY_PATH):
        app.get(base)(list_tasks)
        app.get(base + "/{tid}")(get_task)
        app.get(base + "/{tid}/result")(get_result)
        app.delete(base + "/{tid}")(del_task)

    return app
