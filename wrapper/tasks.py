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
TTL_S = float(os.environ.get("TASK_TTL_S", "1800") or 1800)
QUEUE_MAX = int(os.environ.get("TASK_QUEUE_MAX", "32") or 32)
_GC_EVERY_S = 30.0

# Advertised by /v1/models next to the capability's own endpoints.
ENDPOINTS = [
    {"method": "GET", "path": "/v1/audio/tasks",
     "description": "List tasks (oldest first)"},
    {"method": "GET", "path": "/v1/audio/tasks/{id}",
     "description": "Task status / progress / JSON result"},
    {"method": "GET", "path": "/v1/audio/tasks/{id}/result",
     "description": "Task result (audio for enhance, else JSON)"},
    {"method": "DELETE", "path": "/v1/audio/tasks/{id}",
     "description": "Cancel a running task, or drop a finished one's result"},
]

# Appended to a capability's own description so callers can discover the async mode.
ASYNC_HINT = "async=1 -> 202 + task"

_TERMINAL = ("succeeded", "failed", "canceled")


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

    def doc(self, urls=True):
        d = {"id": self.id, "cap": self.cap, "model": self.model, "status": self.status,
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
        if urls:
            d["poll"] = "/v1/audio/tasks/%s" % self.id
            d["result_url"] = "/v1/audio/tasks/%s/result" % self.id
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
                task.error = {"code": getattr(e, "status_code", 500) or 500,
                              "message": getattr(e, "detail", None) or str(e)}
                log.warning("task %s (%s) failed: %s", task.id, task.cap, e)
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


def to_response(payload):
    # The sync path answers exactly what the job returned, so it stays byte-identical.
    if isinstance(payload, Binary):
        from fastapi.responses import Response

        return Response(content=payload.data, media_type=payload.media_type,
                        headers=payload.headers or None)
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
        doc = task.doc()
        doc["queue_position"] = _runner.queue_position(task)
        return JSONResponse(status_code=202, content={"task": doc})
    try:
        payload = await asyncio.wrap_future(task.future)
    except HTTPException:
        raise
    except Cancelled as e:
        raise HTTPException(status_code=499, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="%s: %s" % (fail, e))
    return to_response(payload)


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

    def _doc(t):
        d = t.doc()
        pos = _runner.queue_position(t)
        if pos is not None:
            d["queue_position"] = pos
        return d

    @app.get("/v1/audio/tasks")
    async def list_tasks():
        return {"object": "list", "running": _runner.running_id,
                "data": [_doc(t) for t in _runner.all()]}

    @app.get("/v1/audio/tasks/{tid}")
    async def get_task(tid: str):
        return _doc(_need(tid))

    @app.get("/v1/audio/tasks/{tid}/result")
    async def get_result(tid: str):
        t = _need(tid)
        if t.status != "succeeded":
            raise HTTPException(status_code=409, detail="task is %s%s" % (
                t.status, ": " + str((t.error or {}).get("message")) if t.error else ""))
        if t.result_kind == "json":
            return JSONResponse(content=t.result)
        if not (t.result_path and os.path.isfile(t.result_path)):
            raise HTTPException(status_code=410, detail="the result was already dropped")
        return FileResponse(t.result_path, media_type=t.content_type, headers=t.headers or None)

    @app.delete("/v1/audio/tasks/{tid}")
    async def del_task(tid: str):
        t = _need(tid)
        if t.status in _TERMINAL:
            _runner.forget(t)
            return {"id": tid, "status": t.status, "dropped": True}
        _runner.cancel(t)
        return {"id": tid, "status": t.status, "canceling": True}

    return app
