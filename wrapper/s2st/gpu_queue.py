import asyncio
from contextlib import asynccontextmanager, contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

DEFAULT = None
SHARED_UPSTREAM = False
MING_PROTECTION_SECONDS = 20.0


def ming_priority_age(path=None):
    selected = path if path is not None else DEFAULT
    if selected is None:
        return None
    try:
        entries = json.loads(Path(selected).with_suffix(".json").read_text())
    except FileNotFoundError:
        return None
    now = time.monotonic()
    ages = [now - x["enqueued"] for x in entries
            if x["label"] == "ming" and x["start"] is not None
            and identity(x["pid"]) == x["start"]]
    age = max(ages, default=0.0)
    return age if age >= MING_PROTECTION_SECONDS else None


def configure(path, *, shared_upstream=False):
    global DEFAULT, SHARED_UPSTREAM
    DEFAULT = Path(path)
    SHARED_UPSTREAM = bool(shared_upstream)


def identity(pid):
    try:
        s = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if s[0] == "Z" else s[19]
    except FileNotFoundError:
        try:
            os.kill(pid, 0)
            return "alive" if not Path("/proc").exists() else None
        except ProcessLookupError:
            return None


class Ticket:
    def __init__(self, label, path=None):
        selected = path if path is not None else DEFAULT
        if selected is None:
            raise RuntimeError("GPU stage queue has not been configured")
        self.path = Path(selected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.token = uuid.uuid4().hex
        self.shared = SHARED_UPSTREAM and label in ("r2t2", "t3po")
        self.entry = {
            "token": self.token,
            "pid": os.getpid(),
            "start": identity(os.getpid()),
            "label": label,
            "shared": self.shared,
            "enqueued": time.monotonic(),
        }
        self.fd = (self.path.with_suffix(".gpu")).open("a")
        self.acquired = False
        self.acquired_at = None
        self.queued_at = time.monotonic()
        self.update(lambda q: q.append(self.entry))

    def update(self, action):
        with self.path.with_suffix(".metadata-lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            queue_path = self.path.with_suffix(".json")
            previous = json.loads(queue_path.read_text()) if queue_path.exists() else []
            queue = previous.copy()
            queue = [
                x for x in queue if identity(x["pid"]) == x["start"] and x["start"] is not None
            ]
            result = action(queue)
            if queue != previous or not queue_path.exists():
                temp = queue_path.with_suffix(".tmp")
                temp.write_text(json.dumps(queue))
                temp.replace(queue_path)
            return result

    def try_acquire(self):
        def claim(queue):
            position = next((i for i, row in enumerate(queue) if row["token"] == self.token), None)
            if position is None:
                return False
            if position and (not self.shared or any(not row.get("shared", False) for row in queue[:position])):
                return False
            try:
                mode = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
                fcntl.flock(self.fd, mode | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            self.acquired = True
            self.acquired_at = time.monotonic()
            return True

        return self.update(claim)

    def close(self):
        try:
            if self.acquired:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.update(
                lambda q: q.__setitem__(slice(None), [x for x in q if x["token"] != self.token])
            )
        finally:
            self.fd.close()
            ended = time.monotonic()
            event = dict(
                self.entry,
                finished=ended,
                acquired=self.acquired,
                wait_seconds=(self.acquired_at or ended) - self.queued_at,
                hold_seconds=ended - self.acquired_at if self.acquired_at is not None else 0,
            )
            with self.path.with_suffix(".events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")


@contextmanager
def stage(label, path=None):
    ticket = Ticket(label, path)
    try:
        while not ticket.try_acquire():
            time.sleep(0.005)
        yield time.monotonic() - ticket.queued_at
    finally:
        ticket.close()


@asynccontextmanager
async def async_stage(label, path=None):
    ticket = Ticket(label, path)
    try:
        while not ticket.try_acquire():
            await asyncio.sleep(0.005)
        yield time.monotonic() - ticket.queued_at
    finally:
        ticket.close()
