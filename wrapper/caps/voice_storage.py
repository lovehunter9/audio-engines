"""Serialized voice-library updates with complete publication of new cards."""
import fcntl
import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

_mutex = threading.RLock()
_held = threading.local()


@contextmanager
def locked(root):
    root = Path(root).resolve()
    with _mutex:
        active = getattr(_held, "roots", None)
        if active is None:
            active = _held.roots = set()
        if root in active:
            yield
            return
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".voices.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            active.add(root)
            try:
                yield
            finally:
                active.remove(root)
                fcntl.flock(handle, fcntl.LOCK_UN)


def serialized(method):
    @wraps(method)
    def call(store, voice_id, *args, **kwargs):
        with locked(store._dir(voice_id).parent):
            return method(store, voice_id, *args, **kwargs)
    return call


def _replace(path, data):
    fd, scratch = tempfile.mkstemp(prefix=".voice-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, path)
    finally:
        if os.path.exists(scratch):
            os.unlink(scratch)


def write_voice(target, meta, wav_bytes=None, transcript=None):
    target = Path(target)
    with locked(target.parent):
        fresh = not target.exists()
        directory = Path(tempfile.mkdtemp(prefix=".voice-", dir=target.parent)) if fresh else target
        try:
            if wav_bytes is not None:
                _replace(directory / "prompt.wav", wav_bytes)
            if transcript is not None:
                _replace(directory / "prompt.txt", transcript)
            _replace(directory / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
            if fresh:
                os.replace(directory, target)
        finally:
            if fresh and directory.exists():
                shutil.rmtree(directory)
