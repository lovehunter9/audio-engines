# One flock on the shared hub cache around the first OpenVINO GPU compile. CUDA and CPU do not take it.
import contextlib
import fcntl
import os

_LOCK_NAME = "ov-gpu-compile.lock"


def uses_gpu(device):
    """True when this compile targets a GPU. An empty device counts: callers here are already on the OpenVINO GPU path."""
    return not (device or "").strip().upper().startswith("CPU")


def _lock_path():
    """The lock lives on the hub cache. Charts mount the shared hostPath there, not at HF_HOME."""
    shared = (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    if not shared:
        root = (os.environ.get("HF_HOME") or "/cache/hf").strip() or "/cache/hf"
        shared = os.path.join(root, "hub")
    return os.path.join(shared, _LOCK_NAME)


@contextlib.contextmanager
def gpu_compile(device):
    """Hold the shared compile lock for one GPU compile. A missing cache directory is logged and the compile still runs."""
    if not uses_gpu(device):
        yield
        return
    path = _lock_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as e:
        print("[ov-compile-lock] unavailable (%s); compiling without it" % e, flush=True)
        yield
        return
    print("[ov-compile-lock] waiting for %s" % path, flush=True)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        print("[ov-compile-lock] holding %s" % path, flush=True)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            print("[ov-compile-lock] released %s" % path, flush=True)
    finally:
        os.close(fd)
