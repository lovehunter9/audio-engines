"""S3 streaming translation with all neural inference on OpenVINO GPU."""
import logging
import threading

from .. import hfgate
from ..contract import EngineArgs
from ..runtime import Runtime
from .s3_s2st import build_app_for

log = logging.getLogger("audio-s3-s2st-ov")
_runtime = Runtime(engine=None)
_state = _runtime.state
_args = EngineArgs()
_PRESET = _args.text("--preset", "adaptive")
_lock = threading.Lock()


def _boot():
    try:
        from huggingface_hub import snapshot_download
        from .s3_ov_engine import OpenVINOEngine
        repo, include, exclude = _runtime._hub_spec()
        snapshot = snapshot_download(repo, local_files_only=True, allow_patterns=include, ignore_patterns=exclude)
        if _PRESET not in ("adaptive", "fixed-0.1", "fixed-0.5"):
            raise ValueError("--preset must be adaptive, fixed-0.1, or fixed-0.5")
        _state.update(engine=OpenVINOEngine(snapshot, _PRESET), ready=True, error=None)
    except Exception as error:
        _state["error"] = hfgate.explain(_runtime.model_repo, error)
        log.exception("S3 OpenVINO runtime failed to initialize")


def build_app(supports):
    _args.warn_unclaimed(log)
    return build_app_for(_runtime, _args, supports, "s3_s2st_ov", _lock)


def run(supports):
    _runtime.serve(supports, _boot, build_app, "S3 OpenVINO GPU", timeout_s=3600, disable_ws_ping=True)
