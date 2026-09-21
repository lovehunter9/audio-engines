# Shared OpenVINO helpers for ov-base caps (Intel iGPU + Arc share one code path).
import os

from .contract import EngineArgs

_args = EngineArgs()
OV_DEVICE = _args.text("--device", "")

_OV_LANG = {
    "en": "English", "zh": "Chinese", "yue": "Chinese", "ja": "Japanese",
    "ko": "Korean", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "english": "English", "chinese": "Chinese", "japanese": "Japanese",
    "korean": "Korean", "german": "German", "french": "French",
    "spanish": "Spanish", "italian": "Italian", "portuguese": "Portuguese",
    "russian": "Russian", "auto": "English",
}


def is_ov():
    """The ov image bakes AUDIO_BASE=ov. Never infer Intel from visible hardware."""
    return (os.environ.get("AUDIO_BASE") or "").strip() == "ov"


def device():
    if OV_DEVICE:
        return OV_DEVICE
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel"):
        return "GPU"
    gpu_raw = (os.environ.get("REQUIRED_GPU_MEMORY") or "").strip()
    if gpu_raw in ("", "0"):
        return "CPU"
    return "GPU"


def language(raw):
    if not raw:
        return "English"
    key = str(raw).strip()
    return _OV_LANG.get(key.lower(), key)
