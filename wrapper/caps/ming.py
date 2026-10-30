# Ming HTTP is the shared ElevenLabs and OpenAI surface. The checkpoint stays in ming_backend.
import logging
import os

from ..runtime import Runtime
from . import tts_el
from .ming_backend import MingBackend, load_model

log = logging.getLogger("audio-ming")

_runtime = Runtime(model=None, backend=None, library=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def _load():
    if not MODEL_REPO:
        raise RuntimeError("MODEL_SOURCE is empty")
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)
    log.info("loading %s from %s", MODEL_NAME or MODEL_REPO, path)
    backend = MingBackend(load_model(path))
    backend.speak_native({"voice_id": "US_James", "ip": "US_James",
                           "labels": {"language": "en"}}, "Hello.")
    return backend


def build_app(supports):
    return tts_el.build_app(supports, module="ming", openai_alias=True, task_legacy=True)


def run(supports):
    tts_el.run(supports, module="ming", watchdog="Ming-3.0-tts", load=_load,
               sample_rate=24000, openai_alias=True, task_legacy=True)
