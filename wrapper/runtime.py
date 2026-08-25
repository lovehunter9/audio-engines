import json
import logging
import os
import threading

import uvicorn

from . import watchdog

def _served_name(fallback):
    # llm-init v1.5 requires GET /v1/models data[].id to equal the model card
    # name exactly. A chart MODEL_NAME that only differs in case (openbmb vs
    # OpenBMB) makes the engine look ready while the data plane stays 503.
    path = os.environ.get("MODEL_SPEC_PATH") or "/run/llm-init/model-spec.json"
    try:
        with open(path, encoding="utf-8") as f:
            name = (json.load(f) or {}).get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return fallback


class Runtime:
    def __init__(self, default_model, default_repo=None, **state):
        self.model_name = _served_name(os.environ.get("MODEL_NAME", default_model))
        # MODEL_SOURCE may list several repos; the first is what this engine serves. A segment can
        # also carry llm-init's inline flags (--include / --subdir / --revision), which a
        # single-file GGUF out of a many-model repo needs, so the repo is only the first token.
        source = (os.environ.get("MODEL_SOURCE", "").split(",")[0] or "").strip()
        if source.startswith("hf://"):
            tokens = source[5:].split()
            self.model_repo = tokens[0] if tokens else (default_repo or self.model_name)
        else:
            self.model_repo = source or default_repo or self.model_name
        self.port = int(os.environ.get("ENGINE_PORT", "8000"))
        self.log_level = os.environ.get("LOG_LEVEL", "info").lower()
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO)
        )
        self.state = {"ready": False, "error": None}
        self.state.update(state)

    def serve(
        self,
        supports,
        load,
        build_app,
        watchdog_name,
        *,
        load_on_main=False,
        disable_ws_ping=False,
        timeout_s=None,
    ):
        ready = lambda: self.state["ready"]
        failed = lambda: self.state["error"]
        # A base whose first load is legitimately hours long must be able to say so.
        deadline = {} if timeout_s is None else {"timeout_s": timeout_s}
        if load_on_main:
            watchdog.arm(ready, failed, watchdog_name, **deadline)
            load()
        else:
            threading.Thread(target=load, daemon=True).start()
            watchdog.arm(ready, failed, watchdog_name, **deadline)
        app = build_app(supports)
        kwargs = {}
        if disable_ws_ping:
            kwargs.update(ws_ping_interval=None, ws_ping_timeout=None)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self.port,
            log_level=self.log_level,
            **kwargs,
        )
