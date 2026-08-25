import logging
import os
import threading

import uvicorn

from . import watchdog


class Runtime:
    def __init__(self, **state):
        # Chart MODEL_NAME / MODEL_SOURCE only. Empty env stays empty: this process never
        # invents an id or a repo (e.g. Qwen3-TTS CustomVoice, Voxtral-4B-TTS, etc.).
        self.model_name = (os.environ.get("MODEL_NAME") or "").strip()
        # MODEL_SOURCE may list several repos; the first is what this engine serves.
        source = (os.environ.get("MODEL_SOURCE", "").split(",")[0] or "").strip()
        self.model_repo = source[5:] if source.startswith("hf://") else source
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
