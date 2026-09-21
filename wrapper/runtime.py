import logging
import os
import threading
import time

import uvicorn

from . import watchdog

log = logging.getLogger("audio-runtime")

# Same ceiling as llm-init's wait_for_sentinel; the load watchdog's 1800s is for reading the snapshot, not download.
_SNAPSHOT_WAIT_S = 3600.0
_SNAPSHOT_POLL_S = 5.0


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

    def _hub_spec(self):
        # Keep --include / --exclude so a partial llm-init fetch is not judged as a missing snapshot.
        tokens = (self.model_repo or "").split()
        if not tokens or "/" not in tokens[0]:
            return "", None, None
        include, exclude = [], []
        i = 1
        while i < len(tokens):
            if tokens[i] in ("--include", "--exclude") and i + 1 < len(tokens):
                (include if tokens[i] == "--include" else exclude).append(tokens[i + 1])
                i += 2
            else:
                i += 1
        return tokens[0], include or None, exclude or None

    def _snapshot_ready(self):
        repo, include, exclude = self._hub_spec()
        if not repo:
            return True
        from huggingface_hub import snapshot_download

        kw = {
            "local_files_only": True,
            "cache_dir": os.environ.get("HF_HUB_CACHE"),
            "token": os.environ.get("HF_TOKEN") or None,
        }
        if include:
            kw["allow_patterns"] = include
        if exclude:
            kw["ignore_patterns"] = exclude
        try:
            snapshot_download(repo, **kw)
            return True
        except Exception:
            return False

    def wait_for_snapshot(self, timeout_s=_SNAPSHOT_WAIT_S, interval_s=_SNAPSHOT_POLL_S):
        repo, _, _ = self._hub_spec()
        t0 = time.time()
        while not self._snapshot_ready():
            elapsed = time.time() - t0
            if elapsed >= timeout_s:
                self.state["waiting"] = None
                self.state["error"] = "timed out waiting for a local snapshot of %s" % repo
                log.error("%s", self.state["error"])
                return False
            self.state["waiting"] = "waiting for a local snapshot of %s" % repo
            log.info("%s (elapsed=%.0fs)", self.state["waiting"], elapsed)
            time.sleep(interval_s)
        self.state["waiting"] = None
        return True

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
        def ready():
            return self.state["ready"]

        def failed():
            if self.state.get("waiting"):
                return None
            return self.state["error"]

        # A base whose first load is legitimately hours long must be able to say so.
        deadline = {} if timeout_s is None else {"timeout_s": timeout_s}

        def boot():
            # Arm after the snapshot is on disk so download time does not eat the load deadline.
            if not self.wait_for_snapshot():
                watchdog.arm(ready, failed, watchdog_name, **deadline)
                return
            watchdog.arm(ready, failed, watchdog_name, **deadline)
            load()

        if load_on_main:
            boot()
        else:
            threading.Thread(target=boot, daemon=True).start()
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
