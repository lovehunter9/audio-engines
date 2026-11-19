"""Run the official Tencent CLI entry beside the unchanged MING implementation."""

import argparse
import asyncio
import ctypes
import os
from pathlib import Path
import threading
import time

from fastapi import HTTPException


class Native:
    def __init__(self, library, model, root):
        self.library = ctypes.CDLL(str(library))
        self.call = self.library.audio_s2st_run
        self.call.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ]
        self.call.restype = ctypes.c_int
        self.model = Path(model)
        self.root = Path(root).resolve()
        self.directory = None
        self.thread = None
        self.code = None
        self.broken = None

    def start(self, body):
        from .tencent import command

        if self.broken:
            raise HTTPException(503, self.broken)
        if self.thread is not None and self.thread.is_alive():
            raise HTTPException(409, "native session is still active")
        directory = Path(body["directory"]).resolve()
        if directory.name != "native" or directory.parent.parent != self.root:
            raise HTTPException(400, "native session must use the owned cache directory")
        argv = body["command"]
        allowed = [command(self.model, d, "/opt/llama/bin/llama-mtmd-cli") for d in ("en2zh", "zh2en")]
        if argv not in allowed:
            raise HTTPException(400, "native arguments differ from the validated policy")
        pcm = directory / "input.pcm"
        import stat
        if not stat.S_ISFIFO(pcm.stat().st_mode):
            raise HTTPException(400, "native input must be the session PCM pipe")
        self.directory = directory
        self.code = None
        values = (ctypes.c_char_p * len(argv))(*(os.fsencode(v) for v in argv))

        def run():
            try:
                self.code = self.call(
                    len(argv), values, os.fsencode(pcm),
                    os.fsencode(directory / "translation.tsv"),
                    os.fsencode(directory / "translation.release"),
                )
            except Exception as error:
                self.broken = "native entry failed: " + str(error)
                self.code = -12

        self.thread = threading.Thread(target=run, name="official-tencent", daemon=True)
        self.thread.start()
        return {"started": True, "gpu_pid": os.getpid()}

    def status(self):
        return {"running": self.thread is not None and self.thread.is_alive(),
                "returncode": self.code, "error": self.broken, "gpu_pid": os.getpid()}

    async def close(self, directory):
        if self.directory is None:
            return self.status()
        if Path(directory).resolve() != self.directory:
            raise HTTPException(409, "native session ownership mismatch")
        (self.directory / "translation.release").touch()
        deadline = time.monotonic() + 60
        while self.thread.is_alive() and time.monotonic() < deadline:
            # Also unblock a FIFO open if the public client disconnected during startup.
            try:
                descriptor = os.open(self.directory / "input.pcm", os.O_WRONLY | os.O_NONBLOCK)
                os.close(descriptor)
            except OSError:
                pass
            await asyncio.sleep(0.01)
        if self.thread.is_alive():
            self.broken = "native cleanup exceeded its deadline"
            raise HTTPException(503, self.broken)
        result = self.status()
        self.directory = None
        if self.code != 0:
            raise HTTPException(500, "native entry failed: " + str(self.code))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--translation-model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument("--port", type=int, default=18091)
    options = parser.parse_args()
    from .ming_worker import build

    app = build(options)
    native = Native("/opt/llama/lib/libaudio-s2st-native.so", options.translation_model,
                    options.cache.parent)

    @app.get("/internal/tencent/health")
    async def health():
        if native.broken:
            raise HTTPException(503, native.broken)
        return dict(status="ready", gpu_pid=os.getpid(), inference="official CLI shared entry")

    @app.post("/internal/tencent/start")
    async def start(body: dict):
        return native.start(body)

    @app.get("/internal/tencent/status")
    async def status():
        return native.status()

    @app.post("/internal/tencent/close")
    async def close(body: dict):
        return await native.close(body["directory"])

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=options.port, log_level="info")


if __name__ == "__main__":
    main()
