"""Own the official ASR, native translation and unchanged MING in one GPU PID."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import threading
import time
import urllib.request


class NativeServer:
    def __init__(self, library, command):
        self.library = ctypes.CDLL(str(library))
        self.run = self.library.audio_t3po_run
        self.run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.run.restype = ctypes.c_int
        self.stop = self.library.audio_t3po_stop
        self.stop.argtypes = []
        self.stop.restype = None
        self.code = None
        values = (ctypes.c_char_p * len(command))(*(os.fsencode(s) for s in command))

        def invoke():
            self.code = self.run(len(command), values)

        self.thread = threading.Thread(target=invoke, name="official-t3po-server", daemon=True)

    def start(self):
        self.thread.start()

    def assert_alive(self):
        if not self.thread.is_alive():
            raise RuntimeError("Official T3PO server exited: " + str(self.code))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--translation-model", type=Path, required=True)
    parser.add_argument("--recognition-model", type=Path, required=True)
    parser.add_argument("--vad-model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument("--asr-chunk-ms", type=int, default=2000)
    parser.add_argument("--asr-chunk-ms-zh", type=int)
    parser.add_argument("--port", type=int, default=18091)
    options = parser.parse_args()
    if options.asr_chunk_ms not in (800, 1600, 2000, 4000):
        raise ValueError("Unsupported ASR delivery increment")
    if options.asr_chunk_ms_zh is not None and options.asr_chunk_ms_zh not in (800, 1600, 2000, 4000):
        raise ValueError("Unsupported Chinese ASR delivery increment")
    root = options.root.resolve()
    (root / "results").mkdir(parents=True, exist_ok=True)
    from .gpu_queue import configure

    configure(root / "gpu-stage-fifo")
    for folder in ["sources", "models"]:
        (root / folder).mkdir(exist_ok=True)
    links = {
        root / "sources/Confucius4-R2T2": Path("/opt/youdao/r2t2"),
        root / "sources/Confucius4-T3PO": Path("/opt/youdao/t3po"),
        root / "models/R2T2-HF": options.recognition_model,
        root / "models/Stream-VAD": options.vad_model,
    }
    for path, target in links.items():
        target = target.resolve()
        if not target.is_dir():
            raise ValueError("Required local snapshot is absent: " + str(target))
        if path.is_symlink() and path.resolve() != target:
            path.unlink()
        if not path.exists():
            path.symlink_to(target, target_is_directory=True)
        if path.resolve() != target:
            raise ValueError("Owned model link has an unexpected target: " + str(path))
    from .ming_worker import build

    app = build(options)
    from .config import YOUDAO_TRANSLATION

    command = [
        "/opt/llama/bin/llama-server",
        *YOUDAO_TRANSLATION.arguments(options.translation_model),
        "--host",
        "127.0.0.1",
        "--port",
        "18092",
        "--alias",
        "Confucius4-T3PO",
        "--metrics",
        "-lv",
        # The official core loader's INFO callback uses verbosity 4.
        {"debug": "5", "info": "4", "warn": "2", "error": "1"}.get(
            os.environ.get("LOG_LEVEL", "info").lower(), "4"
        ),
    ]
    native = NativeServer("/opt/llama/lib/libaudio-t3po-entry.so", command)
    native.start()
    for _ in range(1200):
        native.assert_alive()
        try:
            with urllib.request.urlopen("http://127.0.0.1:18092/health", timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        raise TimeoutError("Native T3PO initialization")
    from .r2t2_worker import serve

    ready = threading.Event()
    errors = []

    def asr_entry():
        try:
            serve(
                argparse.Namespace(
                    root=str(root),
                    port=18094,
                    backend="transformers",
                    memory_fraction=0.20,
                    context=4096,
                    chunk_ms=options.asr_chunk_ms,
                    chinese_chunk_ms=options.asr_chunk_ms_zh,
                ),
                ready,
            )
        except BaseException as exc:
            errors.append(repr(exc))
            (root / "results/asr-thread-error.json").write_text(json.dumps({"error": repr(exc)}))
            os.kill(os.getpid(), signal.SIGTERM)
            raise

    thread = threading.Thread(target=asr_entry, name="official-r2t2", daemon=True)
    thread.start()
    from fastapi import HTTPException

    @app.get("/internal/youdao/health")
    async def health():
        native.assert_alive()
        if errors or not thread.is_alive():
            raise HTTPException(503, errors[-1] if errors else "ASR thread exited")
        if not ready.is_set():
            raise HTTPException(503, "ASR loading and language warmup")
        return dict(status="ready", gpu_pid=os.getpid(), asr_chunk_ms=options.asr_chunk_ms,
                    asr_chunk_ms_zh=options.asr_chunk_ms_zh or options.asr_chunk_ms)

    stop = threading.Event()

    def monitor():
        while not stop.wait(2):
            try:
                native.assert_alive()
                if not thread.is_alive():
                    raise RuntimeError("ASR thread exited")
            except BaseException as exc:
                (root / "results/worker-error.json").write_text(json.dumps({"error": repr(exc)}))
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=monitor, name="resident-health", daemon=True).start()
    import uvicorn

    try:
        uvicorn.run(app, host="127.0.0.1", port=options.port, log_level="info")
    finally:
        stop.set()
        native.stop()
        native.thread.join(15)


if __name__ == "__main__":
    main()
