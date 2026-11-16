"""Expose only arrived PCM to the author's unchanged prefix generator."""

import threading
import numpy as np


class LiveWave:
    def __init__(self, maximum_samples=16000 * 300, idle_timeout=30):
        self._condition = threading.Condition()
        self._maximum = maximum_samples
        self._storage = np.empty(min(16000, maximum_samples), np.float32)
        self._count = 0
        self._required = 1
        self._step = None
        self._finished = False
        self._error = None
        self._idle_timeout = idle_timeout

    def configure(self, chunk_samples, lookahead_samples):
        if chunk_samples <= 0 or lookahead_samples < 0:
            raise ValueError("Invalid original prefix schedule")
        with self._condition:
            self._step = chunk_samples
            self._required = chunk_samples + lookahead_samples + 1

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("Non-finite source PCM")
        with self._condition:
            if self._finished or self._error is not None:
                raise ValueError("Source session has ended")
            size = self._count + len(samples)
            if size > self._maximum:
                raise ValueError("Source session exceeds the supported duration")
            if size > len(self._storage):
                storage = np.empty(
                    min(self._maximum, max(size, 2 * len(self._storage))), np.float32
                )
                storage[: self._count] = self._storage[: self._count]
                self._storage = storage
            self._storage[self._count : size] = samples
            self._count = size
            self._condition.notify_all()

    def finish(self):
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def cancel(self, error=None):
        with self._condition:
            self._error = error or RuntimeError("Source session disconnected")
            self._condition.notify_all()

    def __len__(self):
        with self._condition:
            while self._count < self._required and not self._finished and self._error is None:
                if not self._condition.wait(self._idle_timeout):
                    raise TimeoutError("Source PCM did not arrive before the idle deadline")
            if self._error is not None:
                raise self._error
            return self._count

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.start not in (None, 0) or key.step not in (None, 1):
            raise ValueError("The original runner must request contiguous source prefixes")
        with self._condition:
            if self._error is not None:
                raise self._error
            stop = self._count if key.stop is None else int(key.stop)
            if stop > self._count or stop < 0:
                raise ValueError("Source prefix includes unavailable audio")
            if self._step is None:
                raise RuntimeError("The original prefix schedule has not been installed")
            self._required = max(self._required, stop + self._step + 1)
            return self._storage[:stop].copy()

    def snapshot(self):
        with self._condition:
            if not self._finished:
                raise RuntimeError("Source PCM is not complete")
            return self._storage[: self._count].copy()


class StreamInput:
    def __init__(self):
        import queue

        self.packets = queue.Queue()
        self.samples = 0
        self.ended = False

    def append(self, data):
        if self.ended:
            raise ValueError("Source session has ended")
        if len(data) % 2:
            raise ValueError("PCM frames must contain little-endian int16 samples")
        self.samples += len(data) // 2
        if self.samples > 16000 * 300:
            raise ValueError("Source session exceeds 300 seconds")
        self.packets.put(data)

    def finish(self):
        if not self.samples:
            raise ValueError("Source PCM is empty")
        self.ended = True
        self.packets.put(None)

    def cancel(self):
        self.ended = True
        self.packets.put(RuntimeError("Source session disconnected"))

    def read(self):
        import queue

        try:
            result = self.packets.get(timeout=30)
        except queue.Empty as error:
            raise TimeoutError("Source PCM idle timeout") from error
        if isinstance(result, BaseException):
            raise result
        return result


def observe_pcm(pcm, callback):
    original = pcm.decode

    def decode(*args, **kwargs):
        before = len(pcm.waves)
        result = original(*args, **kwargs)
        for samples in pcm.waves[before:]:
            data = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            if data:
                callback("audio", data)
        return result

    pcm.decode = decode
    return original


def mount_stream(app, runtime, lock, language):
    import asyncio
    import json
    import anyio
    from fastapi import HTTPException, WebSocket

    @app.websocket("/v1/audio/translations/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        incoming = None
        receiver = worker = None
        locked = False
        try:
            if not runtime.state["ready"]:
                raise RuntimeError(runtime.state["error"] or "engine not ready")
            engine = runtime.state["engine"]
            locked = lock.acquire(blocking=False)
            if not locked:
                raise RuntimeError("Another translation session is active")
            await ws.send_json({"event": "ready", "input_sample_rate": 16000, "sample_rate": 24000})
            config = await asyncio.wait_for(ws.receive_json(), 30)
            if not isinstance(config, dict) or config.get("sample_rate", 16000) != 16000:
                raise ValueError("Source PCM must use 16000 Hz")
            source, target = (
                language(config.get("source_lang", "English")),
                language(config.get("target_lang", "Chinese")),
            )
            if source == target:
                raise ValueError("source_lang and target_lang must differ")
            incoming = StreamInput()
            loop = asyncio.get_running_loop()
            outputs = asyncio.Queue()

            def emit(kind, value):
                loop.call_soon_threadsafe(outputs.put_nowait, (kind, value))

            async def receive():
                try:
                    while True:
                        message = await asyncio.wait_for(ws.receive(), 30)
                        if message["type"] == "websocket.disconnect":
                            raise RuntimeError("Source session disconnected")
                        if message.get("bytes") is not None:
                            incoming.append(message["bytes"])
                        elif message.get("text") is not None:
                            packet = json.loads(message["text"])
                            if not isinstance(packet, dict) or packet.get(
                                "event", packet.get("type")
                            ) not in ("end", "stop", "finish", "done"):
                                raise ValueError("Expected source PCM or an end event")
                            incoming.finish()
                            return
                except Exception as error:
                    incoming.cancel()
                    emit("error", str(error))

            async def infer():
                try:
                    await asyncio.to_thread(engine.translate_live, incoming, source, target, emit)
                    emit("done", None)
                except Exception as error:
                    emit("error", str(error))

            receiver = asyncio.create_task(receive())
            worker = asyncio.create_task(infer())
            while True:
                kind, value = await outputs.get()
                if kind == "audio":
                    await ws.send_bytes(value)
                elif kind == "text":
                    await ws.send_json({"text": value, "finished": False})
                elif kind == "done":
                    await ws.send_json({"text": "", "finished": True, "sample_rate": 24000})
                    break
                else:
                    raise RuntimeError(value)
        except Exception as error:
            try:
                await ws.send_json(
                    {"error": error.detail if isinstance(error, HTTPException) else str(error)}
                )
            except RuntimeError:
                pass
        finally:
            with anyio.CancelScope(shield=True):
                if incoming is not None:
                    incoming.cancel()
                if receiver is not None and not receiver.done():
                    receiver.cancel()
                try:
                    if worker is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(worker), 90)
                        except asyncio.TimeoutError:
                            runtime.state.update(ready=False, error="Source session cancellation timed out")
                            await asyncio.to_thread(runtime.state["engine"].close)
                finally:
                    if locked:
                        lock.release()
                try:
                    await ws.close()
                except RuntimeError:
                    pass
