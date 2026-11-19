import asyncio
from dataclasses import dataclass
import hashlib
import tempfile


@dataclass(frozen=True)
class Stream:
    direction: str
    sample_rate: int = 16000

    def __post_init__(self):
        if self.direction not in ("zh2en", "en2zh"):
            raise ValueError("direction must be zh2en or en2zh")
        if self.sample_rate != 16000:
            raise ValueError("stream input must be mono 16 kHz PCM16")


class Input:
    def __init__(self, stream, capacity=16, max_frame_bytes=51200,
                 spool_directory=None, max_samples=900 * 16000):
        if capacity < 1 or max_frame_bytes < 2 or max_samples < 1:
            raise ValueError("invalid stream limits")
        self.stream = stream
        self.queue = asyncio.Queue(maxsize=capacity)
        self.max_frame_bytes = max_frame_bytes
        self.finished = False
        self.exhausted = False
        self.samples = 0
        self.digest = hashlib.sha256()
        self.lock = asyncio.Lock()
        self.max_samples = max_samples
        self._spool = tempfile.TemporaryFile(dir=spool_directory) if spool_directory is not None else None
        self._written = self._read = 0
        self._available = asyncio.Event()

    async def feed(self, data):
        if not isinstance(data, bytes) or not data or len(data) % 2:
            raise ValueError("audio frames must contain whole PCM16 samples")
        if len(data) > self.max_frame_bytes:
            raise ValueError("audio frame exceeds the input bound")
        async with self.lock:
            if self.finished:
                raise ValueError("audio after end of input")
            if self.samples + len(data) // 2 > self.max_samples:
                raise ValueError("session exceeds the input duration limit")
            if self._spool is None:
                await self.queue.put(data)
            else:
                # Drain valid uploads independently of GPU speed. A bounded disk
                # spool avoids a full TCP receive window timing out in the mesh.
                self._spool.seek(self._written)
                self._spool.write(data)
                self._written += len(data)
                self._available.set()
            self.samples += len(data) // 2
            self.digest.update(data)

    async def finish(self):
        async with self.lock:
            if self.finished:
                raise ValueError("duplicate end of input")
            if self._spool is None:
                await self.queue.put(None)
            self.finished = True
            self._available.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.exhausted:
            raise StopAsyncIteration
        if self._spool is not None:
            await self._available.wait()
            async with self.lock:
                if self._read == self._written and self.finished:
                    self.exhausted = True
                    self.close()
                    raise StopAsyncIteration
                self._spool.seek(self._read)
                data = self._spool.read(min(self.max_frame_bytes, self._written - self._read))
                self._read += len(data)
                if self._read == self._written and not self.finished:
                    self._available.clear()
                return data
        data = await self.queue.get()
        self.queue.task_done()
        if data is None:
            self.exhausted = True
            raise StopAsyncIteration
        return data

    def close(self):
        self.finished = self.exhausted = True
        self._available.set()
        if self._spool is not None:
            self._spool.close()
            self._spool = None
