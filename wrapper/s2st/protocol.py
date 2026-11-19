import asyncio
from dataclasses import dataclass
import hashlib


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
    def __init__(self, stream, capacity=16, max_frame_bytes=51200):
        if capacity < 1 or max_frame_bytes < 2:
            raise ValueError("invalid stream limits")
        self.stream = stream
        self.queue = asyncio.Queue(maxsize=capacity)
        self.max_frame_bytes = max_frame_bytes
        self.finished = False
        self.exhausted = False
        self.samples = 0
        self.digest = hashlib.sha256()
        self.lock = asyncio.Lock()

    async def feed(self, data):
        if not isinstance(data, bytes) or not data or len(data) % 2:
            raise ValueError("audio frames must contain whole PCM16 samples")
        if len(data) > self.max_frame_bytes:
            raise ValueError("audio frame exceeds the input bound")
        async with self.lock:
            if self.finished:
                raise ValueError("audio after end of input")
            await self.queue.put(data)
            self.samples += len(data) // 2
            self.digest.update(data)

    async def finish(self):
        async with self.lock:
            if self.finished:
                raise ValueError("duplicate end of input")
            await self.queue.put(None)
            self.finished = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.exhausted:
            raise StopAsyncIteration
        data = await self.queue.get()
        self.queue.task_done()
        if data is None:
            self.exhausted = True
            raise StopAsyncIteration
        return data
