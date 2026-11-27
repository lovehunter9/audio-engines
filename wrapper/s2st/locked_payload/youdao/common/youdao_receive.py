"""Drain ordered upstream events without tying WebSocket reads to synthesis."""

import asyncio


class BoundedReceiver:
    def __init__(self, socket, byte_limit=64 * 1024 * 1024, message_limit=65536):
        if min(byte_limit, message_limit) <= 0:
            raise ValueError("Receiver limits must be positive")
        self.queue = asyncio.Queue()
        self.byte_limit = byte_limit
        self.message_limit = message_limit
        self.queued_bytes = 0
        self.peak_bytes = 0
        self.peak_messages = 0
        self.received_messages = 0
        self.task = asyncio.create_task(self._read(socket))

    async def _read(self, socket):
        try:
            while True:
                value = await socket.recv()
                size = len(value.encode("utf-8")) if isinstance(value, str) else len(value)
                if self.queued_bytes + size > self.byte_limit or self.queue.qsize() >= self.message_limit:
                    raise BufferError("Youdao upstream exceeded its bounded delivery queue")
                self.queued_bytes += size
                self.queue.put_nowait((value, size, None))
                self.received_messages += 1
                self.peak_bytes = max(self.peak_bytes, self.queued_bytes)
                self.peak_messages = max(self.peak_messages, self.queue.qsize())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.queue.put_nowait((None, 0, error))

    async def recv(self):
        value, size, error = await self.queue.get()
        self.queue.task_done()
        self.queued_bytes -= size
        if error is not None:
            raise error
        return value

    def stats(self):
        return dict(byte_limit=self.byte_limit, message_limit=self.message_limit,
                    peak_bytes=self.peak_bytes, peak_messages=self.peak_messages,
                    received_messages=self.received_messages)

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        self.queued_bytes = 0
