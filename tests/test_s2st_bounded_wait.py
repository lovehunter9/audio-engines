import asyncio
import unittest
from types import SimpleNamespace

from wrapper.s2st.bounded_wait import Deadline, patch_bridge


FLUSH = object()


class Engine:
    def __init__(self):
        self.pending_source = ""
        self.flushed = []

    @property
    def stats(self):
        return dict(buffer_units=len(self.pending_source))

    async def feed(self, text):
        self.pending_source += text
        return await self.flush() if text == "done" else []

    async def flush(self):
        if not self.pending_source:
            return []
        text, self.pending_source = self.pending_source, ""
        self.flushed.append(text)
        return [dict(type="translation", text=text)]


class Bridge:
    def __init__(self):
        engine = Engine()
        self._translation = SimpleNamespace(engine=engine, feed=engine.feed, flush=engine.flush)
        self._translation_queue = asyncio.Queue()
        self._idle_task = None
        self._ending = False
        self.events = []
        self.visible = asyncio.Event()

    def _cancel_idle_timer(self):
        task, self._idle_task = self._idle_task, None
        if task is not None and not task.done():
            task.cancel()

    async def send(self, event):
        self.events.append(event)
        if event["type"] == "translation":
            self.visible.set()


patch_bridge(Bridge, FLUSH, 0.08)


async def source(bridge, text, reset=False):
    await bridge._on_asr_update(SimpleNamespace(text=text, reset=reset))
    await bridge._translation_queue.join()


async def close(bridge, task):
    bridge._ending = True
    await bridge._translation_queue.put(None)
    await task


async def main():
    bridge = Bridge()
    task = asyncio.create_task(bridge._translate_worker())
    await source(bridge, "first ")
    first_task = bridge._idle_task
    await asyncio.sleep(0.03)
    await source(bridge, "second")
    assert bridge._idle_task is first_task
    await asyncio.wait_for(bridge.visible.wait(), 1)
    assert bridge._translation.engine.flushed == ["first second"]
    assert len([event for event in bridge.events if event["type"] == "wait_deadline"]) == 1
    await close(bridge, task)

    bridge = Bridge()
    task = asyncio.create_task(bridge._translate_worker())
    await source(bridge, "done")
    assert bridge._idle_task is None
    old_generation = bridge._pending_generation - 1
    await source(bridge, "tail ")
    await bridge._translation_queue.put(Deadline(old_generation, 0))
    await bridge._translation_queue.join()
    assert bridge._translation.engine.flushed == ["done"]
    await source(bridge, "reset", reset=True)
    assert bridge._translation.engine.flushed == ["done", "tail reset"]
    await close(bridge, task)
    await asyncio.sleep(0.1)
    assert bridge._translation_queue.empty()

    bridge = Bridge()
    task = asyncio.create_task(bridge._translate_worker())
    await source(bridge, "last tail")
    await close(bridge, task)
    assert bridge._idle_task is None and bridge._translation.engine.pending_source == "last tail"
    await bridge._translation.flush()
    assert bridge._translation.engine.flushed == ["last tail"]
    await asyncio.sleep(0.1)
    assert bridge._translation_queue.empty()
    print(
        "Pending deadline, non-resetting ASR timer, stale token, native reset flush, EOS drain, and task cleanup passed."
    )


class BoundedWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_deadline_and_native_eos_cleanup(self):
        await main()


if __name__ == "__main__":
    unittest.main()
