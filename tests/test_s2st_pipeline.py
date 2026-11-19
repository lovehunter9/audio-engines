import asyncio
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from wrapper.s2st.processes import Worker, Workers
from wrapper.s2st.protocol import Input, Stream


class InputTests(unittest.IsolatedAsyncioTestCase):
    async def test_backpressure_preserves_samples_and_eos_order(self):
        source = Input(Stream("zh2en"), capacity=1)
        await source.feed(b"\1\0")
        second = asyncio.create_task(source.feed(b"\2\0"))
        await asyncio.sleep(0.01)
        self.assertFalse(second.done())
        self.assertEqual(await anext(source), b"\1\0")
        await second
        ending = asyncio.create_task(source.finish())
        await asyncio.sleep(0.01)
        self.assertFalse(ending.done())
        self.assertEqual(await anext(source), b"\2\0")
        await ending
        with self.assertRaises(StopAsyncIteration):
            await anext(source)
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(anext(source), 0.1)
        self.assertEqual(source.samples, 2)
        self.assertEqual(source.digest.hexdigest(), hashlib.sha256(b"\1\0\2\0").hexdigest())
        with self.assertRaises(ValueError):
            await source.feed(b"\0\0")
        with self.assertRaises(ValueError):
            await source.finish()

    async def test_invalid_frames_never_enter_the_queue(self):
        source = Input(Stream("en2zh"), max_frame_bytes=4)
        for data in (b"", b"1", b"123456", "00"):
            with self.assertRaises(ValueError):
                await source.feed(data)
        self.assertTrue(source.queue.empty())
        self.assertEqual(source.samples, 0)

    async def test_canceled_producer_does_not_record_unconsumed_samples(self):
        source = Input(Stream("zh2en"), capacity=1)
        await source.feed(b"00")
        task = asyncio.create_task(source.feed(b"11"))
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(source.samples, 1)
        self.assertEqual(await anext(source), b"00")

    def test_reject_unimplemented_language_and_audio_formats(self):
        with self.assertRaises(ValueError):
            Stream("fr2en")
        with self.assertRaises(ValueError):
            Stream("en2zh", 24000)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_child_never_becomes_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            workers = Workers(folder)
            with self.assertRaisesRegex(RuntimeError, "worker exited"):
                await workers.start(
                    Worker(
                        "failed",
                        (sys.executable, "-c", "raise SystemExit(7)"),
                        "http://127.0.0.1:1/health",
                        timeout=2,
                    )
                )
            self.assertFalse(workers.processes)
            self.assertFalse(workers.logs)

    async def test_startup_timeout_reaps_process(self):
        with tempfile.TemporaryDirectory() as folder:
            workers = Workers(folder)
            marker = Path(folder) / "pid"
            code = (
                "import os,time;from pathlib import Path;Path(%r).write_text(str(os.getpid()));time.sleep(60)"
                % str(marker)
            )
            with self.assertRaises(TimeoutError):
                await workers.start(
                    Worker(
                        "waiting",
                        (sys.executable, "-c", code),
                        "http://127.0.0.1:1/health",
                        timeout=0.3,
                    )
                )
            self.assertTrue(marker.exists())
            import os

            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)
            self.assertFalse(workers.processes)


if __name__ == "__main__":
    unittest.main()
