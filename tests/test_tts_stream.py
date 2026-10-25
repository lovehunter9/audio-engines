# A streamed synthesis whose client leaves must give the engine back.
#
# Every check runs while the event loop is still up: asyncio.run() cancels whatever is pending on
# the way out, which unblocks a stuck producer and would let the old bug pass. The server's loop
# never ends that way.
import asyncio
import threading
import time
import unittest
from unittest import mock

from wrapper.caps import tts


def _chunks(n, lock, produced):
    with lock:
        for i in range(n):
            time.sleep(0.005)
            produced.append(i)
            yield i


class StreamCancelTest(unittest.TestCase):
    def _drop_after_first(self, n):
        """(lock was free again, chunks produced), both taken before the loop ends."""
        lock, produced = threading.Lock(), []

        async def run():
            agen = tts._aiter(lambda: _chunks(n, lock, produced))
            self.assertEqual(await agen.__anext__(), 0)
            await agen.aclose()  # the client hung up after the first chunk
            free = await asyncio.to_thread(lock.acquire, True, 3)
            if free:
                lock.release()
            return free, len(produced)

        return asyncio.run(run())

    def test_lock_is_released_when_the_sentence_outruns_the_queue(self):
        # Used to park the producer on a full queue with the lock held: every later request hung.
        free, _ = self._drop_after_first(40)
        self.assertTrue(free, "producer still holds the generation lock")

    def test_producer_stops_instead_of_finishing_the_sentence(self):
        _, produced = self._drop_after_first(400)
        self.assertLess(produced, 40)

    def test_a_cancelled_consumer_releases_the_lock(self):
        lock, produced = threading.Lock(), []

        async def run():
            async def consume():
                async for _ in tts._aiter(lambda: _chunks(400, lock, produced)):
                    await asyncio.sleep(10)  # stuck sending, as on a dead socket

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            free = await asyncio.to_thread(lock.acquire, True, 3)
            if free:
                lock.release()
            return free

        self.assertTrue(asyncio.run(run()), "producer still holds the generation lock")

    def test_a_full_read_still_yields_every_item_in_order(self):
        lock, produced = threading.Lock(), []

        async def run():
            return [x async for x in tts._aiter(lambda: _chunks(30, lock, produced))]

        self.assertEqual(asyncio.run(run()), list(range(30)))

    def test_errors_reach_the_consumer(self):
        def boom():
            yield 1
            raise ValueError("decode failed")

        async def run():
            return [x async for x in tts._aiter(boom)]

        with self.assertRaisesRegex(ValueError, "decode failed"):
            asyncio.run(run())


class FrameBudgetTest(unittest.TestCase):
    def test_budget_scales_with_text_and_respects_the_caller(self):
        short = tts._gen_kwargs({"input": "好的。"})["max_new_tokens"]
        long = tts._gen_kwargs({"input": "好" * 80})["max_new_tokens"]
        self.assertGreater(long, short)
        # 80 Chinese characters were measured at ~0.18 s each: the bound must leave room for that.
        self.assertGreater(long, 80 * 0.23 * tts.FRAME_HZ * 1.5)
        self.assertEqual(tts._gen_kwargs({"input": "hi", "max_new_tokens": 99})["max_new_tokens"], 99)

    def test_never_above_the_static_cache(self):
        self.assertLessEqual(tts._frame_budget("字" * 10000), tts.MAX_SEQ_LEN)

    def test_off_switches(self):
        with mock.patch.object(tts, "SECONDS_PER_CHAR", 0):
            self.assertNotIn("max_new_tokens", tts._gen_kwargs({"input": "你好"}))
        with mock.patch.object(tts, "FRAME_BUDGET", False):
            self.assertNotIn("max_new_tokens", tts._gen_kwargs({"input": "你好"}))


if __name__ == "__main__":
    unittest.main()
