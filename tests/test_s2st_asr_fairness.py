import asyncio
import time
import unittest

from wrapper.s2st.youdao_asr_fairness import apply


SOURCE = '''async def asr_stream_api_v1(work):
    async def processor():
        batch = 0
        while batch < 5:
            item = 0
            while item < 4:
                work(batch, item)
                item += 1
            batch += 1
    await processor()
'''


class ASRFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, source):
        namespace = {'asyncio': asyncio}
        exec(source, namespace)
        calls, heartbeats = [], []
        done = asyncio.Event()

        def model(batch, item):
            time.sleep(.006)
            calls.append((batch, item))

        async def heartbeat():
            while not done.is_set():
                await asyncio.sleep(.001)
                if not done.is_set():
                    heartbeats.append(len(calls))

        task = asyncio.create_task(heartbeat())
        try:
            await namespace['asr_stream_api_v1'](model)
        finally:
            done.set()
            await task
        return calls, heartbeats

    async def test_backlogged_inference_keeps_heartbeats_alive_without_changing_calls(self):
        original, starved = await self.exercise(SOURCE)
        patched, responsive = await self.exercise(apply(SOURCE))
        self.assertEqual(original, [(batch, item) for batch in range(5) for item in range(4)])
        self.assertEqual(patched, original)
        self.assertFalse(starved)
        self.assertGreater(len(responsive), 2)
        self.assertTrue(all(0 < count < len(patched) for count in responsive))

    def test_changed_upstream_loop_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'processor loops'):
            apply(SOURCE.replace('while item < 4:', 'if item < 4:'))
