import asyncio
import hashlib
import os
import unittest
from unittest.mock import patch

from wrapper.s2st.youdao_admission import SourceAdmission


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class SourceAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_burst_upload_is_lossless_and_never_exceeds_its_admission_budget(self):
        clock = Clock()
        admission = SourceAdmission(1.25, clock=clock, sleep=clock.sleep)
        frames = [bytes([n % 256]) * (n % 17 + 2300) * 2 for n in range(90)]
        received = []
        for frame in frames:
            await admission.admit(len(frame) // 2)
            received.append(frame)
            self.assertLessEqual(admission.samples / 16000, 4 + 1.25 * clock.now + 1e-9)
        self.assertGreater(clock.now, 0)
        self.assertEqual(admission.samples, sum(map(len, frames)) // 2)
        self.assertEqual(hashlib.sha256(b"".join(received)).digest(), hashlib.sha256(b"".join(frames)).digest())
        self.assertAlmostEqual(admission.wait_seconds, clock.now)

    async def test_ordinary1x_input_adds_no_wait(self):
        clock = Clock()
        admission = SourceAdmission(1.25, clock=clock, sleep=clock.sleep)
        for _ in range(900):
            clock.now += .16
            await admission.admit(2560)
        self.assertEqual(clock.waits, [])
        self.assertEqual(admission.wait_seconds, 0)
        self.assertEqual(admission.samples, 900 * 2560)

    async def test_disabled_mode_does_not_read_a_clock_or_sleep(self):
        def unexpected():
            raise AssertionError("Disabled admission changed scheduling")
        admission = SourceAdmission(clock=unexpected, sleep=unexpected)
        await admission.admit(16000 * 900)
        self.assertEqual(admission.samples, 16000 * 900)
        self.assertEqual(admission.wait_seconds, 0)

    async def test_cancelled_wait_does_not_admit_the_pending_frame(self):
        waiting = asyncio.Event()
        async def sleep(seconds):
            waiting.set()
            await asyncio.Event().wait()
        admission = SourceAdmission(1, clock=lambda: 0.0, sleep=sleep)
        await admission.admit(4 * 16000)
        task = asyncio.create_task(admission.admit(16000))
        await asyncio.wait_for(waiting.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(admission.samples, 4 * 16000)

    def test_invalid_options_fail_before_model_loading(self):
        from wrapper.s2st import server
        for value in ['nan', 'inf', '-1', '.9', 'invalid', '']:
            flag = '--max-source-speed' + (' ' + value if value else '')
            with self.subTest(value=value), patch.dict(os.environ, {'AUDIO_BASE': 's2st-youdao', 'ENGINE_ARGS': flag}):
                with self.assertRaises(ValueError):
                    server.build_app(['audio_s2s'])
        with patch.dict(os.environ, {'AUDIO_BASE': 's2st-tencent', 'ENGINE_ARGS': '--max-source-speed 1.25'}):
            with self.assertRaises(ValueError):
                server.build_app(['audio_s2s'])


if __name__ == '__main__':
    unittest.main()
