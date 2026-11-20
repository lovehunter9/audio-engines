"""Optional source admission pacing; audio and neural chunk boundaries stay intact."""
import asyncio
import math
import time


class SourceAdmission:
    def __init__(self, max_speed=0.0, *, clock=time.monotonic, sleep=asyncio.sleep):
        self.max_speed = self.validate(max_speed)
        self.clock, self.sleep = clock, sleep
        self.started = None
        self.samples = 0
        self.wait_seconds = 0.0

    @staticmethod
    def validate(value):
        if isinstance(value, bool):
            raise ValueError("Maximum source speed must be zero or at least1x")
        speed = float(value)
        if not math.isfinite(speed) or speed < 0 or 0 < speed < 1:
            raise ValueError("Maximum source speed must be zero or at least1x")
        return speed

    async def admit(self, samples):
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise ValueError("Admission requires positive source sample counts")
        total = self.samples + samples
        if self.max_speed:
            if self.started is None:
                self.started = self.clock()
            deadline = self.started + max(0.0, total / 16000 - 4.0) / self.max_speed
            delay = deadline - self.clock()
            if delay > 0:
                started = self.clock()
                while delay > 0:
                    await self.sleep(delay)
                    delay = deadline - self.clock()
                self.wait_seconds += self.clock() - started
        self.samples = total
