"""Excluded warm-up of the unchanged TTS path, using a causal source prefix."""
import asyncio
from pathlib import Path
import sys
import time
import wave

import numpy as np
from frozen_ming_sink import FrozenMingSink


async def main():
    root = Path('/home/olares/s2st-original-check-20261123')
    sink = FrozenMingSink(Path(sys.argv[1]), 'zh-en')
    try:
        await sink.start()
        sink.begin_source(time.monotonic())
        with wave.open(str(root / 'inputs/zh2en-01-tech.wav'), 'rb') as reader:
            pcm = reader.readframes(160000)
        sink.feed_source(np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768)
        await sink.text('This is a short warm-up sentence. ')
        await sink.finish(10)
    finally:
        await sink.close()


if __name__ == '__main__':
    asyncio.run(main())
