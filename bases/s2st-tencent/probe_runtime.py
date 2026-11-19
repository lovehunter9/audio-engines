import asyncio
import shutil

from simulinfer.tts import Speech
from simulinfer.tts_audio import TempoStream


async def main():
    assert Speech is not None
    assert shutil.which("ffmpeg"), "System FFmpeg is required for online playout"
    output = bytearray()

    async def receive(pcm):
        output.extend(pcm)

    stream = TempoStream(24000, 1.1, receive)
    try:
        await stream.start()
        await stream.write(b"\0" * 48000)
        await stream.finish()
        assert output and len(output) % 2 == 0, "No complete PCM from the upstream tempo transport"
    finally:
        await stream.close()


asyncio.run(main())
