import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from wrapper.s2st.protocol import Input, Stream
from wrapper.s2st.tencent import Tencent


FAKE = r"""import os,time,sys
from pathlib import Path
p=Path(os.environ["BENCHMARK_EVENTS_PATH"])
data=sys.stdin.buffer.read()
rows=[("begin",0,""),("audio",len(data)/32000,""),("token",len(data)/32000,"e4"),("token",len(data)/32000,"bda0e5a5bd"),("finished",len(data)/32000,""),("resident_wait",len(data)/32000,"")]
with p.open("w") as f:
 for kind,seconds,text in rows:
  f.write("1\t%s\t1\t%.6f\t0\t%s\n" % (kind,seconds,text));f.flush()
while not Path(os.environ["BENCHMARK_HOLD_EXIT"]).exists():time.sleep(.01)
"""


class NativeProducerTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_watermark_utf8_and_residency(self):
        from types import ModuleType
        from unittest.mock import patch

        translation = ModuleType("simulinfer.translation")
        translation.INSTRUCTIONS = {"zh-en": "translate"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "native"
            binary.write_text("#!" + sys.executable + "\n" + FAKE)
            binary.chmod(0o755)
            source = Input(Stream("zh2en"))
            await source.feed(b"\1\0" * 800)
            await source.finish()
            consumed = []

            async def emit_source(pcm):
                consumed.append(pcm)

            producer = Tencent(root / "model.gguf", root / "session", str(binary))
            with patch.dict(sys.modules, {"simulinfer.translation": translation}):
                rows = [row async for row in producer.run(source, emit_source)]
            self.assertEqual("".join(row["text"] for row in rows), "你好")
            self.assertEqual(b"".join(consumed), b"\1\0" * 800)
            self.assertEqual(producer.consumed_samples, 800)
            self.assertIsNone(producer.process.returncode)
            await producer.close()
            self.assertEqual(producer.process.returncode, 0)

    async def test_disconnect_terminates_owned_worker(self):
        from types import ModuleType
        from unittest.mock import patch

        translation = ModuleType("simulinfer.translation")
        translation.INSTRUCTIONS = {"zh-en": "translate"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "native"
            binary.write_text("#!" + sys.executable + "\n" + FAKE)
            binary.chmod(0o755)
            source = Input(Stream("zh2en"))
            producer = Tencent(root / "model.gguf", root / "session", str(binary))

            async def ignore(pcm):
                pass

            async def consume():
                async for row in producer.run(source, ignore):
                    pass

            with patch.dict(sys.modules, {"simulinfer.translation": translation}):
                task = asyncio.create_task(consume())
                for _ in range(100):
                    if producer.process is not None:
                        break
                    await asyncio.sleep(0.005)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertIsNotNone(producer.process.returncode)
            self.assertTrue(producer.feeder.done())


if __name__ == "__main__":
    unittest.main()
