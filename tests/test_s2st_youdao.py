import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from wrapper.s2st.protocol import Input, Stream
from wrapper.s2st.youdao import Youdao
from wrapper.s2st.youdao_word_tail import SpeechWordTail


class WordTailTests(unittest.TestCase):
    def test_dangling_preposition_waits_for_the_actual_next_words(self):
        guard = SpeechWordTail("zh2en")
        self.assertEqual(guard.feed("delivering the ball to "), "delivering the ball ")
        self.assertEqual(
            guard.feed("Actually, Manchester United now, "), "to Actually, Manchester United now, "
        )
        self.assertEqual(guard.feed("", final=True), "")
        self.assertEqual(guard.received, guard.delivered)

    def test_arbitrary_stream_splits_preserve_every_character(self):
        text = "The train leaves from platform three at eleven. The next one is a bus. "
        for split in range(len(text) + 1):
            guard = SpeechWordTail("zh2en")
            output = guard.feed(text[:split]) + guard.feed(text[split:]) + guard.feed("", True)
            self.assertEqual(output, text)
        guard = SpeechWordTail("zh2en")
        self.assertEqual(guard.feed("The train leaves from "), "The train leaves ")
        self.assertEqual(guard.feed("platform "), "from platform ")

    def test_final_open_words_are_flushed_without_new_text(self):
        guard = SpeechWordTail("zh2en")
        self.assertEqual(guard.feed("The result is a "), "The result ")
        self.assertEqual(guard.feed("", True), "is a ")
        guard = SpeechWordTail("zh2en")
        self.assertEqual(guard.feed("passing to."), "passing ")
        self.assertEqual(guard.feed("", True), "to.")

    def test_chinese_is_immediate(self):
        guard = SpeechWordTail("en2zh")
        for text in ["我们拼命尝试", "去修复", "。", "UTF-8 is a "]:
            self.assertEqual(guard.feed(text), text)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, direction="zh2en", *, future=False, dropped=False, send_error=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            source = Input(Stream(direction))
            pcm = b"\1\0" * 160
            await source.feed(pcm)
            await source.finish()
            pending = asyncio.Queue()
            await pending.put({"type": "init_ok"})
            socket = SimpleNamespace(closed=False, input=bytearray())
            text = (
                "The next train leaves at eleven."
                if direction == "zh2en"
                else "下一班列车十一点出发。"
            )

            async def send(value):
                if isinstance(value, bytes):
                    if send_error:
                        raise RuntimeError("input transport failed")
                    socket.input.extend(value)
                    return
                if json.loads(value).get("type") != "end":
                    return
                count = len(socket.input) // 2
                mark = count + int(future)
                observer = root / "results/asr-observer.jsonl"
                observer.write_text(
                    json.dumps(
                        {
                            "event": "streaming_transcribe_no_reset",
                            "input_samples": count - int(dropped),
                        }
                    )
                    + "\n"
                )
                (root / "results/target-stream-observer.jsonl").write_text(
                    json.dumps({"event": "translation_request"})
                    + "\n"
                    + json.dumps(
                        {
                            "event": "translation_response",
                            "started_monotonic_seconds": 2.0,
                            "finished_monotonic_seconds": 3.5,
                        }
                    )
                    + "\n"
                )
                for event in [
                    {
                        "type": "translation_delta",
                        "text": text,
                        "canonical_text": text,
                        "request_source_samples": mark,
                    },
                    {
                        "type": "speech_text",
                        "text": text + (" " if direction == "zh2en" else ""),
                        "request_source_samples": mark,
                    },
                    {"type": "translation", "text": text},
                    {"type": "ended", "stats": {"buffer_units": 0}},
                ]:
                    await pending.put(event)

            async def recv():
                return json.dumps(await pending.get())

            async def close():
                socket.closed = True

            async def wait_closed():
                pass

            socket.send, socket.recv = send, recv
            socket.close, socket.wait_closed = close, wait_closed

            async def connect(*args, **kwargs):
                return socket

            websockets = ModuleType("websockets")
            websockets.connect = connect
            translation = ModuleType("inference.translation")
            translation.join_translation_segments = lambda parts, direction: "".join(parts).strip()
            inference = ModuleType("inference")
            inference.translation = translation
            health_calls = []

            class Client:
                def __init__(self, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def get(self, url):
                    health_calls.append(url)
                    return SimpleNamespace(
                        raise_for_status=lambda: None,
                        json=lambda: {"active_speech_sessions": 0},
                        text="llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n",
                    )

            delivered = []

            async def consume(value):
                delivered.append(value)

            adapter = Youdao(root / "session", root)
            with (
                patch.dict(
                    sys.modules,
                    {
                        "websockets": websockets,
                        "inference": inference,
                        "inference.translation": translation,
                    },
                ),
                patch("wrapper.s2st.youdao.httpx.AsyncClient", Client),
            ):
                try:
                    result = [row async for row in adapter.run(source, consume)]
                    self.assertEqual(result[0]["text"], text)
                    self.assertEqual(b"".join(delivered), pcm)
                    self.assertEqual(socket.input, pcm + bytes(16000))
                    self.assertEqual(
                        adapter.completion_metadata["asr_source_consumed_samples"], 160
                    )
                    self.assertEqual(
                        adapter.completion_metadata["asr_model_consumed_samples"], 8160
                    )
                    self.assertTrue(health_calls)
                    return result
                finally:
                    self.assertTrue(socket.closed)
                    self.assertTrue(adapter.feeder.done())

    async def test_complete_input_and_padding_are_distinct_in_both_directions(self):
        await self.exercise("zh2en")
        await self.exercise("en2zh")

    async def test_future_speaker_reference_is_rejected_including_one_sample(self):
        with self.assertRaisesRegex(RuntimeError, "before arrival"):
            await self.exercise(future=True)

    async def test_lost_asr_input_cannot_pass_eos(self):
        with self.assertRaisesRegex(RuntimeError, "complete original input"):
            await self.exercise(dropped=True)

    async def test_failed_input_does_not_wait_for_a_translation_timeout(self):
        with self.assertRaisesRegex(RuntimeError, "input transport failed"):
            await asyncio.wait_for(self.exercise(send_error=True), 1)


if __name__ == "__main__":
    unittest.main()
