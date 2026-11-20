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
from wrapper.s2st.youdao_source import SourceClauses
from wrapper.s2st import youdao_source
from wrapper.s2st.youdao_numbers import boundary, decimal_separator


class DeliverySettingsTests(unittest.TestCase):
    def test_unknown_asr_delivery_increment_is_rejected_before_loading(self):
        from wrapper.s2st import server
        import os

        with patch.dict(os.environ, {"AUDIO_BASE": "s2st-youdao", "ENGINE_ARGS": "--asr-chunk-ms 3000"}):
            with self.assertRaisesRegex(ValueError, "ASR delivery increment"):
                server.build_app(["audio_s2s"])


class WordTailTests(unittest.TestCase):
    def test_translation_policy_is_explicit_and_low_force_is_not_selected(self):
        self.assertEqual(Youdao("/unused", "/unused", latency_mode="native").latency_mode, "native")
        with self.assertRaises(ValueError):
            Youdao("/unused", "/unused", latency_mode="low")

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


class TranslationQualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_script_is_preserved_and_reported_without_inference_retry(self):
        import httpx
        from wrapper.s2st.youdao_target import CompletionStream, TargetText

        original = "A 副 product."
        packets = [
            {"choices": [{"delta": {"content": original}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join("data: " + json.dumps(value) + "\n\n" for value in packets)
        body += "data: [DONE]\n\n"
        calls, delivered, observed = [], [], []

        async def handler(request):
            calls.append(request)
            return httpx.Response(200, text=body)

        async def emit(event):
            delivered.append(event)

        async def observe(event):
            observed.append(event)

        target = TargetText("zh2en", emit, lambda parts, direction: "".join(parts), lambda raw: (None, raw))
        native = SimpleNamespace(model="test", max_new_tokens=256, force_max_new_tokens=256,
                                 logit_bias=None, repetition_penalty=None, api_key=None, url="http://test/v1/chat/completions")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            raw = await CompletionStream(native, "zh2en", target, observe=observe, client=client)("source")
        self.assertEqual(raw, original)
        self.assertEqual("".join(e.get("canonical_text", "") for e in delivered), original)
        self.assertEqual(len(calls), 1)
        warnings = [e for e in observed if e["event"] == "translation_quality_warning"]
        self.assertEqual([e["text"] for e in warnings], [original])


class SourceNumbersTests(unittest.TestCase):
    def test_chinese_quantity_scale_cannot_be_sent_separately(self):
        guard = SourceClauses("zh2en")
        self.assertEqual(guard.feed("太阳的半径笼统地说是七十", 0), [])
        first = guard.feed("", 32000)
        self.assertEqual(first[0][0], "太阳的半径笼统地说是")
        second = guard.feed("万公里。", 64000)
        self.assertEqual(second[0][0], "七十万公里。")
        self.assertEqual(guard.received, guard.committed)

    def test_mixed_numeric_scales_are_retained_as_a_whole(self):
        cases = [
            ("距离为70万", "距离为"),
            ("人口为1.4 million", "人口为"),
            ("距离为3.26万", "距离为"),
            ("质量为2x10^30", "质量为"),
            ("It is one point four million ", "It is "),
            ("这是第70", "这是"),
        ]
        for value, expected in cases:
            self.assertEqual(value[: boundary(value, len(value))], expected)

    def test_every_decimal_split_preserves_source_characters(self):
        raw = "总人口是1.4 million people。"
        for split in range(len(raw) + 1):
            guard = SourceClauses("zh2en")
            parts = guard.feed(raw[:split], 0)
            parts += guard.feed(raw[split:], 32000)
            parts += guard.feed("", 64000, final=True)
            self.assertEqual("".join(text for text, _ in parts), raw)
            self.assertFalse(any(text.endswith("1.") for text, _ in parts[:-1]))

    def test_final_number_is_not_dropped_and_ordinary_period_is_unchanged(self):
        guard = SourceClauses("zh2en")
        guard.feed("这是第七十", 0)
        self.assertEqual(guard.feed("", 32000, final=True), [("这是第七十", True)])
        self.assertFalse(decimal_separator("Done. Next", 4))


class AsrFinalityTests(unittest.IsolatedAsyncioTestCase):
    async def test_vad_cannot_force_a_half_word_but_reset_drains_all_text(self):
        sent = []
        marker = object()

        class Engine:
            async def feed(self, text):
                return []

        class ASR:
            async def finish(self):
                pass

        class Session:
            def _cancel_idle_timer(self):
                pass

            async def send(self, event):
                sent.append(event)

        server = ModuleType("inference.server")
        server.SpeechSession, server._FLUSH = Session, marker
        asr = ModuleType("inference.asr")
        asr.R2T2StreamingASR = ASR
        translation = ModuleType("inference.translation")
        translation.TranslationEngine = Engine
        inference = ModuleType("inference")
        inference.server, inference.asr = server, asr
        with patch.dict(
            sys.modules,
            {
                "inference": inference,
                "inference.server": server,
                "inference.asr": asr,
                "inference.translation": translation,
            },
        ):
            youdao_source.install()
            session = Session()
            session.direction = "zh2en"
            session._translation_queue = asyncio.Queue()
            await session._on_asr_update(
                SimpleNamespace(text="，太阳半", source_samples=16000, reset=False, speech_end=True)
            )
            pending = []
            while not session._translation_queue.empty():
                pending.append(session._translation_queue.get_nowait())
            self.assertFalse(any(item is marker for item in pending))
            self.assertNotIn("太阳半", [item.text for item in pending])
            await session._on_asr_update(
                SimpleNamespace(
                    text="径看得出来吗？", source_samples=48000, reset=False, speech_end=True
                )
            )
            self.assertEqual(session._translation_queue.get_nowait().text, "太阳半径看得出来吗？")
            self.assertTrue(session._translation_queue.empty())
            await session._on_asr_update(
                SimpleNamespace(
                    text="这是第七十", source_samples=64000, reset=False, speech_end=True
                )
            )
            self.assertTrue(session._translation_queue.empty())
            await session._on_asr_update(
                SimpleNamespace(text="", source_samples=96000, reset=True, speech_end=False)
            )
            self.assertEqual(session._translation_queue.get_nowait().text, "这是第七十")
            self.assertIs(session._translation_queue.get_nowait(), marker)
            self.assertEqual(session._source_clauses.received, session._source_clauses.committed)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, direction="zh2en", *, future=False, dropped=False, send_error=False,
                       initial_events=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "results").mkdir()
            source = Input(Stream(direction))
            pcm = b"\1\0" * 160
            await source.feed(pcm)
            await source.finish()
            pending = asyncio.Queue()
            for event in initial_events if initial_events is not None else [{"type": "init_ok"}]:
                await pending.put(event)
            socket = SimpleNamespace(closed=False, input=bytearray(), initialized=False)
            text = (
                "The next train leaves at eleven."
                if direction == "zh2en"
                else "下一班列车十一点出发。"
            )

            async def send(value):
                if isinstance(value, bytes):
                    self.assertTrue(socket.initialized, "PCM must wait for the official init_ok")
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
                event = await pending.get()
                if event.get("type") == "init_ok":
                    socket.initialized = True
                return json.dumps(event)

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
                    self.assertEqual(result[0]["source_samples"], 160)
                    self.assertEqual(result[0]["asr_wire_samples"], 8160)
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
                    if adapter.feeder is not None:
                        self.assertTrue(adapter.feeder.done())

    async def test_loading_progress_before_init_ok_is_not_an_inference_retry(self):
        await self.exercise(initial_events=[
            {"type": "loading", "component": "translation"},
            {"type": "loading", "component": "asr"},
            {"type": "init_ok"},
        ])

    async def test_real_initialization_error_is_not_ignored_as_loading(self):
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            await self.exercise(initial_events=[
                {"type": "loading", "component": "translation"},
                {"type": "error", "message": "backend unavailable"},
            ])

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


class SourceLanguageIncrementTests(unittest.IsolatedAsyncioTestCase):
    async def test_direction_routing_does_not_leak_between_sessions_or_change_warmup(self):
        from wrapper.s2st.youdao_asr_chunking import apply

        source = '''SAMPLING_RATE = 16000
CHUNK_ASR_SECONDS = 2.0
CHUNK_ASR_SIZE = 32000
STEP_MS = 2000
WARMUP = (STEP_MS, CHUNK_ASR_SECONDS, CHUNK_ASR_SIZE)
async def asr_stream_api_v1(header):
    async def processor():
        try:
            language = header.get("language", "zhen")
            return STEP_MS, CHUNK_ASR_SECONDS, CHUNK_ASR_SIZE
        finally:
            pass
    return await processor()
'''
        namespace = {}
        exec(apply(source, 2000, 4000), namespace)
        self.assertEqual(namespace['WARMUP'], (2000, 2.0, 32000))
        entry = namespace['asr_stream_api_v1']
        self.assertEqual(await entry({'language': 'Chinese'}), (4000, 4.0, 64000))
        self.assertEqual(await entry({'language': 'English'}), (2000, 2.0, 32000))
        self.assertEqual(await entry({}), (2000, 2.0, 32000))
        with self.assertRaisesRegex(ValueError, 'source-language parser'):
            apply(source.replace('header.get("language", "zhen")', 'header.get("lang")'), 2000, 4000)
