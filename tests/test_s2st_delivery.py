import base64
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from wrapper.s2st.delivery import CompleteWords, install


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def test_complete_segment_releases_punctuation_and_last_word(self):
        state = CompleteWords(True)
        self.assertEqual(state.feed("Look at wh"), "Look at ")
        self.assertEqual(state.feed("at comes next."), "what comes ")
        self.assertEqual(state.feed("", final=True), "next.")
        self.assertEqual(state.feed(" Yes, that's right.", final=True), " Yes, that's right.")
        self.assertEqual(state.received, state.committed)

    def speaker(self, failure=False):
        class Speech:
            def __init__(self):
                self.language = "en"
                self.s = SimpleNamespace(tts_backend="ming")
                self.paired = False
                self.seq = 1
                self.rate = 24000
                self.bytes = 0
                self.events = []
                self.texts = []
                self.emit = self.capture

            async def capture(self, event):
                self.events.append(event)

            async def text(self, value):
                self.texts.append(value)

            async def finish(self):
                return None

            def _speech_body(self, value):
                return dict(input=value)

            async def _synthesize(self, value):
                await self.emit(dict(event="audio", seq=1, data=base64.b64encode(b"0123").decode()))
                self.bytes += 4
                self.visible_during_synthesis = len(self.events)
                if failure:
                    raise RuntimeError("native generation failed before EOS")
                await self.emit(dict(event="tts_end", seq=1))

        module = ModuleType("simulinfer.tts")
        module.Speech = Speech
        with patch.dict(sys.modules, {"simulinfer.tts": module}):
            install()
        return Speech()

    async def test_validated_phrase_publishes_original_chunks_once(self):
        speech = self.speaker()
        await speech._synthesize("One sentence.")
        self.assertEqual(speech.visible_during_synthesis, 0)
        self.assertEqual(
            [event["event"] for event in speech.events], ["tts_piece_validated", "audio", "tts_end"]
        )
        self.assertEqual(base64.b64decode(speech.events[1]["data"]), b"0123")
        self.assertEqual(speech.bytes, 4)

    async def test_failed_phrase_withholds_all_pcm_and_never_retries(self):
        speech = self.speaker(failure=True)
        with self.assertRaisesRegex(RuntimeError, "before EOS"):
            await speech._synthesize("One sentence.")
        self.assertEqual(speech.bytes, 0)
        self.assertEqual([event["event"] for event in speech.events], ["tts_attempt_failed"])
        self.assertEqual(speech.events[0]["published_audio_bytes"], 0)
        self.assertFalse(speech.events[0]["retry"])
        self.assertEqual(speech.emit, speech.capture)

    async def test_native_segment_preserves_all_delivered_text(self):
        speech = self.speaker()
        await speech.text("at wh")
        await speech.text("at comes next.")
        await speech.text(" Yes, that's right. ")
        await speech.finish()
        self.assertEqual("".join(speech.texts), "at what comes next. Yes, that's right. ")
        self.assertTrue(speech.events[-1]["identical"])


if __name__ == "__main__":
    unittest.main()
