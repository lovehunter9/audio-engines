# Official Qwen speakers stay in that backend's directory. The shared pack stays the four cards.
import os
import tempfile
import unittest

import numpy as np

from wrapper.caps import qwen3_voice, tts_el
from wrapper.caps.firered import FireRedBackend


def _tone(text, speaker, language):
    return np.zeros(1600, dtype="float32"), 16000


class Qwen3VoiceCardsTest(unittest.TestCase):
    def test_official_speakers_are_not_in_the_shared_pack(self):
        shared = {c["voice_id"] for c in tts_el.builtin_presets("firered")}
        official = {row[0] for row in qwen3_voice.SPEAKERS}
        self.assertEqual(shared, {"zh-f", "zh-m", "en-f", "en-m"})
        self.assertTrue(official.isdisjoint(shared))
        self.assertEqual(len(official), 9)

    def test_seed_keeps_official_speakers_off_firered(self):
        root = tempfile.mkdtemp(prefix="qwen3voice-")
        native = os.path.join(root, ".qwen-native")
        qwen3_voice.bake_native(native, generate=_tone)
        self.assertTrue(os.path.isfile(os.path.join(native, "Vivian", "prompt.wav")))
        self.assertTrue(os.path.isfile(os.path.join(native, "Vivian", "prompt.txt")))

        class Backend:
            uses_shared_pack = True

            def native_voices_dir(self):
                return native

            def native_presets(self):
                return tts_el.load_voice_cards(native, category="premade")

        planted = tts_el.seed_builtins("qwen3_voice", os.path.join(root, "voices"), backend=Backend())
        ids = {c["voice_id"] for c in planted}
        self.assertTrue({row[0] for row in qwen3_voice.SPEAKERS} <= ids)
        self.assertTrue({"zh-f", "en-m"} <= ids)
        with open(os.path.join(root, "voices", "Vivian", "prompt.txt"), encoding="utf-8") as handle:
            transcript = handle.read()
        self.assertEqual(transcript, qwen3_voice.SPEAKERS[0][2])

        firered = tts_el.seed_builtins("firered", os.path.join(root, "firered"), backend=FireRedBackend(None))
        firered_ids = {c["voice_id"] for c in firered}
        self.assertEqual(firered_ids, {"zh-f", "zh-m", "en-f", "en-m"})
        self.assertFalse(os.path.isdir(os.path.join(root, "firered", "Vivian")))


if __name__ == "__main__":
    unittest.main()
