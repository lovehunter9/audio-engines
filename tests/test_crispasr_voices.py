import unittest

from wrapper.crispasr_voices import VOXTRAL_PRESETS, listed_voices


class VoxtralVoiceListTest(unittest.TestCase):
    def test_session_enumeration_wins(self):
        self.assertEqual(listed_voices(["tara", "leo"], "voxtral-tts"), ["tara", "leo"])

    def test_voxtral_empty_session_uses_the_twenty_presets(self):
        names = listed_voices([], "voxtral-tts")
        self.assertEqual(names, list(VOXTRAL_PRESETS))
        self.assertEqual(len(names), 20)
        self.assertIn("de_female", names)
        self.assertIn("casual_male", names)

    def test_other_backend_stays_empty(self):
        self.assertEqual(listed_voices([], "orpheus"), [])
        self.assertEqual(listed_voices(None, "qwen3-tts"), [])
