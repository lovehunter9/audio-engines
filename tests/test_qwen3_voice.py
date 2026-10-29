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


class QwenKnobTest(unittest.TestCase):
    def test_default_knobs_leave_sampling_alone(self):
        gen, speed, seed = qwen3_voice._sampling(None, "你好", "")
        self.assertEqual(gen["temperature"], 0.9)
        self.assertTrue(gen["do_sample"])
        self.assertNotIn("instruct", gen)
        self.assertNotIn("top_p", gen)
        self.assertEqual(speed, 1.0)
        self.assertIsNone(seed)

    def test_each_knob_moves_the_generator(self):
        _slow, speed, _seed = qwen3_voice._sampling({"speed": 0.7}, "你好", "")
        self.assertAlmostEqual(speed, 0.7)
        self.assertNotIn("instruct", _slow)
        styled, _, _ = qwen3_voice._sampling({"style": 0.8}, "你好", "")
        self.assertIn("语气夸张", styled["instruct"])
        tight, _, _ = qwen3_voice._sampling({"similarity_boost": 1.0}, "你好", "")
        self.assertLess(tight["temperature"], 0.9)
        self.assertEqual(tight["top_p"], 0.85)
        wild, _, seed = qwen3_voice._sampling({"stability": 0.1}, "你好", "")
        self.assertGreater(wild["temperature"], 0.9)
        self.assertIsNotNone(seed)
        locked, _, locked_seed = qwen3_voice._sampling({"stability": 1.0}, "你好", "")
        self.assertFalse(locked["do_sample"])
        self.assertIsNone(locked_seed)
        loose, _, _ = qwen3_voice._sampling({"use_speaker_boost": False}, "你好", "")
        self.assertGreater(loose["temperature"], 0.9)

    def test_openai_format_names_are_accepted(self):
        self.assertEqual(tts_el._output_format("wav").kind, "wav")
        self.assertEqual(tts_el._output_format("pcm").kind, "pcm")
        self.assertEqual(tts_el._output_format("flac").kind, "flac")
        self.assertEqual(tts_el._output_format("ogg").kind, "opus")
        self.assertEqual(tts_el._output_format("mp3").kind, "mp3")

    def test_run_asks_for_the_openai_routes(self):
        seen = {}

        def fake_run(*args, **kwargs):
            seen["kwargs"] = kwargs

        original = qwen3_voice.tts_el.run
        qwen3_voice.tts_el.run = fake_run
        try:
            qwen3_voice.run(["tts", "tts_clone", "tts_design"])
        finally:
            qwen3_voice.tts_el.run = original
        self.assertTrue(seen["kwargs"].get("openai_alias"))
        self.assertTrue(seen["kwargs"].get("task_legacy"))

    def test_legacy_task_alias_is_mounted(self):
        from fastapi.testclient import TestClient

        with TestClient(qwen3_voice.build_app(["tts", "tts_clone", "tts_design"])) as client:
            spec = client.get("/api/engine-spec").json()["endpoints"]
            legacy = {(row["method"], row["path"]) for row in spec if row["path"].startswith("/v1/audio/tasks")}
            self.assertEqual(legacy, {
                ("GET", "/v1/audio/tasks"),
                ("GET", "/v1/audio/tasks/{id}"),
                ("GET", "/v1/audio/tasks/{id}/result"),
                ("DELETE", "/v1/audio/tasks/{id}"),
            })
            self.assertTrue(all(row.get("deprecated") for row in spec
                                if row["path"].startswith("/v1/audio/tasks")))
            listed = client.get("/v1/audio/tasks")
            self.assertEqual(listed.status_code, 200)
            same = client.get("/v1/tasks")
            self.assertEqual(same.status_code, 200)
            self.assertEqual(listed.json(), same.json())


if __name__ == "__main__":
    unittest.main()
