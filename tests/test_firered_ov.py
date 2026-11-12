import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from wrapper.caps import firered


TEXT = (
    "Before we leave for the station tomorrow morning, please check that every window "
    "is closed and the kitchen lights are off. I will put the tickets beside your blue "
    "notebook, so we can have breakfast together and walk to the bus stop without rushing."
)


class NativeStopTest(unittest.TestCase):
    def test_normal_stop_does_not_append_guessed_suffix(self):
        wave = np.zeros(int(12.64 * 24000), dtype="float32")
        self.assertTrue(firered._unsaid(TEXT, wave, 24000))
        sentences = [TEXT]
        self.assertEqual(firered._queue_rest(
            sentences, 0, 1, TEXT, wave, 24000, {"stop_reason": "threshold"}), 1)
        self.assertEqual(sentences, [TEXT])

    def test_limit_and_legacy_paths_keep_existing_recovery(self):
        wave = np.zeros(int(12.64 * 24000), dtype="float32")
        for generation in (None, {"stop_reason": "unknown"}, {"stop_reason": "max_steps"}):
            with self.subTest(generation=generation):
                sentences = [TEXT]
                self.assertEqual(firered._queue_rest(
                    sentences, 0, 1, TEXT, wave, 24000, generation), 2)
                self.assertEqual(sentences[1], firered._unsaid(TEXT, wave, 24000))

    def test_stop_at_the_last_iteration_is_still_a_normal_stop(self):
        prompt = SimpleNamespace(shape=(1, 368, 128))
        output = SimpleNamespace(shape=(1, 368 + 399 * 4, 128))
        summary = firered._firered_generation_summary(
            output, {"latents_out": prompt}, 4, .99)
        self.assertEqual(summary["stop_reason"], "threshold")
        self.assertEqual(summary["generated_steps"], 399)
        self.assertEqual(summary["prompt_frames"], 368)

    def test_real_limit_and_unknown_score_are_distinct(self):
        output = SimpleNamespace(shape=(1, 400 * 4, 128))
        for score, expected in ((.1, "max_steps"), (None, "unknown")):
            summary = firered._firered_generation_summary((output, []), {}, 4, score)
            self.assertEqual(summary["stop_reason"], expected)
            self.assertEqual(summary["generated_steps"], 400)

    def test_context_transcript_matches_a_normally_completed_wave(self):
        wave = np.zeros(24000 * 5, dtype="float32")
        model = SimpleNamespace(_ov_last_generation={"stop_reason": "threshold"})
        backend = firered.FireRedBackend(model)
        with mock.patch.object(firered.tts_long, "tail_pair", return_value=("tail", wave)) as tail:
            backend._carry(TEXT, wave, 24000)
            self.assertEqual(tail.call_args.args[0], TEXT)
        del model._ov_last_generation
        with mock.patch.object(firered.tts_long, "tail_pair", return_value=("tail", wave)) as tail:
            backend._carry(TEXT, wave, 24000)
            self.assertEqual(tail.call_args.args[0], TEXT[:30])

if __name__ == "__main__":
    unittest.main()
