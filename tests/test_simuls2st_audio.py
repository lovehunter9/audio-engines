import unittest
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

from wrapper.caps import simuls2st_audio


@unittest.skipIf(torch is None, "requires torch and numpy")
class PromptAlignmentTest(unittest.TestCase):
    def make_agent(self, codes=24, mel_frames=100):
        inference = SimpleNamespace(
            voicebox=SimpleNamespace(cond_scale_factor=4),
            audio_tokenizer=Mock(return_value=np.arange(codes)),
            extract_mel_feature=Mock(return_value=torch.arange(mel_frames).float().reshape(1, -1, 1)),
        )
        seen = {}

        def decode(generated_codes, prompt_codes, prompt_mel):
            seen.update(generated=generated_codes.clone(), prompt=prompt_mel.clone())
            frames = (prompt_codes.shape[1] + generated_codes.shape[1]) * 4 - prompt_mel.shape[1]
            return torch.zeros(frames)

        inference.codes_to_mel = decode
        inference.mel_to_audio = lambda mel: mel.numpy()
        original = Mock(return_value=np.array([-1], dtype=np.float32))
        agent = SimpleNamespace(
            use_prompt_mel=True, inference=inference, model_device=torch.device("cpu"),
            _synthesize_codes=original,
            _build_flow_matching_prompt_audio=Mock(return_value=(torch.zeros(32000), torch.zeros(48000))),
        )
        simuls2st_audio.install(agent)
        return agent, seen, original

    def test_two_second_prompt_preserves_every_generated_frame(self):
        agent, seen, _ = self.make_agent()
        codes = torch.arange(10).reshape(1, -1)
        audio = agent._synthesize_codes(codes, None, None)
        self.assertEqual(audio.shape, (40,))
        self.assertTrue(torch.equal(seen["generated"], codes))
        self.assertEqual(seen["prompt"][0, -1, 0], 95)

    def test_longer_reference_keeps_the_same_target_duration(self):
        agent, _, _ = self.make_agent(codes=49, mel_frames=200)
        self.assertEqual(len(agent._synthesize_codes(torch.zeros(1, 13), None, None)), 52)

    def test_one_code_tail_needs_no_inserted_silence_codes(self):
        agent, seen, _ = self.make_agent()
        self.assertEqual(len(agent._synthesize_codes(torch.tensor([[123]]), None, None)), 4)
        self.assertEqual(seen["generated"].tolist(), [[123]])

    def test_short_acoustic_prompt_is_padded_only_inside_the_prefix(self):
        agent, seen, _ = self.make_agent(mel_frames=94)
        self.assertEqual(len(agent._synthesize_codes(torch.zeros(1, 1), None, None)), 4)
        self.assertEqual(seen["prompt"][0, -3:, 0].tolist(), [93, 93, 93])

    def test_empty_output_does_not_load_or_decode_a_prompt(self):
        agent, _, _ = self.make_agent()
        self.assertEqual(agent._synthesize_codes(torch.empty(1, 0), None, None).size, 0)
        agent._build_flow_matching_prompt_audio.assert_not_called()

    def test_disabled_acoustic_prompt_preserves_native_behavior(self):
        agent, _, original = self.make_agent()
        agent.use_prompt_mel = False
        codes = torch.ones(1, 3)
        self.assertEqual(agent._synthesize_codes(codes, None, None).tolist(), [-1])
        original.assert_called_once_with(codes, None, None)

    def test_empty_reference_fails_without_producing_fake_audio(self):
        agent, _, _ = self.make_agent(codes=0)
        with self.assertRaisesRegex(RuntimeError, "empty conditioning prefix"):
            agent._synthesize_codes(torch.ones(1, 2), None, None)


if __name__ == "__main__":
    unittest.main()
