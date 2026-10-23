"""CFG mix, left-pad split, and the depth draw. Skipped when torch is not installed."""
import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from wrapper.caps.breeze import (
        _cfg_mix, _depth_cfg_token, _ov_real_rows,
        _sim_logits, _sim_temperature, _sim_top_p,
    )


class BreezeOvCfgTest(unittest.TestCase):
    def test_mix_matches_the_official_formula(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        cond = torch.tensor([2.0, 0.0])
        uncond = torch.tensor([0.0, 2.0])
        got = _cfg_mix(cond, uncond, 1.5)
        self.assertTrue(torch.allclose(got, torch.tensor([3.0, -1.0])))

    def test_rows_drop_a_left_pad(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        embeds = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        rows = _ov_real_rows(embeds, mask)
        self.assertEqual(tuple(rows[0].shape), (1, 2, 2))
        self.assertEqual(tuple(rows[1].shape), (1, 4, 2))
        self.assertTrue(torch.equal(rows[0], embeds[0:1, 2:]))

    def test_rows_reject_a_hole(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        embeds = torch.zeros(2, 4, 2)
        mask = torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]])
        with self.assertRaises(RuntimeError):
            _ov_real_rows(embeds, mask)

    def test_sim_matches_the_fast_path_at_scale_one(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        cond = torch.tensor([[1.0, 3.0, 2.0]])
        self.assertTrue(torch.equal(_sim_logits(cond, 1.0), cond))
        self.assertEqual(_sim_temperature(0.8, 1.0), 0.8)
        self.assertAlmostEqual(_sim_top_p(0.9, 1.0), 0.9)

    def test_sim_sharpens_above_one_and_flattens_below(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        cond = torch.tensor([[0.0, 2.0, 1.0]])
        got = _sim_logits(cond, 2.0)
        self.assertTrue(torch.allclose(got, torch.tensor([[-4.0, 6.0, 1.0]])))
        self.assertLess(_sim_temperature(1.0, 2.0), 1.0)
        self.assertLess(_sim_top_p(0.8, 2.0), 0.8)
        self.assertGreater(_sim_temperature(1.0, 0.7), 1.0)
        self.assertGreater(_sim_top_p(0.5, 0.7), 0.5)

    def test_depth_cfg_greedy_and_masked_tail(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        scores = torch.tensor([0.1, 0.2, 3.0, 0.0])
        tok = _depth_cfg_token(scores, 1.0, 0, 1.0, 0, 4, 4)
        self.assertEqual(tok, 2)
        masked = _depth_cfg_token(torch.tensor([0.0, 9.0]), 1.0, 0, 1.0, 0, 1, 2)
        self.assertEqual(masked, 0)

    def test_four_knobs_gate_the_short_frame_and_speed_does_not(self):
        try:
            from wrapper.caps.breeze import _knobs_request_fast
        except Exception as exc:
            self.skipTest("breeze deps missing: %s" % exc)
        self.assertFalse(_knobs_request_fast(None))
        self.assertFalse(_knobs_request_fast({}))
        self.assertFalse(_knobs_request_fast({"speed": 1.25}))
        self.assertFalse(_knobs_request_fast({
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": 0.5,
        }))
        self.assertTrue(_knobs_request_fast({"stability": 0.0}))
        self.assertTrue(_knobs_request_fast({"similarity_boost": 1.0}))
        self.assertTrue(_knobs_request_fast({"style": 1.0}))
        self.assertTrue(_knobs_request_fast({"use_speaker_boost": False}))

    def test_knob_tail_uses_each_later_books_quiet_code(self):
        try:
            import numpy as np

            from wrapper.caps.breeze import _knob_tail_codes
        except Exception as exc:
            self.skipTest("breeze deps missing: %s" % exc)
        drawn = np.array([3, 4, 5, 6])
        quiet = list(range(100, 116))
        got = _knob_tail_codes(drawn, 15, quiet)
        self.assertEqual(got.tolist(), [3, 4, 5, 6] + list(range(105, 116)))
        self.assertTrue(np.array_equal(_knob_tail_codes(drawn[:4], 4, quiet), drawn))
        with self.assertRaises(RuntimeError):
            _knob_tail_codes(drawn, 15, quiet[:5])

    def test_depth_cfg_sample_keeps_the_only_finite_logit(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        scores = torch.tensor([float("-inf"), 4.0, float("-inf")])
        tok = _depth_cfg_token(scores, 1.0, 0, 1.0, 1, 3, 3)
        self.assertEqual(tok, 1)


if __name__ == "__main__":
    unittest.main()
