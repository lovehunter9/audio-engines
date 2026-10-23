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
        self.assertTrue(torch.equal(_sim_logits(cond, 2.0), cond))
        self.assertTrue(torch.equal(_sim_logits(cond, 0.4), cond))
        self.assertLess(_sim_temperature(1.0, 2.0), 1.0)
        self.assertLess(_sim_top_p(0.8, 2.0), 0.8)
        # A flat knob keeps the model's temperature and closes an open nucleus.
        self.assertLessEqual(_sim_temperature(1.0, 0.7), 1.0)
        self.assertAlmostEqual(_sim_temperature(0.9, 0.85), 0.9)
        self.assertGreater(_sim_top_p(0.5, 0.7), 0.5)
        self.assertAlmostEqual(_sim_top_p(1.0, 0.85), 0.95)

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

    def test_short_frame_only_when_the_knob_sharpens(self):
        try:
            from wrapper.caps.breeze import _knob_uses_short_frame
        except Exception as exc:
            self.skipTest("breeze deps missing: %s" % exc)
        self.assertFalse(_knob_uses_short_frame(False, 4.0))
        self.assertFalse(_knob_uses_short_frame(True, 1.0))
        self.assertFalse(_knob_uses_short_frame(True, 0.85))
        self.assertFalse(_knob_uses_short_frame(True, 0.4))
        self.assertTrue(_knob_uses_short_frame(True, 1.15))
        self.assertTrue(_knob_uses_short_frame(True, 1.825))

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

    def test_collapsed_short_frame_repeats_the_previous_draw(self):
        try:
            import numpy as np

            from wrapper.caps.breeze import _knob_keep_speech
        except Exception as exc:
            self.skipTest("breeze deps missing: %s" % exc)
        quiet = [0, 0, 0, 0, 0]
        # Book 0 is the backbone. Depth slots read books 1..4. Floor energy is 1+1+1+1.
        norms = [[1.0, 9.0]] + [[1.0, 8.0, 30.0] for _ in range(4)]
        prev = np.array([2, 2, 2, 2])
        collapsed = np.array([0, 0, 0, 0])
        held, did = _knob_keep_speech(collapsed, quiet, norms, prev)
        self.assertTrue(did)
        self.assertTrue(np.array_equal(held, prev))
        speech = np.array([2, 2, 2, 2])
        kept, did = _knob_keep_speech(speech, quiet, norms, prev)
        self.assertFalse(did)
        self.assertTrue(np.array_equal(kept, speech))
        first, did = _knob_keep_speech(collapsed, quiet, norms, None)
        self.assertFalse(did)
        self.assertTrue(np.array_equal(first, collapsed))

    def test_depth_cfg_sample_keeps_the_only_finite_logit(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        scores = torch.tensor([float("-inf"), 4.0, float("-inf")])
        tok = _depth_cfg_token(scores, 1.0, 0, 1.0, 1, 3, 3)
        self.assertEqual(tok, 1)


if __name__ == "__main__":
    unittest.main()
