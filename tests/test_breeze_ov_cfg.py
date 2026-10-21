"""CFG mix, left-pad split, and the depth draw. Skipped when torch is not installed."""
import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from wrapper.caps.breeze import _cfg_mix, _depth_cfg_token, _ov_real_rows


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

    def test_depth_cfg_greedy_and_masked_tail(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        scores = torch.tensor([0.1, 0.2, 3.0, 0.0])
        tok = _depth_cfg_token(scores, 1.0, 0, 1.0, 0, 4, 4)
        self.assertEqual(tok, 2)
        masked = _depth_cfg_token(torch.tensor([0.0, 9.0]), 1.0, 0, 1.0, 0, 1, 2)
        self.assertEqual(masked, 0)

    def test_depth_cfg_sample_keeps_the_only_finite_logit(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        scores = torch.tensor([float("-inf"), 4.0, float("-inf")])
        tok = _depth_cfg_token(scores, 1.0, 0, 1.0, 1, 3, 3)
        self.assertEqual(tok, 1)


if __name__ == "__main__":
    unittest.main()
