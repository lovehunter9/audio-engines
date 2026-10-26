"""CFG row split and the one-pass depth retune. Skipped when torch is not installed."""
import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    from wrapper.caps.breeze import (
        _ov_real_rows, _sim_logits, _sim_temperature, _sim_top_p,
    )


class BreezeOvCfgTest(unittest.TestCase):
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

    def test_sim_keeps_logits_and_temperature_and_closes_a_flat_nucleus(self):
        if not _HAS_TORCH:
            self.skipTest("torch is not installed")
        cond = torch.tensor([[0.0, 2.0, 1.0]])
        self.assertTrue(torch.equal(_sim_logits(cond, 2.0), cond))
        self.assertTrue(torch.equal(_sim_logits(cond, 0.4), cond))
        self.assertEqual(_sim_temperature(1.0, 2.0), 1.0)
        self.assertLess(_sim_top_p(0.8, 2.0), 0.8)
        self.assertEqual(_sim_temperature(1.0, 0.7), 1.0)
        self.assertAlmostEqual(_sim_temperature(0.9, 0.85), 0.9)
        self.assertGreater(_sim_top_p(0.5, 0.7), 0.5)
        self.assertAlmostEqual(_sim_top_p(1.0, 0.85), 0.95)
        with self.assertRaises(RuntimeError):
            _sim_logits(cond, None)
        with self.assertRaises(RuntimeError):
            _sim_temperature(1.0, None)


if __name__ == "__main__":
    unittest.main()
