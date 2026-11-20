import json
import os
import tempfile
import unittest

from wrapper.caps import torch_asr as t


class KindTest(unittest.TestCase):
    def test_firered_is_its_checkpoint_and_cmvn(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ("model.pth.tar", "cmvn.ark"):
                open(os.path.join(d, f), "w").close()
            self.assertEqual(t.kind_of(d), t.KIND_FIRERED)

    def test_mimo_is_named_by_its_architecture(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"architectures": ["MiMoV2ASRForCausalLM"], "model_type": "qwen2"}, f)
            self.assertEqual(t.kind_of(d), t.KIND_MIMO)

    def test_anything_else_is_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"architectures": ["Qwen2ForCausalLM"]}, f)
            self.assertIsNone(t.kind_of(d))


class PlanBatchesTest(unittest.TestCase):
    def test_longest_first_and_capped_by_count(self):
        self.assertEqual(t.plan_batches([1, 5, 3, 4, 2], 2, 100), [[1, 3], [2, 4], [0]])

    def test_padded_size_caps_a_batch(self):
        # Padding to the longest: three spans of 40 would be 120 > 100.
        self.assertEqual(t.plan_batches([40, 40, 40], 8, 100), [[0, 1], [2]])

    def test_a_span_alone_is_never_refused(self):
        self.assertEqual(t.plan_batches([500], 8, 100), [[0]])


class FlashShimTest(unittest.TestCase):
    def test_matches_attention_inside_each_packed_sequence(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("no torch")
        torch.manual_seed(0)
        q, k, v = (torch.randn(7, 2, 4) for _ in range(3))
        cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        out = t.flash_attn_varlen_func(q, k, v, cu, cu, 4, 4, causal=False)
        for a, b in ((0, 3), (3, 7)):
            want = F.scaled_dot_product_attention(
                q[a:b].transpose(0, 1), k[a:b].transpose(0, 1), v[a:b].transpose(0, 1)).transpose(0, 1)
            self.assertTrue(torch.allclose(out[a:b], want, atol=1e-6))

    def test_refuses_windowed_attention(self):
        with self.assertRaises(NotImplementedError):
            t.flash_attn_varlen_func(None, None, None, None, None, 0, 0, window_size=(8, 8))

    def test_installed_shim_survives_find_spec(self):
        import importlib.util
        import sys

        saved = sys.modules.pop("flash_attn", None)
        try:
            t._install_flash_attn_shim()
            self.assertIsNotNone(importlib.util.find_spec("flash_attn"))
            from flash_attn import flash_attn_varlen_func
            self.assertIs(flash_attn_varlen_func, t.flash_attn_varlen_func)
        finally:
            sys.modules.pop("flash_attn", None)
            if saved is not None:
                sys.modules["flash_attn"] = saved


if __name__ == "__main__":
    unittest.main()
