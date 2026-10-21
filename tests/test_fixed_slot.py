"""One fixed-slot backbone step must match a full causal prefill. The module still imports when torch is absent."""
import importlib.util
import tempfile
import unittest

from wrapper import tts_ov

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_OV = importlib.util.find_spec("openvino") is not None
if _HAS_TORCH:
    import torch
    import torch.nn as nn


if _HAS_TORCH:
    class _Attn(nn.Module):
        def __init__(self, hidden, n_q, n_kv, head_dim):
            super().__init__()
            self.head_dim = head_dim
            self.scaling = head_dim ** -0.5
            self.q_proj = nn.Linear(hidden, n_q * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
            self.o_proj = nn.Linear(n_q * head_dim, hidden, bias=False)
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)

    class _Layer(nn.Module):
        def __init__(self, hidden, n_q, n_kv, head_dim):
            super().__init__()
            self.self_attn = _Attn(hidden, n_q, n_kv, head_dim)
            self.input_layernorm = nn.LayerNorm(hidden)
            self.post_attention_layernorm = nn.LayerNorm(hidden)
            self.mlp = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.ReLU())

    class _Rotary(nn.Module):
        def __init__(self, head_dim):
            super().__init__()
            inv = torch.arange(0, head_dim, 2).float()
            self.register_buffer("inv", inv, persistent=False)

        def forward(self, hidden, pos):
            ang = pos.to(dtype=hidden.dtype).unsqueeze(-1) * self.inv.to(hidden.dtype)
            cos = torch.cos(ang).repeat_interleave(2, dim=-1)
            sin = torch.sin(ang).repeat_interleave(2, dim=-1)
            return cos, sin

    class _Inner(nn.Module):
        def __init__(self, n_layers, hidden, n_q, n_kv, head_dim):
            super().__init__()
            self.layers = nn.ModuleList(
                [_Layer(hidden, n_q, n_kv, head_dim) for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(hidden)
            self.rotary_emb = _Rotary(head_dim)
            self.config = type("C", (), {"num_hidden_layers": n_layers})()

    def _build():
        torch.manual_seed(0)
        inner = _Inner(n_layers=2, hidden=32, n_q=4, n_kv=2, head_dim=8)
        inner.eval()
        return inner


class FixedSlotMatchTest(unittest.TestCase):
    @unittest.skipUnless(_HAS_TORCH, "needs torch")
    def test_slot_steps_match_full_prefill(self):
        inner = _build()
        full = tts_ov.causal_kv_module(inner, False).eval()
        slot = tts_ov.causal_fixed_slot_module(inner, 8).eval()
        emb = torch.randn(1, 5, 32)
        with torch.inference_mode():
            whole = full(emb, torch.ones(1, 5, dtype=torch.long))
            cache = [torch.zeros(1, 2, 8, 8) for _ in range(4)]
            got = []
            for i in range(5):
                mask = torch.zeros(1, 1, 8, dtype=torch.long)
                mask[0, 0, : i + 1] = 1
                out = slot(
                    emb[:, i:i + 1],
                    mask,
                    torch.tensor([[i]]),
                    torch.tensor([i]),
                    *cache,
                )
                got.append(out[0])
                cache = list(out[1:])
        hidden = torch.cat(got, dim=1)
        self.assertTrue(torch.allclose(hidden, whole[0], atol=1e-5, rtol=1e-5))
        for layer in range(2):
            self.assertTrue(torch.allclose(
                cache[2 * layer][:, :, :5], whole[1 + 2 * layer], atol=1e-5, rtol=1e-5,
            ))
            self.assertTrue(torch.allclose(
                cache[2 * layer + 1][:, :, :5], whole[2 + 2 * layer], atol=1e-5, rtol=1e-5,
            ))

    @unittest.skipUnless(_HAS_TORCH, "needs torch")
    def test_one_prefill_matches_full(self):
        inner = _build()
        full = tts_ov.causal_kv_module(inner, False).eval()
        slot = tts_ov.causal_fixed_slot_module(inner, 8).eval()
        emb = torch.randn(1, 5, 32)
        mask = torch.zeros(1, 5, 8, dtype=torch.long)
        for i in range(5):
            mask[0, i, : i + 1] = 1
        with torch.inference_mode():
            whole = full(emb, torch.ones(1, 5, dtype=torch.long))
            out = slot(
                emb, mask,
                torch.arange(5).unsqueeze(0),
                torch.arange(5),
                *[torch.zeros(1, 2, 8, 8) for _ in range(4)],
            )
        self.assertTrue(torch.allclose(out[0], whole[0], atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(_HAS_TORCH and _HAS_OV, "needs torch and openvino")
    def test_compiled_slot_matches_torch(self):
        inner = _build()
        slot = tts_ov.causal_fixed_slot_module(inner, 8).eval()
        emb = torch.randn(1, 6, 32)
        past = [torch.zeros(1, 2, 8, 8) for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            _, xml, stamp = tts_ov.ir_paths(tmp, "slot", ".ov-slot-test")
            compiled = tts_ov.compile_causal(
                slot,
                (
                    torch.zeros(1, 4, 32),
                    torch.ones(1, 4, 8, dtype=torch.long),
                    torch.arange(4).unsqueeze(0),
                    torch.arange(4),
                    *past,
                ),
                xml, stamp, "CPU",
                dynamize_ranks=(1, 2, 3),
            )
            runner = tts_ov.FixedSlotRunner(compiled, 2, 8, 2, 8)
            with torch.inference_mode():
                got = runner.prefill(emb[:, :5])
                nxt = runner.step(emb[:, 5:6])
                whole = tts_ov.causal_kv_module(inner, False)(
                    emb, torch.ones(1, 6, dtype=torch.long),
                )
        self.assertEqual(runner.slot, 6)
        # Export width is 4. A frozen width fails the longer prefill; f32 sits near 1e-3 of torch.
        joined = torch.cat([got, nxt], dim=1)
        self.assertTrue(torch.allclose(joined, whole[0], atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
