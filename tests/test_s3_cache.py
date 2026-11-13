from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from wrapper.caps.s3_core import Engine


class GateCacheTests(unittest.TestCase):
    def test_reuse_repeated_prompt_and_rebuild_changed_prefix(self):
        engine = Engine.__new__(Engine)
        engine.torch = SimpleNamespace(equal=np.array_equal, cat=lambda xs: np.concatenate(xs))
        calls = []

        def forward(x, cache=None):
            calls.append((x.copy(), cache))
            output = SimpleNamespace(past_key_values=len(calls))
            return output, x[0].copy(), x[:, -1].copy()

        engine._forward = forward
        state = SimpleNamespace(
            gate_prefix=None, gate_cache=None, gate_hidden=None, gate_result=None
        )
        first = np.arange(6).reshape(1, 3, 2)
        engine._gate_forward(state, first)
        extended = np.arange(10).reshape(1, 5, 2)
        output = engine._gate_forward(state, extended)
        np.testing.assert_array_equal(calls[1][0], extended[:, 3:])
        self.assertEqual(calls[1][1], 1)
        np.testing.assert_array_equal(output[1], extended[0])
        self.assertIs(engine._gate_forward(state, extended), output)
        self.assertEqual(len(calls), 2)
        changed = extended.copy()
        changed[0, 0, 0] = -1
        engine._gate_forward(state, changed)
        self.assertIsNone(calls[-1][1])
        np.testing.assert_array_equal(calls[-1][0], changed)

    def test_published_adaptive_policy_and_explicit_override(self):
        engine = Engine.__new__(Engine)
        engine.threshold = None
        play = SimpleNamespace(threshold=Mock(return_value=(0.73, {})))
        state = SimpleNamespace(speech=SimpleNamespace(pcm=SimpleNamespace(play=play)))
        self.assertEqual(engine._threshold(state), 0.73)
        play.threshold.assert_called_once_with(0.1, 0.99, 1.0, 0.5)
        engine.threshold = 0.9
        self.assertEqual(engine._threshold(state), 0.9)
        engine.threshold = None
        state.speech = None
        self.assertEqual(engine._threshold(state), 0.5)


if __name__ == "__main__":
    unittest.main()
