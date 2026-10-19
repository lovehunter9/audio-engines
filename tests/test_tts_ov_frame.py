"""The residual-code frame is one infer. Sliding attention is not this graph."""
import inspect
import unittest

from wrapper.caps import tts_ov
from wrapper.caps import tts_ov_frame
from wrapper.caps.tts_ov_frame import residual_steps


class _Cfg:
    def __init__(self, kinds, groups):
        self.layer_types = kinds
        self.num_code_groups = groups


class PredictorFrameTest(unittest.TestCase):
    def test_load_installs_the_frame(self):
        src = inspect.getsource(tts_ov._load)
        self.assertIn("tts_ov_frame.install", src)
        self.assertIn("load_in_8bit=True", src)
        self.assertIn("compress_frame_int8", src)

    def test_frame_stamp_is_the_int8_graph(self):
        src = inspect.getsource(tts_ov_frame.install)
        self.assertIn("qwen3tts-predictor-frame-v2", src)
        self.assertIn("compress(xml)", src)
        self.assertNotIn("codec", inspect.getsource(tts_ov.compress_frame_int8))

    def test_full_attention_counts_the_residual_codes(self):
        self.assertEqual(residual_steps(_Cfg(["full_attention"] * 5, 16)), 15)

    def test_sliding_attention_is_refused(self):
        with self.assertRaises(RuntimeError):
            residual_steps(_Cfg(["full_attention", "sliding_attention"], 16))

    def test_a_single_group_is_refused(self):
        with self.assertRaises(RuntimeError):
            residual_steps(_Cfg(["full_attention"], 1))


if __name__ == "__main__":
    unittest.main()
