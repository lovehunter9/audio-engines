"""The residual-code frame is one infer. Sliding attention is not this graph."""
import inspect
import os
import unittest
from unittest.mock import patch

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
        self.assertIn("load_in_8bit=load_8bit", src)
        self.assertIn("compress_frame_int8 if int8 else None", src)

    def test_0_6b_still_requests_int8_and_1_7b_does_not(self):
        small = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        large = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        self.assertFalse(tts_ov._exporter_auto_int8(small))
        self.assertTrue(tts_ov._exporter_auto_int8(large))
        self.assertTrue(tts_ov._load_in_8bit(True, small))
        self.assertFalse(tts_ov._load_in_8bit(True, large))
        self.assertFalse(tts_ov._load_in_8bit(False, small))
        self.assertFalse(tts_ov._load_in_8bit(False, large))

    def test_only_the_integrated_gpu_uses_int8(self):
        with patch.dict(os.environ, {"OLARES_GPU_MODE": "intel"}):
            self.assertTrue(tts_ov._igpu_int8())
        with patch.dict(os.environ, {"OLARES_GPU_MODE": "intel-gpu"}):
            self.assertFalse(tts_ov._igpu_int8())
        with patch.dict(os.environ, {"OLARES_GPU_MODE": "Intel"}):
            self.assertTrue(tts_ov._igpu_int8())

    def test_frame_stamp_follows_the_weight_format(self):
        self.assertEqual(
            tts_ov_frame.frame_cache_names(1024, 1024, None),
            ("qwen3tts-predictor-frame-v1.xml", ".qwen3tts-predictor-frame-v1"))
        self.assertEqual(
            tts_ov_frame.frame_cache_names(1024, 1024, object()),
            ("qwen3tts-predictor-frame-v2.xml", ".qwen3tts-predictor-frame-v2"))
        xml, stamp = tts_ov_frame.frame_cache_names(2048, 1024, None)
        self.assertEqual(xml, "qwen3tts-predictor-frame-h2048-1024-v1.xml")
        self.assertNotIn("qwen3tts-predictor-frame-v1.xml", (xml,))
        self.assertNotEqual(stamp, ".qwen3tts-predictor-frame-v1")
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
