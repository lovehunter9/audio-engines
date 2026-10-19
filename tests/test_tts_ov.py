# The Intel loader must not replace the CUDA one until its own image calls run().
import os
import unittest
from unittest import mock

import numpy as np

from wrapper.caps import tts
from wrapper.caps import tts_ov


class _FakeOV:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.array([0.25, -0.25], dtype="float32"), 24000


class OpenVINOAdapterTest(unittest.TestCase):
    def test_import_leaves_the_cuda_loader_in_place(self):
        self.assertEqual(tts._load.__module__, "wrapper.caps.tts")
        self.assertIn("faster_qwen3_tts", tts._load.__code__.co_names)

    def test_custom_voice_and_clone_use_the_qwen_tts_spellings(self):
        ov = _FakeOV()
        model = tts_ov._OpenVINOQwen3TTS(ov, 24000)
        audio, sr = model.generate_custom_voice(
            text="你好", speaker="aiden", language="Chinese", instruct=None, chunk_size=8)
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape, (2,))
        self.assertNotIn("chunk_size", ov.calls[0])
        self.assertEqual(ov.calls[0]["speaker"], "aiden")

        model.generate_voice_clone(
            text="hi", language="English", ref_audio="/tmp/ref.wav", ref_text="",
            xvec_only=True, chunk_size=4)
        clone = ov.calls[1]
        self.assertTrue(clone["x_vector_only_mode"])
        self.assertIsNone(clone["ref_text"])
        self.assertNotIn("xvec_only", clone)
        self.assertNotIn("chunk_size", clone)

    def test_streaming_yields_the_whole_sentence_once(self):
        model = tts_ov._OpenVINOQwen3TTS(_FakeOV(), 24000)
        chunks = list(model.generate_custom_voice_streaming(
            text="Hi.", speaker="aiden", language="English", instruct=None, chunk_size=8))
        self.assertEqual(len(chunks), 1)
        audio, sr, timing = chunks[0]
        self.assertEqual(sr, 24000)
        self.assertIsNone(timing)
        self.assertGreater(audio.shape[0], 0)

    def test_intel_mode_selects_gpu_unless_device_is_set(self):
        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel", "ENGINE_ARGS": ""},
                             clear=False):
            fresh = tts.EngineArgs("")
            with mock.patch.object(tts, "_args", fresh):
                self.assertEqual(tts_ov._device(), "GPU")
                asked = tts.EngineArgs("--device CPU")
                with mock.patch.object(tts, "_args", asked):
                    self.assertEqual(tts_ov._device(), "CPU")

    def test_a_non_intel_pod_without_a_device_fails_closed(self):
        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "nvidia"}, clear=False):
            with mock.patch.object(tts, "_args", tts.EngineArgs("")):
                with self.assertRaises(RuntimeError):
                    tts_ov._device()


if __name__ == "__main__":
    unittest.main()
