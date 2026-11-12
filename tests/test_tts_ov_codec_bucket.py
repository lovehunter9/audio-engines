"""The Intel codec pads each window to a fixed frame count and crops the wave back."""
import unittest

import numpy as np

from wrapper.caps.tts_ov import _bucketing_forward, _codec_bucket


class CodecBucketTest(unittest.TestCase):
    def test_frames_land_on_the_next_bucket(self):
        self.assertEqual(_codec_bucket(1), 64)
        self.assertEqual(_codec_bucket(38), 64)
        self.assertEqual(_codec_bucket(64), 64)
        self.assertEqual(_codec_bucket(97), 128)
        self.assertEqual(_codec_bucket(256), 256)
        self.assertEqual(_codec_bucket(300), 384)
        self.assertEqual(_codec_bucket(400), 400)

    def test_a_short_window_is_padded_and_the_wave_keeps_the_real_frames(self):
        seen = []

        def orig(codes):
            seen.append(int(codes.shape[-1]))
            width = int(codes.shape[-1]) * 1920
            return np.zeros(codes.shape[:-1] + (width,), dtype=np.float32)

        forward = _bucketing_forward(orig)
        wav = forward(np.zeros((1, 16, 97), dtype=np.int64))
        self.assertEqual(seen, [128])
        self.assertEqual(wav.shape[-1], 97 * 1920)

    def test_a_window_already_on_a_bucket_is_not_padded(self):
        seen = []

        def orig(codes):
            seen.append((int(codes.shape[-1]), int(codes.sum())))
            width = int(codes.shape[-1]) * 1920
            return np.ones(codes.shape[:-1] + (width,), dtype=np.float32)

        forward = _bucketing_forward(orig)
        codes = np.ones((1, 16, 64), dtype=np.int64)
        wav = forward(codes)
        self.assertEqual(seen, [(64, 16 * 64)])
        self.assertEqual(wav.shape[-1], 64 * 1920)

    def test_the_pad_is_zeros_on_the_right(self):
        seen = []

        def orig(codes):
            seen.append(codes.copy())
            return np.zeros((codes.shape[-1] * 4,), dtype=np.float32)

        forward = _bucketing_forward(orig)
        codes = np.arange(6, dtype=np.int64).reshape(1, 1, 6)
        forward(codes)
        padded = seen[0]
        self.assertEqual(int(padded.shape[-1]), 64)
        self.assertTrue(np.array_equal(padded[..., :6], codes))
        self.assertEqual(int(padded[..., 6:].sum()), 0)


class StaticCodecTest(unittest.TestCase):
    def test_compiles_exact_bucket_shape_and_reuses_only_matching_shapes(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        from wrapper.caps.tts_ov import _static_codec_forward

        class Model:
            shape = None

            def clone(self):
                return Model()

            def reshape(self, shapes):
                self.shape = tuple(shapes[0])

        calls = []
        buffer = np.zeros((1, 1, 4), dtype=np.float32)

        def compile_model(model, device, options):
            calls.append((model.shape, device, options))

            def infer(inputs):
                self.assertEqual(inputs[0].shape, model.shape)
                self.assertTrue(inputs[0].flags.c_contiguous)
                buffer.fill(inputs[0].sum())
                return {0: buffer}
            return infer

        graph = Model()
        part = SimpleNamespace(model=graph, _device="GPU", ov_config={"CACHE_DIR": "/cache"})
        modules = {"torch": SimpleNamespace(from_numpy=lambda x: x),
                   "optimum.intel.openvino.modeling_base": SimpleNamespace(
                       core=SimpleNamespace(compile_model=compile_model))}
        with patch.dict("sys.modules", modules), patch(
                "wrapper.ov_compile_lock.gpu_compile", return_value=nullcontext()) as lock:
            forward = _static_codec_forward(part)
            first = forward(np.ones((1, 16, 384), dtype=np.int64))
            second = forward(np.full((1, 16, 384), 2, dtype=np.int64))
            third = forward(np.full((1, 16, 256), 3, dtype=np.int64))
        self.assertEqual(calls, [((1, 16, 384), "GPU", {"CACHE_DIR": "/cache"}),
                                 ((1, 16, 256), "GPU", {"CACHE_DIR": "/cache"})])
        self.assertEqual(lock.call_count, 2)
        self.assertIsNone(graph.shape)
        np.testing.assert_array_equal(first, 16 * 384)
        np.testing.assert_array_equal(second, 2 * 16 * 384)
        np.testing.assert_array_equal(third, 3 * 16 * 256)
        self.assertFalse(np.shares_memory(first, third))

    def test_discrete_gpu_keeps_the_original_decoder(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from wrapper.caps import tts_ov

        original = Mock(return_value=np.zeros((1, 1, 384), dtype=np.float32))
        original._codec_buckets = False
        decoder = SimpleNamespace(forward=original)
        ov = SimpleNamespace(codec_decoder=object(), model=SimpleNamespace(
            speech_tokenizer=SimpleNamespace(model=SimpleNamespace(decoder=decoder))))
        with patch.dict(os.environ, {"OLARES_GPU_MODE": "intel-gpu"}), patch.object(
                tts_ov, "_static_codec_forward") as static:
            tts_ov._install_codec_buckets(ov)
            decoder.forward(np.zeros((1, 16, 300), dtype=np.int64))
        static.assert_not_called()
        self.assertEqual(original.call_args.args[0].shape[-1], 384)

if __name__ == "__main__":
    unittest.main()
