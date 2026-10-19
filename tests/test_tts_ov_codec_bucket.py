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
