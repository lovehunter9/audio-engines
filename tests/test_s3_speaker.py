import unittest

import numpy as np

try:
    import openvino as ov
except ImportError:
    ov = None

from wrapper.caps.s3_ov_pcm import _speaker_pooling


@unittest.skipIf(ov is None, "OpenVINO is required for speaker pooling equivalence")
class SpeakerPoolingTests(unittest.TestCase):
    def test_short_and_partial_windows_match_the_original_runtime_divisor(self):
        from openvino import opset13 as ops
        source = ops.parameter([1, 2, -1], np.float32)
        pool = ops.avg_pool(source, [100], [0], [0], [100], True, "ceil")
        model = ov.Model([pool], [source])
        self.assertEqual(_speaker_pooling(model), 1)
        compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        rng = np.random.default_rng(13)
        for length in (1, 63, 100, 101, 263, 391, 692):
            with self.subTest(length=length):
                values = rng.normal(size=(1, 2, length)).astype(np.float32)
                padded = np.pad(values, ((0, 0), (0, 0), (0, -length % 100)))
                expected = padded.reshape(1, 2, -1, 100).mean(axis=-1)
                np.testing.assert_allclose(compiled([values])[0], expected, rtol=2e-5, atol=2e-6)
