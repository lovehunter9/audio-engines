import unittest

import numpy as np

try:
    import openvino as ov
except ImportError:
    ov = None

from wrapper.caps.ov_gpu_graph import promote_1d_convolutions


@unittest.skipIf(ov is None, "OpenVINO runtime is required for numerical graph equivalence")
class ConvolutionPromotionTests(unittest.TestCase):
    def test_original_convolution_variants_preserve_samples(self):
        from openvino import opset13 as ops
        rng = np.random.default_rng(41)
        core = ov.Core()
        for kind in ("convolution", "group_convolution", "convolution_backprop_data", "group_convolution_backprop_data"):
            for length in (17, 32):
                with self.subTest(kind=kind, length=length):
                    source = ops.parameter([1, 4, length], np.float32)
                    shape = [2, 2, 2, 3] if kind.startswith("group") else [4, 4, 3]
                    weight = rng.normal(0, .1, shape).astype(np.float32)
                    convolution = getattr(ops, kind)(source, weight, strides=[2], pads_begin=[1], pads_end=[1], dilations=[1])
                    original = ov.Model([convolution], [source])
                    transformed = original.clone()
                    self.assertEqual(promote_1d_convolutions(transformed), 1)
                    data = rng.normal(size=[1, 4, length]).astype(np.float32)
                    expected = core.compile_model(original, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})([data])[0].copy()
                    actual = core.compile_model(transformed, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})([data])[0]
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
