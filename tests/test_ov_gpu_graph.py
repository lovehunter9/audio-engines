import unittest
from types import SimpleNamespace

import numpy as np

try:
    import openvino as ov
except ImportError:
    ov = None

from wrapper.caps.ov_gpu_graph import infer_gpu, promote_1d_convolutions


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


class RequestIsolationTests(unittest.TestCase):
    def test_shapes_use_fresh_requests_and_outputs_own_their_samples(self):
        requests = []
        class Request:
            def infer(self, inputs, share_inputs=False):
                self.data = inputs[0]
            def get_output_tensor(self, index):
                return SimpleNamespace(data=self.data)
        def create():
            value = Request()
            requests.append(value)
            return value
        compiled = SimpleNamespace(create_infer_request=create, outputs=[0])
        first = np.arange(32, dtype=np.float32)
        a = infer_gpu(compiled, [first])[0]
        b = infer_gpu(compiled, [np.arange(40, dtype=np.float32)])[0]
        first[:] = -1
        np.testing.assert_array_equal(a, np.arange(32, dtype=np.float32))
        np.testing.assert_array_equal(b, np.arange(40, dtype=np.float32))
        self.assertEqual(len(requests), 2)
        self.assertIsNot(requests[0], requests[1])
