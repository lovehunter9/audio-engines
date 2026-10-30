"""Ming OpenVINO wiring, with no weights and no GPU."""
import os
import unittest
from pathlib import Path
from unittest import mock

from wrapper.caps import ming_backend, ming_ov


ROOT = Path(__file__).resolve().parents[1]


class BucketTests(unittest.TestCase):
    def test_aggregator_prompt_batch_is_dynamic(self):
        class _T:
            def __init__(self, shape):
                self.shape = shape

        self.assertEqual(ming_ov.AGGREGATOR_DYNAMIC, (0,))
        self.assertEqual(
            ming_ov._export_shapes(_T((1, 1, 64)), dynamic=(0,)), [(-1, 1, 64)])
        self.assertEqual(ming_ov._export_shapes(_T((1, 1, 64))), [(1, 1, 64)])
        with self.assertRaises(ValueError):
            ming_ov._export_shapes(_T((1, 1, 64)), dynamic=(3,))
        text = (ROOT / "wrapper/caps/ming_ov.py").read_text()
        self.assertIn("dynamic=AGGREGATOR_DYNAMIC", text)

    def test_campplus_span_is_two_hundred_frames(self):
        self.assertEqual(ming_ov.campplus_span(288), (44, 244, 0))
        self.assertEqual(ming_ov.campplus_span(200), (0, 200, 0))
        self.assertEqual(ming_ov.campplus_span(80), (0, 80, 120))
        with self.assertRaises(ValueError):
            ming_ov.campplus_span(0)

    def test_smallest_bucket(self):
        self.assertEqual(ming_ov.bucket_for(1, ming_ov.VAE_FRAME_BUCKETS), 64)
        self.assertEqual(ming_ov.bucket_for(64, ming_ov.VAE_FRAME_BUCKETS), 64)
        self.assertEqual(ming_ov.bucket_for(65, ming_ov.VAE_FRAME_BUCKETS), 128)
        with self.assertRaises(RuntimeError):
            ming_ov.bucket_for(ming_ov.VAE_FRAME_BUCKETS[-1] + 1, ming_ov.VAE_FRAME_BUCKETS)

    def test_load_delegates_only_for_the_intel_base(self):
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "mingov"}):
            with mock.patch("wrapper.caps.ming_ov.load_model", return_value="ov") as load:
                self.assertEqual(ming_backend.load_model("inclusionAI/Ming-3.0-tts-0.5B"), "ov")
                load.assert_called_once_with("inclusionAI/Ming-3.0-tts-0.5B")

    def test_saved_xml_shape_rejects_the_dynamic_vae(self):
        body = '<layer type="Parameter"><data shape="?,?,?" /></layer>'
        path = ROOT / "tests" / "_vae_shape.xml"
        path.write_text(body)
        try:
            self.assertEqual(ming_ov._saved_input_shapes(str(path)), [(-1, -1, -1)])
        finally:
            path.unlink()

    def test_onnxruntime_stub_survives_find_spec(self):
        import importlib.util
        import sys

        saved = sys.modules.pop("onnxruntime", None)
        try:
            ming_ov._hide_cpu_onnxruntime()
            self.assertIsNotNone(importlib.util.find_spec("onnxruntime"))
        finally:
            sys.modules.pop("onnxruntime", None)
            if saved is not None:
                sys.modules["onnxruntime"] = saved

    def test_gpu_is_required_and_kv_is_stateful(self):
        text = (ROOT / "wrapper/caps/ming_ov.py").read_text()
        self.assertIn("require_gpu", text)
        self.assertIn("refusing to compile it on GPU", text)
        self.assertIn("_saved_input_shapes", text)
        self.assertIn("def _static_frames", text)
        self.assertIn("index_select", text)
        self.assertIn("DeviceKvRunner", text)
        self.assertIn("compile_causal", text)
        self.assertNotIn("MakeStateful", text)
        self.assertNotIn("CPUExecutionProvider", text)
        self.assertNotIn('compile_model(model, "CPU"', text)
        docker = (ROOT / "bases/mingov/deps.Dockerfile").read_text()
        self.assertNotIn("nvidia/cuda", docker)
        self.assertIn("openvino", docker)
        self.assertIn("download.pytorch.org/whl/cpu", docker)


if __name__ == "__main__":
    unittest.main()
