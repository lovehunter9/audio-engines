# The Intel path is an image switch. These checks do not import NeMo or OpenVINO.
import os
import unittest
from unittest import mock

from wrapper.caps import diar_nemotron_ov


class NemotronOvGateTests(unittest.TestCase):
    def test_cuda_image_does_not_select_openvino(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_BASE", None)
            self.assertFalse(diar_nemotron_ov.wants())
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "nemotron"}, clear=False):
            self.assertFalse(diar_nemotron_ov.wants())

    def test_ov_image_selects_openvino(self):
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "nemotron-ov"}, clear=False):
            self.assertTrue(diar_nemotron_ov.wants())

    def test_intel_mode_asks_for_gpu_and_a_cpu_box_does_not(self):
        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel"}, clear=False):
            self.assertEqual(diar_nemotron_ov.device(), "GPU")
        with mock.patch.dict(
            os.environ, {"OLARES_GPU_MODE": "", "REQUIRED_GPU_MEMORY": ""}, clear=False
        ):
            self.assertEqual(diar_nemotron_ov.device(), "CPU")

    def test_cuda_load_still_picks_cuda_before_the_ov_hook(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "wrapper/caps/diar_nemotron.py",
        )
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        cuda = text.index('dev = "cuda" if torch.cuda.is_available() else "cpu"')
        hook = text.index('== "nemotron-ov"')
        self.assertLess(cuda, hook)
        self.assertIn("from .diar_nemotron_ov import install as _install_ov", text)
