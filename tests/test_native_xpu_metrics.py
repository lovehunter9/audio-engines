import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from wrapper import gpu


class MetricsTests(unittest.TestCase):
    def test_xpu_metrics_use_own_allocator_and_declared_quota(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("capacity or inventory must not be queried")

        fake = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            xpu=SimpleNamespace(
                is_available=lambda: True,
                memory_allocated=lambda: 1234,
                get_device_properties=forbidden,
                device_count=forbidden,
            ),
        )
        with (
            patch.dict(sys.modules, torch=fake),
            patch.dict(os.environ, AUDIO_BASE="seamlessxpu", REQUIRED_GPU_MEMORY="16Gi"),
        ):
            text = gpu.gpu_metrics_text()
        self.assertIn("gpu_present 1", text)
        self.assertIn("gpu_mem_used_bytes 1234", text)
        self.assertIn("gpu_mem_total_bytes 17179869184", text)
        self.assertIn("configured scheduling quota", text)
