"""The OpenVINO startup lock and the one speakrs SIGSEGV retry. CUDA load and speakrs' single start must not gain either."""
import inspect
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from wrapper import ov_compile_lock
from wrapper.caps import align, diar_speakrs, stt_stream
from wrapper import watchdog


class CompileLockTest(unittest.TestCase):
    def test_default_lock_sits_on_the_shared_hub_directory(self):
        env = {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf/hub",
            "HUGGINGFACE_HUB_CACHE": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                ov_compile_lock._lock_path(),
                "/cache/hf/hub/ov-gpu-compile.lock",
            )

    def test_without_hub_cache_the_lock_is_still_under_the_hub_directory(self):
        env = {"HF_HOME": "/cache/hf", "HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                ov_compile_lock._lock_path(),
                "/cache/hf/hub/ov-gpu-compile.lock",
            )

    def test_cpu_does_not_take_the_lock(self):
        self.assertFalse(ov_compile_lock.uses_gpu("CPU"))
        self.assertFalse(ov_compile_lock.uses_gpu("cpu"))
        self.assertTrue(ov_compile_lock.uses_gpu("GPU"))
        self.assertTrue(ov_compile_lock.uses_gpu("GPU.0"))

    def test_a_second_gpu_compile_waits_until_the_first_releases(self):
        cache = tempfile.mkdtemp()
        os.environ["HF_HUB_CACHE"] = cache
        order = []
        started = threading.Event()

        def hold():
            with ov_compile_lock.gpu_compile("GPU"):
                order.append("hold")
                started.set()
                time.sleep(0.2)
                order.append("release")

        def wait():
            self.assertTrue(started.wait(2))
            with ov_compile_lock.gpu_compile("GPU"):
                order.append("next")

        threads = [threading.Thread(target=hold), threading.Thread(target=wait)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(order, ["hold", "release", "next"])

    def test_a_cpu_compile_does_not_wait_on_the_gpu_lock(self):
        cache = tempfile.mkdtemp()
        os.environ["HF_HUB_CACHE"] = cache
        order = []
        started = threading.Event()

        def hold():
            with ov_compile_lock.gpu_compile("GPU"):
                started.set()
                time.sleep(0.2)
                order.append("gpu-done")

        def cpu():
            self.assertTrue(started.wait(2))
            with ov_compile_lock.gpu_compile("CPU"):
                order.append("cpu")

        threads = [threading.Thread(target=hold), threading.Thread(target=cpu)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(order, ["cpu", "gpu-done"])


class CudaPathUntouchedTest(unittest.TestCase):
    def test_cuda_loads_do_not_call_the_compile_lock(self):
        self.assertNotIn("gpu_compile", inspect.getsource(stt_stream._load_blocking))
        self.assertNotIn("gpu_compile", inspect.getsource(align._load))
        self.assertNotIn("gpu_compile", inspect.getsource(diar_speakrs._Child._start_once))

    def test_openvino_loads_hold_the_compile_lock(self):
        self.assertIn("gpu_compile", inspect.getsource(stt_stream._load_ov))
        self.assertIn("gpu_compile", inspect.getsource(align._load_ov))
        self.assertIn("gpu_compile", inspect.getsource(diar_speakrs._Child._start_openvino))


class _Proc:
    def __init__(self, rc, line):
        self._rc = rc
        self._line = line
        self.stdin = None
        self.stdout = self
        self.returncode = None

    def readline(self):
        line, self._line = self._line, ""
        return line

    def poll(self):
        return None if self._rc is None else self._rc

    def wait(self, timeout=None):
        if self._rc is None:
            time.sleep(30)
            return 0
        self.returncode = self._rc
        return self._rc


class SpeakrsStartupTest(unittest.TestCase):
    def setUp(self):
        self.mode = diar_speakrs.EXECUTION_MODE
        self.exit = diar_speakrs._exit
        self.grace = diar_speakrs._EXIT_GRACE_S
        self.exits = []
        diar_speakrs._exit = self.exits.append
        diar_speakrs._EXIT_GRACE_S = 0.0
        diar_speakrs._stopping.clear()
        self._env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": tempfile.mkdtemp()}, clear=False)
        self._env.start()

    def tearDown(self):
        self._env.stop()
        diar_speakrs.EXECUTION_MODE = self.mode
        diar_speakrs._exit = self.exit
        diar_speakrs._EXIT_GRACE_S = self.grace
        diar_speakrs._stopping.clear()

    def _child(self, procs):
        child = diar_speakrs._Child(["speakrs-engine"])

        def popen(*_args, **_kwargs):
            return procs.pop(0)

        child._popen = lambda: setattr(child, "_proc", popen()) or child._proc
        return child

    def test_cuda_does_not_retry_a_startup_sigsegv(self):
        diar_speakrs.EXECUTION_MODE = "cuda"
        child = self._child([_Proc(-11, "")])
        with self.assertRaises(RuntimeError):
            child.start()
        time.sleep(0.05)
        self.assertEqual(self.exits, [watchdog.EXIT_CODE])

    def test_openvino_retries_one_startup_sigsegv_in_process(self):
        diar_speakrs.EXECUTION_MODE = "openvino"
        ready = _Proc(None, '{"ready": true, "device": "GPU", "load_seconds": 1.5}\n')
        child = self._child([_Proc(-11, ""), ready])
        with mock.patch.object(diar_speakrs, "_openvino_models_dir", lambda d: d):
            hello = child.start()
        self.assertTrue(hello.get("ready"))
        self.assertEqual(self.exits, [])
        self.assertIs(child._proc, ready)

    def test_a_second_openvino_sigsegv_still_exits(self):
        diar_speakrs.EXECUTION_MODE = "openvino"
        child = self._child([_Proc(-11, ""), _Proc(-11, "")])
        with self.assertRaises(RuntimeError):
            child.start()
        self.assertEqual(self.exits, [watchdog.EXIT_CODE])

    def test_openvino_cpu_does_not_take_the_retry(self):
        diar_speakrs.EXECUTION_MODE = "openvino:CPU"
        self.assertFalse(diar_speakrs._openvino_gpu_mode())
        diar_speakrs.EXECUTION_MODE = "openvino:GPU.1"
        self.assertTrue(diar_speakrs._openvino_gpu_mode())
        diar_speakrs.EXECUTION_MODE = "cuda"
        self.assertFalse(diar_speakrs._openvino_gpu_mode())


if __name__ == "__main__":
    unittest.main()
