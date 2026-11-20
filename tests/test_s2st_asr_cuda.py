from contextlib import contextmanager
import unittest

from wrapper.s2st.asr_cuda import AsrCuda


class Stream:
    def __init__(self, cuda, pointer):
        self.cuda, self.cuda_stream = cuda, pointer

    def wait_stream(self, other):
        self.cuda.events.append(("wait", other.cuda_stream))

    def synchronize(self):
        self.cuda.events.append(("join", self.cuda_stream))


class Cuda:
    def __init__(self):
        self.events = []
        self.default = Stream(self, 0)
        self.current = self.default

    def Stream(self, device):
        assert device == 0
        self.events.append(("create", device))
        return Stream(self, 41)

    def current_stream(self, device):
        assert device == 0
        return self.current

    def default_stream(self, device):
        assert device == 0
        return self.default

    @contextmanager
    def stream(self, value):
        previous = self.current
        self.current = value
        self.events.append(("enter", value.cuda_stream))
        try:
            yield
        finally:
            self.events.append(("exit", value.cuda_stream))
            self.current = previous


class AsrCudaTest(unittest.TestCase):
    def test_disabled_does_not_create_or_join_any_stream(self):
        cuda = Cuda()
        with AsrCuda(cuda).scope() as metadata:
            self.assertFalse(metadata["asr_cuda_stream_separate"])
            self.assertEqual(metadata["asr_cuda_stream"], 0)
        self.assertEqual(cuda.events, [])

    def test_model_readiness_waits_once_and_each_call_joins_before_release(self):
        cuda = Cuda()
        execution = AsrCuda(cuda, True)
        for _ in range(2):
            with execution.scope() as metadata:
                self.assertEqual(cuda.current.cuda_stream, 41)
                self.assertEqual(metadata["asr_cuda_stream"], 41)
                self.assertEqual(metadata["asr_default_cuda_stream"], 0)
                self.assertFalse(metadata["asr_stream_synchronized_before_release"])
                cuda.events.append(("inference", 41))
            self.assertTrue(metadata["asr_stream_synchronized_before_release"])
            self.assertIs(cuda.current, cuda.default)
        self.assertEqual(cuda.events, [
            ("create", 0), ("wait", 0), ("enter", 41), ("inference", 41),
            ("join", 41), ("exit", 41), ("enter", 41), ("inference", 41),
            ("join", 41), ("exit", 41),
        ])

    def test_failure_still_joins_and_restores_callers_stream(self):
        cuda = Cuda()
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            with AsrCuda(cuda, True).scope() as metadata:
                raise RuntimeError("model failed")
        self.assertTrue(metadata["asr_stream_synchronized_before_release"])
        self.assertEqual(cuda.events[-2:], [("join", 41), ("exit", 41)])
        self.assertIs(cuda.current, cuda.default)
