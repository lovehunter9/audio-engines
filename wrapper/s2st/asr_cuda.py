"""Optional ASR-only CUDA stream; model inputs and math stay unchanged."""
from contextlib import contextmanager


class AsrCuda:
    def __init__(self, cuda, enabled=False):
        self.cuda = cuda
        self.stream = cuda.Stream(device=0) if enabled else None
        self.prepared = False

    @contextmanager
    def scope(self):
        current = self.cuda.current_stream(0)
        default = self.cuda.default_stream(0)
        metadata = dict(
            asr_cuda_stream_separate=self.stream is not None,
            asr_cuda_stream=int((self.stream or current).cuda_stream),
            asr_default_cuda_stream=int(default.cuda_stream),
            asr_stream_synchronized_before_release=False,
        )
        if self.stream is None:
            yield metadata
            return
        if not self.prepared:
            self.stream.wait_stream(current)
            self.prepared = True
        with self.cuda.stream(self.stream):
            try:
                yield metadata
            finally:
                self.stream.synchronize()
                metadata["asr_stream_synchronized_before_release"] = True
