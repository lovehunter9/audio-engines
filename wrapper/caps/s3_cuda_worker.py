"""Carry the original single-GPU hooks into native spawned workers."""
from vllm.v1.worker.gpu_worker import Worker
from .s3_native_compat import install_worker_compat


class NativeWorker(Worker):
    def __init__(self, *args, **kwargs):
        install_worker_compat()
        super().__init__(*args, **kwargs)
