"""Bound checkpoint deserialization memory without changing GPU inference."""
import hashlib
import json
import threading
import uuid
from pathlib import Path


def prepare_checkpoint_loader(root):
    root = Path(root)
    path = root / "inference/streaming/streaming_codex_to_wav.py"
    original = path.read_text()
    before = '    state_dict = torch.load(str(ckpt_path), map_location=device)'
    after = '    state_dict = torch.load(str(ckpt_path), map_location="cpu", mmap=True)'
    if original.count(after) == 1 and before not in original:
        return
    if original.count(before) != 1:
        raise RuntimeError("The pinned original checkpoint loader changed")
    updated = original.replace(before, after)
    path.write_text(updated)
    (root / "runtime-compatibility.json").write_text(json.dumps({
        "path": str(path.relative_to(root)),
        "reason": "Select checkpoint entries on the host before copying the unchanged model to CUDA",
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
    }, indent=2))


_original_reduce = None


_local_payloads = {}
_local_lock = threading.Lock()


def _rebuild_local_tensor(key, unused_a, unused_b, unused_c, unused_d, unused_e, device):
    with _local_lock:
        tensor = _local_payloads.pop(key)
    if tensor.device.index != int(device):
        raise RuntimeError("Original S3 CUDA transport changed devices")
    return tensor


def _reduce_local_tensor(value):
    if not value.is_cuda:
        return _original_reduce(value)
    key = uuid.uuid4().hex
    with _local_lock:
        _local_payloads[key] = value
    return _rebuild_local_tensor, (key, 0, 0, 0, 0, 0, value.device.index or 0)


def install_worker_compat():
    import importlib
    import os
    import sys
    import s3_runtime_plugin
    s3_runtime_plugin.register()
    if os.environ.get("S3_NATIVE_TRANSPORT") == "local":
        from torch.multiprocessing import reductions
        global _original_reduce
        _original_reduce = getattr(reductions, "_s3_original_reduce", reductions.reduce_tensor)
        reductions._s3_original_reduce = _original_reduce
        reduce = _reduce_local_tensor
        reductions.reduce_tensor = reduce
        for name in ("gpu_transport_entry", "gpu_native_entry", "prefix_gpu_entry", "capture_gpu_entry", "capture_gpu_worker", "prefix_gpu_worker"):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "reduce_tensor"):
                module.reduce_tensor = reduce
    importlib.import_module("s3_syncgraphs_entry")
    from single_gpu_worker import install_model_hooks
    install_model_hooks()
