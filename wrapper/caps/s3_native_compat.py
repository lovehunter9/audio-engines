"""Platform data transport and checkpoint loading around the original S3 runner."""
import hashlib
import importlib
import json
import os
import sys
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
        "reason": "Map the unchanged checkpoint before moving the selected original model to CUDA",
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
    }, indent=2))


_original_reduce = None


def _rebuild_host_tensor(raw, shape, dtype, unused_a, unused_b, unused_c, device):
    import torch
    host = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    return host.to(device=torch.device("cuda", int(device)))


def _reduce_host_tensor(value):
    if not value.is_cuda:
        return _original_reduce(value)
    import torch
    tensor = value.detach().contiguous()
    raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
    return _rebuild_host_tensor, (raw, tuple(tensor.shape), tensor.dtype, 0, 0, 0, tensor.device.index or 0)


def install_worker_compat():
    import s3_runtime_plugin
    s3_runtime_plugin.register()
    if os.environ.get("S3_NATIVE_TRANSPORT") == "host":
        from torch.multiprocessing import reductions
        global _original_reduce
        _original_reduce = getattr(reductions, "_s3_original_reduce", reductions.reduce_tensor)
        reductions._s3_original_reduce = _original_reduce
        reductions.reduce_tensor = _reduce_host_tensor
        for name in ("gpu_transport_entry", "gpu_native_entry", "prefix_gpu_entry", "capture_gpu_entry", "capture_gpu_worker", "prefix_gpu_worker"):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "reduce_tensor"):
                module.reduce_tensor = _reduce_host_tensor
    importlib.import_module("s3_syncgraphs_entry")
    from capture_gpu_worker import install_capture_hooks
    install_capture_hooks()
