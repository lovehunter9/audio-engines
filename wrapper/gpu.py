# GET /metrics for every wrapper, using the exact gauge names llm-init relays (NOT audio_gpu_*).
import os
import re

_QUANTITY = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]i?)?\s*$")
_UNITS = {"": 1, "K": 10 ** 3, "M": 10 ** 6, "G": 10 ** 9, "T": 10 ** 12,
          "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40}


def _quota_bytes():
    """The platform's REQUIRED_GPU_MEMORY as bytes; 0 when unset, zero or unparseable (CPU-only)."""
    m = _QUANTITY.match(os.environ.get("REQUIRED_GPU_MEMORY", "") or "")
    return int(float(m.group(1)) * _UNITS[m.group(2) or ""]) if m else 0


def quota_mib():
    """The platform's GPU quota in MiB, for sizing work to the slice we were given; 0 on CPU."""
    return _quota_bytes() // (2 ** 20)


def _xpu_memory():
    """(used, total, util) from torch.xpu; None when no XPU is visible.

    Intel's runtime exposes no NVML equivalent for compute utilization, so that gauge stays 0.
    """
    try:
        import torch

        xpu = getattr(torch, "xpu", None)
        if not (xpu and xpu.is_available() and xpu.device_count() > 0):
            return None
        mem_get_info = getattr(xpu, "mem_get_info", None)
        if mem_get_info:
            free, total = mem_get_info()
            return int(total - free), int(total), 0.0
        # Older torch.xpu has no mem_get_info; the device properties still carry the total.
        total = int(xpu.get_device_properties(0).total_memory)
        return int(xpu.memory_allocated()), total, 0.0
    except Exception:
        return None


def _has_cuda():
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def is_xpu():
    """True when this install runs on an Intel accelerator, per OLARES_GPU_MODE from the chart.

    Which accelerator to use is the user's choice at install time, not something to probe: a
    machine can have an NVIDIA card AND an Intel iGPU, and picking `intel` there is a decision,
    not a fallback. So the mode is told to us; only when nothing told us (plain helm install, or a
    chart with no Intel mode) do we fall back to what the hardware happens to show.
    """
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode:
        return mode.startswith("intel")
    return not _has_cuda() and _xpu_memory() is not None


def visible_memory_bytes():
    """What the accelerator reports as this device's total, or 0 when there is no device to ask."""
    if is_xpu():
        xpu = _xpu_memory()
        return xpu[1] if xpu else 0
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return int(torch.cuda.mem_get_info()[1])
    except Exception:
        pass
    return 0


def memory_fraction(reserve=0.85, lo=0.1, hi=0.9):
    """The HAMi quota expressed as vLLM's --gpu-memory-utilization, i.e. a share of what CUDA shows.

    Where HAMi virtualizes the memory view the two numbers coincide and this is just `reserve`;
    where it does not, the engine sees the whole card and the quota has to be scaled down to its
    slice of it. None means there is nothing to derive from, so the caller keeps its own default.
    """
    quota, visible = _quota_bytes(), visible_memory_bytes()
    if not quota or not visible:
        return None
    return min(hi, max(lo, quota * reserve / visible))


def gpu_metrics_text():
    present, used, total, util = 0, 0, 0, 0.0
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            present = 1
            free, tot = torch.cuda.mem_get_info()  # bytes; the whole card unless HAMi virtualizes it
            total = int(tot)
            used = int(tot - free)
            try:
                util = float(torch.cuda.utilization()) / 100.0  # needs pynvml; 0 if absent
            except Exception:
                util = 0.0
    except Exception:
        present = 0
    # An Intel install reports its XPU even where a CUDA device is visible too: the mode is the
    # user's choice, so these gauges have to describe the device this engine was told to use.
    if is_xpu() or not present:
        xpu = _xpu_memory()
        if xpu:
            used, total, util = xpu
            present = 1
    lines = []

    def g(name, help_, val):
        lines.append("# HELP %s %s" % (name, help_))
        lines.append("# TYPE %s gauge" % name)
        lines.append("%s %s" % (name, val))

    g("gpu_present", "1 if a CUDA device is visible to this engine, else 0", present)
    g("gpu_mem_used_bytes", "GPU memory in use on this engine's device slice (bytes)", used)
    g("gpu_mem_total_bytes", "GPU memory CUDA reports to this engine (bytes)", total)
    g("gpu_util_ratio", "GPU compute utilization 0..1 (0 when unavailable)", "%.4f" % util)
    return "\n".join(lines) + "\n"


def mount_metrics(app):
    # Register GET /metrics on a FastAPI/Starlette app (Prometheus text exposition).
    from fastapi.responses import PlainTextResponse

    @app.get("/metrics")
    def _metrics():
        return PlainTextResponse(
            gpu_metrics_text(), media_type="text/plain; version=0.0.4"
        )

    return app
