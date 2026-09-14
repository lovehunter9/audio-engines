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


def cuda_visible():
    """True when this process can see a CUDA device, without needing CUDA torch.

    NVML first (no CUDA context). /dev/nvidia0 next. torch last, for images that
    still have it. All three miss on a CPU-only box.
    """
    if _nvml_stats() is not None:
        return True
    if os.path.exists("/dev/nvidia0"):
        return True
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def visible_memory_bytes():
    """What CUDA reports as this device's total, or 0 when there is no device to ask."""
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


def _nvml_stats():
    """(used, total, util) straight from NVML, or None when there is no device to ask.

    The fallback for an image that carries no torch: an engine whose runtime links CUDA itself
    (ggml, ONNX Runtime) has no reason to install a whole torch stack for four gauges. NVML also
    needs no CUDA context of its own, so it costs no GPU memory to ask.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() <= 0:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu) / 100.0
            except Exception:
                util = 0.0
            return int(mem.used), int(mem.total), util
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def gpu_metrics_text(nvml_fallback=False):
    present, used, total, util = 0, 0, 0, 0.0
    slice_scoped = True
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
    if not present and nvml_fallback:
        # torch is absent or saw no device; ask NVML before reporting zeros. Zeros would read as
        # "a card is there and idle", which is the one answer that must not be guessed.
        #
        # The two paths do NOT measure the same thing, and the help text says so rather than
        # pretending otherwise: torch goes through the CUDA driver, which is where memory
        # virtualization intercepts, so it sees this container's slice. NVML is a different
        # interface and may report the whole card, including memory other pods are using.
        stats = _nvml_stats()
        if stats is not None:
            present, slice_scoped = 1, False
            used, total, util = stats
    lines = []

    def g(name, help_, val):
        lines.append("# HELP %s %s" % (name, help_))
        lines.append("# TYPE %s gauge" % name)
        lines.append("%s %s" % (name, val))

    scope = ("on this engine's device slice" if slice_scoped
             else "on the whole device as NVML reports it, which may include other pods")
    g("gpu_present", "1 if a CUDA device is visible to this engine, else 0", present)
    g("gpu_mem_used_bytes", "GPU memory in use %s (bytes)" % scope, used)
    g("gpu_mem_total_bytes", "GPU memory reported to this engine %s (bytes)"
      % ("by CUDA" if slice_scoped else "by NVML"), total)
    g("gpu_util_ratio", "GPU compute utilization 0..1 (0 when unavailable)", "%.4f" % util)
    return "\n".join(lines) + "\n"


def mount_metrics(app, nvml_fallback=False):
    """Register GET /metrics on a FastAPI/Starlette app (Prometheus text exposition).

    nvml_fallback is opt-in because the two sources do not measure the same thing (see
    gpu_metrics_text). Defaulting it on would change what an existing engine's gauges mean
    the day someone adds nvidia-ml-py to its image for an unrelated reason, and nothing
    about that day would look like a change: same gauge names, same shape, wider scope.
    """
    from fastapi.responses import PlainTextResponse

    @app.get("/metrics")
    def _metrics():
        return PlainTextResponse(
            gpu_metrics_text(nvml_fallback), media_type="text/plain; version=0.0.4"
        )

    return app
