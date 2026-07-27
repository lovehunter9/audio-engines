# GET /metrics for every wrapper, using the exact gauge names llm-init relays (NOT audio_gpu_*).


def gpu_metrics_text():
    present, used, total, util = 0, 0, 0, 0.0
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            present = 1
            free, tot = torch.cuda.mem_get_info()  # bytes, current device (HAMi slice)
            total = int(tot)
            used = int(tot - free)
            try:
                util = float(torch.cuda.utilization()) / 100.0  # needs pynvml; 0 if absent
            except Exception:
                util = 0.0
    except Exception:
        present = 0
    lines = []

    def g(name, help_, val):
        lines.append("# HELP %s %s" % (name, help_))
        lines.append("# TYPE %s gauge" % name)
        lines.append("%s %s" % (name, val))

    g("gpu_present", "1 if a CUDA device is visible to this engine, else 0", present)
    g("gpu_mem_used_bytes", "GPU memory in use on this engine's device slice (bytes)", used)
    g("gpu_mem_total_bytes", "GPU memory visible to this engine (HAMi quota, bytes)", total)
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
