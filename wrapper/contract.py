# The llm-init contract (README): /v1/models 503s until loaded, plus /health; /metrics is gpu.py.
import os


def parse_supports():
    # The clone-time capability CSV, relayed verbatim by llm-init for the engine to self-report.
    raw = os.environ.get("MODEL_SUPPORTS", "") or ""
    out = []
    for tok in raw.split(","):
        t = tok.strip()
        if t and t not in out:
            out.append(t)
    return out


def models_payload(model_name, mode, supports, endpoints):
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "mode": mode,
                "supports": supports,
                "endpoints": endpoints,
            }
        ],
    }


def register(app, *, model_name, mode, supports, endpoints, is_ready, error=None,
             task_api=False):
    # is_ready: () -> bool ; error: () -> str|None (last load error, for detail).
    from fastapi import HTTPException

    # The self-report lists itself, since a caller should see every path the engine serves.
    advertised = [{"method": "GET", "path": "/v1/models",
                   "description": "Model self-report (id / mode / supports / endpoints)"}]
    advertised.extend(endpoints)
    # Mounted and advertised together, so the task API can never be one without the other.
    if task_api:
        from . import tasks

        tasks.mount(app)
        advertised.extend(tasks.ENDPOINTS)

    def _err():
        try:
            return error() if callable(error) else error
        except Exception:
            return None

    @app.get("/v1/models")
    def models():
        if not is_ready():
            raise HTTPException(status_code=503, detail=_err() or "model not loaded yet")
        return models_payload(model_name, mode, supports, advertised)

    @app.get("/health")
    def health():
        if is_ready():
            return {"status": "ok"}
        raise HTTPException(status_code=503, detail=_err() or "loading model")

    @app.get("/healthz")
    def healthz():
        return health()

    @app.get("/readyz")
    def readyz():
        return health()

    return app
