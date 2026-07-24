# llm-init engine contract surface shared by every audio-engine wrapper.
#
# Registers the endpoints llm-init actually consumes, with llama-server
# semantics so llm-init's uniform proxy probes work unchanged:
#   GET /v1/models  -> 200 + {data:[{id,mode,supports,endpoints}]} ONLY when the
#                      model is loaded; 503 while loading. llm-init WaitAlive
#                      polls this until 200 (== ready) and Ready matches id.
#   GET /health|/healthz|/readyz -> 200 when ready, else 503 (engine's own).
# GET /metrics (gpu_*) is mounted separately via gpu.mount_metrics.
import os


def parse_supports():
    # MODEL_SUPPORTS is the clone-time capability CSV, passed through by llm-init
    # verbatim; the engine self-reports it (no llm-init interpretation).
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


def register(app, *, model_name, mode, supports, endpoints, is_ready, error=None):
    # is_ready: () -> bool ; error: () -> str|None (last load error, for detail).
    from fastapi import HTTPException

    def _err():
        try:
            return error() if callable(error) else error
        except Exception:
            return None

    @app.get("/v1/models")
    def models():
        if not is_ready():
            raise HTTPException(status_code=503, detail=_err() or "model not loaded yet")
        return models_payload(model_name, mode, supports, endpoints)

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
