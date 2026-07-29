# The llm-init contract (README): /v1/models 503s until loaded, plus /health; /metrics is gpu.py.
import datetime
import os
import re
import time

from . import catalog

# Ollama-style coarse capability, carried alongside the fine-grained audio ones.
COARSE_CAPABILITY = "audio"

# How capability keys travel in env, model specs and the gateway; bare names are the internal form.
SUPPORTS_PREFIX = "supports_"

# Every engine here decodes to 16 kHz mono before inference.
SAMPLE_RATE = 16000

_SIZE_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])(?:[-_.]|$)", re.IGNORECASE)
_WEIGHT_FORMATS = (
    (".safetensors", "safetensors"), (".gguf", "gguf"), (".onnx", "onnx"), (".nemo", "nemo"),
    (".ckpt", "pytorch"), (".pth", "pytorch"), (".pt", "pytorch"), (".bin", "pytorch"),
)
_START = time.time()
_disk_facts = {}


def parse_supports():
    """The clone-time MODEL_SUPPORTS CSV as bare capability names, plus the tokens that are not keys.

    Capabilities travel in the one supports_ vocabulary llm-init and the gateway already use, and
    are bare everywhere inside the engine (module names, Ollama capabilities, endpoint tables).
    """
    raw = os.environ.get("MODEL_SUPPORTS", "") or ""
    caps, bad = [], []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        if not t.startswith(SUPPORTS_PREFIX) or len(t) == len(SUPPORTS_PREFIX):
            bad.append(t)
        elif t[len(SUPPORTS_PREFIX):] not in caps:
            caps.append(t[len(SUPPORTS_PREFIX):])
    return caps, bad


def base_name():
    """The base image identity, as baked in at build time."""
    return (os.environ.get("AUDIO_BASE") or "").strip()


def _cache_dir(repo):
    cache = (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
             or "/cache/hf/hub")
    return os.path.join(cache, "models--" + repo.replace("/", "--"))


def _disk(repo):
    """Bytes / mtime / weight format read off the HF cache once; zeros when it is not there."""
    if repo in _disk_facts:
        return _disk_facts[repo]
    size, mtime, exts = 0, 0.0, set()
    for root, _dirs, files in os.walk(_cache_dir(repo)):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            # Blobs carry the bytes, the snapshot symlinks carry the real filenames.
            size += st.st_size
            mtime = max(mtime, st.st_mtime)
            exts.add(os.path.splitext(name)[1].lower())
    fmt = next((f for ext, f in _WEIGHT_FORMATS if ext in exts), "")
    facts = (size, mtime or _START, fmt)
    if size:
        _disk_facts[repo] = facts
    return facts


def _parameter_size(*names):
    """Ollama's "1.7B" style label, taken from the model id when it spells one out."""
    for name in names:
        m = _SIZE_TOKEN.search(name or "")
        if m:
            return "%s%s" % (m.group(1), m.group(2).upper())
    return ""


def models_payload(model_name, capabilities, description="", repo=None, module="",
                   model_format=None, quantization=None):
    """Ollama models[] and OpenAI data[] for the one model this process serves."""
    repo = repo or model_name
    size, mtime, fmt = _disk(repo)
    fmt = model_format or fmt
    family = catalog.FAMILIES.get(module, "")
    modified = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()
    owner = "audio-%s" % base_name() if base_name() else "audio-engines"
    details = {
        "parent_model": "",
        "format": fmt,
        "family": family,
        "families": [family] if family else [],
        "parameter_size": _parameter_size(model_name, repo),
        "quantization_level": quantization or "",
    }
    return {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "modified_at": modified,
                "size": size,
                "digest": "",
                "type": "model",
                "description": description,
                "tags": [],
                "capabilities": capabilities,
                "parameters": "",
                "details": details,
            }
        ],
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(mtime),
                "owned_by": owner,
                "meta": {
                    "family": family,
                    "format": fmt,
                    "size": size,
                    "sample_rate": SAMPLE_RATE,
                },
            }
        ],
    }


# Routes register() always mounts, reported so the self-report covers the engine's whole data plane, not just its capabilities.
CONTRACT_ENDPOINTS = [
    {"method": "GET", "path": "/v1/models",
     "description": "Model list (Ollama + OpenAI shape, with capabilities); 503 until the model is loaded",
     "available": True},
]


def register(app, *, model_name, module, served, is_ready, error=None, task_api=False,
             repo=None, model_format=None, quantization=None):
    # is_ready: () -> bool ; error: () -> str|None (last load error, for detail).
    from fastapi import HTTPException

    base = base_name()
    declared, _bad = parse_supports()
    capabilities = [COARSE_CAPABILITY] + list(served)
    spec = {
        "schema_version": 1,
        "base": base,
        "model": model_name,
        "implements": catalog.implements(base),
        "declares": declared,
        "serves": list(served),
        "endpoints": CONTRACT_ENDPOINTS + catalog.spec_endpoints(base, served, declared),
    }
    # Mounted and advertised together, so the task API can never be one without the other.
    if task_api:
        from . import tasks

        tasks.mount(app)
        spec["endpoints"].extend(dict(e, available=True) for e in tasks.ENDPOINTS)

    def _err():
        try:
            return error() if callable(error) else error
        except Exception:
            return None

    @app.get("/v1/models")
    def models():
        if not is_ready():
            raise HTTPException(status_code=503, detail=_err() or "model not loaded yet")
        return models_payload(model_name, capabilities, description=app.title, repo=repo,
                              module=module, model_format=model_format, quantization=quantization)

    @app.get("/api/engine-spec")
    def engine_spec():
        # Never gated on load: llm-init reads the contract while the model is still downloading.
        return spec

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
