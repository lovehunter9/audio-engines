# The llm-init contract (README): /v1/models 503s until loaded, plus /health; /metrics is gpu.py.
import datetime
import json
import os
import re
import shlex
import time

from . import catalog

# Ollama-style coarse capability, carried alongside the fine-grained audio ones.
COARSE_CAPABILITY = "audio"

# How capability keys travel in env, model specs and the gateway; bare names are the internal form.
SUPPORTS_PREFIX = "supports_"

# The analysis default; generation caps synthesize at their own rate and pass it to register().
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


_NUMBER = re.compile(r"^-\d")


class EngineArgs:
    """ENGINE_ARGS, llm-init's single channel for engine-native flags on the engine container.

    Every tunable is a flag in here, so a new knob costs nothing in env names. Caps claim the flags
    they understand; whatever is left over passes through to a child engine untouched, keeping the
    engine's own spelling.
    """

    def __init__(self, raw=None):
        if raw is None:
            raw = os.environ.get("ENGINE_ARGS", "") or ""
        self._vals, self._spans, self._claimed = {}, [], set()
        # Flags whose value was present but unreadable. warn_unclaimed reports them: a
        # number that will not parse falls back to the default, and a default is exactly
        # what a working engine looks like, so nothing else would ever say it happened.
        self._unreadable = []
        toks = shlex.split(raw)
        i = 0
        while i < len(toks):
            tok = toks[i]
            if not tok.startswith("-"):
                self._spans.append((None, [tok]))
                i += 1
                continue
            flag, eq, inline = tok.partition("=")
            key = self._key(flag)
            if eq:
                value, span, i = inline, [tok], i + 1
            elif i + 1 < len(toks) and not self._is_flag(toks[i + 1]):
                value, span, i = toks[i + 1], [tok, toks[i + 1]], i + 2
            else:
                value, span, i = True, [tok], i + 1  # bare flag, e.g. --enforce-eager
            # shlex strips the quotes, so re-JSON anything still shaped like an object or array.
            if isinstance(value, str) and value[:1] in "{[":
                value = self._as_json(value)
                if eq:
                    span = ["%s=%s" % (flag, value)]
                elif len(span) == 2:
                    span = [span[0], value]
            self._vals[key] = value
            self._spans.append((key, span))

    @staticmethod
    def _key(name):
        return name.lstrip("-").replace("-", "_")

    @staticmethod
    def _is_flag(tok):
        return tok.startswith("-") and not _NUMBER.match(tok)

    @staticmethod
    def _as_json(value):
        """Keep valid JSON; repair the common shlex-stripped form {0:{max_num_seqs:10}}."""
        try:
            json.loads(value)
            return value
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        # shlex ate the quotes: quote bare keys (idents + integers) before ':'
        fixed = re.sub(r"([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', value)
        fixed = re.sub(r"([{\[,]\s*)(\d+)\s*:", r'\1"\2":', fixed)
        try:
            json.loads(fixed)
            return fixed
        except (TypeError, ValueError, json.JSONDecodeError):
            return value

    def text(self, name, default=None):
        key = self._key(name)
        self._claimed.add(key)
        val = self._vals.get(key)
        return default if val is None or val is True else str(val)

    def given(self, name):
        """Was this flag present at all, with or without a value.

        text() cannot answer it: a valueless flag and an absent one both come back as the
        default, so anything that wants to say "you set this and it does nothing here" has
        to ask separately.
        """
        return self._key(name) in self._vals

    def _bare(self, name):
        """The flag carries no usable value -- `--flag`, or `--flag=` with nothing after it.

        Both spellings mean the same mistake and neither survives text(), which answers the
        default for them exactly as for a flag nobody passed. A number that needs a value has
        to tell those apart, or the likeliest typo of all is the one that reports nothing.
        """
        val = self._vals.get(self._key(name))
        return val is True or (isinstance(val, str) and not val.strip())

    def number(self, name, default):
        raw = self.text(name)
        if self._bare(name):
            self._unreadable.append((name, None, default))
            return float(default)
        try:
            return float(raw or default)
        except (TypeError, ValueError):
            self._unreadable.append((name, raw, default))
            return float(default)

    def count(self, name, default):
        raw = self.text(name)
        if self._bare(name):
            self._unreadable.append((name, None, default))
            return int(default)
        try:
            return int(float(raw or default))
        except (TypeError, ValueError):
            self._unreadable.append((name, raw, default))
            return int(default)

    # The spellings a value may use to mean on and off. Public because a cap that has to tell
    # "off" from "a value I do not recognise" needs the same lists switch() decides by, and a
    # second copy of them drifts the day one side gains a spelling.
    ON_WORDS = ("1", "true", "yes", "on")
    OFF_WORDS = ("0", "false", "no", "off")

    def switch(self, name, default=False):
        key = self._key(name)
        self._claimed.add(key)
        if self._key(name) not in self._vals:
            return bool(default)
        # `--flag=` is the bare flag with a stray equals sign, not a value that means off.
        # Reading it as off flips the meaning of the flag silently -- a chart rendering
        # `--flag={{ .Values.x }}` with x unset turns the feature OFF while its author reads
        # the template as turning it on. _bare() already decides this for numbers; a switch
        # has to answer the same question the same way.
        if self._bare(name):
            return True
        return str(self._vals.get(key)).strip().lower() in self.ON_WORDS

    def passthrough(self):
        """The flags no cap claimed, ready to hand to a child engine's argv."""
        out = []
        for key, span in self._spans:
            if key is None or key not in self._claimed:
                out.extend(span)
        return out

    def warn_unclaimed(self, log):
        """For caps that run no child engine: a typo would otherwise vanish without a trace."""
        rest = self.passthrough()
        if rest:
            log.warning("ignoring ENGINE_ARGS flags this engine does not take: %s", " ".join(rest))
        for name, raw, default in self._unreadable:
            if raw is None:
                # Quoting a placeholder here sends the reader grepping their ENGINE_ARGS for
                # a string that is not in it, and "not a number" is not what went wrong.
                log.warning("ENGINE_ARGS %s was given with no value; using %s", name, default)
            else:
                log.warning("ENGINE_ARGS %s=%r is not a number; using %s", name, raw, default)


def cache_dir(repo):
    """Where the shared HF cache keeps this repo. Public because more than one module needs it."""
    cache = (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
             or "/cache/hf/hub")
    return os.path.join(cache, "models--" + repo.replace("/", "--"))


def _disk(repo):
    """Bytes / mtime / weight format read off the HF cache once; zeros when it is not there."""
    if repo in _disk_facts:
        return _disk_facts[repo]
    size, mtime, exts = 0, 0.0, set()
    for root, _dirs, files in os.walk(cache_dir(repo)):
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
                   model_format=None, quantization=None, sample_rate=None):
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
                    "sample_rate": sample_rate or SAMPLE_RATE,
                },
            }
        ],
    }


# Routes register() always mounts, reported so the self-report covers the engine's whole data plane, not just its capabilities.
CONTRACT_ENDPOINTS = [
    {"method": "GET", "path": "/v1/models",
     "description": "Model list (Ollama + OpenAI shape, with capabilities); 503 until the model is loaded",
     "available": True,
     **catalog.describe_endpoint("", "", "GET", "/v1/models", False)},
]


def register(app, *, model_name, module, served, is_ready, error=None, task_api=False,
             task_legacy=True, repo=None, model_format=None, quantization=None,
             sample_rate=None, endpoint_available=None):
    # is_ready: () -> bool ; error: () -> str|None (last load error, for detail).
    from fastapi import HTTPException

    base = base_name()
    declared, _bad = parse_supports()
    capabilities = [COARSE_CAPABILITY] + list(served)
    spec = {
        "schema_version": 2,
        "base": base,
        "model": model_name,
        "implements": catalog.implements(base),
        "declares": declared,
        "serves": list(served),
        "endpoints": [dict(endpoint) for endpoint in CONTRACT_ENDPOINTS]
        + catalog.spec_endpoints(base, served, declared),
    }
    # Mounted and advertised together, so the task API can never be one without the other.
    if task_api:
        from . import tasks

        tasks.mount(app, legacy=task_legacy)
        spec["endpoints"].extend(
            dict(e, available=True) for e in tasks.advertised(legacy=task_legacy))
    for endpoint in spec["endpoints"]:
        described = catalog.describe_endpoint(
            "", "", endpoint["method"], endpoint["path"], endpoint.get("async_supported", False))
        for key, value in described.items():
            endpoint.setdefault(key, value)

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
                              module=module, model_format=model_format, quantization=quantization,
                              sample_rate=sample_rate)

    @app.get("/api/engine-spec")
    def engine_spec():
        # Never gated on load: llm-init reads the contract while the model is still downloading.
        if endpoint_available is None:
            return spec
        report = dict(spec)
        report["endpoints"] = []
        for source in spec["endpoints"]:
            endpoint = dict(source)
            try:
                available, reason = endpoint_available(endpoint)
            except Exception:
                available, reason = endpoint.get("available", False), "dynamic availability unavailable"
            endpoint["available"] = bool(endpoint.get("available", False) and available)
            if not endpoint["available"] and reason:
                endpoint["reason"] = reason
            report["endpoints"].append(endpoint)
        return report

    @app.get("/api/engine-capacity")
    def engine_capacity():
        # Every audio process owns one model and funnels synchronous and
        # asynchronous inference through the same single worker.
        return {"max_concurrency": 1}

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
