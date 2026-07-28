# audio-fasterwhisper deps: everything but the wrapper, so a wrapper change costs one small layer.
# Rebuilt only when THIS file changes: its content hash is the tag the append step looks for.
FROM docker.io/beclab/harveyff-whisper-webui:v1.0.7

# Everything goes into the one python that owns faster_whisper; this block's three traps are in README.
RUN set -eux; \
    PY=; \
    for cand in "$(command -v python3 || true)" /Whisper-WebUI/venv/bin/python3 /usr/bin/python3; do \
        [ -n "$cand" ] && [ -x "$cand" ] || continue; \
        if "$cand" -c "import faster_whisper" >/dev/null 2>&1; then PY="$cand"; break; fi; \
    done; \
    : "${PY:?no interpreter in this image can import faster_whisper}"; \
    "$PY" -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    "$PY" -m pip install --no-cache-dir --root-user-action=ignore "transformers>=4.56"; \
    printf '#!/bin/sh\nexec %s "$@"\n' "$PY" > /usr/local/bin/audio-python; \
    chmod 755 /usr/local/bin/audio-python; \
    audio-python -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    audio-python -c "from ctranslate2.converters import TransformersConverter; \
import transformers as t; v=tuple(int(x) for x in t.__version__.split('.')[:2]); \
assert v >= (4, 56), t.__version__"; \
    command -v ffmpeg >/dev/null

LABEL org.opencontainers.image.title="audio-fasterwhisper-deps"
