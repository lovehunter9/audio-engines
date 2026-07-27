# audio-fasterwhisper: Whisper family on CTranslate2. Build context is the repo ROOT.
FROM docker.io/beclab/harveyff-whisper-webui:v1.0.7

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-fasterwhisper" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

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

WORKDIR /app
COPY wrapper /app/wrapper
# LOAD_TIMEOUT_S is generous here: first-use CT2 conversion is legitimate multi-GB work (README).
ENV AUDIO_BASE=fasterwhisper \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOAD_TIMEOUT_S=5400 \
    WRAPPER_PORT=8000

# Serves directly here; the chart overrides command with its sentinel-wait shell, same module.
CMD ["audio-python", "-m", "wrapper.app"]
