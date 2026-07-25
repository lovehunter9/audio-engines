# audio-fasterwhisper base: Whisper family on CTranslate2 (stt). Build context
# is the repo ROOT (COPY wrapper/...).
#   docker build -f bases/fasterwhisper/Dockerfile -t <reg>/audio-fasterwhisper:<tag> .
FROM docker.io/beclab/harveyff-whisper-webui:v1.0.7

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-fasterwhisper" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# This image has several pythons, so pick the one that owns the engine and put
# everything in THAT one. --ignore-installed because the venv is built with
# --system-site-packages: pip sees fastapi in the system python, reports
# "already satisfied", installs nothing, and the venv still cannot import it at
# runtime. The base's CUDA-matched ctranslate2 + faster_whisper build must never
# be replaced by a generic wheel, so only the web layer is added — and
# transformers only if absent, since it is what CT2 converts FROM (the wrapper
# converts transformers-format Whisper checkpoints on first load).
#
# audio-python must be a WRAPPER SCRIPT, never a symlink: the engine lives in a
# venv, and CPython decides it is in one by looking for pyvenv.cfg next to the
# executable. A symlink in /usr/local/bin has no pyvenv.cfg beside it, so python
# starts against the system prefix, the venv's site-packages never enter sys.path
# and the wrapper dies on `import fastapi` — while a build check run against the
# interpreter's real path passes. Hence the final check goes THROUGH audio-python:
# verify the exact command the chart executes, not an equivalent-looking one.
RUN set -eux; \
    PY=; \
    for cand in "$(command -v python3 || true)" /Whisper-WebUI/venv/bin/python3 /usr/bin/python3; do \
        [ -n "$cand" ] && [ -x "$cand" ] || continue; \
        if "$cand" -c "import faster_whisper" >/dev/null 2>&1; then PY="$cand"; break; fi; \
    done; \
    : "${PY:?no interpreter in this image can import faster_whisper}"; \
    "$PY" -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    "$PY" -c "import transformers" >/dev/null 2>&1 \
        || "$PY" -m pip install --no-cache-dir --root-user-action=ignore transformers; \
    printf '#!/bin/sh\nexec %s "$@"\n' "$PY" > /usr/local/bin/audio-python; \
    chmod 755 /usr/local/bin/audio-python; \
    audio-python -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    audio-python -c "from ctranslate2.converters import TransformersConverter"; \
    command -v ffmpeg >/dev/null

WORKDIR /app
COPY wrapper /app/wrapper
ENV AUDIO_BASE=fasterwhisper \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WRAPPER_PORT=8000

# Default: serve directly (local dev / docker run). The Olares chart overrides
# command with a thin sentinel-wait shell (templates/wrappers.yaml) that execs
# this same module after llm-init finishes the model download.
CMD ["audio-python", "-m", "wrapper.app"]
