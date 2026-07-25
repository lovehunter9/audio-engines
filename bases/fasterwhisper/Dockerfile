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
# be replaced by a generic wheel, so only the web layer is added.
#
# The final import check covers the WEB deps too, so this fails the BUILD rather
# than a user's clone — an earlier version only checked the engine deps and shipped
# an image whose wrapper died on `import fastapi`.
RUN set -eux; \
    PY=; \
    for cand in "$(command -v python3 || true)" /Whisper-WebUI/venv/bin/python3 /usr/bin/python3; do \
        [ -n "$cand" ] && [ -x "$cand" ] || continue; \
        if "$cand" -c "import faster_whisper" >/dev/null 2>&1; then PY="$cand"; break; fi; \
    done; \
    : "${PY:?no interpreter in this image can import faster_whisper}"; \
    "$PY" -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    "$PY" -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    command -v ffmpeg >/dev/null; \
    ln -sf "$PY" /usr/local/bin/audio-python

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
