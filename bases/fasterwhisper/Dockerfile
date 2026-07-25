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

# The base ships a CUDA-matched ctranslate2 + faster_whisper build that a
# generic wheel must never replace, so only the web layer is added here.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"

# Fail the BUILD, not a user's clone, if the base ever stops shipping what the
# wrapper needs (CT2 runtime, HF cache reader, ffmpeg for batch slicing).
# `audio-python` is the single interpreter name every base image and every
# chart's sentinel shell uses (some bases keep their deps in a venv).
RUN python3 -c "import faster_whisper, huggingface_hub, torch" \
    && command -v ffmpeg >/dev/null \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

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
