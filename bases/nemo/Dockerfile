# audio-nemo base: NVIDIA NeMo runtime for Streaming Sortformer (diar_stream).
# Build context is the repo ROOT (COPY wrapper/...).
#   docker build -f bases/nemo/Dockerfile -t <reg>/audio-nemo:<tag> .
FROM docker.io/beclab/nvidia-nemo:26.02

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-nemo" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# NeMo lives in the image's own venv, NOT in the system python, so install the
# web layer into THAT interpreter and expose it as `audio-python` — the single
# name every base image and every chart's sentinel shell uses.
RUN PY=/opt/venv/bin/python \
    && [ -x "$PY" ] || PY=$(command -v python3) \
    && "$PY" -m pip install --no-cache-dir --root-user-action=ignore \
        python-multipart websockets "fastapi>=0.110" "uvicorn>=0.29" \
    && "$PY" -c "import nemo.collections.asr, numpy, fastapi, uvicorn, websockets" \
    && ln -sf "$PY" /usr/local/bin/audio-python

WORKDIR /app
COPY wrapper /app/wrapper
ENV AUDIO_BASE=nemo \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WRAPPER_PORT=8000

# Default: serve directly (local dev / docker run). The Olares chart overrides
# command with a thin sentinel-wait shell (templates/wrappers.yaml) that execs
# this same module after llm-init finishes the model download.
CMD ["audio-python", "-m", "wrapper.app"]
