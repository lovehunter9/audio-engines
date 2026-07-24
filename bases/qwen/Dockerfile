# audio-qwen base: Qwen ASR family (stt / stt_stream / align) on qwen-asr's
# in-process vLLM engine. Build context is the repo ROOT (COPY wrapper/...).
#   docker build -f bases/qwen/Dockerfile -t <reg>/audio-qwen:<tag> .
FROM docker.io/beclab/vllm-vllm-openai:v0.23.0-cu129

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-qwen" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# BUILD-TIME deps (no runtime pip). qwen-asr pulls flask->blinker but the image
# ships a distutils blinker 1.4 pip won't uninstall; reinstall it as pip-managed
# first so the qwen-asr install doesn't abort. Install PLAIN qwen-asr (NOT
# [vllm], which re-resolves the whole vLLM stack) since vLLM is already present.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed blinker \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        qwen-asr soundfile librosa av websockets python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29"

WORKDIR /app
COPY wrapper /app/wrapper
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WRAPPER_PORT=8000

# Default: serve directly (local dev / docker run). The Olares chart overrides
# command with a thin sentinel-wait shell (templates/wrappers.yaml) that execs
# this same module after llm-init finishes the model download.
CMD ["python3", "-m", "wrapper.app"]
