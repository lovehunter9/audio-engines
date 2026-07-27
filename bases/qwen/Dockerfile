# audio-qwen: Qwen ASR family on qwen-asr's in-process vLLM. Build context is the repo ROOT.
FROM docker.io/beclab/vllm-vllm-openai:v0.23.0-cu129

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-qwen" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# Build-time deps only. blinker first (distutils 1.4 blocks qwen-asr); plain qwen-asr, not [vllm].
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed blinker \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        qwen-asr soundfile librosa av websockets python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29" \
    && python3 -c "import qwen_asr, soundfile, librosa, fastapi, uvicorn, multipart, websockets" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

WORKDIR /app
COPY wrapper /app/wrapper
ENV AUDIO_BASE=qwen \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WRAPPER_PORT=8000

# Serves directly here; the chart overrides command with its sentinel-wait shell, same module.
CMD ["audio-python", "-m", "wrapper.app"]
