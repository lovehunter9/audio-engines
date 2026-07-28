# audio-qwen deps: everything but the wrapper, so a wrapper change costs one small layer (README).
# Rebuilt only when THIS file changes: its content hash is the tag the append step looks for.
FROM docker.io/beclab/vllm-vllm-openai:v0.23.0-cu129

# Build-time deps only. blinker first (distutils 1.4 blocks qwen-asr); plain qwen-asr, not [vllm].
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed blinker \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        qwen-asr soundfile librosa av websockets python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29" \
    && python3 -c "import qwen_asr, soundfile, librosa, fastapi, uvicorn, multipart, websockets" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-qwen-deps"
