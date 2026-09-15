# Pyannote / Silero / SpeechBrain: own image, no shared audio-runtime.
# Stay on pyannote.audio 3.x: 4.x pulls torch 2.14 + a second CUDA 13 stack
# and changes the Pipeline API. Torch goes on first so pip does not bring
# that stack in, then strip drops any leftover nvidia-* that ldd does not map.
ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN set -eux; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url "${IDX}"; \
    python3 -m pip install --no-cache-dir \
        "pyannote.audio>=3.3.0,<4" speechbrain silero-vad omegaconf soundfile \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    python3 -m pip uninstall -y \
        matplotlib pandas optuna pyannoteai-sdk \
        opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
        opentelemetry-exporter-otlp-proto-grpc \
        opentelemetry-exporter-otlp-proto-http \
        opentelemetry-exporter-otlp-proto-common \
        opentelemetry-proto opentelemetry-semantic-conventions \
        || true; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    ldconfig; \
    python3 -c "import torch, pyannote.audio, silero_vad, omegaconf, speechbrain, soundfile, fastapi, uvicorn, multipart; \
print('TARGETARCH', '''${TARGETARCH}''', 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'pyannote.audio', getattr(pyannote.audio, '__version__', '?')); \
assert torch.version.cuda, 'lost CUDA torch after pip install'; \
v=getattr(pyannote.audio, '__version__', '0'); \
assert v.startswith('3.'), 'expected pyannote.audio 3.x, got %s' % v"; \
    python3 -c "import soundfile as sf; \
    print('libsndfile', sf.__libsndfile_version__); \
    [print(c, s, sf.check_format(c, s)) for c, s in \
     (('FLAC', 'PCM_16'), ('OGG', 'OPUS'), ('OGG', 'VORBIS'))]"

LABEL org.opencontainers.image.title="audio-pyannote-deps"
