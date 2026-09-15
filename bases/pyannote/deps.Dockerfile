# Pyannote / Silero / SpeechBrain: own image, no shared audio-runtime.
# 4.x is the line that still imports on current cu128 / cu130 torch
# (2.9+ dropped torchaudio.AudioMetaData, which 3.x still names).
# Install our torch first, then pin that pair so pip cannot let 4.x
# pull torch 2.14 and a second CUDA 13 stack. strip drops leftover
# nvidia-* that ldd does not map.
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
    python3 -m pip freeze | grep -E '^(torch|torchaudio)==' > /tmp/torch.pin; \
    python3 -m pip install --no-cache-dir -c /tmp/torch.pin \
        "pyannote.audio>=4,<5" speechbrain silero-vad omegaconf soundfile \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    # 4.x imports telemetry on `import pyannote.audio` (OTLP HTTP exporter).
    # pandas and the opentelemetry stack stay; they are not CUDA bloat.
    python3 -m pip uninstall -y \
        matplotlib optuna pyannoteai-sdk || true; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh /tmp/torch.pin

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
assert v.startswith('4.'), 'expected pyannote.audio 4.x, got %s' % v"; \
    python3 -c "import soundfile as sf; \
    print('libsndfile', sf.__libsndfile_version__); \
    [print(c, s, sf.check_format(c, s)) for c, s in \
     (('FLAC', 'PCM_16'), ('OGG', 'OPUS'), ('OGG', 'VORBIS'))]"

LABEL org.opencontainers.image.title="audio-pyannote-deps"
