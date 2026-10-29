# audio-qwen3voice deps: faster-qwen3-tts. amd64 uses the official CUDA runtime; arm64 uses slim plus cu130.
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base-amd64
# ffmpeg on both arches: without it audioread has no backend for the mp4/m4a a phone records.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1; \
    rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim-bookworm AS base-arm64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    python3 -c "import torch; \
assert torch.version.cuda, 'arm64 qwen3tts deps must be CUDA torch, got %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FROM base-${TARGETARCH}
ARG TARGETARCH

# Pin faster-qwen3-tts==0.3.2 and torch>=2.5.1: 0.4+ pulls transformers 5 and breaks qwen-tts init.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "faster-qwen3-tts==0.3.2" \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets \
    && python3 -c "\
import os, shutil, sys, torch, soundfile, fastapi, httpx, websockets; \
from faster_qwen3_tts import FasterQwen3TTS; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: audioread has no backend and non-wav reference audio dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda); \
assert tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:3]) >= (2, 5, 1), \
    'faster-qwen3-tts needs torch>=2.5.1 for CUDA graph capture, got %s' % torch.__version__; \
assert arch != 'arm64' or torch.version.cuda, 'arm64 lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-qwen3voice-deps"
