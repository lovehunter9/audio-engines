# audio-qwen3tts deps: faster-qwen3-tts in-process, no vLLM.
#
# vLLM-Omni served this model as two engine processes, each with its own CUDA context and KV pool:
# upstream #2318 measured 22 GB for a 0.6B model whose weights are 2.6 GB. faster-qwen3-tts is
# plain torch.cuda.CUDAGraph in one process, ~4.4 GB for the 1.7B, and it is the only path here
# that streams at all — the stock Qwen3-TTS repo has no streaming.
#
# amd64: official pytorch CUDA runtime (single-arch upstream).
# arm64: no pytorch/pytorch CUDA tag — same recipe as pyannote/mossspeech arm64 (slim + cu130).
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

# CUDA-graph capture is unreliable on torch<=2.5.0, so the floor is 2.5.1, asserted below.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "faster-qwen3-tts>=0.3.2" \
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

LABEL org.opencontainers.image.title="audio-qwen3tts-deps"
