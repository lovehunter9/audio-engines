# audio-breeze deps: pin breeze-tts, no vLLM; amd64=pytorch CUDA, arm64=slim+cu130.
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base-amd64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git; \
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
assert torch.version.cuda, 'arm64 breeze deps must be CUDA torch, got %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FROM base-${TARGETARCH}
ARG TARGETARCH

# breezeblue-ai/breeze-tts HEAD as of 2026-08-31 (infer.py + breeze_infer.runtime).
ARG BREEZE_REF=ca632ce6c4d05f7985da4eab29b1a5d445b43f7b

# Upstream pins transformers 4.57.3 and qwen-tts 0.1.1. torch stays as the base ships it.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers==4.57.3" "qwen-tts==0.1.1" \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.115" "uvicorn>=0.30" httpx python-multipart websockets

RUN set -eux; \
    mkdir -p /opt/breeze-tts /tmp/breeze-src; \
    cd /tmp/breeze-src; \
    git init -q .; \
    git remote add origin https://github.com/breezeblue-ai/breeze-tts.git; \
    git fetch -q --depth 1 origin "${BREEZE_REF}"; \
    git checkout -q FETCH_HEAD; \
    cp -r breeze_infer models configs /opt/breeze-tts/; \
    echo "${BREEZE_REF}" > /opt/breeze-tts/COMMIT; \
    rm -rf /tmp/breeze-src; \
    python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'breeze.pth'), 'w').write('/opt/breeze-tts\n'); \
print('sys.path entry installed via', os.path.join(p, 'breeze.pth'))"

RUN env -u PYTHONPATH python3 -c "\
import os, shutil, torch, transformers; \
from breeze_infer.runtime import load_runtime; \
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference audio that is not wav dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert transformers.__version__ == '4.57.3', \
    'Breeze TTS 2 targets transformers 4.57.3, got %s' % transformers.__version__; \
assert arch != 'arm64' or torch.version.cuda, 'arm64 lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-breeze-deps"
