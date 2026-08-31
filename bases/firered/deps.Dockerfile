# audio-firered deps: pin FireRedTTS3, no vLLM / flash_attn; amd64=pytorch CUDA, arm64=slim+cu130.
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base-amd64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git curl; \
    rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim-bookworm AS base-arm64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git curl build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    python3 -m pip install --no-cache-dir --no-deps torchcodec; \
    python3 -c "import torch; \
assert torch.version.cuda, 'arm64 firered deps must be CUDA torch, got %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FROM base-${TARGETARCH}
ARG TARGETARCH

# FireRedTeam/FireRedTTS3 HEAD as of 2026-08-31 (Instruct API + generate_tts).
ARG FIRERED_REF=1d32ba780da6af37a71bdfd9c68c12003e908a46

# Pin transformers 5.6.2; use fasttext-wheel (the sdist needs a C++ compiler the CUDA runtime image lacks).
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers==5.6.2" einops regex python-dotenv wetext fasttext-wheel \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets

RUN set -eux; \
    mkdir -p /opt/fireredtts3 /tmp/firered-src; \
    cd /tmp/firered-src; \
    git init -q .; \
    git remote add origin https://github.com/FireRedTeam/FireRedTTS3.git; \
    git fetch -q --depth 1 origin "${FIRERED_REF}"; \
    git checkout -q FETCH_HEAD; \
    cp -r fireredtts3 /opt/fireredtts3/; \
    mkdir -p /opt/fireredtts3/fireredtts3/utils/llm_tn/models; \
    curl -fsSL -o /opt/fireredtts3/fireredtts3/utils/llm_tn/models/lid.176.ftz \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz; \
    echo "${FIRERED_REF}" > /opt/fireredtts3/COMMIT; \
    rm -rf /tmp/firered-src; \
    python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'fireredtts3.pth'), 'w').write('/opt/fireredtts3\n'); \
print('sys.path entry installed via', os.path.join(p, 'fireredtts3.pth'))"

RUN env -u PYTHONPATH python3 -c "\
import os, shutil, torch, transformers, fasttext; \
from fireredtts3.core import FireRedTTS3Instruct; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference audio that is not wav dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert transformers.__version__.startswith('5.6.'), \
    'FireRedTTS3-Instruct config targets transformers 5.6.x, got %s' % transformers.__version__; \
assert arch != 'arm64' or torch.version.cuda, 'arm64 lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-firered-deps"
