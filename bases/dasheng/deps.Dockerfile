# audio-dasheng deps: Dasheng-AudioGen (Xiaomi LLM PLUS + SJTU X-LANCE) in-process.
#
# A diffusion transformer, not an LLM — no vLLM, no child engine process, no KV pool. The whole
# model is `AutoModel.from_pretrained(..., trust_remote_code=True)` plus a flan-t5-large text
# encoder and the dashengtokenizer codec, all three loaded in this process.
#
# transformers 5.x is not merely untested: the upstream card says "Not compatible with
# transformers 5.x", and the model ships custom modeling code that calls the 4.x API. Pinned
# below rather than left to resolve to whatever is newest on build day.
#
# amd64: official pytorch CUDA runtime (single-arch upstream).
# arm64: no pytorch/pytorch CUDA tag — same recipe as the other bases here (slim + cu130).
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base-amd64
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
assert torch.version.cuda, 'arm64 dasheng deps must be CUDA torch, got %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FROM base-${TARGETARCH}
ARG TARGETARCH

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers>=4.51,<5" einops \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets \
    && python3 -c "\
import os, shutil, torch, transformers, einops, soundfile, fastapi, httpx; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: librosa cannot decode anything libsndfile refuses'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert int(transformers.__version__.split('.')[0]) < 5, \
    'Dasheng-AudioGen custom modeling code targets transformers 4.x, got %s' % transformers.__version__; \
assert arch != 'arm64' or torch.version.cuda, 'arm64 lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-dasheng-deps"
