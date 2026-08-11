# audio-soulx deps: SoulX-Podcast (Soul AI Lab) in-process, no vLLM.
#
# Upstream publishes no PyPI package and ships no setup.py, so pip cannot install it from git
# either. The source is cloned here at a pinned commit and put on PYTHONPATH — it is never copied
# into this repository. Upgrading is a one-line SHA bump; patching, if it ever comes to that,
# belongs in a .patch file applied right after the checkout.
#
# Two things about this model cost more than they look:
#
#   * The audio tokenizer is NOT in the model repo. s3tokenizer.load_model() reaches out to
#     ModelScope over plain urllib for a ~480 MB onnx and caches it under XDG_CACHE_HOME. That
#     fetch obeys neither HF_HUB_OFFLINE nor llm-init, so on an offline node it is a hard boot
#     failure. It is baked in below, and XDG_CACHE_HOME is pinned so the cache is still found at
#     runtime no matter what HOME the pod gets.
#   * transformers is pinned to upstream's exact version. The LLM stage is a Qwen2.5 backbone
#     driven through a hand-rolled engine that reaches into transformers internals.
#
# amd64: official pytorch CUDA runtime (single-arch upstream).
# arm64: no pytorch/pytorch CUDA tag — same recipe as the other bases here (slim + cu130).
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base-amd64
# git for the pinned checkout; ffmpeg because audioread has no backend for uploads without it.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git; \
    rm -rf /var/lib/apt/lists/*

# torchaudio 2.9 (what arm64 resolves) rebuilds load() on torchcodec; --no-deps keeps our torch.
FROM python:3.11-slim-bookworm AS base-arm64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    python3 -m pip install --no-cache-dir --no-deps torchcodec; \
    python3 -c "import torch; \
assert torch.version.cuda, 'arm64 soulx deps must be CUDA torch, got %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FROM base-${TARGETARCH}
ARG TARGETARCH

# Soul-AILab/SoulX-Podcast @ main, 2025-12-11. Pinned: upstream pins no dependency of its own.
ARG SOULX_REF=5ac9c0e1cfe596396200c7d38e3fd53b7b3fbf4b

# Must hold at runtime too: s3tokenizer reads it, and the pod's HOME is not the build's /root.
ENV XDG_CACHE_HOME=/opt/cache
# No PYTHONPATH here: the appended wrapper layer sets it to /app, so /opt/soulx rides a .pth.

# torch/torchaudio stay as the base ships them: pip would resolve its own and swap CUDA for CPU.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers==4.57.1" "accelerate>=1.6" \
        s3tokenizer diffusers onnxruntime \
        "huggingface-hub>=0.34" soundfile librosa scipy numpy einops \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets

# One-commit checkout; the touches turn upstream's namespace dirs into an importable package.
RUN set -eux; \
    mkdir -p /opt/soulx /tmp/soulx-src; \
    cd /tmp/soulx-src; \
    git init -q .; \
    git remote add origin https://github.com/Soul-AILab/SoulX-Podcast.git; \
    git fetch -q --depth 1 origin "${SOULX_REF}"; \
    git checkout -q FETCH_HEAD; \
    cp -r soulxpodcast /opt/soulx/; \
    cp LICENSE /opt/soulx/soulxpodcast/LICENSE; \
    touch /opt/soulx/soulxpodcast/__init__.py /opt/soulx/soulxpodcast/models/__init__.py; \
    echo "${SOULX_REF}" > /opt/soulx/soulxpodcast/COMMIT; \
    rm -rf /tmp/soulx-src; \
    python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'soulx.pth'), 'w').write('/opt/soulx\n'); \
print('sys.path entry installed via', os.path.join(p, 'soulx.pth'))"

# Tokenizer baked in here (see the header); size-checked because a proxy error page writes fine.
RUN set -eux; \
    python3 -c "import s3tokenizer; s3tokenizer.load_model('speech_tokenizer_v2_25hz'); \
print('s3tokenizer weights cached')"; \
    test "$(stat -c%s /opt/cache/s3tokenizer/speech_tokenizer_v2_25hz.onnx)" -gt 100000000; \
    chmod -R a+rX /opt/cache

# Import and decode probe, PYTHONPATH cleared so it runs under the environment the pod gets.
RUN env -u PYTHONPATH python3 -c "\
import os, shutil, tempfile, wave, torch, torchaudio, transformers, onnxruntime, s3tokenizer; \
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input; \
from soulxpodcast.config import SamplingParams; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: audioread has no backend and non-wav reference audio dies'; \
_p=os.path.join(tempfile.mkdtemp(), 'probe.wav'); \
_w=wave.open(_p, 'wb'); _w.setnchannels(1); _w.setsampwidth(2); _w.setframerate(16000); \
_w.writeframes(bytes(1600)); _w.close(); \
_wav, _sr=torchaudio.load(_p); \
assert _sr == 16000 and _wav.numel() == 800, \
    'reference-audio decode probe returned %s Hz %s' % (_sr, tuple(_wav.shape)); \
print('reference-audio decode ok, torchaudio', torchaudio.__version__); \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'torchaudio', torchaudio.__version__, 'transformers', transformers.__version__); \
assert transformers.__version__ == '4.57.1', \
    'SoulX drives transformers internals; pin moved to %s' % transformers.__version__; \
assert arch != 'arm64' or torch.version.cuda, 'arm64 lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-soulx-deps"
