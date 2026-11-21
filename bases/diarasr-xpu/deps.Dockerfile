# audio-diarasr-xpu deps: the diarasr wrapper on PyTorch XPU, for discrete Intel Arc.
#
# The same transformers remote code as bases/diarasr, on XPU instead of CUDA; no OpenVINO
# export. The OpenCL ICD is not optional even though torch drives the card through Level
# Zero: oneDNN takes its OpenCL path for some shapes (an 11000-row Linear, the stft behind
# the feature extractor, parts of SDPA), and without the ICD those fail with "could not
# create a primitive". Intel GPU is x86_64, so amd64 only.
FROM ubuntu:24.04
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ARG NEO_VER=26.31.39395.13
ARG IGC_TAG=v2.40.13
ARG IGC_DEB=2.40.13+22418
RUN set -eux; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "diarasr-xpu base is amd64-only (TARGETARCH=${TARGETARCH})" >&2; \
        exit 1; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-pip python3-venv \
        libsndfile1 ffmpeg libze1 ocl-icd-libopencl1 clinfo; \
    mkdir -p /tmp/neo && cd /tmp/neo; \
    curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-core-2_${IGC_DEB}_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-opencl-2_${IGC_DEB}_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-ocloc_${NEO_VER}-0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-opencl-icd_${NEO_VER}-0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libigdgmm12_22.10.0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libze-intel-gpu1_${NEO_VER}-0_amd64.deb"; \
    printf '%s\n' \
        "ebd795e9fddf303a9b24b7f04545d8ddd9ad1f85b3d0cb1166476fab24da6d44  intel-igc-core-2_${IGC_DEB}_amd64.deb" \
        "4f990874efc11c3f6091a663b08aef576c4af592dcd8f12e116f8c2fc92d34d9  intel-igc-opencl-2_${IGC_DEB}_amd64.deb" \
        "12c5e61ed1dca5cbf38494e280abf88100a451580d57c44f601a17d9727e465e  intel-ocloc_${NEO_VER}-0_amd64.deb" \
        "5a9c9e8fdca8a2f9e22754b1a4618c7babf21d7c3ab3503c680005007c7a8c44  intel-opencl-icd_${NEO_VER}-0_amd64.deb" \
        "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  libigdgmm12_22.10.0_amd64.deb" \
        "1722943f81b576b9bb8d61016464208f48ce533dc3bf24ad39605293115cc289  libze-intel-gpu1_${NEO_VER}-0_amd64.deb" \
        | sha256sum -c; \
    dpkg -i /tmp/neo/*.deb; \
    apt-get install -y -f --no-install-recommends; \
    rm -rf /tmp/neo /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# The XPU index first, so the later installs keep this torch rather than pulling a CUDA one.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        --index-url https://download.pytorch.org/whl/xpu "torch==2.15.0"

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers>=5.6,<6" "safetensors>=0.6.2" accelerate \
        "huggingface-hub>=0.34" soundfile librosa soxr numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart \
    && ! python3 -m pip list 2>/dev/null | grep -i '^nvidia-' \
    && python3 -c "\
import shutil, torch, transformers, librosa, fastapi; \
from transformers import VibeVoiceAsrForConditionalGeneration; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
assert torch.version.cuda is None and getattr(torch.version, 'xpu', None), \
    'diarasr-xpu deps must be XPU torch, got %s' % torch.__version__; \
v = tuple(int(x) for x in transformers.__version__.split('.')[:2]); \
assert (5, 6) <= v < (6, 0), transformers.__version__; \
print('torch', torch.__version__, 'xpu', torch.version.xpu, 'transformers', transformers.__version__)"

LABEL org.opencontainers.image.title="audio-diarasr-xpu-deps" \
      audio.compute_runtime="26.31.39395.13"
