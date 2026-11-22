# audio-diarasr-sgl-xpu deps: the diarasr-sgl wrapper on SGLang-Omni's XPU build, for discrete Intel Arc.
#
# On an Arc Pro B70 a 14-minute meeting takes 65 s against 235 s for transformers in
# bases/diarasr-xpu. The wrapper has to ask for the intel_xpu attention backend: the default triton
# decode attention slows down linearly with context and ends up slower than transformers. There
# is no DFlash here; sglang-omni refuses it off CUDA.
#
# What the XPU runtime needs beyond bases/diarasr-xpu:
# - libze-dev: oneCCL dlopens the unversioned libze_loader.so, and Triton's Intel backend compiles
#   its launcher against level_zero/ze_api.h. Without either the engine fails at startup.
# - ldconfig after the pip installs: oneCCL's plugins land in /usr/local/lib.
# - python3-dev and a compiler: Triton builds that launcher at runtime.
FROM ubuntu:24.04
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300

ARG NEO_VER=26.31.39395.13
ARG IGC_TAG=v2.40.13
ARG IGC_DEB=2.40.13+22418
ARG ZE_VER=1.34.0
ARG UV_VERSION=0.13.9
RUN set -eux; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "diarasr-sgl-xpu base is amd64-only (TARGETARCH=${TARGETARCH})" >&2; \
        exit 1; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 python3-dev python3-pip build-essential \
        libsndfile1 ffmpeg ocl-icd-libopencl1 clinfo libnuma1; \
    mkdir -p /tmp/neo && cd /tmp/neo; \
    curl -fsSL -O "https://github.com/oneapi-src/level-zero/releases/download/v${ZE_VER}/libze1_${ZE_VER}+u24.04_amd64.deb"; \
    curl -fsSL -O "https://github.com/oneapi-src/level-zero/releases/download/v${ZE_VER}/libze-dev_${ZE_VER}+u24.04_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-core-2_${IGC_DEB}_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-opencl-2_${IGC_DEB}_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-ocloc_${NEO_VER}-0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-opencl-icd_${NEO_VER}-0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libigdgmm12_22.10.0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libze-intel-gpu1_${NEO_VER}-0_amd64.deb"; \
    printf '%s\n' \
        "45210e4549cd965ad7b9f160eefe53cbcdba01af341bea7ac7ba0aedeeb613ba  libze1_${ZE_VER}+u24.04_amd64.deb" \
        "2813bb3b06d0d8c0f3c771051defabdd74da2e454661cd654ebabcfb5ab21c01  libze-dev_${ZE_VER}+u24.04_amd64.deb" \
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
    test -e /usr/lib/x86_64-linux-gnu/libze_loader.so; \
    test -f /usr/include/level_zero/ze_api.h; \
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C /usr/local/bin --strip-components=1; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python

ARG SGLANG_REF=v0.5.23
ARG SGLANG_OMNI_REF=2ba53e5fbdb5975bab2e307f44740c61fcda0cfa
# Both projects ship a separate pyproject for XPU; it replaces the CUDA one before the install.
RUN set -eux; \
    XPU=https://download.pytorch.org/whl/xpu; \
    I="uv pip install --index-strategy unsafe-best-match"; \
    $I --extra-index-url $XPU torch==2.14.0+xpu torchvision==0.29.0+xpu torchaudio==2.11.0+xpu \
        torchcodec==0.17.0 "setuptools>=77.0.0" setuptools-scm wheel; \
    git clone -q --depth 1 --branch "$SGLANG_REF" https://github.com/sgl-project/sglang.git /tmp/sglang; \
    cp /tmp/sglang/python/pyproject_xpu.toml /tmp/sglang/python/pyproject.toml; \
    $I --no-build-isolation --extra-index-url $XPU /tmp/sglang/python; \
    $I --no-deps xgrammar==0.1.33; \
    git init -q /tmp/sglang-omni; \
    git -C /tmp/sglang-omni fetch -q --depth 1 https://github.com/sgl-project/sglang-omni "$SGLANG_OMNI_REF"; \
    git -C /tmp/sglang-omni checkout -q FETCH_HEAD; \
    cp /tmp/sglang-omni/pyproject_xpu.toml /tmp/sglang-omni/pyproject.toml; \
    $I --no-build-isolation --extra-index-url $XPU /tmp/sglang-omni; \
    $I --no-deps openai-whisper==20250625; \
    $I "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart soundfile; \
    rm -rf /tmp/sglang /tmp/sglang-omni; \
    ldconfig; \
    ! uv pip list 2>/dev/null | grep -i '^nvidia-'; \
    command -v sgl-omni; \
    python3 -c "\
import shutil, torch, sglang, sglang_omni, fastapi; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
assert torch.version.cuda is None and getattr(torch.version, 'xpu', None), torch.__version__; \
from sglang_omni.models.moss_transcribe_diarize import config; \
print('torch', torch.__version__, 'xpu', torch.version.xpu, 'sglang', sglang.__version__)"

LABEL org.opencontainers.image.title="audio-diarasr-sgl-xpu-deps" \
      audio.compute_runtime="26.31.39395.13" \
      audio.sglang_omni="2ba53e5fbdb5975bab2e307f44740c61fcda0cfa"
