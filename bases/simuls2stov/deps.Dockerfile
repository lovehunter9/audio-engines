# SimulS2ST-Omni on Intel OpenVINO. The CUDA image stays bases/simuls2st. amd64 only.
#
# iGPU and Arc share this image. The chart sets OLARES_GPU_MODE to intel or intel-gpu;
# both compile for OpenVINO device GPU. Weights stay out of the image.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ARG NEO_VER=26.22.38646.4
ARG IGC_TAG=v2.36.3
ARG IGC_DEB=2.36.3+21719
ARG SIMUL_SHA=c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libsndfile1 \
        ffmpeg \
        libze1 \
        ocl-icd-libopencl1 \
        clinfo \
    && mkdir -p /tmp/neo && cd /tmp/neo \
    && curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-core-2_${IGC_DEB}_amd64.deb" \
    && curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-opencl-2_${IGC_DEB}_amd64.deb" \
    && curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-ocloc_${NEO_VER}-0_amd64.deb" \
    && curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-opencl-icd_${NEO_VER}-0_amd64.deb" \
    && curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libigdgmm12_22.10.0_amd64.deb" \
    && curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libze-intel-gpu1_${NEO_VER}-0_amd64.deb" \
    && printf '%s\n' \
        "9e0975ac75015b431ebb2da81a802b9fd1e28a3c270313a97569cd1e6a6c6048  intel-igc-core-2_${IGC_DEB}_amd64.deb" \
        "350a52331e784bb7fb9ed42e993b5c44b7e6562fc74d2cf3102b29b6a576fa85  intel-igc-opencl-2_${IGC_DEB}_amd64.deb" \
        "25745841e66279c7ff1ff6c2d4da9ec911685c12e1c9c5609d99e4d54a364dc0  intel-ocloc_${NEO_VER}-0_amd64.deb" \
        "6fdac2e8a2aacf844ebfd90521bf7102b3ebb44f69c1bced1a9785a7ce96a3c2  intel-opencl-icd_${NEO_VER}-0_amd64.deb" \
        "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  libigdgmm12_22.10.0_amd64.deb" \
        "8bef9f24e03f826f93c076081bda13c6ac3afbd9e42b9fb8f298fab652330e2f  libze-intel-gpu1_${NEO_VER}-0_amd64.deb" \
        | sha256sum -c \
    && dpkg -i /tmp/neo/*.deb \
    && apt-get install -y -f --no-install-recommends \
    && rm -rf /tmp/neo /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        openvino

RUN git clone https://github.com/hasaki321/SimulS2ST-Omni.git /opt/simuls2st \
    && git -C /opt/simuls2st checkout "$SIMUL_SHA" \
    && git -C /opt/simuls2st submodule update --init external/SimulEval \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        -e /opt/simuls2st/external/SimulEval \
        "transformers==4.52.3" "peft==0.18.1" "qwen-omni-utils==0.0.8" \
        "accelerate==1.12.0" "numpy==1.26.4" "safetensors==0.7.0" \
        "soundfile==0.13.1" "sentencepiece==0.2.1" "einops==0.8.1" \
        "scipy==1.15.3" "librosa==0.11.0" \
        "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0" \
        "easydict==1.13" "pyworld==0.3.5" "json5>=0.9.0" "ruamel.yaml==0.18.16" \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets \
        "numpy==1.26.4" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore --force-reinstall \
        torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -c "\
import torch, torchaudio, transformers, peft, openvino; \
assert torch.__version__.startswith('2.8.'), torch.__version__; \
assert not torch.cuda.is_available(), 'simuls2stov must not ship CUDA torch'; \
assert torch.version.cuda is None, torch.version.cuda; \
print('torch', torch.__version__, 'openvino', openvino.__version__, \
      'torchaudio', torchaudio.__version__, 'transformers', transformers.__version__, \
      'peft', peft.__version__)"

ENV SIMULS2ST_ROOT=/opt/simuls2st \
    PYTHONPATH=/opt/simuls2st

RUN python3 -c "\
from src.agents.simuleval_omni_talker_s2st_agent import OmniTalkerStreamingS2STAgent; \
import src.voicebox.melspec, src.voicebox.tokenizer, src.voicebox.voicebox_model; \
from src.voicebox.util import load_config; \
print('agent', OmniTalkerStreamingS2STAgent.__name__)"

LABEL org.opencontainers.image.title="audio-simuls2st-ov-deps" \
      org.opencontainers.image.source="https://github.com/hasaki321/SimulS2ST-Omni" \
      org.opencontainers.image.revision="c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c" \
      audio.compute_runtime="26.22.38646.4"
