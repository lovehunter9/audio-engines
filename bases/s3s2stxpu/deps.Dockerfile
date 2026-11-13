FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ARG NEO_VER=26.22.38646.4
ARG IGC_TAG=v2.36.3
ARG IGC_DEB=2.36.3+21719
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libsndfile1-dev \
        libtbb-dev \
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

RUN python3 -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
ARG S3_REVISION=cb7439d89863e0b3524b744758b8098cc33a024e
RUN git clone https://github.com/QwenAudio/S3-S2ST.git /opt/s3 && git -C /opt/s3 checkout "$S3_REVISION"
RUN printf '%s\n' 'torch==2.8.0+xpu' 'torchaudio==2.8.0+xpu' 'numpy==1.26.4' 'scipy==1.15.3' 'tbb==2022.1.0' > /opt/s3-constraints.txt && python3 -m pip install --no-cache-dir -c /opt/s3-constraints.txt '/opt/s3[inference]' scipy fastapi uvicorn httpx python-multipart websockets
COPY bases/s3s2st/extract_native.py /opt/extract_s3_native.py
RUN python3 /opt/extract_s3_native.py
ENV PYTHONPATH=/opt/s3/native:/opt/s3/vendor/FunASR:/opt/s3/inference/streaming:/opt/s3
ENV S3_ROOT=/opt/s3
RUN python3 -c "import torch; from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig,Qwen2Model,Qwen2Config; from s3_encoder import SenseVoiceEncoder; from s3_adaptor import Transformer; import funasr.models.cosyvoice.flow_matching_model,funasr.models.cosyvoice.causal_bigvgan; assert torch.__version__.startswith('2.8.'); print('S3 native modules loaded')" && ln -sf "$(command -v python3)" /usr/local/bin/audio-python
