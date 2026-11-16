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
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "openvino==2026.4.1" "optimum-intel==2.2.0" "transformers==5.2.0" onnx

ARG S3_REVISION=1ca52ea064ec2081f12675d853525d394571bf73
RUN git init /opt/s3 \
    && git -C /opt/s3 remote add origin https://github.com/QwenAudio/S3-S2ST.git \
    && git -C /opt/s3 fetch --depth 1 origin "$S3_REVISION" \
    && git -C /opt/s3 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/s3 rev-parse HEAD)" = "$S3_REVISION" \
    && printf '%s\n' "$S3_REVISION" > /opt/s3/COMMIT \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "numpy==1.26.4" "safetensors==0.5.3" "onnxruntime==1.28.0" \
        "omegaconf==2.3.0" "hydra-core==1.3.2" "einops==0.8.2" \
        "rotary-embedding-torch==0.8.9" soundfile librosa sentencepiece PyYAML \
        kaldiio tiktoken tqdm fastapi uvicorn httpx python-multipart websockets \
    && python3 -c "import torch, openvino, transformers; from optimum.intel import OVModelForCausalLM; assert torch.version.cuda is None; assert transformers.__version__ == '5.2.0'; print(torch.__version__, openvino.__version__, transformers.__version__)"

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore more-itertools \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore --no-deps "openai-whisper==20250625"
COPY wrapper/caps/s3_ov_source.py /opt/s3_adapter/s3_ov_source.py
RUN PYTHONPATH=/opt/s3_adapter python3 -c "from s3_ov_source import neural_definitions, definitions; e, a, s = neural_definitions('/opt/s3'); assert e.SenseVoiceEncoder and a.Transformer and s.enable_true_streaming_encoder; import whisper; print('original neural definitions and calibrated frontend dependencies loaded')"

LABEL org.opencontainers.image.title="S3 OpenVINO graph export dependencies" \
      org.opencontainers.image.source="https://github.com/QwenAudio/S3-S2ST" \
      org.opencontainers.image.revision="1ca52ea064ec2081f12675d853525d394571bf73"
CMD ["sleep", "infinity"]
