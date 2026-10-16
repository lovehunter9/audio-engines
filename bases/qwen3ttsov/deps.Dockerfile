# audio-qwen3tts-ov deps. Ubuntu + OpenVINO; Intel GPU is amd64 only.
# The CUDA image stays bases/qwen3tts (faster-qwen3-tts). This one does not install it.
#
# Text2SpeechPipeline landed in openvino-genai after the 2026.4.1 wheel. optimum-intel
# d7c6f7ef53 (GPU fix, 2026-10-14) exports the checkpoint and runs qwen_tts on those graphs,
# against the openvino wheel this image already installs. qwen-tts 0.1.1 needs transformers
# 4.57.3, so this image does not take the ASR image's transformers 5.13 pin.
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
        libsndfile1 \
        ffmpeg \
        sox \
        libsox-fmt-all \
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

# Matched CPU wheels. The CPU index has no torchaudio next to torch 2.14, and a PyPI
# torchaudio would replace this torch with a CUDA build.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.9.1" "torchaudio==2.9.1"

# qwen-tts is installed without its deps: gradio is the demo UI, and its torch pin would
# fight the CPU wheel above. transformers stays on 4.57.3, the release qwen-tts 0.1.1 requires.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        openvino \
        openvino-genai \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
        "huggingface_hub" \
        librosa \
        soundfile \
        einops \
        "fastapi>=0.110" \
        "uvicorn>=0.29" \
        python-multipart \
        websockets \
        numpy \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "qwen-tts==0.1.1" --no-deps \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "git+https://github.com/huggingface/optimum-intel.git@d7c6f7ef53b8584df63eef7b143aff0dc1d03fca" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers==4.57.3" \
        "pyparsing==3.3.2" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.9.1" "torchaudio==2.9.1" \
    && ! python3 -m pip list 2>/dev/null | grep -i '^nvidia-' \
    && python3 -c "import pyparsing, openvino, torch, torchaudio, transformers, qwen_tts, librosa, soundfile, fastapi, uvicorn, huggingface_hub, numpy; \
from optimum.intel import OVModelForTextToSpeechSeq2Seq; \
assert pyparsing.__version__ == '3.3.2', pyparsing.__version__; \
assert transformers.__version__ == '4.57.3', transformers.__version__; \
assert torch.version.cuda is None, torch.__version__; \
print('torch', torch.__version__); \
print('torchaudio', torchaudio.__version__); \
print('transformers', transformers.__version__); \
print('openvino', openvino.__version__); \
print('qwen_tts', 'ok'); \
print('qwen3_tts', OVModelForTextToSpeechSeq2Seq.__name__)"

LABEL org.opencontainers.image.title="audio-qwen3tts-ov-deps" \
      audio.compute_runtime="26.22.38646.4"
