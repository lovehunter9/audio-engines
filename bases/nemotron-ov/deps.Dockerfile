# Nemotron 3 Diarization on OpenVINO. Intel GPU is amd64 only.
# Sibling of bases/nemotron, not a variant of it: no CUDA, no triton, no slim.sh.
# Same NeMo commit. PyPI nemo-toolkit 3.0.0 has no rope TransformerEncoder.
ARG NEMO_REF=5d641ef5048bee7a496e5cee8289a18215570b1a
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

ARG NEO_VER=26.22.38646.4
ARG IGC_TAG=v2.36.3
ARG IGC_DEB=2.36.3+21719
ARG NEMO_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
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

# CPU torch: the image exports IR on first load. Inference is OpenVINO. Do not install a CUDA wheel.
# Noble's pip is a Debian package with no RECORD, so -U pip cannot uninstall it. NeMo's wheel
# is Metadata-Version 2.4, which pip 24.0 does not read; overlay pip>=24.1 instead.
RUN python3 -m pip install --no-cache-dir --ignore-installed 'pip>=24.1' \
    && python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip freeze | grep -E '^(torch|torchaudio)==' > /tmp/torch.pin \
    && python3 -m pip install --no-cache-dir -c /tmp/torch.pin \
        "nemo-toolkit[asr] @ https://github.com/NVIDIA-NeMo/Speech/archive/${NEMO_REF}.tar.gz" \
        numpy openvino \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29" websockets \
    && rm -f /tmp/torch.pin \
    && python3 -c "\
import torch; \
from nemo.collections.asr.models import SortformerEncLabelModel; \
import openvino; \
assert not torch.cuda.is_available(), 'CPU torch picked up CUDA'; \
assert SortformerEncLabelModel.restore_from.__name__; \
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder; \
TransformerEncoder(feat_in=16, d_model=32, n_heads=2, n_layers=1, self_attention_model='rope'); \
print('nemotron-ov torch', torch.__version__, 'openvino', openvino.__version__)"

LABEL org.opencontainers.image.title="audio-nemotron-ov-deps"
