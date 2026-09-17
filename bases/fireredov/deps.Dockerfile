# FireRedTTS3-Instruct Intel OpenVINO. Official loop + RedAE stay in Python.
# CUDA bases/firered is untouched. Intel GPU is amd64 only.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ARG NEO_VER=26.22.38646.4
ARG IGC_TAG=v2.36.3
ARG IGC_DEB=2.36.3+21719
ARG FIRERED_REF=1d32ba780da6af37a71bdfd9c68c12003e908a46
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
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
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        openvino \
        "transformers==5.6.2" einops regex python-dotenv wetext fasttext-wheel \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets \
    && mkdir -p /opt/fireredtts3 /tmp/firered-src \
    && cd /tmp/firered-src \
    && git init -q . \
    && git remote add origin https://github.com/FireRedTeam/FireRedTTS3.git \
    && git fetch -q --depth 1 origin "${FIRERED_REF}" \
    && git checkout -q FETCH_HEAD \
    && cp -r fireredtts3 /opt/fireredtts3/ \
    && mkdir -p /opt/fireredtts3/fireredtts3/utils/llm_tn/models \
    && curl -fsSL -o /opt/fireredtts3/fireredtts3/utils/llm_tn/models/lid.176.ftz \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
    && echo "${FIRERED_REF}" > /opt/fireredtts3/COMMIT \
    && cd / && rm -rf /tmp/firered-src \
    && python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'fireredtts3.pth'), 'w').write('/opt/fireredtts3\n')" \
    && python3 -c "\
import os, shutil, torch, transformers, openvino, fasttext; \
from fireredtts3.core import FireRedTTS3Instruct; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
assert transformers.__version__.startswith('5.6.'), transformers.__version__; \
assert not torch.cuda.is_available(), 'fireredov must not ship CUDA torch'; \
print('torch', torch.__version__, 'openvino', openvino.__version__, \
      'transformers', transformers.__version__)"

LABEL org.opencontainers.image.title="audio-firered-ov-deps" \
      audio.compute_runtime="26.22.38646.4"
