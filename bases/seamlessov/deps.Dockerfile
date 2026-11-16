FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ARG NEO_VER=26.22.38646.4
ARG IGC_TAG=v2.36.3
ARG IGC_DEB=2.36.3+21719
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        libsndfile1 \
        libtbb12 \
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

FROM runtime AS build
ARG FAIRSEQ_REVISION=76015a1280d72ef1f94535637579907da929297c
ARG SEAMLESS_REVISION=9a081e935c29c6b1e0e05bcbf3fcb623ad306524
RUN apt-get update && apt-get install -y --no-install-recommends git python3-dev build-essential cmake ninja-build pkg-config libsndfile1-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir "setuptools==80.10.2" wheel "packaging==23.2" "numpy==1.26.4" \
    && git init /opt/fairseq2 \
    && git -C /opt/fairseq2 remote add origin https://github.com/facebookresearch/fairseq2.git \
    && git -C /opt/fairseq2 fetch --depth 1 origin "$FAIRSEQ_REVISION" \
    && git -C /opt/fairseq2 checkout --detach FETCH_HEAD \
    && git -C /opt/fairseq2 submodule update --init --recursive --depth 1 \
    && sed -i 's/THPVariable_Wrap(at::TensorBase var)/THPVariable_Wrap(const at::TensorBase \&var)/' /opt/fairseq2/fairseq2n/python/src/fairseq2n/bindings/type_casters/torch.cc \
    && cmake -GNinja -S /opt/fairseq2/fairseq2n -B /opt/fairseq2/fairseq2n/build \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
         -DFAIRSEQ2N_INSTALL_STANDALONE=ON -DFAIRSEQ2N_USE_CUDA=OFF \
         -DFAIRSEQ2N_THREAD_LIB=tbb \
    && cmake --build /opt/fairseq2/fairseq2n/build --parallel 2 \
    && cd /opt/fairseq2/fairseq2n/python \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/fairseq2 \
    && git init /opt/seamless \
    && git -C /opt/seamless remote add origin https://github.com/facebookresearch/seamless_communication.git \
    && git -C /opt/seamless fetch --depth 1 origin "$SEAMLESS_REVISION" \
    && git -C /opt/seamless checkout --detach FETCH_HEAD \
    && printf '%s\n' "$SEAMLESS_REVISION" > /opt/seamless/COMMIT \
    && python3 -m pip install --no-cache-dir \
         "jiwer==3.1.0" "overrides==7.7.0" "sacrebleu==2.5.1" "torcheval==0.0.7" \
         "datasets==2.18.0" "pyarrow==17.0.0" "simuleval==1.1.4" \
         "librosa==0.11.0" "soundfile==0.13.1" "scipy==1.15.3" fire \
         "huggingface_hub==0.36.2" "safetensors==0.7.0" "omegaconf==2.3.0" \
         "tiktoken==0.12.0" "more-itertools==10.8.0" \
         fastapi uvicorn httpx python-multipart websockets \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
         "openai-whisper==20240930" "sonar-space==0.2.1" /opt/seamless \
    && python3 -m pip install --no-cache-dir "openvino==2026.4.1" \
    && git -C /opt/seamless diff --exit-code HEAD \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ -prune -exec rm -rf '{}' +

FROM runtime AS release
COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=build /opt/seamless/COMMIT /opt/seamless/COMMIT
RUN python3 -c "import torch,openvino,fairseq2; from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent; assert torch.version.cuda is None; print(openvino.__version__,fairseq2.__version__,SeamlessStreamingS2STAgent.__name__)"
LABEL org.opencontainers.image.source="https://github.com/facebookresearch/seamless_communication" \
      org.opencontainers.image.revision="9a081e935c29c6b1e0e05bcbf3fcb623ad306524"
