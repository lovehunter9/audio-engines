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
RUN python3 -m pip install --no-cache-dir "cmake>=3.21,<4" ninja "setuptools<81" wheel packaging "tbb==2022.1.0" "tbb-devel==2022.1.0" "numpy<2"

FROM runtime AS native
ARG FAIRSEQ2_REVISION=c664a42ac30c5ba3ab23382327903cf690ea1ffb
RUN git clone https://github.com/facebookresearch/fairseq2.git /opt/fairseq2 && git -C /opt/fairseq2 checkout "$FAIRSEQ2_REVISION" && git -C /opt/fairseq2 submodule update --init --recursive && sed -i 's/THPVariable_Wrap(at::TensorBase var)/THPVariable_Wrap(const at::TensorBase\& var)/' /opt/fairseq2/fairseq2n/python/src/fairseq2n/bindings/type_casters/torch.cc
RUN cd /opt/fairseq2/fairseq2n && cmake -GNinja -B build -DFAIRSEQ2N_USE_CUDA=OFF -DFAIRSEQ2N_PYTHON_DEVEL=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 2
RUN cd /opt/fairseq2/fairseq2n/python && python3 -m pip wheel --no-deps --no-build-isolation -w /wheels . && cd /opt/fairseq2 && python3 -m pip wheel --no-deps --no-build-isolation -w /wheels .

FROM runtime
COPY --from=native /wheels /wheels
RUN python3 -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
ARG SEAMLESS_REVISION=9a081e935c29c6b1e0e05bcbf3fcb623ad306524
RUN git clone https://github.com/facebookresearch/seamless_communication.git /opt/seamless && git -C /opt/seamless checkout "$SEAMLESS_REVISION" && python3 -m pip install --no-cache-dir --no-deps /opt/seamless
RUN printf '%s\n' 'torch==2.8.0+xpu' 'torchaudio==2.8.0+xpu' 'fairseq2==0.2.1' 'fairseq2n==0.2.1' 'numpy==1.26.4' 'scipy==1.15.3' 'tbb==2022.1.0' 'packaging==23.2' > /opt/seamless-constraints.txt
RUN python3 -m pip install --no-cache-dir -c /opt/seamless-constraints.txt "datasets==2.18.0" "pyarrow==17.0.0" "simuleval==1.1.4" fire "librosa==0.11.0" openai-whisper "sonar-space==0.2.1" scipy soundfile huggingface_hub safetensors einops fastapi uvicorn httpx python-multipart websockets
RUN python3 -c "import torch, fairseq2, fairseq2n; from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent; assert torch.__version__.startswith('2.8.'); assert torch.version.cuda is None; assert fairseq2.__version__ == '0.2.1'; print(SeamlessStreamingS2STAgent.__name__)"
ENV SEAMLESS_ROOT=/opt/seamless
