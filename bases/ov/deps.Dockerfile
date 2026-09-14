# audio-ov deps (hash-tagged rebuilds). Intel GPU only exists on amd64; CI passes a single-arch slice.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# OpenVINO's GPU plugin talks to the device through Level Zero / OpenCL.
# Ubuntu 24.04's intel-opencl-icd 23.43 does not know Arrow Lake-S (PCI 8086:7D67),
# so clinfo reports 0 platforms and the plugin loads CPU only. Pin Intel's
# compute-runtime 26.22 (Arrow Lake production, Ubuntu 24.04, i915 prelim) plus
# the matching IGC. The chart still has to mount /dev/dri.
#
# Skip debug .ddeb files. Checksums from the upstream release notes.
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

# No CUDA torch. CPU torch is only for first-start OpenVINO IR export.
# Do not `pip install --upgrade pip`: Ubuntu's pip has no RECORD file and the upgrade aborts the build.
# ASR: openvino_genai.ASRPipeline + optimum export (qwen3_asr in transformers 5.13).
# Align: OVModelForQwen3ASRForcedAligner from openvino-dev-samples/optimum-intel until upstream merges.
COPY bases/ov/patches/apply_qwen3_asr_batch.py /tmp/apply_qwen3_asr_batch.py

# cmake defaults to Unix Makefiles (needs `make`). ninja is faster once CMAKE_GENERATOR=Ninja.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        make \
        ninja-build \
        g++ \
        python3-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        openvino \
        openvino-genai \
        huggingface_hub \
        librosa \
        soundfile \
        "fastapi>=0.110" \
        "uvicorn>=0.29" \
        python-multipart \
        websockets \
        numpy \
        "safetensors>=0.8.0" \
        "transformers>=5.13,<5.14" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "git+https://github.com/openvino-dev-samples/optimum-intel.git@add-qwen3-asr-hf-and-forced-aligner" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "qwen-asr==0.0.6" \
    && GENAI_VER="$(python3 -c 'import openvino_genai as g; print(getattr(g, "__version__", "") or "")')" \
    && GENAI_TAG="${GENAI_VER%%-*}" \
    && echo "pip openvino-genai ${GENAI_VER} -> tag ${GENAI_TAG}" \
    && mkdir -p /tmp/genai && cd /tmp/genai \
    && git clone --depth 1 --recurse-submodules --shallow-submodules --branch "${GENAI_TAG}" \
        https://github.com/openvinotoolkit/openvino.genai.git src \
    && python3 /tmp/apply_qwen3_asr_batch.py /tmp/genai/src \
    && export CMAKE_GENERATOR=Ninja \
    && unset CFLAGS CXXFLAGS \
    && export CMAKE_ARGS="-DENABLE_SAMPLES=OFF -DENABLE_JS=OFF -DENABLE_GGUF_SUPPORT=OFF" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        setuptools wheel ninja pybind11 \
        "py-build-cmake==0.5.0" \
        "pybind11-stubgen==2.5.5" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore --force-reinstall --no-deps --no-build-isolation \
        /tmp/genai/src \
    && rm -rf /tmp/genai /tmp/apply_qwen3_asr_batch.py \
    && python3 -c "import openvino, openvino_genai, optimum, transformers, qwen_asr, librosa, soundfile, fastapi, uvicorn, huggingface_hub, numpy; \
from optimum.intel import OVModelForQwen3ASRForcedAligner; \
print('openvino', openvino.__version__); \
print('openvino_genai', getattr(openvino_genai, '__version__', 'ok')); \
print('transformers', transformers.__version__); \
print('qwen_asr', 'ok'); \
print('forced_aligner', OVModelForQwen3ASRForcedAligner.__name__)"

LABEL org.opencontainers.image.title="audio-ov-deps" \
      audio.compute_runtime="26.22.38646.4"
