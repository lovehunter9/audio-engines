ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04 AS amd64-build
FROM docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 AS arm64-build
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04 AS amd64-release
FROM docker.io/nvidia/cuda:12.9.1-base-ubuntu24.04 AS arm64-release

FROM ${TARGETARCH}-build AS build
ARG TARGETARCH
ARG FAIRSEQ_REVISION=76015a1280d72ef1f94535637579907da929297c
ARG SEAMLESS_REVISION=9a081e935c29c6b1e0e05bcbf3fcb623ad306524
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential cmake ninja-build pkg-config \
        libsndfile1-dev libtbb-dev ffmpeg ca-certificates binutils \
    && rm -rf /var/lib/apt/lists/* \
    && if [ "$TARGETARCH" = arm64 ]; then idx=cu129; else idx=cu128; fi \
    && python3 -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
         --index-url "https://download.pytorch.org/whl/$idx" \
    && python3 -m pip install --no-cache-dir "setuptools==80.10.2" wheel "packaging==23.2" "numpy==1.26.4" \
    && git init /opt/fairseq2 \
    && git -C /opt/fairseq2 remote add origin https://github.com/facebookresearch/fairseq2.git \
    && git -C /opt/fairseq2 fetch --depth 1 origin "$FAIRSEQ_REVISION" \
    && git -C /opt/fairseq2 checkout --detach FETCH_HEAD \
    && git -C /opt/fairseq2 submodule update --init --recursive --depth 1 \
    && if [ "$TARGETARCH" = arm64 ]; then threadlib=''; else threadlib=tbb; fi \
    && sed -i 's/THPVariable_Wrap(at::TensorBase var)/THPVariable_Wrap(const at::TensorBase \&var)/' /opt/fairseq2/fairseq2n/python/src/fairseq2n/bindings/type_casters/torch.cc \
    && cmake -GNinja -S /opt/fairseq2/fairseq2n -B /opt/fairseq2/fairseq2n/build \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
         -DFAIRSEQ2N_INSTALL_STANDALONE=ON -DFAIRSEQ2N_USE_CUDA=ON \
         -DFAIRSEQ2N_THREAD_LIB="$threadlib" -DCMAKE_CUDA_ARCHITECTURES='80-virtual;100-real;120-real' \
    && cmake --build /opt/fairseq2/fairseq2n/build --parallel 2 \
    && cd /opt/fairseq2/fairseq2n/python \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/fairseq2 \
    && git init /opt/seamless \
    && git -C /opt/seamless remote add origin https://github.com/facebookresearch/seamless_communication.git \
    && git -C /opt/seamless fetch --depth 1 origin "$SEAMLESS_REVISION" \
    && git -C /opt/seamless checkout --detach FETCH_HEAD \
    && printf '%s\n' "$SEAMLESS_REVISION" > /opt/seamless/COMMIT \
    && printf '%s\n' 'numpy==1.26.4' 'packaging==23.2' 'pandas==2.2.3' 'numba==0.61.2' 'llvmlite==0.44.0' > /tmp/seamless-constraints.txt \
    && export PIP_CONSTRAINT=/tmp/seamless-constraints.txt \
    && python3 -m pip install --no-cache-dir \
         "jiwer==3.1.0" "overrides==7.7.0" "sacrebleu==2.5.1" "torcheval==0.0.7" \
         "datasets==2.18.0" "pyarrow==17.0.0" "simuleval==1.1.4" \
         "librosa==0.11.0" "soundfile==0.13.1" "scipy==1.15.3" fire \
         "huggingface_hub==0.36.2" "safetensors==0.7.0" "omegaconf==2.3.0" \
         "tiktoken==0.12.0" "more-itertools==10.8.0" \
         fastapi uvicorn httpx python-multipart websockets \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
         "openai-whisper==20240930" "sonar-space==0.2.1" /opt/seamless
COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN export LD_LIBRARY_PATH="$(python3 -c 'import site,pathlib; p=pathlib.Path(site.getsitepackages()[0]); print(":".join(str(x) for x in (p/"nvidia").glob("*/lib")))')" \
    && sh /tmp/strip_unused_cuda.sh \
    && git -C /opt/seamless diff --exit-code HEAD \
    && rm -rf /opt/fairseq2 /opt/seamless/.git \
    && find /usr/local/lib/python3.12 -type d \( -name __pycache__ -o -name tests \) -prune -exec rm -rf '{}' +

FROM ${TARGETARCH}-release AS release
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 ffmpeg libsndfile1 libtbb12 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python
COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=build /opt/cuda-stubs /usr/local/lib
COPY --from=build /opt/seamless/COMMIT /opt/seamless/COMMIT
RUN ldconfig \
    && python3 -c "import torch, torchaudio, fairseq2, fairseq2n; from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent; assert torch.version.cuda; print(torch.__version__, fairseq2.__version__, SeamlessStreamingS2STAgent.__name__)"
LABEL org.opencontainers.image.source="https://github.com/facebookresearch/seamless_communication" \
      org.opencontainers.image.revision="9a081e935c29c6b1e0e05bcbf3fcb623ad306524"
