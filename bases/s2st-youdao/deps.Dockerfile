ARG CUDA_VERSION=13.0.3
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS native
ARG LLAMA_REF=ad6c66839af3c5646fba8c6c2e2087a1e4e38948
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake ninja-build g++ && rm -rf /var/lib/apt/lists/*
RUN git init -q /src && cd /src && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch -q --depth 1 origin ${LLAMA_REF} && git checkout -q FETCH_HEAD && rm -rf .git
# These are the three already-tested min_tokens compatibility files, not a new
COPY bases/s2st-youdao/server-task.h /src/tools/server/server-task.h
COPY bases/s2st-youdao/server-schema.cpp /src/tools/server/server-schema.cpp
COPY bases/s2st-youdao/server-context.cpp /src/tools/server/server-context.cpp
ARG LOCKED_ROLE_SHA=704fa2b3772d923cf1c160699f43dac2cf655ac59bcbf940bf5a83981c617778
ARG LOCKED_NATIVE_PATCH_SHA=328e84b9df523ab1b83678b21c2f5c554aa35024437aa870e0c9dc809074de2c
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_COMPRESSION_MODE=size '-DCMAKE_CUDA_ARCHITECTURES=86;89;90;120' '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF && cmake --build /build --target llama-ui-assets --parallel 1 && cmake --build /build --target llama-server --parallel 4 && mkdir /native && cp /build/bin/llama-server /native/ && cp -a /build/bin/lib*.so* /native/ && cp /src/LICENSE /native/LICENSE && strip --strip-unneeded /native/llama-server /native/*.so*

# Reuse only the validated MING Python distribution. ASR has a separate
FROM docker.io/beclab/audio-s2st-tencent:deps-7bc3825f4572 AS python

ARG CUDA_VERSION
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-pip python3-venv gcc g++ ffmpeg libsndfile1 ca-certificates libnuma1 && rm -rf /var/lib/apt/lists/*
COPY --from=python /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=native /native /opt/youdao-native
COPY bases/s2st-youdao/role-lock.json /opt/audio-role-lock.json
COPY bases/s2st-youdao/apply-locked-role.py /opt/install-locked-role.py
RUN python3 /opt/install-locked-role.py --lock /opt/audio-role-lock.json --role ming-frozen --python /usr/bin/python3 && python3 -m venv --system-site-packages /opt/venvs/frontend && /opt/venvs/frontend/bin/python /opt/install-locked-role.py --lock /opt/audio-role-lock.json --role frontend --python /opt/venvs/frontend/bin/python
COPY bases/s2st-youdao/apply-asr-role.py /opt/install-asr-role.py
COPY bases/s2st-youdao/cuda-wheels.json /opt/asr-cuda-wheels.json
RUN python3 /opt/install-asr-role.py --lock /opt/audio-role-lock.json
COPY bases/s2st-youdao/apply-layout.py /opt/prepare-image-layout.py
RUN python3 /opt/prepare-image-layout.py --family youdao
RUN --mount=from=native,source=/usr/local/cuda/lib64/stubs,target=/ci-cuda-stubs,ro LD_LIBRARY_PATH=/ci-cuda-stubs:/opt/youdao-native /opt/youdao-native/llama-server --version && python3 -c "import torch; assert torch.__version__=='2.13.0+cu130'; assert torch.backends.cuda.is_built(); import torchaudio; assert torchaudio.__version__=='2.11.0+cu130'" && find /usr/local/lib/python3.12 /opt/venvs -type d -name __pycache__ -prune -exec rm -rf '{}' ';' && rm -rf /root/.cache/pip
LABEL org.opencontainers.image.title="audio-s2st-youdao-deps"
