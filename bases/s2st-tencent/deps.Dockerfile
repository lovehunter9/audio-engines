ARG CUDA_VERSION=13.0.3
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS native
ARG TENCENT_REF=d38e66aac7bb18101f417338b795e4546a5a1a59
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake ninja-build python3 && rm -rf /var/lib/apt/lists/*
RUN git init -q /src && cd /src && git remote add origin https://github.com/lihy715/llama.cpp.Hyv3Speech.git && git fetch -q --depth 1 origin ${TENCENT_REF} && git checkout -q FETCH_HEAD && rm -rf .git
COPY bases/s2st-tencent/native-entry.cpp /src/tools/mtmd/mtmd-cli.cpp
COPY bases/s2st-tencent/tencent_revised_native_body.inc /src/tools/mtmd/tencent_revised_native_body.inc
COPY bases/s2st-tencent/native-audio.cpp /src/tools/mtmd/mtmd-audio.cpp
# EXPORT_FINGERPRINTS: native body and role lock hashes are inserted by the
ARG LOCKED_ROLE_SHA=704fa2b3772d923cf1c160699f43dac2cf655ac59bcbf940bf5a83981c617778
ARG LOCKED_NATIVE_BODY_SHA=d8c3e4bf13f684e480b317e9bd3eda2cd01d7bcda2377fe50e43381c5cf7c17c
RUN python3 -c "from pathlib import Path; p=Path('/src/tools/mtmd/CMakeLists.txt'); p.write_text(p.read_text()+'\ntarget_include_directories(llama-mtmd-cli PRIVATE /src/vendor/nlohmann)\n')" && ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_COMPRESSION_MODE=size '-DCMAKE_CUDA_ARCHITECTURES=86;89;90;120' '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_HTML=OFF && cmake --build /build --target llama-mtmd-cli --parallel 4 && mkdir /native && cp /build/bin/llama-mtmd-cli /native/llama-mtmd-native-rpc && cp -a /build/bin/lib*.so* /native/ && cp /src/LICENSE /native/LICENSE && strip --strip-unneeded /native/llama-mtmd-native-rpc /native/*.so*

# This existing multi-architecture image supplies the validated frozen MING
FROM docker.io/beclab/audio-s2st-tencent:deps-7bc3825f4572 AS python

ARG CUDA_VERSION
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-pip python3-venv gcc g++ ffmpeg libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=python /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=native /native /opt/tencent-native
COPY bases/s2st-tencent/role-lock.json /opt/audio-role-lock.json
COPY bases/s2st-tencent/apply-locked-role.py /opt/install-locked-role.py
RUN python3 /opt/install-locked-role.py --lock /opt/audio-role-lock.json --role ming-frozen --python /usr/bin/python3 && python3 -m venv --system-site-packages /opt/venvs/frontend && /opt/venvs/frontend/bin/python /opt/install-locked-role.py --lock /opt/audio-role-lock.json --role frontend --python /opt/venvs/frontend/bin/python
COPY bases/s2st-tencent/apply-layout.py /opt/prepare-image-layout.py
RUN python3 /opt/prepare-image-layout.py --family tencent
RUN --mount=from=native,source=/usr/local/cuda/lib64/stubs,target=/ci-cuda-stubs,ro LD_LIBRARY_PATH=/ci-cuda-stubs:/opt/tencent-native /opt/tencent-native/llama-mtmd-native-rpc --version && python3 -c "import torch; assert torch.__version__=='2.13.0+cu130'; assert torch.backends.cuda.is_built(); import torchaudio; assert torchaudio.__version__=='2.11.0+cu130'" && find /usr/local/lib/python3.12 /opt/venvs -type d -name __pycache__ -prune -exec rm -rf '{}' ';' && rm -rf /root/.cache/pip
LABEL org.opencontainers.image.title="audio-s2st-tencent-deps"
