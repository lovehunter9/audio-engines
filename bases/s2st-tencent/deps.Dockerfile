ARG CUDA_VERSION=13.0.3
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS native
ARG TARGETARCH
ARG TENCENT_REF=d38e66aac7bb18101f417338b795e4546a5a1a59
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake ninja-build python3 && rm -rf /var/lib/apt/lists/*
RUN git init -q /src && cd /src && git remote add origin https://github.com/lihy715/llama.cpp.Hyv3Speech.git && git fetch -q --depth 1 origin ${TENCENT_REF} && git checkout -q FETCH_HEAD && rm -rf .git
COPY bases/s2st-tencent/apply-native.py bases/s2st-tencent/native-io.patch /patch/
RUN python3 /patch/apply-native.py /src && cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF '-DCMAKE_CUDA_ARCHITECTURES=86;89;90;120' -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_HTML=OFF && cmake --build /build --target llama-mtmd-cli --parallel 4 && mkdir -p /native/bin /native/lib && cp /src/LICENSE /native/LICENSE && cp /build/bin/llama-mtmd-cli /native/bin/ && find /build/bin -maxdepth 1 -type f -name '*.so*' -exec cp -a '{}' /native/lib/ ';' && find /build/bin -maxdepth 1 -type l -name '*.so*' -exec cp -a '{}' /native/lib/ ';' && strip --strip-unneeded /native/bin/* /native/lib/*.so*

ARG CUDA_VERSION
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04 AS python
ARG TARGETARCH
ARG MING_REF=028c1eace193e7acdb256453da89b6e9207e052b
ARG SIMULINFER_REF=7efcb18c521a7025b37d4bc2230a1df8ade83950
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates && rm -rf /var/lib/apt/lists/*
# These are the imported versions in the validated host environment. torchaudio is intentionally installed without resolving Torch backwards.
RUN python3 -m pip install 'torch==2.13.0+cu130' --index-url https://download.pytorch.org/whl/cu130 && python3 -m pip install --no-deps 'torchaudio==2.11.0+cu130' --index-url https://download.pytorch.org/whl/cu130
COPY bases/s2st-tencent/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-deps fireredvad==0.0.2 && python3 -m pip install -r /tmp/requirements.txt && python3 -c "import torch; assert torch.__version__ == '2.13.0+cu130'"
RUN git init -q /opt/ming-tts && cd /opt/ming-tts && git remote add origin https://github.com/inclusionAI/Ming-3.0-tts.git && git fetch -q --depth 1 origin ${MING_REF} && git checkout -q FETCH_HEAD && rm -rf .git data figures cookbooks examples docker ming_tiny && printf '%s\n' ${MING_REF} > COMMIT
RUN git init -q /tmp/hy && cd /tmp/hy && git remote add origin https://github.com/Tencent-Hunyuan/Hy-SimulMT.git && git fetch -q --depth 1 origin ${SIMULINFER_REF} && git checkout -q FETCH_HEAD && mkdir -p /opt/upstream && cp -a SimulInfer/simulinfer /opt/upstream/ && find . -maxdepth 1 -iname 'LICENSE*' -exec cp '{}' /opt/upstream/ ';' && printf '%s\n' ${SIMULINFER_REF} > /opt/upstream/COMMIT && rm -rf /tmp/hy
# Preserve Triton, CUDA headers and the compiler path needed by the accepted compiled MING flow.
RUN python3 -c "import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()['purelib']); (p/'s2st-upstream.pth').write_text('/opt/ming-tts\n/opt/upstream\n')" && find /usr/local/lib/python3.12 -type d -name __pycache__ -prune -exec rm -rf '{}' ';' && rm -rf /root/.cache/pip /tmp/pip-*

ARG CUDA_VERSION
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04 AS release
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUDA_HOME=/opt/cuda-runtime CUDA_PATH=/opt/cuda-runtime
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev gcc g++ ffmpeg libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/audio-python
COPY --from=python /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=python /opt/ming-tts /opt/ming-tts
COPY --from=python /opt/upstream /opt/upstream
COPY --from=native /native /opt/llama
RUN python3 -c "import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'; assert (p/'include/cuda.h').exists(); pathlib.Path('/opt/cuda-runtime').symlink_to(p)"
RUN python3 -c "import glob,pathlib; paths=glob.glob('/usr/local/lib/python3.12/*-packages/nvidia/**/lib',recursive=True); pathlib.Path('/etc/ld.so.conf.d/s2st.conf').write_text('/opt/llama/lib\n'+'\n'.join(paths)+'\n')" && ldconfig && python3 -c "import torch,torchaudio,onnxruntime,ming_tts,simulinfer,importlib.util; assert torch.__version__=='2.13.0+cu130'; assert torchaudio.__version__=='2.11.0+cu130'; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers(); assert importlib.util.find_spec('vllm_omni') is None; print(torch.__version__,torchaudio.__version__,onnxruntime.__version__)" && /opt/llama/bin/llama-mtmd-cli --version
LABEL org.opencontainers.image.title="audio-s2st-tencent-deps"
