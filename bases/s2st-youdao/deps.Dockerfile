ARG CUDA_VERSION=13.0.3
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS native
ARG TARGETARCH
ARG LLAMA_REF=ad6c66839af3c5646fba8c6c2e2087a1e4e38948
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake ninja-build g++ && rm -rf /var/lib/apt/lists/*
RUN git init -q /src && cd /src && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch -q --depth 1 origin ${LLAMA_REF} && git checkout -q FETCH_HEAD && rm -rf .git
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_COMPRESSION_MODE=size '-DCMAKE_CUDA_ARCHITECTURES=86;89;90;120' '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs' -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF && cmake --build /build --target llama-ui-assets --parallel 1 && cmake --build /build --target llama-server --parallel 4
COPY bases/s2st-youdao/native-entry.cpp /src/native-entry.cpp
RUN g++ -std=c++17 -O3 -DNDEBUG -shared -fPIC -pthread -I/src/common -I/src/include -I/src/ggml/include -I/src/vendor /src/native-entry.cpp -L/build/bin -Wl,-rpath,/opt/llama/lib -llama-server-impl -llama-common -lllama -o /build/bin/libaudio-t3po-entry.so && mkdir -p /native/bin /native/lib && cp /src/LICENSE /native/LICENSE && cp /build/bin/llama-server /native/bin/ && find /build/bin -maxdepth 1 -type f -name '*.so*' -exec cp -a '{}' /native/lib/ ';' && find /build/bin -maxdepth 1 -type l -name '*.so*' -exec cp -a '{}' /native/lib/ ';' && strip --strip-unneeded /native/bin/* /native/lib/*.so*

# Only the validated Python/MING files are copied into the final runtime.
# Tencent's native libraries and its image layers are not retained there.
FROM docker.io/lovehunter9/audio-s2st-tencent:s2st-tencent-dev-20261120-9@sha256:0d7d7fec91e693995fe889a7b7cba0bcf738f5f944feb921fadbcaf85733d119 AS ming

FROM docker.io/python:3.12-slim-bookworm AS asr
ARG R2T2_REF=62757213befa282e7fbfe1bfde3c29577781527c
ARG T3PO_REF=0ac9d0c806f8f139ebaa83d394ffbd5057ff3ae3
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
COPY bases/s2st-youdao/requirements-asr.txt bases/s2st-youdao/requirements-web.txt /requirements/
RUN python -m pip install --no-cache-dir --no-deps --target /asr-source -r /requirements/requirements-asr.txt && python -m pip install --no-cache-dir --no-deps --target /web -r /requirements/requirements-web.txt
COPY bases/s2st-youdao/apply-asr-namespace.py /apply-asr-namespace.py
RUN python /apply-asr-namespace.py --source /asr-source --output /asr-private
RUN git init -q /t3po && cd /t3po && git remote add origin https://github.com/netease-youdao/Confucius4-T3PO.git && git fetch -q --depth 1 origin ${T3PO_REF} && GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD && mkdir -p /upstream/t3po && cp -a inference LICENSE /upstream/t3po/ && printf '%s\n' ${T3PO_REF} > /upstream/t3po/COMMIT && rm -rf /t3po
RUN git init -q /r2t2 && cd /r2t2 && git remote add origin https://github.com/netease-youdao/Confucius4-R2T2.git && git fetch -q --depth 1 origin ${R2T2_REF} && GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD && mkdir -p /upstream/r2t2 && cp -a r2t2 ws_server.py README.md pyproject.toml /upstream/r2t2/ && printf '%s\n' ${R2T2_REF} > /upstream/r2t2/COMMIT && rm -rf /r2t2
RUN find /asr-private /web /upstream -type d -name __pycache__ -prune -exec rm -rf '{}' ';'

ARG CUDA_VERSION
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUDA_HOME=/opt/cuda-runtime CUDA_PATH=/opt/cuda-runtime
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev gcc g++ ffmpeg libsndfile1 ca-certificates && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/audio-python
COPY --from=ming /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=ming /opt/ming-tts /opt/ming-tts
COPY --from=ming /opt/upstream /opt/upstream
COPY --from=native /native /opt/llama
COPY --from=asr /asr-private /opt/asr-private
COPY --from=asr /web /opt/youdao-web
COPY --from=asr /upstream /opt/youdao
RUN python3 -c "import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()['purelib']); c=p/'nvidia/cu13'; assert (c/'include/cuda.h').exists(); pathlib.Path('/opt/cuda-runtime').symlink_to(c); (p/'youdao-web.pth').write_text('/opt/youdao-web\n/opt/youdao/t3po\n')"
RUN python3 -c "import glob,pathlib; paths=glob.glob('/usr/local/lib/python3.12/*-packages/nvidia/**/lib',recursive=True); pathlib.Path('/etc/ld.so.conf.d/s2st.conf').write_text('/opt/llama/lib\n'+'\n'.join(paths)+'\n')" && ldconfig
COPY bases/s2st-youdao/probe_runtime.py /opt/s2st-youdao-probe.py
RUN --mount=from=native,source=/usr/local/cuda/lib64/stubs,target=/ci-cuda-stubs,ro LD_LIBRARY_PATH=/ci-cuda-stubs python3 /opt/s2st-youdao-probe.py && LD_LIBRARY_PATH=/ci-cuda-stubs /opt/llama/bin/llama-server --version
LABEL org.opencontainers.image.title="audio-s2st-youdao-deps"
