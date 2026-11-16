ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04 AS amd64
FROM docker.io/nvidia/cuda:12.9.1-base-ubuntu24.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH
ARG SIMUL_SHA=c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential binutils ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    if [ "$TARGETARCH" = arm64 ]; then idx=cu129; else idx=cu128; fi; \
    python3 -m pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url "https://download.pytorch.org/whl/$idx"; \
    printf '%s\n' 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' \
        'numpy==1.26.4' > /tmp/simuls-constraints.txt; \
    git init /opt/simuls2st; \
    git -C /opt/simuls2st remote add origin https://github.com/hasaki321/SimulS2ST-Omni.git; \
    git -C /opt/simuls2st fetch --depth 1 origin "$SIMUL_SHA"; \
    git -C /opt/simuls2st checkout --detach FETCH_HEAD; \
    test "$(git -C /opt/simuls2st rev-parse HEAD)" = "$SIMUL_SHA"; \
    git -C /opt/simuls2st submodule update --init --depth 1 external/SimulEval; \
    python3 -m pip install --no-cache-dir -c /tmp/simuls-constraints.txt \
        /opt/simuls2st/external/SimulEval \
        "transformers==4.52.3" "peft==0.18.1" "qwen-omni-utils==0.0.8" \
        "accelerate==1.12.0" "numpy==1.26.4" "safetensors==0.7.0" \
        "soundfile==0.13.1" "sentencepiece==0.2.1" "einops==0.8.1" \
        "scipy==1.15.3" "librosa==0.11.0" \
        "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0" \
        "easydict==1.13" "pyworld==0.3.5" "json5>=0.9.0" \
        "ruamel.yaml==0.18.16" "six==1.17.0" \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets; \
    printf '%s\n' "$SIMUL_SHA" > /opt/simuls2st/COMMIT; \
    rm -rf /opt/simuls2st/.git; \
    rm -f /opt/simuls2st/external/SimulEval/.git; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    SIMULS2ST_ROOT=/opt/simuls2st
COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
COPY --from=build /opt/simuls2st /opt/simuls2st
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends python3 ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    python3 -c "import pathlib,site; p=pathlib.Path(site.getsitepackages()[0]); p.mkdir(parents=True,exist_ok=True); p.joinpath('simuls2st.pth').write_text('/opt/simuls2st\\n')"; \
    ldconfig
COPY bases/simuls2st/probe_runtime.py /opt/probe_simuls2st_runtime.py
RUN env -u PYTHONPATH python3 /opt/probe_simuls2st_runtime.py --arch "$TARGETARCH"
LABEL org.opencontainers.image.title="audio-simuls2st-deps" \
      org.opencontainers.image.source="https://github.com/hasaki321/SimulS2ST-Omni" \
      org.opencontainers.image.revision="c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c"
