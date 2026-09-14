# FasterWhisper on the shared slim runtime.
# amd64: pip CT2 CUDA wheel. arm64: compile CT2 on cudnn-devel, copy the closure out
# so the 4.3 GB devel image is not the final floor.
ARG RUNTIME_IMAGE=docker.io/lovehunter9/audio-runtime:slim1
ARG TARGETARCH
ARG CT2_REF=v4.6.0

# ----- arm64 builder (skipped on amd64) -----
FROM docker.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04 AS ct2-builder
ARG CT2_REF
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_ARCH_LIST="8.7;8.9;9.0+PTX" \
    PIP_BREAK_SYSTEM_PACKAGES=1
COPY bases/fasterwhisper/probe_ct2_cuda.py /opt/probe_ct2_cuda.py
COPY bases/fasterwhisper/collect_ct2_runtime.py /opt/collect_ct2_runtime.py
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        git cmake ninja-build build-essential pkg-config \
        libopenblas-dev ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    git clone --recursive --depth 1 --branch "${CT2_REF}" \
        https://github.com/OpenNMT/CTranslate2.git /tmp/CT2; \
    cmake -S /tmp/CT2 -B /tmp/CT2/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_DNNL=OFF -DWITH_RUY=ON \
        -DBUILD_CLI=OFF -DOPENMP_RUNTIME=COMP \
        -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}"; \
    cmake --build /tmp/CT2/build --parallel "$(nproc)"; \
    cmake --install /tmp/CT2/build; \
    ldconfig; \
    python3 -m pip install --no-cache-dir -r /tmp/CT2/python/install_requirements.txt; \
    mkdir -p /opt/ct2-wheels; \
    (cd /tmp/CT2/python && python3 setup.py bdist_wheel -d /opt/ct2-wheels); \
    python3 -m pip install --no-cache-dir /opt/ct2-wheels/ctranslate2-*.whl; \
    python3 /opt/collect_ct2_runtime.py /opt/ct2-runtime; \
    rm -rf /tmp/CT2

# ----- amd64: pip faster-whisper + official CT2 CUDA wheel -----
FROM ${RUNTIME_IMAGE} AS base-amd64
ARG TARGETARCH
COPY bases/fasterwhisper/probe_ct2_cuda.py /opt/probe_ct2_cuda.py
RUN set -eux; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "faster-whisper" huggingface_hub "transformers>=4.56" \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    python3 -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)" \
        || python3 -m pip install --no-cache-dir --force-reinstall \
            torch --index-url https://download.pytorch.org/whl/cu128; \
    python3 /opt/probe_ct2_cuda.py; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    audio-python -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    audio-python -c "from ctranslate2.converters import TransformersConverter; \
import transformers as t; v=tuple(int(x) for x in t.__version__.split('.')[:2]); \
assert v >= (4, 56), t.__version__"; \
    command -v ffmpeg >/dev/null

# ----- arm64: runtime + copied CT2/CUDA closure (no devel leftover) -----
FROM ${RUNTIME_IMAGE} AS base-arm64
ARG TARGETARCH
COPY --from=ct2-builder /opt/ct2-wheels /opt/ct2-wheels
COPY --from=ct2-builder /opt/ct2-runtime /opt/ct2-runtime
COPY bases/fasterwhisper/probe_ct2_cuda.py /opt/probe_ct2_cuda.py
ENV LD_LIBRARY_PATH=/opt/ct2-runtime/lib
RUN set -eux; \
    echo /opt/ct2-runtime/lib > /etc/ld.so.conf.d/ct2.conf; \
    ldconfig; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "faster-whisper" huggingface_hub "transformers>=4.56" \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
        /opt/ct2-wheels/ctranslate2-*.whl; \
    python3 -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)" \
        || python3 -m pip install --no-cache-dir --force-reinstall \
            torch --index-url https://download.pytorch.org/whl/cu130; \
    python3 /opt/probe_ct2_cuda.py; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    audio-python -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    audio-python -c "from ctranslate2.converters import TransformersConverter; \
import transformers as t; v=tuple(int(x) for x in t.__version__.split('.')[:2]); \
assert v >= (4, 56), t.__version__"; \
    command -v ffmpeg >/dev/null

FROM base-${TARGETARCH}
LABEL org.opencontainers.image.title="audio-fasterwhisper-deps"
