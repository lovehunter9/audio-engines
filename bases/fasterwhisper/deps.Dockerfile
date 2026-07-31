# audio-fasterwhisper deps (hash-tagged rebuilds). amd64: harveyff+CT2 zero-diff. arm64: CUDA CT2 from source + cu130 torch.
ARG TARGETARCH

# ----- amd64: byte-stable recipe (same steps as the pre-split single-FROM file) -----
FROM docker.io/beclab/harveyff-whisper-webui:v1.0.7 AS base-amd64
RUN set -eux; \
    PY=; \
    for cand in "$(command -v python3 || true)" /Whisper-WebUI/venv/bin/python3 /usr/bin/python3; do \
        [ -n "$cand" ] && [ -x "$cand" ] || continue; \
        if "$cand" -c "import faster_whisper" >/dev/null 2>&1; then PY="$cand"; break; fi; \
    done; \
    : "${PY:?no interpreter in this image can import faster_whisper}"; \
    "$PY" -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    "$PY" -m pip install --no-cache-dir --root-user-action=ignore "transformers>=4.56"; \
    printf '#!/bin/sh\nexec %s "$@"\n' "$PY" > /usr/local/bin/audio-python; \
    chmod 755 /usr/local/bin/audio-python; \
    audio-python -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    audio-python -c "from ctranslate2.converters import TransformersConverter; \
import transformers as t; v=tuple(int(x) for x in t.__version__.split('.')[:2]); \
assert v >= (4, 56), t.__version__"; \
    command -v ffmpeg >/dev/null

# ----- arm64: CUDA CT2 from source + CUDA torch (cu130) -----
FROM docker.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04 AS base-arm64
ARG CT2_REF=v4.6.0
# FindCUDA arch list: 8.7;8.9;9.0+PTX (semicolon-separated; covers GB10 via PTX JIT).
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_ARCH_LIST="8.7;8.9;9.0+PTX"
# Probe lives in its own file so try/except is not smashed by Dockerfile line continuations.
COPY bases/fasterwhisper/probe_ct2_cuda.py /opt/probe_ct2_cuda.py
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        git cmake ninja-build build-essential pkg-config \
        libopenblas-dev ffmpeg libsndfile1 ca-certificates; \
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
    rm -rf /tmp/CT2; \
    # CUDA torch for device=auto; faster-whisper may pull a CPU CT2 wheel — put ours back.
    python3 -m pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu130; \
    python3 -m pip install --no-cache-dir \
        "faster-whisper" huggingface_hub "transformers>=4.56" \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps /opt/ct2-wheels/ctranslate2-*.whl; \
    python3 /opt/probe_ct2_cuda.py; \
    printf '#!/bin/sh\nexec python3 "$@"\n' > /usr/local/bin/audio-python; \
    chmod 755 /usr/local/bin/audio-python; \
    audio-python -c "import faster_whisper, huggingface_hub, torch, fastapi, uvicorn, multipart"; \
    audio-python -c "from ctranslate2.converters import TransformersConverter; \
import transformers as t; v=tuple(int(x) for x in t.__version__.split('.')[:2]); \
assert v >= (4, 56), t.__version__"; \
    command -v ffmpeg >/dev/null

FROM base-${TARGETARCH}
LABEL org.opencontainers.image.title="audio-fasterwhisper-deps"
