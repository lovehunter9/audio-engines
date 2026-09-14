# Shared CUDA+torch floor for the non-vLLM engines (breeze / firered / pyannote / fasterwhisper).
# nvidia/cuda *-base is ~0.1 GB; pip torch brings its own CUDA libs. *-runtime + pip torch
# would ship CUDA twice (~2 GB extra for nothing).
ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64
FROM ${TARGETARCH}

ARG TARGETARCH
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# cu128 on amd64, cu130 on arm64 (GB10). Same package names so family images do not re-pin torch.
RUN set -eux; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        torch torchaudio --index-url "${IDX}"; \
    python3 -c "import torch, torchaudio; \
assert torch.version.cuda, 'runtime image lost CUDA torch: %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', '''${TARGETARCH}''')"

LABEL org.opencontainers.image.title="audio-runtime" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}"
