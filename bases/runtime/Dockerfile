# Shared CUDA+torch floor for breeze / firered / pyannote / fasterwhisper.
# nvidia/cuda *-base is ~0.1 GB; pip torch brings its own CUDA libs. *-runtime +
# pip torch would ship CUDA twice.
#
# build installs and strips. release is a fresh base that only receives the
# stripped site-packages and CUDA stubs — gcc, pip, and strip leftovers stay
# off the published image.
ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url "${IDX}"; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh; \
    python3 -c "import torch, torchaudio; \
assert torch.version.cuda, 'runtime image lost CUDA torch: %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', '''${TARGETARCH}''')"

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/

RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python \
    && ldconfig; \
    python3 -c "import torch, torchaudio; \
assert torch.version.cuda, 'release image lost CUDA torch: %s' % (torch.__version__,); \
print('release torch', torch.__version__, 'cuda', torch.version.cuda, 'arch', '''${TARGETARCH}''')"

LABEL org.opencontainers.image.title="audio-runtime" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}"
