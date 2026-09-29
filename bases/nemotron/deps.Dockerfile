# Nemotron 3 Diarization: CUDA runtime + nemo-toolkit[asr], not beclab/nvidia-nemo.
# That image is the NGC training container (~55 GB unpacked). This base installs only
# what SortformerEncLabelModel.restore_from needs, then strips unused CUDA libs.
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
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url "${IDX}"; \
    python3 -m pip freeze | grep -E '^(torch|torchaudio)==' > /tmp/torch.pin; \
    python3 -m pip install --no-cache-dir -c /tmp/torch.pin \
        "nemo-toolkit[asr]==3.0.0" \
        numpy \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29" websockets; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh /tmp/torch.pin

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    ldconfig; \
    python3 -c "\
import os, shutil, torch; \
from nemo.collections.asr.models import SortformerEncLabelModel; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda, 'lost CUDA torch after pip install'; \
assert SortformerEncLabelModel.restore_from.__name__"

LABEL org.opencontainers.image.title="audio-nemotron-deps"
