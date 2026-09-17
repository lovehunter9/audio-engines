# Qwen3-ASR + ForcedAligner: cuda-base + CUDA torch + qwen-asr (transformers).
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
        numpy \
        "transformers==4.57.6" "accelerate==1.12.0" \
        "nagisa==0.2.11" "soynlp==0.0.493" \
        librosa soundfile \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29" websockets; \
    python3 -m pip install --no-cache-dir --no-deps qwen-asr; \
    python3 -m pip uninstall -y gradio gradio-client flask sox || true; \
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
import importlib.util, os, shutil, torch; \
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda, 'lost CUDA torch after pip install'; \
assert Qwen3ASRModel.from_pretrained.__name__; \
assert Qwen3ForcedAligner.from_pretrained.__name__; \
assert importlib.util.find_spec('gradio') is None; \
assert importlib.util.find_spec('flask') is None; \
assert importlib.util.find_spec('sox') is None"

LABEL org.opencontainers.image.title="audio-qwen-deps"
