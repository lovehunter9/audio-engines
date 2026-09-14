# Breeze: own image, no shared audio-runtime. build/release so git and pip
# stay off the published layers.
ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH
ARG BREEZE_REF=ca632ce6c4d05f7985da4eab29b1a5d445b43f7b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 ca-certificates git; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    python3 -m pip install --no-cache-dir \
        "transformers==4.57.3" "qwen-tts==0.1.1" \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.115" "uvicorn>=0.30" httpx python-multipart websockets; \
    python3 -m pip install --no-cache-dir --force-reinstall \
        torch torchaudio --index-url "${IDX}"; \
    mkdir -p /opt/breeze-tts /tmp/breeze-src; \
    cd /tmp/breeze-src; \
    git init -q .; \
    git remote add origin https://github.com/breezeblue-ai/breeze-tts.git; \
    git fetch -q --depth 1 origin "${BREEZE_REF}"; \
    git checkout -q FETCH_HEAD; \
    cp -r breeze_infer models configs /opt/breeze-tts/; \
    echo "${BREEZE_REF}" > /opt/breeze-tts/COMMIT; \
    rm -rf /tmp/breeze-src; \
    python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'breeze.pth'), 'w').write('/opt/breeze-tts\n')"; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
COPY --from=build /opt/breeze-tts /opt/breeze-tts
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    ldconfig; \
    env -u PYTHONPATH python3 -c "\
import os, shutil, torch, transformers; \
from breeze_infer.runtime import load_runtime; \
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference audio that is not wav dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert transformers.__version__ == '4.57.3', \
    'Breeze TTS 2 targets transformers 4.57.3, got %s' % transformers.__version__; \
assert torch.version.cuda, 'lost CUDA torch after pip install'"

LABEL org.opencontainers.image.title="audio-breeze-deps"
