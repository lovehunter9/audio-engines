# FireRed: own image, no shared audio-runtime.
ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH
ARG FIRERED_REF=1d32ba780da6af37a71bdfd9c68c12003e908a46

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 ca-certificates git curl; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url "${IDX}"; \
    python3 -m pip install --no-cache-dir \
        "transformers==5.6.2" einops regex python-dotenv wetext fasttext-wheel \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets; \
    mkdir -p /opt/fireredtts3 /tmp/firered-src; \
    cd /tmp/firered-src; \
    git init -q .; \
    git remote add origin https://github.com/FireRedTeam/FireRedTTS3.git; \
    git fetch -q --depth 1 origin "${FIRERED_REF}"; \
    git checkout -q FETCH_HEAD; \
    cp -r fireredtts3 /opt/fireredtts3/; \
    mkdir -p /opt/fireredtts3/fireredtts3/utils/llm_tn/models; \
    curl -fsSL -o /opt/fireredtts3/fireredtts3/utils/llm_tn/models/lid.176.ftz \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz; \
    echo "${FIRERED_REF}" > /opt/fireredtts3/COMMIT; \
    cd /; \
    rm -rf /tmp/firered-src; \
    python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'fireredtts3.pth'), 'w').write('/opt/fireredtts3\n')"; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
COPY --from=build /opt/fireredtts3 /opt/fireredtts3
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    ldconfig; \
    env -u PYTHONPATH python3 -c "\
import os, shutil, torch, transformers, fasttext; \
from fireredtts3.core import FireRedTTS3Instruct; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference audio that is not wav dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert transformers.__version__.startswith('5.6.'), \
    'FireRedTTS3-Instruct config targets transformers 5.6.x, got %s' % transformers.__version__; \
assert torch.version.cuda, 'lost CUDA torch after pip install'"

LABEL org.opencontainers.image.title="audio-firered-deps"
