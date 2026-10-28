# Ming-3.0-tts-0.5B. CUDA torch, then strip unused nvidia-*. No flash-attn and no pynini.
ARG TARGETARCH
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH
ARG MING_REF=028c1eace193e7acdb256453da89b6e9207e052b

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
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url "${IDX}"; \
    python3 -m pip install --no-cache-dir \
        "transformers==4.46.3" "accelerate==1.3.0" \
        "x_transformers==2.9.2" "diffusers==0.33.0" \
        onnxruntime "soundfile==0.12.1" numpy einops inflect \
        "huggingface-hub>=0.34" tqdm loguru packaging \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets; \
    mkdir -p /opt/ming-tts /tmp/ming-src; \
    cd /tmp/ming-src; \
    git init -q .; \
    git remote add origin https://github.com/inclusionAI/Ming-3.0-tts.git; \
    git fetch -q --depth 1 origin "${MING_REF}"; \
    git checkout -q FETCH_HEAD; \
    rm -rf .git data figures cookbooks examples docker ming_tiny; \
    cp -a . /opt/ming-tts/; \
    echo "${MING_REF}" > /opt/ming-tts/COMMIT; \
    cd /; \
    rm -rf /tmp/ming-src; \
    python3 -c "import sysconfig, os; \
p = sysconfig.get_paths()['purelib']; \
open(os.path.join(p, 'ming.pth'), 'w').write('/opt/ming-tts\n')"; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
COPY --from=build /opt/ming-tts /opt/ming-tts
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    ldconfig; \
    env -u PYTHONPATH python3 -c "\
import os, shutil, torch, transformers, ming_tts; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference audio that is not wav dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert transformers.__version__ == '4.46.3', \
    'Ming-3.0-tts-0.5B targets transformers 4.46.3, got %s' % transformers.__version__; \
assert torch.version.cuda, 'lost CUDA torch after pip install'; \
assert ming_tts.MingTTS"

LABEL org.opencontainers.image.title="audio-ming-deps"
