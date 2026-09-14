# Breeze on the shared slim runtime (no official pytorch/pytorch 4.3 GB image).
ARG RUNTIME_IMAGE=docker.io/lovehunter9/audio-runtime:slim1
FROM ${RUNTIME_IMAGE}
ARG TARGETARCH

ARG BREEZE_REF=ca632ce6c4d05f7985da4eab29b1a5d445b43f7b

# torch stays the runtime image's unless a dep overwrites it with CPU. git is fetch-only.
# Reinstall + strip must share this RUN or the fat CUDA layer comes back.
COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends git; \
    python3 -m pip install --no-cache-dir \
        "transformers==4.57.3" "qwen-tts==0.1.1" \
        "huggingface-hub>=0.34" soundfile librosa numpy \
        "fastapi>=0.115" "uvicorn>=0.30" httpx python-multipart websockets; \
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
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    python3 -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)" \
        || python3 -m pip install --no-cache-dir --force-reinstall \
            torch torchaudio --index-url "${IDX}"; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh; \
    apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN env -u PYTHONPATH python3 -c "\
import os, shutil, torch, transformers; \
from breeze_infer.runtime import load_runtime; \
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference audio that is not wav dies'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda, \
      'transformers', transformers.__version__); \
assert transformers.__version__ == '4.57.3', \
    'Breeze TTS 2 targets transformers 4.57.3, got %s' % transformers.__version__; \
assert torch.version.cuda, 'lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-breeze-deps"
