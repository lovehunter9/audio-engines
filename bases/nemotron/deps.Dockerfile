# Nemotron 3 Diarization: CUDA runtime + nemo-toolkit[asr], not beclab/nvidia-nemo.
# That image is the NGC training container (~55 GB unpacked). This base installs only
# what SortformerEncLabelModel.restore_from needs, then strips unused CUDA libs.
ARG TARGETARCH
# nemo-toolkit 3.0.0 (PyPI latest) has no rope TransformerEncoder, which this checkpoint's encoder needs.
ARG NEMO_REF=5d641ef5048bee7a496e5cee8289a18215570b1a
FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu22.04 AS amd64
FROM docker.io/nvidia/cuda:13.0.3-base-ubuntu22.04 AS arm64

FROM ${TARGETARCH} AS build
ARG TARGETARCH
ARG NEMO_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
# The rope encoder runs torch.compile(flex_attention) on CUDA; the shared strip drops triton, so torch's own pin goes back after it.
# Jammy's pip 22.0.2 reads no Requires-Dist from the Speech source wheel (Metadata-Version 2.4), so pip is upgraded first.
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
    python3 -m pip install --no-cache-dir -U pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url "${IDX}"; \
    python3 -m pip freeze | grep -E '^(torch|torchaudio)==' > /tmp/torch.pin; \
    python3 -m pip install --no-cache-dir -c /tmp/torch.pin \
        "nemo-toolkit[asr] @ https://github.com/NVIDIA-NeMo/Speech/archive/${NEMO_REF}.tar.gz" \
        numpy \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29" websockets; \
    sh /tmp/strip_unused_cuda.sh; \
    TRITON=$(python3 -c "import importlib.metadata as m; print(next(r.split(';')[0].strip() for r in (m.requires('torch') or []) if r.startswith(('triton', 'pytorch-triton'))))"); \
    python3 -m pip install --no-cache-dir --no-deps \
        --index-url "${IDX}" --extra-index-url https://pypi.org/simple "${TRITON}"; \
    rm -f /tmp/strip_unused_cuda.sh /tmp/torch.pin

FROM ${TARGETARCH} AS release
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
COPY --from=build /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=build /opt/cuda-stubs/ /usr/local/lib/
# gcc, libc6-dev and python3-dev: triton builds its CUDA launcher with the host compiler at first use.
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev gcc libc6-dev ffmpeg libsndfile1 ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python; \
    ldconfig; \
    python3 -c "\
import os, shutil, sysconfig, torch, triton; \
from nemo.collections.asr.models import SortformerEncLabelModel; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda, 'lost CUDA torch after pip install'; \
assert SortformerEncLabelModel.restore_from.__name__; \
assert shutil.which('gcc'), 'no gcc for triton'; \
assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')), 'no Python.h for triton'; \
print('triton', triton.__version__); \
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder; \
TransformerEncoder(feat_in=16, d_model=32, n_heads=2, n_layers=1, self_attention_model='rope'); \
print('rope TransformerEncoder ok')"

LABEL org.opencontainers.image.title="audio-nemotron-deps"
