# audio-diarasr-sgl deps: MOSS-Transcribe-Diarize on SGLang-Omni (wrapper/caps/diar_sgl.py).
#
# The wrapper starts `sgl-omni serve` as a child process and keeps the llm-init contract itself.
# On an RTX 5090 Laptop a 14-minute meeting takes 13.8 s with the DFlash2 draft against 102 s for
# transformers in bases/diarasr; local-model-bench moss-transcribe-diarize has the runs.
#
# There is no CUDA toolkit here, only the pip wheels, and the four pins below are what made that
# work on the benchmark box:
# - flashinfer-jit-cache: flashinfer otherwise compiles its kernels at first start, and the pip
#   nvcc 13.5 rejects the cccl headers flashinfer ships.
# - nvcc, crt and nvvm 13.0: sglang still compiles the DFlash sampling kernel at runtime, and cccl
#   refuses an nvcc newer than the 13.0 runtime headers torch brings.
# - lib64 and libcudart.so in nvidia/cu13: that JIT links -lcudart through CUDA_HOME.
# - python3-dev and a compiler: cdifflib (an sglang-omni dependency) builds from source, and the
#   JIT needs a host compiler.
FROM ubuntu:24.04
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300

ARG SGLANG_OMNI_REF=2ba53e5fbdb5975bab2e307f44740c61fcda0cfa
ARG UV_VERSION=0.13.9
RUN set -eux; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "diarasr-sgl base is amd64-only (TARGETARCH=${TARGETARCH})" >&2; \
        exit 1; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 python3-dev python3-pip build-essential \
        ffmpeg libsndfile1 libnuma1; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C /usr/local/bin --strip-components=1; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# The patch skips the CUDA IPC prewarm in a stage process that never sends to another process.
# Under HAMi time slicing every allocation is managed memory, which has no IPC handle, so the
# prewarm alone failed MOSS's single terminal stage.
COPY bases/diarasr-sgl/patches/ /tmp/sglang-omni-patches/
RUN git clone --filter=blob:none https://github.com/sgl-project/sglang-omni /tmp/sglang-omni \
    && git -C /tmp/sglang-omni checkout -q "${SGLANG_OMNI_REF}" \
    && git -C /tmp/sglang-omni apply /tmp/sglang-omni-patches/*.patch \
    && uv pip install --prerelease=allow /tmp/sglang-omni \
    && rm -rf /tmp/sglang-omni /tmp/sglang-omni-patches \
    && uv pip install --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==0.7.2" \
    && uv pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart soundfile nvidia-ml-py

ENV CUDA_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
RUN set -eux; \
    cu=$(python3 -c "import os, nvidia; print(os.path.join(list(nvidia.__path__)[0], 'cu13'))"); \
    test "$cu" = "$CUDA_HOME"; \
    ln -sfn lib "$cu/lib64"; \
    ln -sf libcudart.so.13 "$cu/lib/libcudart.so"; \
    "$cu/bin/nvcc" --version | tail -1; \
    command -v sgl-omni; \
    python3 -c "\
import shutil, torch, sglang, sglang_omni, flashinfer, fastapi; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
assert torch.version.cuda and torch.version.cuda.startswith('13.'), torch.__version__; \
from sglang_omni.models.moss_transcribe_diarize import config; \
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, may_send_out_of_process; \
stages = config.MossTranscribeDiarizePipelineConfig(model_path='x').stages; \
assert not any(may_send_out_of_process(StageLaunchConfig(stage_name=s.name, next_stages=s.next, \
    route_fn=s.route_fn, stream_targets=list(s.stream_to))) for s in stages), 'MOSS would prewarm CUDA IPC'; \
print('torch', torch.__version__, 'sglang', sglang.__version__, 'flashinfer', flashinfer.__version__)"

# bf16 needs no FP8 GEMM, and deep_gemm asserts on CUDA_HOME at import on sm_120.
ENV SGLANG_ENABLE_JIT_DEEPGEMM=0

LABEL org.opencontainers.image.title="audio-diarasr-sgl-deps" \
      audio.sglang_omni="2ba53e5fbdb5975bab2e307f44740c61fcda0cfa"
