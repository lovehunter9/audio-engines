# audio-qwen-xpu deps (hash-tagged rebuilds): Qwen3-ASR on Intel's XPU vLLM, driven by the same
# qwen-asr wrapper as the CUDA `qwen` base. amd64 only — there is no Intel GPU on arm64.
FROM docker.io/intel/vllm:0.21.0-ubuntu24.04-20260805

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The torch in this image links oneCCL, so `import torch` fails with "libccl.so.1: cannot open
# shared object file" unless oneAPI's setvars has run: a hard requirement, not a tuning step. Its
# interpreter also lives in /opt/venv rather than being the system python. Both facts go into the
# audio-python shim, so every entry point gets a working interpreter — not just the chart's start
# script. setvars is skipped when SETVARS_COMPLETED says an outer shell already sourced it.
RUN set -eux; \
    [ -f /opt/intel/oneapi/setvars.sh ]; \
    PY="$(command -v python3)"; \
    printf '%s\n' \
      '#!/bin/bash' \
      '# The image torch links oneCCL; without setvars it cannot find libccl.so.1.' \
      'if [ -f /opt/intel/oneapi/setvars.sh ] && [ -z "${SETVARS_COMPLETED:-}" ]; then' \
      '    . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true' \
      'fi' \
      "exec ${PY} \"\$@\"" \
      > /usr/local/bin/audio-python; \
    chmod 755 /usr/local/bin/audio-python; \
    audio-python -c "import torch; print('torch', torch.__version__, 'has xpu:', hasattr(torch, 'xpu'))"

# Build-time deps only. blinker first (distutils 1.4 blocks qwen-asr); plain qwen-asr, not [vllm].
# The image's XPU torch/vLLM is the whole point of this base, so a dependency that replaces either
# with a wheel from PyPI has to fail the build here rather than lose the GPU at runtime.
RUN set -eux; \
    T0="$(audio-python -c 'import torch; print(torch.__version__)')"; \
    V0="$(audio-python -c 'import vllm; print(vllm.__version__)')"; \
    audio-python -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed blinker; \
    audio-python -m pip install --no-cache-dir --root-user-action=ignore \
        qwen-asr soundfile librosa av websockets python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29"; \
    T1="$(audio-python -c 'import torch; print(torch.__version__)')"; \
    V1="$(audio-python -c 'import vllm; print(vllm.__version__)')"; \
    [ "$T0" = "$T1" ] || { echo "pip replaced the image's torch: $T0 -> $T1"; exit 1; }; \
    [ "$V0" = "$V1" ] || { echo "pip replaced the image's vLLM: $V0 -> $V1"; exit 1; }; \
    audio-python -c "import torch; assert hasattr(torch, 'xpu'), 'torch has no xpu namespace'"; \
    audio-python -c "import qwen_asr, soundfile, librosa, fastapi, uvicorn, multipart, websockets"

# qwen-asr targets one shape of vLLM's multimodal data parser: v0.23 takes the processor's
# _get_data_parser, v0.16 wants it on ProcessingInfo (the `qwen` base patches that for arm64).
# This image is 0.21, i.e. between the two, so report which shape it is: a mismatch would otherwise
# surface as an unexplained load failure on the first Intel machine we get.
RUN audio-python -c "\
import inspect; \
import vllm; \
print('vllm', vllm.__version__); \
import qwen_asr.core.vllm_backend.qwen3_asr as m; \
src = inspect.getsource(m); \
print('qwen-asr defines _get_data_parser:', '_get_data_parser' in src); \
import vllm.multimodal.processing as p; \
info = getattr(p, 'BaseProcessingInfo', None); \
print('vLLM BaseProcessingInfo:', info); \
print('  .get_data_parser:', hasattr(info, 'get_data_parser')); \
print('  .build_data_parser:', hasattr(info, 'build_data_parser')); \
proc = getattr(p, 'BaseMultiModalProcessor', None); \
print('vLLM BaseMultiModalProcessor._get_data_parser:', hasattr(proc, '_get_data_parser'))"

LABEL org.opencontainers.image.title="audio-qwen-xpu-deps"
