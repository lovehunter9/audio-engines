# audio-qwen-xpu deps (hash-tagged rebuilds): Qwen3-ASR on Intel's XPU vLLM, driven by the same
# qwen-asr wrapper as the CUDA `qwen` base. amd64 only — there is no Intel GPU on arm64.
#
# NOT Intel's newest tag. 0.21.0-xpu reports vllm 0.21.1.dev18 but pins transformers==5.8.0 and has
# already dropped vllm.inputs.data, so it is cut from a vLLM main NEWER than the 0.23.0 our CUDA
# base runs; qwen-asr 0.0.6 fails to import there (it needs transformers 4.x and that module).
# 0.17.0-xpu is Intel's last transformers-4.x image, i.e. the newest one qwen-asr can meet.
FROM docker.io/intel/vllm:0.17.0-xpu

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The torch in this image links oneCCL, so `import torch` fails with "libccl.so.1: cannot open
# shared object file" unless oneAPI's setvars has run: a hard requirement, not a tuning step. Its
# interpreter also lives in /opt/venv rather than being the system python. Both facts go into the
# audio-python shim, so every entry point gets a working interpreter — not just the chart's start
# script. setvars is skipped when SETVARS_COMPLETED says an outer shell already sourced it.
RUN set -eux; \
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

# The version story, printed before anything can fail: these are answers a CI log can give and the
# Intel machine we do not have yet cannot. `pip check` is the one that caught 0.21: it names the
# transformers pin that qwen-asr and Intel's vLLM disagree on.
RUN set -x; \
    audio-python -c "import torch; print('torch', torch.__version__, 'xpu:', hasattr(torch, 'xpu'))" || true; \
    audio-python -c "import vllm; print('vllm', vllm.__version__)" || true; \
    audio-python -c "import transformers; print('transformers', transformers.__version__)" || true; \
    audio-python -m pip check || true

# qwen-asr targets one shape of vLLM's multimodal data parser: v0.23 takes the processor's
# _get_data_parser, v0.16 wants it on ProcessingInfo (the `qwen` base patches that for arm64).
# 0.17 sits next to the arm64 case, so this reports which shape it is; if it is the 0.16 one, the
# fix is the patch bases/qwen already carries rather than anything new.
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

# The report above came back as the arm64 case: this vLLM wants get_data_parser on ProcessingInfo,
# while qwen-asr defines _get_data_parser on the processor. Same patch bases/qwen applies for
# vLLM 0.16, with no arch gate because this base only ever has the one case. The source blocks are
# copied verbatim from there on purpose — if qwen-asr drifts, both bases refuse at the same assert
# instead of one of them patching something it no longer understands.
RUN audio-python -c "\
import pathlib; \
import qwen_asr.core.vllm_backend.qwen3_asr as m; \
p=pathlib.Path(m.__file__); \
src=p.read_text(); \
old_proc='''class Qwen3ASRMultiModalProcessor(\n    Qwen3OmniMoeThinkerMultiModalProcessor,\n):\n    def _get_data_parser(self) -> MultiModalDataParser:\n        feature_extractor = self.info.get_feature_extractor()\n        return Qwen3ASRMultiModalDataParser(\n            target_sr=feature_extractor.sampling_rate,\n        )\n'''; \
new_proc='''class Qwen3ASRMultiModalProcessor(\n    Qwen3OmniMoeThinkerMultiModalProcessor,\n):\n'''; \
assert old_proc in src, 'qwen-asr processor block drift; refuse silent patch'; \
src=src.replace(old_proc, new_proc, 1); \
old_info='''    def get_supported_mm_limits(self) -> Mapping[str, int | None]:\n        return {\"audio\": None}\n'''; \
new_info='''    def get_supported_mm_limits(self) -> Mapping[str, int | None]:\n        return {\"audio\": None}\n\n    def get_data_parser(self) -> MultiModalDataParser:\n        feature_extractor = self.get_feature_extractor()\n        return Qwen3ASRMultiModalDataParser(\n            target_sr=feature_extractor.sampling_rate,\n        )\n\n    # vLLM 0.16 error text names build_data_parser; some trees use get_data_parser.\n    def build_data_parser(self) -> MultiModalDataParser:\n        return self.get_data_parser()\n'''; \
assert old_info in src, 'qwen-asr ProcessingInfo block drift; refuse silent patch'; \
src=src.replace(old_info, new_info, 1); \
assert '_get_data_parser' not in src, 'leftover _get_data_parser after patch'; \
p.write_text(src); \
print('patched', p)"

# A separate layer so the module is imported fresh: the patch wrote the file the previous process
# had already loaded, so only a new interpreter proves the result parses and still imports.
RUN audio-python -c "\
import inspect; \
import qwen_asr.core.vllm_backend.qwen3_asr as m; \
src = inspect.getsource(m); \
assert '_get_data_parser' not in src, 'patch did not take'; \
assert 'def get_data_parser' in src, 'get_data_parser missing after patch'; \
print('ok: data parser now hangs off ProcessingInfo, module imports clean')"

LABEL org.opencontainers.image.title="audio-qwen-xpu-deps"
