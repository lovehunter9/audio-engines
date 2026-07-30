# audio-qwen deps (hash-tagged rebuilds). amd64: vLLM v0.23.0-cu129 zero-diff. arm64: v0.16.0-cu130 + qwen-asr data_parser patch.
ARG TARGETARCH
FROM docker.io/beclab/vllm-vllm-openai:v0.23.0-cu129 AS base-amd64
FROM docker.io/beclab/vllm-vllm-openai:v0.16.0-cu130 AS base-arm64

FROM base-${TARGETARCH}
ARG TARGETARCH

# Build-time deps only. blinker first (distutils 1.4 blocks qwen-asr); plain qwen-asr, not [vllm].
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore --ignore-installed blinker \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        qwen-asr soundfile librosa av websockets python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29" \
    && python3 -c "import qwen_asr, soundfile, librosa, fastapi, uvicorn, multipart, websockets" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# arm64-only: migrate qwen-asr's _get_data_parser onto ProcessingInfo for vLLM 0.16+. amd64 never enters this block (TARGETARCH gate) — zero behavior change.
RUN python3 -c "\
import os, pathlib, re, sys; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
print('qwen-asr mm-patch TARGETARCH=', arch); \
sys.exit(0) if arch != 'arm64' else None; \
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

# arm64 build fails if TORCH_CUDA_ARCH_LIST empty; amd64 keeps image-baked arches.
RUN python3 -c "\
import os, sys; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
lst=(os.environ.get('TORCH_CUDA_ARCH_LIST') or '').strip(); \
print('TARGETARCH=', arch, 'CUDA_VERSION=', os.environ.get('CUDA_VERSION',''), 'TORCH_CUDA_ARCH_LIST=', lst); \
sys.exit(0 if (arch != 'arm64' or lst) else 1)"

LABEL org.opencontainers.image.title="audio-qwen-deps"
