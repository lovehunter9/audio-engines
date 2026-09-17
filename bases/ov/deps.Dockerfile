# audio-qwen-ov deps. Intel GPU is amd64 only.
#
# Vendor is lovehunter9/ov-vendor:v0.0.1 (and :v0.0.1-base). This file
# only compiles the C++ patch onto that pin, so a patch-only rebuild is
# compile + crane-append. CI must pass VENDOR and BASE (vendor-image.sh).

ARG VENDOR
ARG BASE
FROM ${VENDOR} AS builder
COPY bases/ov/patches/apply_qwen3_asr_batch.py /tmp/apply_qwen3_asr_batch.py
RUN test -d /opt/genai-src \
    && cp -a /opt/genai-src /tmp/genai \
    && python3 /tmp/apply_qwen3_asr_batch.py /tmp/genai \
    && export CMAKE_GENERATOR=Ninja \
    && unset CFLAGS CXXFLAGS \
    && export CMAKE_ARGS="-DENABLE_SAMPLES=OFF -DENABLE_JS=OFF -DENABLE_GGUF_SUPPORT=OFF" \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore --force-reinstall --no-deps --no-build-isolation \
        /tmp/genai \
    && rm -rf /tmp/genai /tmp/apply_qwen3_asr_batch.py /root/.cache

FROM ${BASE}
COPY --from=builder /usr/local /usr/local
RUN python3 -c "import openvino, openvino_genai, optimum, transformers, qwen_asr, librosa, soundfile, fastapi, uvicorn, huggingface_hub, numpy; \
from optimum.intel import OVModelForQwen3ASRForcedAligner; \
print('openvino', openvino.__version__); \
print('openvino_genai', getattr(openvino_genai, '__version__', 'ok')); \
print('transformers', transformers.__version__); \
print('qwen_asr', 'ok'); \
print('forced_aligner', OVModelForQwen3ASRForcedAligner.__name__)"

LABEL org.opencontainers.image.title="audio-qwen-ov-deps" \
      audio.compute_runtime="26.22.38646.4"
