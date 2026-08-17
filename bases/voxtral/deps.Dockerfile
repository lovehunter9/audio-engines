# audio-voxtral deps: mainline vLLM serving Voxtral Mini Realtime via /v1/realtime.
#
# This is NOT vLLM-Omni. Official stack is `vllm serve --tokenizer-mode mistral` (vLLM recipes
# for mistralai/Voxtral-Mini-4B-Realtime-2602). amd64 keeps the validated cu129 / v0.23 image
# (realtime documented from 0.20). arm64 stays on the general aarch64 CUDA track (v0.16.0-cu130);
# the architecture has been registered since 0.16, but the wrapper fails the load if
# /v1/realtime is missing rather than pretending.
ARG TARGETARCH
FROM docker.io/beclab/vllm-vllm-openai:v0.23.0-cu129 AS base-amd64
FROM docker.io/beclab/vllm-vllm-openai:v0.16.0-cu130 AS base-arm64

FROM base-${TARGETARCH}
ARG TARGETARCH

# mistral_common[audio] is required for --tokenizer-mode mistral. Wrapper surface is FastAPI.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "mistral-common[audio]>=1.9.0" \
        soundfile librosa av httpx websockets python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29" \
    && python3 -c "\
import os, shutil, mistral_common, soundfile, librosa, fastapi, uvicorn, httpx, websockets, multipart; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
print('TARGETARCH', arch, 'mistral_common', getattr(mistral_common, '__version__', '?')); \
assert shutil.which('vllm'), 'vllm CLI missing from the base image'; \
print('vllm bin', shutil.which('vllm'))" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-voxtral-deps"
