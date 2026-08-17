# audio-mosstts deps: official MOSS-TTS-Nano ONNX in-process. Not vLLM-Omni, not audio.cpp.
#
# The model's identity is realtime streaming on a small CPU (or CUDA EP). Official deployment
# dropped PyTorch for inference and ships split ONNX graphs + Realtime Streaming Decode.
# onnx_tts_runtime.py still imports torch/torchaudio to load a reference clip, so a CPU torch
# wheel is enough; the audio comes out of onnxruntime.
#
# amd64: onnxruntime-gpu (CPU EP is in the same wheel).
# arm64: onnxruntime CPU; CUDA EP wheels are not reliably on PyPI for aarch64, and the wrapper
# falls back to cpu if --execution-provider cuda cannot session.
ARG TARGETARCH

FROM python:3.12-slim-bookworm AS base-amd64
FROM python:3.12-slim-bookworm AS base-arm64

FROM base-${TARGETARCH}
ARG TARGETARCH

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "numpy>=1.24" "sentencepiece>=0.1.99" soundfile \
        "huggingface-hub>=0.34" \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        torch==2.7.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cpu; \
    if [ "${TARGETARCH}" = "amd64" ]; then \
        python3 -m pip install --no-cache-dir --root-user-action=ignore "onnxruntime-gpu>=1.20.0"; \
    else \
        python3 -m pip install --no-cache-dir --root-user-action=ignore "onnxruntime>=1.20.0"; \
    fi; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        --no-deps git+https://github.com/OpenMOSS/MOSS-TTS-Nano.git; \
    python3 -c "\
import os, shutil, onnxruntime, torch, torchaudio, soundfile, fastapi, sentencepiece; \
import onnx_tts_runtime, ort_cpu_runtime; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
assert shutil.which('ffmpeg'), 'no ffmpeg: reference clips that are not wav die'; \
print('TARGETARCH', arch, 'ort', onnxruntime.__version__, 'providers', onnxruntime.get_available_providers()); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-mosstts-deps"
