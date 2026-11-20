# audio-diarasr deps: speaker-attributed transcription in one pass (wrapper/caps/diar_asr.py).
#
# MOSS-Transcribe-Diarize ships remote code written against transformers 5.6+, and VibeVoice-ASR
# is native in transformers from 5.3; one pin serves both. No vLLM: both run in this process.
#
# amd64 only for now: these images exist to measure the models next to the meeting pipeline,
# and the measurements run on x86 NVIDIA cards. CUDA 12.8 torch, so Blackwell (sm_120) loads.
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG TARGETARCH

RUN set -eux; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "diarasr base is amd64-only (TARGETARCH=${TARGETARCH})" >&2; \
        exit 1; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "transformers>=5.6,<6" "safetensors>=0.6.2" accelerate \
        "huggingface-hub>=0.34" soundfile librosa soxr numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart nvidia-ml-py \
    && python3 -c "\
import shutil, torch, transformers, librosa, fastapi; \
from transformers import VibeVoiceAsrForConditionalGeneration; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
assert torch.version.cuda, 'diarasr deps must be CUDA torch, got %s' % torch.__version__; \
v = tuple(int(x) for x in transformers.__version__.split('.')[:2]); \
assert (5, 6) <= v < (6, 0), transformers.__version__; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-diarasr-deps"
