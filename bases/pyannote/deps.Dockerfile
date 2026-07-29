# audio-pyannote deps: everything but the wrapper, so a wrapper change costs one small layer (README).
# Rebuilt only when THIS file changes: its content hash is the tag the append step looks for.
#
# amd64: keep the validated maximsachs image. arm64: that mirror has no arm64 manifest, so build
# a generic aarch64 stack (CPU torch wheels here; CUDA aarch64 can replace the arm64 stage later).
ARG TARGETARCH
FROM docker.io/beclab/maximsachs-pyannote_fastapi:4.0.4 AS base-amd64

FROM python:3.11-slim-bookworm AS base-arm64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    python3 -m pip install --no-cache-dir "pyannote.audio>=3.3.0" speechbrain

FROM base-${TARGETARCH}

# Build-time deps only: the amd64 base has torch + pyannote; arm64 stage installed those above.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        silero-vad omegaconf speechbrain soundfile python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29"

# Fail the BUILD, not a clone, if a dep stops resolving. audio-python = the interpreter owning them.
RUN python3 -c "import torch, pyannote.audio, silero_vad, omegaconf, speechbrain, soundfile, fastapi, uvicorn, multipart" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# Log which containers enhance can return; it degrades to FLAC then WAV, so this is informational.
RUN python3 -c "import soundfile as sf; \
    print('libsndfile', sf.__libsndfile_version__); \
    [print(c, s, sf.check_format(c, s)) for c, s in \
     (('FLAC', 'PCM_16'), ('OGG', 'OPUS'), ('OGG', 'VORBIS'))]"

LABEL org.opencontainers.image.title="audio-pyannote-deps"
