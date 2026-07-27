# audio-pyannote: the torch/pyannote family. Build context is the repo ROOT.
FROM docker.io/beclab/maximsachs-pyannote_fastapi:4.0.4

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-pyannote" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# Build-time deps only: the base has torch + pyannote, the other caps need vad/embed/enhance's own.
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

WORKDIR /app
COPY wrapper /app/wrapper
ENV AUDIO_BASE=pyannote \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WRAPPER_PORT=8000

# Serves directly here; the chart overrides command with its sentinel-wait shell, same module.
CMD ["audio-python", "-m", "wrapper.app"]
