# audio-pyannote base: the torch/pyannote family (vad / diar / speaker_embed /
# enhance). Build context is the repo ROOT (COPY wrapper/...).
#   docker build -f bases/pyannote/Dockerfile -t <reg>/audio-pyannote:<tag> .
FROM docker.io/beclab/maximsachs-pyannote_fastapi:4.0.4

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="audio-pyannote" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# BUILD-TIME deps (no runtime pip). The base brings torch + pyannote.audio; the
# other three caps each need one thing it lacks: silero-vad (vad), omegaconf
# (embed — pyannote/embedding's checkpoint pickles an omegaconf config) and
# speechbrain (enhance).
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        silero-vad omegaconf speechbrain soundfile python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29"

# Fail the BUILD, not a user's clone, if a dependency ever stops resolving.
# `audio-python` is the single interpreter name every base image and every
# chart's sentinel shell uses (some bases keep their deps in a venv).
RUN python3 -c "import torch, pyannote.audio, silero_vad, omegaconf, speechbrain, soundfile" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

WORKDIR /app
COPY wrapper /app/wrapper
ENV AUDIO_BASE=pyannote \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WRAPPER_PORT=8000

# Default: serve directly (local dev / docker run). The Olares chart overrides
# command with a thin sentinel-wait shell (templates/wrappers.yaml) that execs
# this same module after llm-init finishes the model download.
CMD ["audio-python", "-m", "wrapper.app"]
