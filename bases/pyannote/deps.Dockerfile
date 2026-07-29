# audio-pyannote deps: everything but the wrapper, so a wrapper change costs one small layer (README).
# Rebuilt only when THIS file changes: its content hash is the tag the append step looks for.
FROM docker.io/beclab/maximsachs-pyannote_fastapi:4.0.4

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

LABEL org.opencontainers.image.title="audio-pyannote-deps"
