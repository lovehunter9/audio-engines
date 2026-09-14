# Pyannote / Silero / SpeechBrain on the shared slim runtime (no 4.6 GB maximsachs image).
ARG RUNTIME_IMAGE=docker.io/lovehunter9/audio-runtime:slim1
FROM ${RUNTIME_IMAGE}
ARG TARGETARCH

# pyannote/speechbrain can pull a CPU torch from PyPI — put CUDA back in this RUN if they do.
# Do not unconditionally force-reinstall: that would add a second torch layer on top of runtime.
# Reinstall + strip must share this RUN or the fat CUDA layer comes back.
COPY bases/runtime/strip_unused_cuda.sh /tmp/strip_unused_cuda.sh
RUN set -eux; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        IDX=https://download.pytorch.org/whl/cu130; \
    else \
        IDX=https://download.pytorch.org/whl/cu128; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends git build-essential; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        "pyannote.audio>=3.3.0" speechbrain silero-vad omegaconf soundfile \
        python-multipart "fastapi>=0.110" "uvicorn>=0.29"; \
    python3 -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)" \
        || python3 -m pip install --no-cache-dir --force-reinstall \
            torch torchaudio --index-url "${IDX}"; \
    sh /tmp/strip_unused_cuda.sh; \
    rm -f /tmp/strip_unused_cuda.sh; \
    apt-get purge -y git build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN python3 -c "import torch, pyannote.audio, silero_vad, omegaconf, speechbrain, soundfile, fastapi, uvicorn, multipart; \
print('TARGETARCH', '''${TARGETARCH}''', 'torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda, 'lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

RUN python3 -c "import soundfile as sf; \
    print('libsndfile', sf.__libsndfile_version__); \
    [print(c, s, sf.check_format(c, s)) for c, s in \
     (('FLAC', 'PCM_16'), ('OGG', 'OPUS'), ('OGG', 'VORBIS'))]"

LABEL org.opencontainers.image.title="audio-pyannote-deps"
