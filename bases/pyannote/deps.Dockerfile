# audio-pyannote deps (hash-tagged rebuilds). amd64: maximsachs zero-diff. arm64: cu130 torch stack (no arm64 upstream mirror).
ARG TARGETARCH
FROM docker.io/beclab/maximsachs-pyannote_fastapi:4.0.4 AS base-amd64

FROM python:3.11-slim-bookworm AS base-arm64
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    python3 -m pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    python3 -m pip install --no-cache-dir "pyannote.audio>=3.3.0" speechbrain; \
    # pyannote/speechbrain deps can silently pull CPU torch from PyPI on aarch64 — put CUDA back.
    python3 -m pip install --no-cache-dir --force-reinstall \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu130; \
    python3 -c "import torch; \
assert torch.version.cuda, 'arm64 pyannote deps must be CUDA torch, got %s' % (torch.__version__,); \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FROM base-${TARGETARCH}
ARG TARGETARCH

# Build-time deps only: the amd64 base has torch + pyannote; arm64 stage installed those above.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        silero-vad omegaconf speechbrain soundfile python-multipart \
        "fastapi>=0.110" "uvicorn>=0.29"

# Import-check deps; re-assert CUDA torch on arm64 after shared pip (PyPI may overwrite).
RUN python3 -c "import torch, pyannote.audio, silero_vad, omegaconf, speechbrain, soundfile, fastapi, uvicorn, multipart; \
import os; \
arch=os.environ.get('TARGETARCH') or '''${TARGETARCH}'''; \
print('TARGETARCH', arch, 'torch', torch.__version__, 'cuda', torch.version.cuda); \
assert arch != 'arm64' or torch.version.cuda, 'arm64 lost CUDA torch after pip install'" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# Log which containers enhance can return; it degrades to FLAC then WAV, so this is informational.
RUN python3 -c "import soundfile as sf; \
    print('libsndfile', sf.__libsndfile_version__); \
    [print(c, s, sf.check_format(c, s)) for c, s in \
     (('FLAC', 'PCM_16'), ('OGG', 'OPUS'), ('OGG', 'VORBIS'))]"

LABEL org.opencontainers.image.title="audio-pyannote-deps"
