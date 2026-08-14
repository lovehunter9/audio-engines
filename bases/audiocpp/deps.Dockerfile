# audio-audiocpp deps: upstream's audio.cpp server binary, plus the Python the wrapper needs.
#
# Nothing is compiled here. Upstream publishes a daily multi-arch (amd64+arm64) image that already
# carries audiocpp_server, its ggml CPU/CUDA backends and all 47 model_specs under /app, so this
# file only pins one of those builds and adds Python on top. Pinned by date+revision rather than
# the rolling :full-cuda12 so a rebuild cannot silently change the engine underneath a chart.
#
# ENTRYPOINT [] is load-bearing, not tidiness: upstream's /app/entrypoint.sh is a multiplexer that
# understands cli|server|perf|parity and exits 1 on anything else. Left in place it would receive
# the appended CMD as argv and answer "Unknown command: audio-python", so the wrapper would never
# start. The wrapper execs /app/audiocpp_server directly and has no use for the multiplexer.
FROM ghcr.io/0xshug0/audio.cpp:full-cuda12-20260813-dd6b089

# Upstream drops to USER ubuntu. Package installs need root, and llm-init's weight cache arrives
# root-owned, so the runtime stays root like every other base here.
USER root

# A venv rather than the system interpreter: Ubuntu ships PEP 668 (externally-managed), and
# --break-system-packages is not spelled the same across the releases upstream may rebase onto.
# ffmpeg and libsndfile1 are already in the upstream image; asserted below instead of reinstalled.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends python3 python3-venv; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m venv /opt/venv; \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip; \
    /opt/venv/bin/pip install --no-cache-dir \
        "huggingface-hub>=0.34" soundfile numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets \
        nvidia-ml-py; \
    ln -sf /opt/venv/bin/python /usr/local/bin/audio-python

# Build-time assertions, so a broken base fails here and not on a node an hour later. No torch on
# purpose: this base runs no Python inference, and pulling torch in just to read GPU memory would
# cost gigabytes, so wrapper/gpu.py reads NVML through nvidia-ml-py instead.
RUN /opt/venv/bin/python -c "\
import json, shutil, subprocess, sys; \
import fastapi, httpx, numpy, soundfile, uvicorn, websockets, multipart; \
import huggingface_hub, pynvml; \
assert shutil.which('ffmpeg'), 'upstream image lost ffmpeg: non-wav reference audio would have no decoder'; \
missing_bins = [p for p in ('/app/audiocpp_server', '/app/audiocpp_cli') if not shutil.which(p)]; \
assert not missing_bins, 'the upstream image lacks %s' % missing_bins; \
caps = subprocess.run(['/app/audiocpp_cli', '--list-loaders', '--json'], capture_output=True, text=True, timeout=120); \
print('loader advertisement here:', len(json.loads(caps.stdout)['loaders']), 'families') if caps.returncode == 0 else \
print('NOTE: --list-loaders exited %s on this builder (no GPU here; the wrapper retries on the node and falls back to model_specs): %s' % (caps.returncode, (caps.stderr or caps.stdout).strip()[:200])); \
fmts = set(soundfile.available_formats()); \
missing = {'WAV', 'FLAC', 'OGG'} - fmts; \
assert not missing, 'libsndfile is missing %s' % sorted(missing); \
print('audiocpp deps ok:', sys.version.split()[0], '| soundfile formats:', len(fmts), '| mp3:', 'MP3' in fmts)"

# The multiplexer goes here, for the reason at the top of this file: the appended CMD has to be
# argv[0], and `crane mutate --cmd` does not touch an inherited entrypoint.
ENTRYPOINT []

LABEL org.opencontainers.image.title="audio-audiocpp-deps"
