# audio-crispasr deps: Voxtral-4B-TTS on CrispASR's ggml runtime (wrapper/caps/crispasr_tts.py).
ARG TARGETARCH

# runtime, not devel: ggml is compiled inside the wheel and only needs libcudart/libcublas here.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
ARG TARGETARCH

ARG CRISPASR_VERSION=0.8.29
# PyPI's `crispasr` is the CPU build, so the CUDA wheel is named by URL instead.
ARG CRISPASR_WHEEL=https://github.com/CrispStrobe/CrispASR/releases/download/v${CRISPASR_VERSION}/crispasr-${CRISPASR_VERSION}%2Bcuda-py3-none-manylinux_2_28_x86_64.whl

# Upstream ships no arm64 CUDA build, and a 4B TTS model on CPU cannot hold a conversation.
RUN set -eux; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "crispasr base is amd64-only (TARGETARCH=${TARGETARCH})" >&2; \
        exit 1; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ffmpeg libsndfile1 \
        curl ca-certificates \
        binutils; \
    rm -rf /var/lib/apt/lists/*

# No torch: ggml links CUDA itself, so VRAM is read through NVML (nvidia-ml-py) instead.
RUN set -eux; \
    python3 -m pip install --no-cache-dir --break-system-packages --root-user-action=ignore \
        "${CRISPASR_WHEEL}" \
        "huggingface-hub>=0.34" \
        nvidia-ml-py \
        numpy soundfile \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets

# CUDA job in CrispASR 0.8.29 leaves GGML_NATIVE=ON; overlay AVX2 libggml-cpu from the same tag so codec CPU fallback does not SIGILL.
RUN set -eux; \
    root="$(python3 -c 'import importlib.util, pathlib; \
spec = importlib.util.find_spec("crispasr"); \
print(pathlib.Path(spec.origin).parent if spec and spec.origin else "", end="")')"; \
    test -n "$root"; \
    curl -fsSL -o /tmp/cpu-libs.tgz \
        "https://github.com/CrispStrobe/CrispASR/releases/download/v${CRISPASR_VERSION}/libcrispasr-linux-x86_64.tar.gz"; \
    mkdir -p /tmp/cpu-libs; \
    tar -xzf /tmp/cpu-libs.tgz -C /tmp/cpu-libs; \
    src="$(find /tmp/cpu-libs -name 'libggml-cpu.so.0.*' -type f | head -1)"; \
    test -n "$src" || { echo "AVX2 bundle has no libggml-cpu" >&2; exit 1; }; \
    for f in "$(dirname "$src")"/libggml-cpu.so*; do cp -a "$f" "$root/"; done; \
    dst="$(find "$root" -maxdepth 1 -name 'libggml-cpu.so.0.*' -type f | head -1)"; \
    test -n "$dst"; \
    if objdump -d "$dst" | grep -E '%zmm|vinserti64x4|ldtilecfg' >/dev/null; then \
        echo "overlaid libggml-cpu still contains AVX-512/AMX" >&2; exit 1; \
    fi; \
    rm -rf /tmp/cpu-libs /tmp/cpu-libs.tgz

# Verified without importing: libggml-cuda.so needs libcuda.so.1, and a CI runner has no driver.
RUN set -eux; \
    python3 -m pip show crispasr | grep -E '^(Name|Version|Location)'; \
    root="$(python3 -c 'import importlib.util, pathlib; \
spec = importlib.util.find_spec("crispasr"); \
print(pathlib.Path(spec.origin).parent if spec and spec.origin else "", end="")')"; \
    test -n "$root" || { echo "crispasr package not importable at all" >&2; exit 1; }; \
    lib="$(ls "$root"/libggml-cuda.so* 2>/dev/null | head -1)"; \
    test -n "$lib" || { echo "no libggml-cuda.so: this is the CPU build" >&2; exit 1; }; \
    readelf -d "$lib" | grep NEEDED; \
    readelf -d "$lib" | grep -q 'libcuda\.so\.1' \
        || { echo "libggml-cuda.so does not link the driver" >&2; exit 1; }; \
    ldconfig -p | grep -E 'libcudart\.so\.12|libcublas\.so\.12'; \
    command -v ffmpeg >/dev/null || { echo "no ffmpeg" >&2; exit 1; }; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-crispasr-deps"
