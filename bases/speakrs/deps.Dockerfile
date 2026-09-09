# audio-speakrs deps (hash-tagged rebuilds).
#
# CUDA 13 on both architectures, and the base has to move with the engine image rather than after
# it: the runtime ONNX Runtime links against is chosen there, and a CUDA 12 base under a CUDA 13
# runtime is missing libcudart.so.13 outright. The two are one change in two files.
#
# No torch. speakrs runs on ONNX Runtime, which links CUDA itself, so a torch stack here would be
# several gigabytes bought for nothing; the four /metrics gauges come from NVML instead (the same
# trade audio-crispasr already makes).
#
# WHY THE ENGINE ARRIVES AS AN IMAGE AND NOT AS SOURCE
# ---------------------------------------------------
# scripts/deps-image.sh tags this image with a hash of THIS FILE plus any sibling probe_*.py, and
# nothing else. A Rust tree sitting next to this file would therefore not move the tag when it
# changed: same tag, different contents -- exactly what hash tagging exists to prevent. Pinning the
# engine image by digest puts the engine's identity inside the hashed file, so rebuilding the
# engine forces a new line here, which forces a new deps tag. The chain holds without anyone
# remembering to bump anything.
ARG ENGINE_IMAGE=docker.io/beclab/speakrs-engine@sha256:fd39309ba31673d92e0afa10fd7f2e080af32711524dea67baeb2be29443c87c
FROM ${ENGINE_IMAGE} AS engine

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libsndfile1 libcudnn9-cuda-13 ca-certificates \
        ffmpeg; \
    rm -rf /var/lib/apt/lists/*

# ffmpeg is the engine's decoder, not a convenience: speakrs takes 16 kHz mono f32 and the Rust
# side shells out to convert, so without it every diarization fails at the first request with
# "could not run ffmpeg" -- while the model loads, /v1/models answers 200 and every contract and
# capability check passes. Nothing short of sending audio finds it.

# The shell only serves HTTP, reads a file header for billing, and reads NVML. Everything that
# touches audio samples or the model happens in the engine process.
RUN python3 -m pip install --no-cache-dir --break-system-packages --root-user-action=ignore \
        "fastapi>=0.110" "uvicorn>=0.29" python-multipart soundfile nvidia-ml-py

COPY --from=engine /usr/local/bin/speakrs-engine /usr/local/bin/speakrs-engine
# ONNX Runtime's CUDA provider libraries travel with the engine that dlopen's them, so the two can
# never be a version apart.
COPY --from=engine /usr/local/lib/onnxruntime/ /usr/local/lib/onnxruntime/
# ort's load-dynamic dlopen's libonnxruntime.so by an explicit search list -- the binary's own
# directory, the cwd, and cargo target dirs -- and never consults the ldconfig cache. ldconfig
# alone therefore left the engine unable to find a library sitting right there.
ENV ORT_DYLIB_PATH=/usr/local/lib/onnxruntime/libonnxruntime.so
RUN ldconfig /usr/local/lib/onnxruntime && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# Import-check the shell's deps, and prove the engine binary is the right architecture and runs.
# A wrong-arch copy survives build and push and only shows up as "exec format error" in a pod.
# ffmpeg is checked here and not left to a probe on purpose: a shell-out dependency shows up in
# no import, no requirements file and no health endpoint, so the first thing that notices is a
# request carrying audio -- long after the image shipped and the engine reported itself ready.
RUN python3 -c "import fastapi, uvicorn, multipart, soundfile, pynvml; \
    print('soundfile', soundfile.__libsndfile_version__)" \
    && ffmpeg -version | head -1 \
    && ffprobe -version | head -1 \
    && /usr/local/bin/speakrs-engine --version

LABEL org.opencontainers.image.title="audio-speakrs-deps"
