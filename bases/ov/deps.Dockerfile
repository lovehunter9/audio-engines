# audio-ov deps (hash-tagged rebuilds). Intel GPU only exists on amd64; CI passes a single-arch slice.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# compute-runtime (Level Zero + OpenCL) is what OpenVINO's GPU plugin talks to.
# clinfo is diagnostic; the chart mounts /dev/dri so the plugin can see the iGPU / Arc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
        libsndfile1 \
        ffmpeg \
        intel-opencl-icd \
        libze1 \
        libze-intel-gpu1 \
        ocl-icd-libopencl1 \
        clinfo \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/python \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# No CUDA torch. optimum[openvino] may pull a CPU torch; that is only for first-start IR export.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore --upgrade pip \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore \
        openvino \
        openvino-genai \
        "optimum[openvino]" \
        transformers \
        huggingface_hub \
        librosa \
        soundfile \
        "fastapi>=0.110" \
        "uvicorn>=0.29" \
        python-multipart \
        websockets \
        numpy \
    && python3 -c "import openvino, openvino_genai, optimum, transformers, librosa, soundfile, fastapi, uvicorn, huggingface_hub, numpy; \
print('openvino', openvino.__version__); \
print('openvino_genai', getattr(openvino_genai, '__version__', 'ok'))"

LABEL org.opencontainers.image.title="audio-ov-deps"
