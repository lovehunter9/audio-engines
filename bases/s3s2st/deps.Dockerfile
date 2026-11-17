FROM docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build

ARG TARGETARCH
ARG TORCHCODEC_REVISION=c47b7dfe05aec660938437febc2055e6caedca5b
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
ARG S3_REVISION=1ca52ea064ec2081f12675d853525d394571bf73
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 python3 python3-pip python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git init /opt/s3 \
    && git -C /opt/s3 remote add origin https://github.com/QwenAudio/S3-S2ST.git \
    && git -C /opt/s3 fetch --depth 1 origin "$S3_REVISION" \
    && git -C /opt/s3 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/s3 rev-parse HEAD)" = "$S3_REVISION" \
    && printf '%s\n' "$S3_REVISION" > /opt/s3/COMMIT

COPY bases/s3s2st/collect_requirements.py /opt/collect_s3_requirements.py
RUN python3 -m pip install --no-cache-dir "torch==2.9.0" "torchaudio==2.9.0" "torchvision==0.24.0" \
        --index-url https://download.pytorch.org/whl/cu129 \
    && python3 -m pip install --no-cache-dir --no-deps "vllm==0.11.1" \
    && python3 -m pip install --no-cache-dir "packaging==25.0" \
    && python3 /opt/collect_s3_requirements.py \
    && python3 -m pip install --no-cache-dir -r /tmp/s3-runtime-requirements.txt

RUN if [ "$TARGETARCH" = arm64 ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            cmake pkg-config libavcodec-dev libavformat-dev libavutil-dev \
            libswresample-dev libavdevice-dev libavfilter-dev libswscale-dev \
        && rm -rf /var/lib/apt/lists/* \
        && python3 -m pip install --no-cache-dir pybind11 \
        && git init /opt/torchcodec \
        && git -C /opt/torchcodec remote add origin https://github.com/pytorch/torchcodec.git \
        && git -C /opt/torchcodec fetch --depth 1 origin "$TORCHCODEC_REVISION" \
        && git -C /opt/torchcodec checkout --detach FETCH_HEAD \
        && I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
            CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)" \
            CMAKE_BUILD_PARALLEL_LEVEL=2 python3 -m pip install --no-cache-dir \
            --no-build-isolation --no-deps /opt/torchcodec \
        && rm -rf /opt/torchcodec; \
    else python3 -m pip install --no-cache-dir "torchcodec==0.8.1"; fi

RUN python3 -m pip install --no-cache-dir -e '/opt/s3[inference]' \
        fastapi uvicorn httpx python-multipart websockets "opencv-python-headless==4.11.0.86" "better-exceptions==0.3.3" "nvtx==0.2.13" \
    && python3 -m pip install --no-cache-dir "setuptools==80.10.2" \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps "openai-whisper==20240930" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

RUN if [ "$TARGETARCH" = arm64 ]; then fa_arch=aarch64; else fa_arch=x86_64; fi \
    && python3 -m pip install --no-cache-dir --no-deps \
       "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_${fa_arch}.whl"

COPY bases/s3s2st/apply_compat.py /opt/apply_s3_compat.py
RUN python3 /opt/apply_s3_compat.py

COPY bases/s3s2st/native_plugin /opt/s3-native-runtime
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/s3-native-runtime

COPY bases/s3s2st/collect_kernels.py /opt/link_s3_kernels.py
RUN python3 /opt/link_s3_kernels.py \
    && bash -c '. /opt/s3/env.sh; python3 -c "import ast, pathlib, vllm, torch; ast.parse(pathlib.Path(vllm.__file__).parent.joinpath(\"engine/omnidrca2_engine.py\").read_text()); assert vllm.__file__.startswith(\"/opt/s3/vendor/\"); assert torch.__version__.startswith(\"2.9.\"); print(vllm.__file__, torch.__version__)"'

COPY bases/s3s2st/collect_runtime.py /opt/collect_s3_runtime.py
RUN python3 /opt/collect_s3_runtime.py

COPY bases/runtime/split_layers.py /opt/split_runtime.py
RUN python3 /opt/split_runtime.py /opt/s3-runtime

FROM ubuntu:24.04 AS release
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/s3-runtime/site \
    CUDA_HOME=/opt/s3-runtime/cuda PATH=/opt/s3-runtime/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/s3-runtime/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip gcc g++ libc6-dev ninja-build \
        ffmpeg libsndfile1 libnuma1 libgomp1 libgl1 libsm6 libxext6 \
        libibverbs1 librdmacm1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python \
    && python3 -c "import pathlib, site; p = pathlib.Path(site.getsitepackages()[0]); p.mkdir(parents=True, exist_ok=True); p.joinpath('s3_runtime.pth').write_text(\"import site; site.addsitedir('/opt/s3-runtime/site')\\n\")"
COPY --from=build /opt/runtime-parts/00/ /
COPY --from=build /opt/runtime-parts/01/ /
COPY --from=build /opt/runtime-parts/02/ /
COPY --from=build /opt/runtime-parts/03/ /
COPY --from=build /opt/runtime-parts/04/ /
COPY --from=build /opt/runtime-parts/05/ /
COPY --from=build /opt/runtime-parts/06/ /
COPY --from=build /opt/runtime-parts/07/ /
COPY --from=build /opt/runtime-parts/08/ /
COPY --from=build /opt/runtime-parts/09/ /
COPY --from=build /opt/runtime-parts/10/ /
COPY --from=build /opt/runtime-parts/11/ /
COPY --from=build /opt/runtime-parts/12/ /
COPY --from=build /opt/runtime-parts/13/ /
COPY --from=build /opt/runtime-parts/14/ /
COPY --from=build /opt/runtime-parts/15/ /
COPY --from=build /opt/runtime-layout.json /opt/runtime-layout.json
COPY --from=build /opt/s3 /opt/s3
COPY bases/s3s2st/warmup /opt/s3-warmup
RUN printf '%s\n' /opt/s3-runtime/lib > /etc/ld.so.conf.d/s3-runtime.conf && ldconfig
COPY bases/s3s2st/probe_runtime.py /opt/probe_s3_runtime.py
RUN env -u PYTHONPATH bash -c '. /opt/s3/env.sh; python3 /opt/probe_s3_runtime.py'
LABEL org.opencontainers.image.title="audio-s3-s2st-deps" \
      org.opencontainers.image.source="https://github.com/QwenAudio/S3-S2ST" \
      org.opencontainers.image.revision="1ca52ea064ec2081f12675d853525d394571bf73" \
      audio.kernel_source="vllm-project/vllm v0.11.1; compatibility requires native inference verification"
