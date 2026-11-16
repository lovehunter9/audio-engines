FROM docker.io/vllm/vllm-openai:v0.11.1 AS build

ARG S3_REVISION=1ca52ea064ec2081f12675d853525d394571bf73
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git init /opt/s3 \
    && git -C /opt/s3 remote add origin https://github.com/QwenAudio/S3-S2ST.git \
    && git -C /opt/s3 fetch --depth 1 origin "$S3_REVISION" \
    && git -C /opt/s3 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/s3 rev-parse HEAD)" = "$S3_REVISION" \
    && printf '%s\n' "$S3_REVISION" > /opt/s3/COMMIT

RUN python3 -m pip install --no-cache-dir -e '/opt/s3[inference]' \
        fastapi uvicorn httpx python-multipart websockets "torchcodec>=0.8,<0.9" "opencv-python-headless==4.11.0.86" "better-exceptions==0.3.3" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

COPY bases/s3s2st/collect_kernels.py /opt/link_s3_kernels.py
RUN python3 /opt/link_s3_kernels.py \
    && bash -c '. /opt/s3/env.sh; python3 -c "import ast, pathlib, vllm, torch; ast.parse(pathlib.Path(vllm.__file__).parent.joinpath(\"engine/omnidrca2_engine.py\").read_text()); assert vllm.__file__.startswith(\"/opt/s3/vendor/\"); assert torch.__version__.startswith(\"2.9.\"); print(vllm.__file__, torch.__version__)"'

COPY bases/s3s2st/collect_runtime.py /opt/collect_s3_runtime.py
RUN python3 /opt/collect_s3_runtime.py

FROM ubuntu:24.04 AS release
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/s3-runtime/site \
    LD_LIBRARY_PATH=/opt/s3-runtime/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip gcc g++ libc6-dev \
        ffmpeg libsndfile1 libnuma1 libgomp1 libgl1 libsm6 libxext6 \
        libibverbs1 librdmacm1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python
COPY --from=build /opt/s3-runtime /opt/s3-runtime
COPY --from=build /opt/s3 /opt/s3
RUN bash -c '. /opt/s3/env.sh; python3 -c "import torch, torchaudio, torchvision, vllm, triton, transformers; assert torch.version.cuda; assert vllm.__file__.startswith(\"/opt/s3/vendor/\"); print(torch.__version__, torch.version.cuda, vllm.__file__, triton.__version__, transformers.__version__)"'
LABEL org.opencontainers.image.title="audio-s3-s2st-deps" \
      org.opencontainers.image.source="https://github.com/QwenAudio/S3-S2ST" \
      org.opencontainers.image.revision="1ca52ea064ec2081f12675d853525d394571bf73" \
      audio.kernel_source="vllm-project/vllm v0.11.1; compatibility requires native inference verification"
