FROM docker.io/vllm/vllm-openai:v0.11.1

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
        fastapi uvicorn httpx python-multipart websockets \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

COPY bases/s3s2st/collect_kernels.py /opt/link_s3_kernels.py
RUN python3 /opt/link_s3_kernels.py \
    && bash -c '. /opt/s3/env.sh; python3 -c "from vllm.engine.omnidrca2_engine import OmniDRCA2Engine; import vllm, torch; assert vllm.__file__.startswith(\"/opt/s3/vendor/\"); assert torch.__version__.startswith(\"2.9.\"); print(OmniDRCA2Engine.__module__, torch.__version__)"'

LABEL org.opencontainers.image.title="audio-s3-s2st-deps" \
      org.opencontainers.image.source="https://github.com/QwenAudio/S3-S2ST" \
      org.opencontainers.image.revision="1ca52ea064ec2081f12675d853525d394571bf73" \
      audio.kernel_source="vllm-project/vllm v0.11.1; compatibility requires native inference verification"
