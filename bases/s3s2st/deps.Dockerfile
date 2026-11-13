FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 build-essential && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
ARG S3_REVISION=cb7439d89863e0b3524b744758b8098cc33a024e
RUN git clone https://github.com/QwenAudio/S3-S2ST.git /opt/s3 && git -C /opt/s3 checkout "$S3_REVISION"
RUN printf '%s\n' 'torch==2.8.0' 'torchaudio==2.8.0' 'numpy==1.26.4' 'scipy==1.15.3' > /opt/s3-constraints.txt && python3 -m pip install --no-cache-dir -c /opt/s3-constraints.txt '/opt/s3[inference]' scipy fastapi uvicorn httpx python-multipart websockets
COPY bases/s3s2st/extract_native.py /opt/extract_s3_native.py
RUN python3 /opt/extract_s3_native.py
RUN python3 -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 's3_native.pth').write_text('/opt/s3/native\n/opt/s3/vendor/FunASR\n/opt/s3/inference/streaming\n/opt/s3\n')"
ENV S3_ROOT=/opt/s3
RUN env PYTHONPATH=/app python3 -c "import torch; from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig,Qwen2Model,Qwen2Config; from s3_encoder import SenseVoiceEncoder; from s3_adaptor import Transformer; import funasr.models.cosyvoice.flow_matching_model,funasr.models.cosyvoice.causal_bigvgan; assert torch.__version__.startswith('2.8.'); print('S3 native modules loaded')" && ln -sf "$(command -v python3)" /usr/local/bin/audio-python
