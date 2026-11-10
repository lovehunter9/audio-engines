# audio-simuls2st deps: SimulS2ST-Omni thinker, in-process, CUDA only.
#
# Official pin is torch 2.6.0+cu124 and transformers 4.52.3. The streaming text agent lives in
# the upstream repo, which has no PyPI package, so the source is cloned at a pinned commit and
# put on PYTHONPATH. Weights stay out of the image: llm-init snapshots HA-SA-ki/SimulS2ST-Omni.
#
# This image serves speech-to-text translation only. VoiceBox, the vocoder and DualCodec are
# not installed; speech out is a later image, not a flag on this one.
#
# amd64 only. The workflow does not build arm64.
FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 git; \
    rm -rf /var/lib/apt/lists/*

# Match the base torch. A bare `pip install torchaudio` would replace it.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

ARG SIMUL_SHA=c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c
RUN set -eux; \
    git clone https://github.com/hasaki321/SimulS2ST-Omni.git /opt/simuls2st; \
    git -C /opt/simuls2st checkout "$SIMUL_SHA"; \
    git -C /opt/simuls2st submodule update --init external/SimulEval; \
    python3 -m pip install --no-cache-dir --root-user-action=ignore \
        -e /opt/simuls2st/external/SimulEval \
        "transformers==4.52.3" "peft==0.18.1" "qwen-omni-utils==0.0.8" \
        "accelerate==1.12.0" "numpy==1.26.4" "safetensors==0.7.0" \
        "soundfile==0.13.1" "sentencepiece==0.2.1" "einops==0.8.1" \
        "scipy==1.15.3" "librosa==0.11.0" \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart websockets; \
    python3 -c "\
import torch, torchaudio, transformers, peft; \
assert torch.__version__.startswith('2.6.'), torch.__version__; \
assert torch.version.cuda.startswith('12.4'), torch.version.cuda; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, \
      'torchaudio', torchaudio.__version__, 'transformers', transformers.__version__, \
      'peft', peft.__version__)"; \
    ln -sf "$(command -v python3)" /usr/local/bin/audio-python

ENV SIMULS2ST_ROOT=/opt/simuls2st \
    PYTHONPATH=/opt/simuls2st

# Import the agent the chart process loads. No weights: from_pretrained stays inside _load.
RUN python3 -c "\
from src.agents.simuleval_omni_talker_s2tt_agent import OmniTalkerStreamingS2TTAgent; \
print('agent', OmniTalkerStreamingS2TTAgent.__name__)"

LABEL org.opencontainers.image.title="audio-simuls2st-deps" \
      org.opencontainers.image.source="https://github.com/hasaki321/SimulS2ST-Omni" \
      org.opencontainers.image.revision="c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c"
