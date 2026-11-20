# audio-torchasr deps: offline STT from upstream research code (wrapper/caps/torch_asr.py).
#
# FireRedASR2-AED (FireRedTeam/FireRedASR2S) and MiMo-V2.5-ASR (XiaomiMiMo/MiMo-V2.5-ASR) ship
# as source trees, not packages, so each is fetched at a pinned commit and put on sys.path with a
# .pth file. FireRed's tree goes in one level down: its top package imports the whole
# VAD/LID/punctuation system, and the recogniser alone is `fireredasr2`. MiMo's `src` is renamed
# to `mimo_v25asr`, a name nothing else claims.
#
# MiMo's modeling code is written against transformers 4.49's generation internals, and FireRed's
# AED path only touches transformers through an import; one pin serves both. MiMo's audio
# tokenizer imports flash-attn, which the wrapper replaces with torch SDPA, so it is not
# installed. CUDA 12.8 torch, so Blackwell (sm_120) loads. amd64 only, like diarasr.
ARG TARGETARCH

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG TARGETARCH
ARG FIRERED_COMMIT=4e7d9aaf4482a47cec1724807026b9b151926eb5
ARG MIMO_COMMIT=876312a76ce0b06c381d8b962decbc082526a0c0

RUN set -eux; \
    if [ "${TARGETARCH}" != "amd64" ]; then \
        echo "torchasr base is amd64-only (TARGETARCH=${TARGETARCH})" >&2; \
        exit 1; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torchaudio==2.8.0" "transformers==4.49.0" "safetensors>=0.4.5" "accelerate>=1.9.0" \
        "huggingface-hub>=0.34,<1" einops kaldiio kaldi_native_fbank sentencepiece \
        soundfile numpy \
        "fastapi>=0.110" "uvicorn>=0.29" httpx python-multipart nvidia-ml-py

RUN set -eux; \
    python3 - <<EOF
import io, os, shutil, site, tarfile, urllib.request

def fetch(repo, commit, dest):
    url = "https://codeload.github.com/%s/tar.gz/%s" % (repo, commit)
    with urllib.request.urlopen(url) as r:
        tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz").extractall("/opt")
    os.rename("/opt/%s-%s" % (repo.split("/")[1], commit), dest)

fetch("FireRedTeam/FireRedASR2S", "${FIRERED_COMMIT}", "/opt/fireredasr2s")
fetch("XiaomiMiMo/MiMo-V2.5-ASR", "${MIMO_COMMIT}", "/opt/mimo")
os.rename("/opt/mimo/src", "/opt/mimo/mimo_v25asr")
sp = site.getsitepackages()[0]
open(os.path.join(sp, "fireredasr2.pth"), "w").write("/opt/fireredasr2s/fireredasr2s\n")
open(os.path.join(sp, "mimo_v25asr.pth"), "w").write("/opt/mimo\n")
EOF

RUN python3 -c "\
import shutil, sys, types, torch, torchaudio, transformers, fastapi; \
sys.modules['flash_attn'] = types.SimpleNamespace(flash_attn_varlen_func=None); \
from fireredasr2 import FireRedAsr2, FireRedAsr2Config; \
from mimo_v25asr.mimo_audio.mimo_audio import MimoAudio; \
assert shutil.which('ffmpeg'), 'no ffmpeg'; \
assert torch.version.cuda, 'torchasr deps must be CUDA torch, got %s' % torch.__version__; \
assert transformers.__version__ == '4.49.0', transformers.__version__; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)" \
    && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

LABEL org.opencontainers.image.title="audio-torchasr-deps"
