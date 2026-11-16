"""Check the release-stage native runtime without downloading weights."""
import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchvision
import transformers
import peft
from src.agents.simuleval_omni_talker_s2st_agent import OmniTalkerStreamingS2STAgent
from src.voicebox import melspec, tokenizer, voicebox_model
from src.voicebox.util import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=("amd64", "arm64"), required=True)
args = parser.parse_args()
expected = "12.9" if args.arch == "arm64" else "12.8"
assert torch.__version__.startswith("2.8."), torch.__version__
assert torchaudio.__version__.startswith("2.8."), torchaudio.__version__
assert torchvision.__version__.startswith("0.23."), torchvision.__version__
assert torch.version.cuda.startswith(expected), torch.version.cuda
assert transformers.__version__ == "4.52.3"
assert peft.__version__ == "0.18.1"
assert np.__version__ == "1.26.4"
assert shutil.which("ffmpeg")
assert not shutil.which("gcc"), "The release image must not retain the compiler"
assert Path("/opt/simuls2st/COMMIT").read_text().strip() == "c7e0b260aa21473d30ffeaa1768c624d4f9bfa3c"
assert melspec.MelSpectrogram and tokenizer.VoiceBoxAudioTokenizer and voicebox_model.VoiceBox and load_config
buffer = io.BytesIO()
sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
buffer.seek(0)
wave, rate = torchaudio.load(buffer)
assert wave.shape == (1, 1600) and rate == 16000
print(json.dumps({"arch": args.arch, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "agent": OmniTalkerStreamingS2STAgent.__name__, "audio_roundtrip": True}))
