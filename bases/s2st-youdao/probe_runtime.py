"""Build/import check only; hardware inference is a separate acceptance gate."""
import ctypes
import importlib.util
from pathlib import Path
import sys
import torch
import torchaudio
import transformers
import tokenizers
import onnxruntime
import ming_tts
import simulinfer
assert torch.__version__=='2.13.0+cu130'
assert torchaudio.__version__=='2.11.0+cu130'
assert transformers.__version__=='4.46.3'
assert tokenizers.__version__=='0.20.3'
assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
assert importlib.util.find_spec('vllm') is None
assert importlib.util.find_spec('vllm_omni') is None
sys.path.insert(0,'/opt/asr-private')
sys.path.insert(0,'/opt/youdao/r2t2')
import asr_transformers
import asr_tokenizers
from qwen_asr import Qwen3ASRModel
from r2t2 import R2T2ASRModel
assert asr_transformers.__version__=='4.57.6'
assert asr_tokenizers.__version__=='0.22.2'
assert transformers.__version__=='4.46.3'
from sanic import Sanic
from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
import kaldi_native_fbank
from inference.translation import join_translation_segments
import imageio_ffmpeg
assert Path(imageio_ffmpeg.get_ffmpeg_exe()).exists()
ctypes.CDLL('/opt/llama/lib/libaudio-t3po-entry.so')
print('MING dependencies retained; private ASR dependencies and native entry import successfully')
