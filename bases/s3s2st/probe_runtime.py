"""Check the minimal runtime without pretending a GPU-free builder can infer."""
import ast
from importlib.metadata import entry_points
from pathlib import Path
import shutil
import tempfile

import torch
import torchaudio
import torchvision
import vllm
import triton
import transformers
import flashinfer
import nvtx
import whisper

assert torch.version.cuda
assert vllm.__file__.startswith("/opt/s3/vendor/")
assert "/wrapper/caps/" not in whisper.__file__
for tool in ("nvcc", "ptxas", "fatbinary", "nvlink", "cudafe++", "bin2c", "ninja", "gcc"):
    assert shutil.which(tool), tool
root = Path(vllm.__file__).parent
for name in ("engine/omnidrca2_engine.py", "vllm_flash_attn/__init__.py", "vllm_flash_attn/flash_attn_interface.py"):
    ast.parse((root/name).read_text())
plugins = {item.name: item for item in entry_points(group="vllm.general_plugins")}
assert "s3_native" in plugins
assert callable(plugins["s3_native"].load())
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary)/"roundtrip.wav"
    torchaudio.save(str(path), torch.zeros(1, 1600), 16000)
    samples, rate = torchaudio.load(str(path))
    assert samples.shape == (1, 1600) and rate == 16000
if torch.cuda.is_available():
    from vllm.engine.omnidrca2_engine import OmniDRCA2Engine
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    assert callable(OmniDRCA2Engine) and callable(flash_attn_varlen_func)
else:
    print("No GPU in the builder: native initialization and inference require the real deployment checks")
print(torch.__version__, torchvision.__version__, triton.__version__, transformers.__version__, flashinfer.__version__, nvtx.__name__)
