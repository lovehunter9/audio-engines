"""Install the sealed ASR closure in its own CUDA12.8 interpreter environment."""
import argparse
import json
from pathlib import Path
import platform
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--lock', required=True)
parser.add_argument('--wheel-lock', default='/opt/asr-cuda-wheels.json')
args = parser.parse_args()
profile = json.loads(Path(args.lock).read_text())['profiles']['youdao-asr']
python = '/opt/venvs/youdao-asr/bin/python'
subprocess.run(['/usr/bin/python3', '-m', 'venv', '/opt/venvs/youdao-asr'], check=True)
versions = profile['resolved_transitive_versions']
cuda_torch = ['torch', 'torchaudio', 'torchvision']
architecture = platform.machine()
catalogue = json.loads(Path(args.wheel_lock).read_text())['wheels']
wheels = [row for row in catalogue if row['arch'] == architecture]
if {row['package'] for row in wheels} != set(cuda_torch) or len(wheels) != 3:
    raise RuntimeError('Missing exact official CUDA artifacts for this architecture')
for row in wheels:
    if row['base_version'] != versions[row['package']] or '#sha256=' not in row['url']:
        raise RuntimeError('CUDA artifact version or immutable hash differs from the frozen role')
subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                *[row['url'] for row in wheels]], check=True)
subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
                *[name + '==' + version for name, version in versions.items() if name not in cuda_torch]], check=True)
subprocess.run([python, '/opt/install-locked-role.py', '--lock', args.lock,
                '--role', 'youdao-asr', '--python', python], check=True)
subprocess.run([python, '-c',
    'import torch,torchaudio,torchvision; '
    'assert torch.__version__=="2.9.1+cu128"; '
    'assert torchaudio.__version__.split("+")[0]=="2.9.1"; '
    'assert torchvision.__version__.split("+")[0]=="0.24.1"; '
    'from torchaudio._extension.utils import _check_cuda_version; '
    'assert _check_cuda_version()==12080; '
    'assert torchvision.extension._check_cuda_version()==12080; '
    'assert torch.version.cuda=="12.8" and torch.backends.cuda.is_built()'], check=True)
