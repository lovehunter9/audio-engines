"""Stage the installed S3 runtime independently of the CUDA development image."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import site
import subprocess
from collections import deque

root = Path('/opt/s3-runtime')
python = root / 'site'
libs = root / 'lib'
python.mkdir(parents=True, exist_ok=True)
libs.mkdir(exist_ok=True)
locations = [Path(p) for p in site.getsitepackages() if Path(p).is_dir()]
for location in reversed(locations):
    shutil.copytree(location, python, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for path in python.rglob('*'):
    if path.is_symlink() and os.path.isabs(os.readlink(path)):
        original = Path(os.readlink(path))
        for location in locations:
            if original.is_relative_to(location):
                target = python / original.relative_to(location)
                path.unlink()
                path.symlink_to(target)
                break
seeds = []
for package in ('torch', 'torchaudio', 'torchvision', 'torchcodec', 'vllm'):
    seeds.extend(p for p in (python / package).rglob('*.so*') if p.is_file())
queue = deque(seeds)
seen = set()
collected = []
while queue:
    path = queue.popleft().resolve()
    if path in seen:
        continue
    seen.add(path)
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if '=>' not in line:
            continue
        parts = line.split('=>', 1)[1].strip().split()
        if not parts or not parts[0].startswith('/'):
            continue
        dependency = Path(parts[0]).resolve()
        if not dependency.exists():
            continue
        if dependency.name.startswith(("libcuda.so", "libnvidia-ml.so")):
            continue
        if any(dependency.is_relative_to(location) for location in locations):
            continue
        if dependency.is_relative_to(python):
            continue
        if str(dependency).startswith(('/usr/local/cuda', '/usr/local/lib')):
            destination = libs / parts[0].split('/')[-1]
            if not destination.exists():
                shutil.copy2(dependency, destination)
                collected.append(str(dependency))
            queue.append(dependency)
for directory in [Path('/usr/local/cuda/lib64'), *Path('/usr/local').glob('cuda-*/targets/*/lib')]:
    if not directory.is_dir():
        continue
    for pattern in ('libcudnn*.so*', 'libnvrtc-builtins.so*'):
        for dependency in directory.glob(pattern):
            if dependency.is_file() and not (libs / dependency.name).exists():
                shutil.copy2(dependency, libs / dependency.name)
                collected.append(str(dependency.resolve()))
source = Path('/opt/s3')
for path in source.rglob('*.so'):
    if path.is_symlink():
        data = path.resolve()
        path.unlink()
        shutil.copy2(data, path)
shutil.rmtree(source / '.git', ignore_errors=True)
shutil.rmtree(python / 'vllm', ignore_errors=True)

def size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())

packages = sorted([{'name':p.name,'bytes':size(p)} for p in python.iterdir() if p.is_dir()],
                  key=lambda x:x['bytes'], reverse=True)
report = {'site_bytes':size(python),'extra_libraries_bytes':size(libs),'source_bytes':size(source),
          'largest_packages':packages[:35], 'preserved_external_libraries':collected,
          'original_source_revision':(source/'COMMIT').read_text().strip()}
(root/'MANIFEST.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
for path in source.rglob('*.so'):
    with path.open('rb') as stream:
        hashlib.file_digest(stream,'sha256')
