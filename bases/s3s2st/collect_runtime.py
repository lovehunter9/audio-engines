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
for pattern in ('flashinfer_jit_cache*', '__pycache__'):
    for unused in python.glob(pattern):
        if unused.is_dir():
            shutil.rmtree(unused)

cuda = root / 'cuda'
cuda_source = Path('/usr/local/cuda').resolve()
required_tools = ('nvcc', 'nvcc.profile', 'ptxas', 'fatbinary', 'nvlink', 'cudafe++', 'bin2c')
(cuda / 'bin').mkdir(parents=True)
for name in required_tools:
    original = cuda_source / 'bin' / name
    if not original.is_file():
        raise FileNotFoundError('Required native CUDA JIT tool: ' + str(original))
    shutil.copy2(original, cuda / 'bin' / name)
shutil.copytree(cuda_source / 'nvvm', cuda / 'nvvm', symlinks=False, dirs_exist_ok=True)
for target in (cuda_source / 'targets').iterdir():
    if not target.is_dir():
        continue
    shutil.copytree(target / 'include', cuda / 'targets' / target.name / 'include', symlinks=False, dirs_exist_ok=True)
    (cuda / 'include').symlink_to(Path('targets') / target.name / 'include')
    destination = cuda / 'targets' / target.name / 'lib'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('libcudadevrt.a', 'libcudart_static.a', 'libculibos.a'):
        if (target / 'lib' / name).is_file():
            shutil.copy2(target / 'lib' / name, destination / name)
    for library in (target / 'lib').glob('libcudart.so*'):
        if library.is_file():
            shutil.copy2(library, destination / library.name)
    (cuda / 'lib64').symlink_to(destination.relative_to(cuda))
    stub = target / 'lib/stubs/libcuda.so'
    if stub.is_file():
        (destination / 'stubs').mkdir(exist_ok=True)
        shutil.copy2(stub, destination / 'stubs/libcuda.so')

for path in python.rglob('*'):
    if path.is_symlink() and os.path.isabs(os.readlink(path)):
        original = Path(os.readlink(path))
        for location in locations:
            if original.is_relative_to(location):
                target = python / original.relative_to(location)
                path.unlink()
                path.symlink_to(target)
                break
seeds = list((python / 'torch').glob('_C*.so'))
seeds.append(python / 'torch/lib/libtorch_global_deps.so')
for package in ('torchaudio', 'torchvision', 'torchcodec', 'vllm'):
    seeds.extend(p for p in (python / package).rglob('*.so*') if p.is_file())
link_environment = dict(os.environ)
link_paths = [str(python / 'torch/lib'), *[str(p) for p in (python / 'nvidia').glob('*/lib')],
              *[str(p) for location in locations for p in (location / 'nvidia').glob('*/lib')]]
link_environment['LD_LIBRARY_PATH'] = ':'.join([*link_paths, os.environ.get('LD_LIBRARY_PATH', '')])
required_cuda_packages = {'cudnn', 'cuda_nvrtc', 'nvjitlink', 'nccl', 'cuda_runtime'}
queue = deque(seeds)
seen = set()
collected = []


def collect_library(original):
    dependency = original.resolve()
    canonical = libs / dependency.name
    if not canonical.exists():
        shutil.copy2(dependency, canonical)
        collected.append(str(dependency))
        queue.append(dependency)
    alias = libs / original.name
    if alias != canonical and not alias.exists():
        alias.symlink_to(canonical.name)


for directory in [Path('/usr/local/cuda/lib64'), *Path('/usr/local').glob('cuda-*/targets/*/lib')]:
    if not directory.is_dir():
        continue
    for pattern in ('libcudnn*.so*', 'libnvrtc-builtins.so*'):
        for dependency in directory.glob(pattern):
            if dependency.is_file():
                collect_library(dependency)
while queue:
    path = queue.popleft().resolve()
    if path in seen:
        continue
    seen.add(path)
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, env=link_environment)
    for line in result.stdout.splitlines():
        if '=>' not in line:
            continue
        parts = line.split('=>', 1)[1].strip().split()
        if not parts or not parts[0].startswith('/'):
            continue
        original = Path(parts[0])
        dependency = original.resolve()
        if not dependency.exists():
            continue
        if dependency.name.startswith(("libcuda.so", "libnvidia-ml.so")):
            continue
        if dependency.is_relative_to(python / 'nvidia'):
            required_cuda_packages.add(dependency.relative_to(python / 'nvidia').parts[0])
            queue.append(dependency)
            continue
        located = False
        for location in locations:
            if dependency.is_relative_to(location / 'nvidia'):
                required_cuda_packages.add(dependency.relative_to(location / 'nvidia').parts[0])
                queue.append(dependency)
                located = True
                break
        if located:
            continue
        if any(dependency.is_relative_to(location) for location in locations):
            queue.append(dependency)
            continue
        if dependency.is_relative_to(python):
            queue.append(dependency)
            continue
        if str(dependency).startswith(('/usr/local/cuda', '/usr/local/lib')):
            collect_library(original)
removed_cuda_packages = []
for directory in (python / 'nvidia').iterdir():
    if directory.is_dir() and directory.name not in required_cuda_packages and directory.name != '__pycache__':
        removed_cuda_packages.append(directory.name)
        shutil.rmtree(directory)
shutil.rmtree(python / 'nvidia_cutlass_dsl', ignore_errors=True)
(python / 'nvidia_cutlass_dsl_packages.pth').unlink(missing_ok=True)
source = Path('/opt/s3')
patches = json.loads((source / 'PATCHES.json').read_text())
changed = subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
if set(changed) != {patch['path'] for patch in patches}:
    raise RuntimeError('Unexpected modification to the pinned original source')
for patch in patches:
    original = subprocess.check_output(['git', '-C', str(source), 'show', 'HEAD:' + patch['path']])
    updated = (source / patch['path']).read_bytes()
    if hashlib.sha256(original).hexdigest() != patch['before_sha256'] or hashlib.sha256(updated).hexdigest() != patch['after_sha256']:
        raise RuntimeError('The declared original-source compatibility patch changed')
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
report = {'site_bytes':size(python),'extra_libraries_bytes':size(libs),'source_bytes':size(source),'jit_toolchain_bytes':size(cuda),
          'largest_packages':packages[:35], 'preserved_external_libraries':collected,
          'original_source_revision':(source/'COMMIT').read_text().strip(),
          'tracked_source_unmodified':False, 'compatibility_patches':patches, 'retained_cuda_packages':sorted(required_cuda_packages),
          'removed_cuda_packages':removed_cuda_packages}
(root/'MANIFEST.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
for record in json.loads((source/'KERNELS.json').read_text())['libraries']:
    path = source / 'vendor/vllm' / record['path']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != record['sha256']:
        raise RuntimeError("The original compiled dependency changed: " + record['path'])
print("Original compiled dependency hashes verified")
