#!/bin/sh
# Drop CUDA pieces eager inference does not load. Same RUN as `pip install torch`
# or Docker keeps the fat layer.
#
# Keep: cublas, cudnn, cuda-runtime, nvjitlink, cufft, nvrtc, curand, plus any
# nvidia *.so on the import-time load chain of torch/_C*.so (cu128 links
# libcusparseLt; cu130 links libcufile). Drop the rest of: nccl, cusolver,
# cusparse, cusparselt, cupti, nvtx, cufile, nvshmem, triton. torch.testing
# stays — import torch pulls it.
set -eu

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
freeze=$(python3 -m pip freeze)
STRIP_CANDIDATES=$(echo "$freeze" | grep -iE '^(nvidia-(nccl|cusolver|cusparse|cusparselt|cuda-cupti|nvtx|cufile|nvshmem)[^[:space:]=]*|triton|pytorch-triton)==' \
    | cut -d= -f1 || true)
export STRIP_CANDIDATES

python3 - "$SITE" <<'PY'
import glob, os, subprocess, sys
from collections import deque

sitep = sys.argv[1]
needed = set()
seen = set()
queue = deque(glob.glob(os.path.join(sitep, "torch", "_C*.so")))
while queue:
    path = queue.popleft()
    if path in seen or not os.path.isfile(path):
        continue
    seen.add(path)
    try:
        out = subprocess.check_output(["ldd", path], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        out = getattr(exc, "output", "") or ""
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("lib"):
            needed.add(parts[0])
        if "=>" not in line:
            continue
        resolved = line.split("=>", 1)[1].strip().split()
        if not resolved:
            continue
        dest = resolved[0]
        if dest.startswith(sitep) and ".so" in os.path.basename(dest):
            queue.append(dest)
print("import-time DT_NEEDED:", " ".join(sorted(needed)))

force_drop = {"triton", "pytorch-triton"}
for pkg in os.environ.get("STRIP_CANDIDATES", "").split():
    if pkg in force_drop:
        print("drop", pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
        continue
    show = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", "-f", pkg], text=True
    )
    sonames = set()
    in_files = False
    for ln in show.splitlines():
        if ln.startswith("Files:"):
            in_files = True
            continue
        if not in_files:
            continue
        rel = ln.strip()
        if not rel:
            continue
        base = os.path.basename(rel)
        if ".so" in base:
            sonames.add(base)
    hit = sonames & needed
    if hit:
        print("keep", pkg, "provides", " ".join(sorted(hit)))
        continue
    print("drop", pkg)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
PY

if [ -d "$SITE" ]; then
    find "$SITE" -type d -name __pycache__ -print0 | xargs -0 -r rm -rf
    if [ -d "$SITE/nvidia" ]; then
        find "$SITE/nvidia" -type d -name include -print0 | xargs -0 -r rm -rf
        find "$SITE/nvidia" -type f \( -name '*.a' -o -name '*.h' \) -delete
    fi
    rm -rf "$SITE/torch/test" 2>/dev/null || true
fi
rm -rf /root/.cache/pip /tmp/pip-*

python3 -c "\
import torch, torchaudio
assert torch.version.cuda, 'strip_unused_cuda dropped CUDA torch'
print('stripped torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

python3 -c "\
import os, site
s = site.getsitepackages()[0]
print('site', s)
for name in sorted(os.listdir(s)):
    if name.startswith(('nvidia', 'torch', 'triton')):
        p = os.path.join(s, name)
        if os.path.isdir(p):
            total = 0
            for root, dirs, files in os.walk(p):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            print('  %6.1f Mi  %s' % (total / 1024 / 1024, name))
"
