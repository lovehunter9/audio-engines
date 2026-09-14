#!/bin/sh
# Drop CUDA pieces eager inference does not load. Same RUN as `pip install torch`
# or Docker keeps the fat layer.
#
# Keep: cublas, cudnn, cuda-runtime, nvjitlink, cufft, nvrtc, curand, plus any
# nvidia *.so on the import-time load chain of torch/_C*.so (cu128 links
# libcusparseLt; cu130 links libcufile). Drop the rest of: nccl, cusolver,
# cusparse, cusparselt, cupti, nvtx, cufile, nvshmem, triton. torch.testing
# stays — import torch pulls it.
#
# nvshmem is only pulled in through libtorch_nvshmem; drop that NEEDED and
# delete the file. nccl is different: libtorch_cuda relocates against ncclRecv
# and friends, so the soname must stay. Replace the 300 Mi wheel with a stub
# that exports the same dynamic symbols. Multi-GPU is not this image's contract.
set -eu

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
PURGE_TOOLS=0
if ! command -v patchelf >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1 \
        || ! command -v readelf >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends patchelf gcc binutils
    PURGE_TOOLS=1
fi

python3 - "$SITE" <<'PY'
import glob, os, subprocess, sys, tempfile

sitep = sys.argv[1]


def needed_of(path):
    try:
        return subprocess.check_output(
            ["patchelf", "--print-needed", path], text=True
        ).split()
    except (OSError, subprocess.CalledProcessError):
        return []


def dyn_defined(path):
    out = subprocess.check_output(["readelf", "-Ws", path], text=True)
    names = []
    for line in out.splitlines():
        if " UND " in line or " UND\t" in line:
            continue
        if " FUNC " not in line and " OBJECT " not in line:
            continue
        if " GLOBAL " not in line and " WEAK " not in line:
            continue
        name = line.split()[-1].split("@")[0]
        if name and name.isidentifier() and name not in ("_init", "_fini"):
            names.append(name)
    return names


nccl_libs = []
for root, _, files in os.walk(os.path.join(sitep, "nvidia")):
    for name in files:
        if name.startswith("libnccl.so"):
            nccl_libs.append(os.path.join(root, name))
if nccl_libs:
    real = next((p for p in nccl_libs if not os.path.islink(p)), nccl_libs[0])
    real = os.path.realpath(real)
    symbols = dyn_defined(real)
    print("stub libnccl.so.2 from", real, "symbols", len(symbols))
    # C, not C++: gcc without g++ has no cc1plus.
    lines = []
    for name in symbols:
        lines.append("void %s() {}" % name)
    src = tempfile.NamedTemporaryFile("w", suffix=".c", delete=False)
    src.write("\n".join(lines) + "\n")
    src.close()
    dest = "/usr/local/lib/libnccl.so.2"
    subprocess.check_call(
        ["gcc", "-shared", "-fPIC", "-Wl,-soname,libnccl.so.2", "-o", dest, src.name]
    )
    os.remove(src.name)
    link = "/usr/local/lib/libnccl.so"
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink("libnccl.so.2", link)
    subprocess.call(["ldconfig"])

drop_needed = (
    "libnvshmem_host.so.3",
    "libnvshmem.so.3",
    "libtorch_nvshmem.so",
)
for root, _, files in os.walk(os.path.join(sitep, "torch")):
    for name in files:
        if ".so" not in name:
            continue
        path = os.path.join(root, name)
        have = needed_of(path)
        for soname in drop_needed:
            if soname in have:
                print("patchelf --remove-needed", soname, path)
                subprocess.check_call(["patchelf", "--remove-needed", soname, path])

for path in glob.glob(os.path.join(sitep, "torch", "lib", "libtorch_nvshmem*")):
    print("delete", path)
    os.remove(path)
PY

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

if [ "$PURGE_TOOLS" = 1 ]; then
    apt-get purge -y patchelf gcc binutils
    apt-get autoremove -y --purge
    rm -rf /var/lib/apt/lists/*
fi

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
nvidia = os.path.join(s, 'nvidia')
if os.path.isdir(nvidia):
    print('nvidia parts')
    for name in sorted(os.listdir(nvidia)):
        p = os.path.join(nvidia, name)
        if not os.path.isdir(p):
            continue
        total = 0
        for root, dirs, files in os.walk(p):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        print('  %6.1f Mi  nvidia/%s' % (total / 1024 / 1024, name))
"
