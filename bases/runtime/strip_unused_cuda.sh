#!/bin/sh
# Drop CUDA pieces eager inference does not load. Same RUN as `pip install torch`
# or Docker keeps the fat layer.
#
# Walk torch/_C*.so with ldd. Keep a nvidia-* / cuda-* wheel only when one of
# its files is the resolved path on that chain. Soname matching is not enough:
# a leftover CUDA 13 wheel can advertise the same libcusparse.so.12 as the
# CUDA 12 wheel the current torch actually maps.
#
# Import-only NVIDIA libs: torch relocates against the soname, but this
# image never uses multi-GPU / GDS / CUPTI. Stub the host lib, then
# force-uninstall the wheel. Keep libtorch_nvshmem.so — that lives in torch.
set -eu
# Callers that `cd` into a dir they then delete leave us without a cwd;
# Ubuntu pip then dies on os.getcwd() in __main__.
cd /

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
PURGE_TOOLS=0
if ! command -v gcc >/dev/null 2>&1 || ! command -v readelf >/dev/null 2>&1; then
    apt-get update
    # libc6-dev is a Recommends of gcc; --no-install-recommends drops it and
    # then `gcc -shared` dies with `cannot find crti.o` (arm64 slim3).
    apt-get install -y --no-install-recommends gcc binutils libc6-dev
    PURGE_TOOLS=1
fi

python3 - "$SITE" <<'PY'
import os, subprocess, sys, tempfile

sitep = sys.argv[1]
stubdir = "/opt/cuda-stubs"
os.makedirs(stubdir, exist_ok=True)
open(os.path.join(stubdir, ".keep"), "w").close()


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
    return sorted(set(names))


def write_stub(real, soname):
    symbols = dyn_defined(real)
    print("stub", soname, "from", real, "symbols", len(symbols))
    src = tempfile.NamedTemporaryFile("w", suffix=".c", delete=False)
    src.write("\n".join("void %s() {}" % name for name in symbols) + "\n")
    src.close()
    dest = os.path.join("/usr/local/lib", soname)
    subprocess.check_call(
        ["gcc", "-shared", "-fPIC", "-Wl,-soname,%s" % soname, "-o", dest, src.name]
    )
    os.remove(src.name)
    staged = os.path.join(stubdir, soname)
    subprocess.check_call(["cp", "-a", dest, staged])
    short = soname.split(".so")[0] + ".so"
    for directory in ("/usr/local/lib", stubdir):
        link = os.path.join(directory, short)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(soname, link)


stub_prefixes = (
    "libnccl.so",
    "libnvshmem_host.so",
    "libcupti.so",
    "libcufile.so",
)
found = {prefix: [] for prefix in stub_prefixes}
nvidia_root = os.path.join(sitep, "nvidia")
if os.path.isdir(nvidia_root):
    for root, _, files in os.walk(nvidia_root):
        for name in files:
            for prefix in stub_prefixes:
                if name.startswith(prefix):
                    found[prefix].append(os.path.join(root, name))
                    break
for prefix, paths in found.items():
    if not paths:
        continue
    real = next((p for p in paths if not os.path.islink(p)), paths[0])
    real = os.path.realpath(real)
    write_stub(real, os.path.basename(real))

subprocess.call(["ldconfig"])
PY

freeze=$(python3 -m pip freeze)
STRIP_CANDIDATES=$(echo "$freeze" | grep -iE '^(nvidia-[^=]+|cuda-toolkit|cuda-bindings|cuda-pathfinder|triton|pytorch-triton)==' \
    | cut -d= -f1 || true)
export STRIP_CANDIDATES

python3 - "$SITE" <<'PY'
import glob, os, subprocess, sys
from collections import deque

sitep = sys.argv[1]
needed = set()
resolved_files = set()
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
            try:
                resolved_files.add(os.path.realpath(dest))
            except OSError:
                pass
print("import-time DT_NEEDED:", " ".join(sorted(needed)))

force_drop = {"triton", "pytorch-triton"}
force_drop_prefix = (
    "nvidia-nccl",
    "nvidia-nvshmem",
    "nvidia-cuda-cupti",
    "nvidia-cufile",
)
for pkg in os.environ.get("STRIP_CANDIDATES", "").split():
    if pkg in force_drop or pkg.startswith(force_drop_prefix):
        print("drop", pkg)
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
        continue
    show = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", "-f", pkg], text=True
    )
    files = []
    in_files = False
    location = ""
    for ln in show.splitlines():
        if ln.startswith("Location:"):
            location = ln.split(":", 1)[1].strip()
            continue
        if ln.startswith("Files:"):
            in_files = True
            continue
        if not in_files:
            continue
        rel = ln.strip()
        if rel:
            files.append(rel)
    hit = []
    for rel in files:
        path = os.path.join(location, rel) if location else os.path.join(sitep, rel)
        try:
            real = os.path.realpath(path)
        except OSError:
            continue
        if real in resolved_files:
            hit.append(os.path.basename(rel))
    if hit:
        print("keep", pkg, "maps", " ".join(sorted(set(hit))))
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
    if [ -d "$SITE/torch" ]; then
        rm -rf "$SITE/torch/include" "$SITE/torch/test" 2>/dev/null || true
        find "$SITE/torch" -type f \( -name '*.a' -o -name '*.h' \) -delete
    fi
fi
rm -rf /root/.cache/pip /tmp/pip-*

if [ "$PURGE_TOOLS" = 1 ]; then
    apt-get purge -y gcc binutils libc6-dev
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
