#!/bin/sh
# Drop CUDA pieces eager inference does not load. Same RUN as `pip install torch`
# or Docker keeps the fat layer.
#
# Keep: cublas, cudnn, cuda-runtime, nvjitlink, cufft, nvrtc, curand, plus any
# nvidia *.so on the import-time load chain of torch/_C*.so (cu128 links
# libcusparseLt; cu130 links libcufile). Drop the rest of: nccl, cusolver,
# cusparse, cusparselt, cupti, nvtx, cufile, triton. torch.testing stays —
# import torch pulls it.
#
# nccl: libtorch_cuda relocates against ncclRecv, so the soname must stay.
# nvshmem: cu130 libtorch_python relocates against
# c10d::nvshmem_extension::is_nvshmem_available — keep libtorch_nvshmem, stub
# only the host wheel. Replace those wheels with stubs that export the same
# dynamic symbols, then force-uninstall the pip package. Multi-GPU is not
# this image's contract.
set -eu

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
    """[(name, version or None), ...] — keep @VER so ld.so does not abort."""
    out = subprocess.check_output(["readelf", "-Ws", path], text=True)
    pairs = []
    for line in out.splitlines():
        if " UND " in line or " UND\t" in line:
            continue
        if " FUNC " not in line and " OBJECT " not in line:
            continue
        if " GLOBAL " not in line and " WEAK " not in line:
            continue
        raw = line.split()[-1]
        if "@@" in raw:
            name, ver = raw.split("@@", 1)
        elif "@" in raw:
            name, ver = raw.split("@", 1)
        else:
            name, ver = raw, None
        if name and name.isidentifier() and name not in ("_init", "_fini"):
            pairs.append((name, ver))
    return sorted(set(pairs))


def write_stub(real, soname):
    pairs = dyn_defined(real)
    names = sorted({name for name, _ in pairs})
    print("stub", soname, "from", real, "symbols", len(names), "pairs", len(pairs))
    src = tempfile.NamedTemporaryFile("w", suffix=".c", delete=False)
    src.write("\n".join("void %s() {}" % name for name in names) + "\n")
    src.close()
    by_ver = {}
    unversioned = set()
    for name, ver in pairs:
        if ver:
            by_ver.setdefault(ver, set()).add(name)
            unversioned.discard(name)
        elif name not in {n for n, v in pairs if v}:
            unversioned.add(name)
    map_lines = []
    versions = sorted(by_ver)
    for i, ver in enumerate(versions):
        glob = "\n".join("    %s;" % s for s in sorted(by_ver[ver]))
        inherit = " %s" % versions[i - 1] if i else ""
        extra = "\n  local: *;" if i == 0 and not unversioned else ""
        map_lines.append("%s {\n  global:\n%s%s\n}%s;" % (ver, glob, extra, inherit))
    if unversioned:
        glob = "\n".join("    %s;" % s for s in sorted(unversioned))
        map_lines.append("{\n  global:\n%s\n  local: *;\n};" % glob)
    mapf = tempfile.NamedTemporaryFile("w", suffix=".map", delete=False)
    mapf.write("\n".join(map_lines) + "\n")
    mapf.close()
    dest = os.path.join("/usr/local/lib", soname)
    subprocess.check_call(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-Wl,-soname,%s" % soname,
            "-Wl,--version-script,%s" % mapf.name,
            "-o",
            dest,
            src.name,
        ]
    )
    os.remove(src.name)
    os.remove(mapf.name)
    staged = os.path.join(stubdir, soname)
    subprocess.check_call(["cp", "-a", dest, staged])
    short = soname.split(".so")[0] + ".so"
    for directory in ("/usr/local/lib", stubdir):
        link = os.path.join(directory, short)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(soname, link)


def first_real(prefix):
    found = []
    for root, _, files in os.walk(os.path.join(sitep, "nvidia")):
        for name in files:
            if name.startswith(prefix):
                found.append(os.path.join(root, name))
    if not found:
        return None
    real = next((p for p in found if not os.path.islink(p)), found[0])
    return os.path.realpath(real)


nccl = first_real("libnccl.so")
if nccl:
    write_stub(nccl, "libnccl.so.2")

host = first_real("libnvshmem_host.so")
if host:
    write_stub(host, "libnvshmem_host.so.3")
plain = first_real("libnvshmem.so")
if plain:
    write_stub(plain, "libnvshmem.so.3")

subprocess.call(["ldconfig"])
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
    if pkg in force_drop or pkg.startswith("nvidia-nccl") or pkg.startswith("nvidia-nvshmem"):
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
