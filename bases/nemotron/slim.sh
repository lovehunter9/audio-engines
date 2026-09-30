#!/bin/sh
# Nemotron only, after the shared strip: drop what the engine never maps, stub what it maps but never calls.
# Evidence: /proc/<engine>/maps after offline, streaming and a 4h job, on 5090 (cu128) and GB10 (cu130).
set -eu
cd /
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

apt-get update
apt-get install -y --no-install-recommends gcc binutils libc6-dev

python3 - "$SITE" <<'PY'
import glob, os, shutil, subprocess, sys, tempfile

site = sys.argv[1]
freed = 0


def gone(path):
    global freed
    if os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        freed += sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
        shutil.rmtree(path)
    elif os.path.exists(path):
        freed += os.path.getsize(path)
        os.remove(path)
    else:
        return
    print("drop", os.path.relpath(path, site))


# Never mapped: cuDNN's legacy sub-libraries (torch drives the graph API), profiler and linalg libs.
# triton/backends/amd stays: triton imports every backend package when it enumerates them.
for pat in ("nvidia/cudnn/lib/libcudnn_adv.so*", "nvidia/cudnn/lib/libcudnn_ops.so*",
            "nvidia/cudnn/lib/libcudnn_cnn.so*", "nvidia/cudnn/lib/libcudnn_ext.so*",
            "nvidia/*/lib/libnvperf_host.so*", "nvidia/*/lib/libnvperf_target.so*",
            "nvidia/*/lib/libcheckpoint.so*", "nvidia/*/lib/libcufftw.so*",
            "torch/lib/libtorch_cuda_linalg.so",
            "triton/backends/nvidia/lib/cupti*",
            "triton/backends/nvidia/bin/cuobjdump*", "triton/backends/nvidia/bin/nvdisasm*"):
    for path in glob.glob(os.path.join(site, pat)):
        gone(path)


def symbols(path):
    out = subprocess.check_output(["readelf", "-W", "--dyn-syms", path], text=True)
    funcs, objs = set(), {}
    for line in out.splitlines():
        f = line.split()
        if len(f) < 8 or f[6] == "UND" or f[4] not in ("GLOBAL", "WEAK"):
            continue
        name = f[7].split("@")[0]
        if not name.isidentifier() or name in ("_init", "_fini"):
            continue
        if f[3] == "FUNC":
            funcs.add(name)
        elif f[3] == "OBJECT":
            objs[name] = max(1, int(f[2], 0))
    return sorted(funcs), objs


def stub(real):
    # Same file name, so torch's RPATH still resolves it; any call aborts loudly instead of computing nothing.
    global freed
    soname = os.path.basename(real)
    funcs, objs = symbols(real)
    src = tempfile.NamedTemporaryFile("w", suffix=".c", delete=False)
    src.write("#include <stdio.h>\n#include <stdlib.h>\n")
    src.write("static void die(const char *f) { fprintf(stderr, \"[nemotron-slim] stubbed %s called\\n\", f); abort(); }\n")
    for name in funcs:
        src.write("void %s(void) { die(\"%s\"); }\n" % (name, name))
    for name, size in objs.items():
        src.write("char %s[%d];\n" % (name, size))
    src.close()
    size = os.path.getsize(real)
    out = real + ".stub"
    subprocess.check_call(["gcc", "-shared", "-fPIC", "-O1", "-Wl,-soname,%s" % soname, "-o", out, src.name])
    os.remove(src.name)
    os.replace(out, real)
    freed += size - os.path.getsize(real)
    print("stub", os.path.relpath(real, site), "funcs", len(funcs), "objs", len(objs))


# Mapped because libtorch_cuda links them, never called: sparse and semi-structured kernels, curand.
for pat in ("nvidia/cusparselt/lib/libcusparseLt.so*", "nvidia/*/lib/libcusparse.so*",
            "nvidia/*/lib/libcurand.so*"):
    for path in glob.glob(os.path.join(site, pat)):
        if not os.path.islink(path):
            stub(path)

print("nemotron slim freed %.0f MB" % (freed / 1e6))
PY

apt-get purge -y --auto-remove gcc binutils libc6-dev
rm -rf /var/lib/apt/lists/*
