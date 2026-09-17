# Copy libctranslate2 + its CUDA NEEDED closure into DEST/lib; harvest cublas even when ldd says not found.
import glob
import os
import shutil
import subprocess
import sys

DEST = sys.argv[1]
LIBDIR = os.path.join(DEST, "lib")
os.makedirs(LIBDIR, exist_ok=True)

SKIP_EXACT = {
    "linux-vdso.so.1",
    "ld-linux-aarch64.so.1",
    "ld-linux-x86-64.so.2",
}
SKIP_PREFIX = (
    "libc.so",
    "libm.so",
    "libdl.so",
    "librt.so",
    "libpthread.so",
    "libgcc_s.so",
    "libstdc++.so",
    "libopenblas",  # image OpenBLAS; a copy would steal sbgemm_ from CPU torch
    "libblas.so",
    "liblapack.so",
    "libgfortran",
    "libquadmath",
    "libcuda.so",  # driver; nvidia-container-runtime injects it on the node
)

# CT2 encode dlopens these; copy even when ldd still says not found.
CUDA_RUNTIME_PREFIXES = (
    "libcublas.so",
    "libcublasLt.so",
    "libcudart.so",
    "libcudnn.so",
    "libcudnn_",
    "libnvJitLink.so",
    "libnvjitlink.so",
)


def is_skip(name: str) -> bool:
    if name in SKIP_EXACT:
        return True
    return any(name.startswith(p) for p in SKIP_PREFIX)


def copy_into(path: str) -> None:
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        return
    dest_real = os.path.join(LIBDIR, os.path.basename(real))
    if not os.path.exists(dest_real):
        shutil.copy2(real, dest_real)
    link_name = os.path.basename(path)
    dest_link = os.path.join(LIBDIR, link_name)
    if link_name != os.path.basename(real) and not os.path.lexists(dest_link):
        os.symlink(os.path.basename(real), dest_link)


roots: list[str] = []
for d in (
    "/usr/local/lib",
    "/usr/lib",
    "/usr/local/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
):
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        if fn.startswith("libctranslate2"):
            roots.append(os.path.join(d, fn))

try:
    import ctranslate2

    pkg = os.path.dirname(ctranslate2.__file__)
    roots.extend(glob.glob(os.path.join(pkg, "*.so")))
except ImportError:
    pass

if not roots:
    raise SystemExit("collect_ct2_runtime: no libctranslate2 or ctranslate2*.so found")


def is_cuda_runtime_lib(name: str) -> bool:
    return any(name.startswith(p) for p in CUDA_RUNTIME_PREFIXES)


def nvidia_wheel_libs() -> list[str]:
    found: list[str] = []
    try:
        import site

        roots_n = [os.path.join(p, "nvidia") for p in site.getsitepackages()]
    except Exception:
        roots_n = []
    for nvidia_root in roots_n:
        if not os.path.isdir(nvidia_root):
            continue
        for dirpath, _, files in os.walk(nvidia_root):
            for name in files:
                if not is_cuda_runtime_lib(name):
                    continue
                if ".so" not in name:
                    continue
                found.append(os.path.join(dirpath, name))
    return found


def system_cudnn_libs() -> list[str]:
    # CUDNN 9 backends are dlopened; ldd on libctranslate2 only sees the stub.
    found: list[str] = []
    for pattern in (
        "/usr/lib/aarch64-linux-gnu/libcudnn*.so*",
        "/usr/lib/x86_64-linux-gnu/libcudnn*.so*",
        "/usr/lib/libcudnn*.so*",
        "/usr/local/cuda/lib64/libcudnn*.so*",
        "/usr/local/cuda/targets/*/lib/libcudnn*.so*",
    ):
        found.extend(glob.glob(pattern))
    return found


seen: set[str] = set()
queue = list(roots)
queue.extend(nvidia_wheel_libs())
queue.extend(system_cudnn_libs())
while queue:
    path = queue.pop()
    if not path or path in seen or not os.path.exists(path):
        continue
    seen.add(path)
    if is_skip(os.path.basename(path)):
        continue
    copy_into(path)
    try:
        out = subprocess.check_output(
            ["ldd", os.path.realpath(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        continue
    for line in out.splitlines():
        if "=>" not in line:
            continue
        right = line.split("=>", 1)[1].strip().split()
        if not right or right[0] == "not":
            continue
        queue.append(right[0])

copied = os.listdir(LIBDIR)
print("collected", len(copied), "files into", LIBDIR)
if not any(name.startswith("libcublas.so") for name in copied):
    raise SystemExit("collect_ct2_runtime: libcublas.so* missing — CT2 encode will 500")
if not any(name.startswith("libcudnn") for name in copied):
    raise SystemExit("collect_ct2_runtime: libcudnn* missing — CT2 encode will abort")
