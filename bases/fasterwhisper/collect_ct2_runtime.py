# Copy libctranslate2 + its non-glibc NEEDED closure into DEST/lib.
# Used so the arm64 CT2 source build does not leave nvidia/cuda *-devel in the final image.
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

seen: set[str] = set()
queue = list(roots)
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

print("collected", len(os.listdir(LIBDIR)), "files into", LIBDIR)
