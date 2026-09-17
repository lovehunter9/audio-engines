# CUDA-built CT2 required (cublas .so.12 or .so.13 on the loader); CUDA torch is not.
import subprocess

import torch
import ctranslate2 as c

assert torch.version.cuda is None, (
    "CUDA torch leaked into the FasterWhisper image: %s" % torch.__version__
)

loader = subprocess.check_output(["ldconfig", "-p"], text=True)
if "libcublas.so.12" not in loader and "libcublas.so.13" not in loader:
    raise SystemExit("fw deps: libcublas.so is not on the loader path")
if "libcudnn.so" not in loader and "libcudnn_" not in loader:
    raise SystemExit("fw deps: libcudnn.so is not on the loader path")
print("loader has libcublas and libcudnn")

try:
    types = set(c.get_supported_compute_types("cuda"))
    assert types, "empty cuda compute types"
    print("ct2", c.__version__, "cuda_types", sorted(types))
    print("cpu torch", torch.__version__, "for TransformersConverter only")
except Exception as e:
    msg = str(e)
    if "not compiled with CUDA" in msg:
        raise SystemExit("fw deps: CPU-only ctranslate2: " + msg)
    print("ct2", c.__version__, "cuda_probe deferred_no_driver:", msg)
    print("cpu torch", torch.__version__, "for TransformersConverter only")
