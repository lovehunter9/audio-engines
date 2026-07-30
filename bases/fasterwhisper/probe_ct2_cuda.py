# Require CUDA-built CT2 + CUDA torch; driver-missing on CI runners is OK, CPU-only CT2 is not.
import ctranslate2 as c
import torch

assert torch.version.cuda, "arm64 fw deps need CUDA torch, got %s" % (torch.__version__,)
try:
    types = set(c.get_supported_compute_types("cuda"))
    assert types, "empty cuda compute types"
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("ct2", c.__version__, "cuda_types", sorted(types))
except Exception as e:
    msg = str(e)
    if "not compiled with CUDA" in msg:
        raise SystemExit("arm64 fw deps: CPU-only ctranslate2: " + msg)
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("ct2", c.__version__, "cuda_probe deferred_no_driver:", msg)
