# CUDA-built CT2 is required. CUDA torch is not: decode is CT2, convert uses CPU torch.
# Driver-missing on CI runners is OK; a CPU-only CT2 wheel is not.
import ctranslate2 as c
import torch

assert torch.version.cuda is None, (
    "CUDA torch leaked into the FasterWhisper image: %s" % torch.__version__
)
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
