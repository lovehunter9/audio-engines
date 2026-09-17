# Shared OpenVINO compile for TTS one-step modules. Loops stay in official Python.
import logging
import os

log = logging.getLogger("audio-tts-ov")


def is_breeze_ov():
    return (os.environ.get("AUDIO_BASE") or "").strip() == "breezeov"


def is_firered_ov():
    return (os.environ.get("AUDIO_BASE") or "").strip() == "fireredov"


def require_gpu():
    from . import ovutil

    device = ovutil.device()
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel") and device.upper() != "GPU":
        raise RuntimeError("tts ov on %s must use GPU, got %s" % (mode, device))
    if device.upper() != "GPU":
        raise RuntimeError("tts ov requires OpenVINO GPU, got %s" % device)
    return device


def ir_paths(src, name, stamp):
    ir_dir = os.path.join(src, "openvino")
    return ir_dir, os.path.join(ir_dir, name + ".xml"), os.path.join(ir_dir, stamp)


def compile_module(mod, example, xml, stamp, device):
    """Export a torch nn.Module once, compile on GPU, return a callable(np)->np."""
    import numpy as np
    import openvino as ov

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        if not isinstance(example, (tuple, list)):
            example = (example,)
        log.info("exporting %s example=%s", xml, [tuple(t.shape) for t in example])
        ov_model = ov.convert_model(mod, example_input=example)
        ov.save_model(ov_model, xml)
        open(stamp, "w").close()
    core = ov.Core()
    compiled = core.compile_model(xml, device)
    log.info("compiled %s on %s", xml, device)

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    return run


def force_cpu_torch_device():
    """Official FireRed __init__ hardcodes torch.device('cuda'). Lie for the load.

    Must stay a type: transformers 5.6 does isinstance(device_map, torch.device).
    A function replacement raises TypeError and the engine never leaves loading.
    """
    import torch

    real = torch.device

    class _Meta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, real)

        def __subclasscheck__(cls, subclass):
            return subclass is cls or issubclass(subclass, real)

    class device(metaclass=_Meta):
        def __new__(cls, *args, **kwargs):
            x = args[0] if args else kwargs.get("type", "cpu")
            if x == "cuda" or (isinstance(x, str) and x.startswith("cuda")):
                return real("cpu")
            return real(*args, **kwargs)

    torch.device = device
    return lambda: setattr(torch, "device", real)


def allow_breeze_fast_on_cpu():
    """FastBreezeStreamingRuntime asserts CUDA. Eager path works on CPU if we skip that."""
    import models.fast_streaming as fs

    if getattr(fs.FastBreezeStreamingRuntime.__init__, "_ov_cpu", False):
        return
    orig = fs.FastBreezeStreamingRuntime.__init__
    real_get = fs._get_device

    class _Lie:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        @property
        def type(self):
            return "cuda"

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __str__(self):
            return str(self._inner)

    def init(self, model, audio_tokenizer, config=None, *, tokenizer=None):
        fs._get_device = lambda m: _Lie(real_get(m))
        try:
            orig(self, model, audio_tokenizer, config, tokenizer=tokenizer)
        finally:
            fs._get_device = real_get
        self.device = real_get(model)

    init._ov_cpu = True
    fs.FastBreezeStreamingRuntime.__init__ = init
