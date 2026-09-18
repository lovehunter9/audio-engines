# Shared OpenVINO compile for TTS one-step modules. Loops stay in official Python.
# Causal AR (Breeze backbone / FireRed Qwen3) is compiled here too: the official
# Python loop still drives sampling, but each transformer step runs on GPU.
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


def flatten_kv(cache):
    """Turn an HF Cache / legacy tuple into [k0, v0, k1, v1, ...]."""
    if cache is None:
        return []
    layers = cache.layers if hasattr(cache, "layers") else cache
    out = []
    for item in layers:
        if hasattr(item, "keys"):
            out.extend([item.keys, item.values])
        else:
            out.extend([item[0], item[1]])
    return out


def unflatten_kv(flat, n_layers):
    return tuple((flat[2 * i], flat[2 * i + 1]) for i in range(n_layers))


def kv_meta(config):
    n_layers = int(config.num_hidden_layers)
    n_kv = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    return n_layers, n_kv, int(head_dim)


def _dynamize_time(ov_model):
    """Prefill embeds are [B,T,H]; cached K/V are [B,kv,T,D]. T must grow."""
    mapping = {}
    for inp in ov_model.inputs:
        shape = inp.get_partial_shape()
        if shape.rank.is_dynamic:
            continue
        rank = shape.rank.get_length()
        if rank == 2 or rank == 3:
            shape[1] = -1
        elif rank == 4:
            shape[2] = -1
        else:
            continue
        try:
            mapping[inp.any_name] = shape
        except Exception:
            mapping[inp] = shape
    if mapping:
        ov_model.reshape(mapping)
    return ov_model


def compile_causal(mod, example, xml, stamp, device):
    """Like compile_module, but dynamize the time axis before save."""
    import numpy as np
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        if not isinstance(example, (tuple, list)):
            example = (example,)
        log.info("exporting %s example=%s", xml, [tuple(t.shape) for t in example])
        mod.eval()
        with torch.inference_mode():
            ov_model = _dynamize_time(ov.convert_model(mod, example_input=example))
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


def causal_full_module(inner):
    """Layer stack only. Official backbone.forward calls create_causal_mask; that
    traces into functorch vmap and dies with unordered_map::at (intel3/intel5)."""
    import torch
    import torch.nn as nn

    if not all(hasattr(inner, n) for n in ("layers", "norm", "rotary_emb")):
        raise RuntimeError("backbone missing layers/norm/rotary_emb for stack export")
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        cfg._attn_implementation = "eager"
    n_layers = int(getattr(cfg, "num_hidden_layers", len(inner.layers)))

    class _Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds, attention_mask):
            t = inputs_embeds.shape[1]
            pos = torch.arange(t, device=inputs_embeds.device).unsqueeze(0)
            cache_pos = pos.reshape(-1)
            causal = torch.triu(
                torch.ones(t, t, dtype=torch.bool, device=inputs_embeds.device), 1
            )
            min_v = torch.finfo(inputs_embeds.dtype).min
            attn = inputs_embeds.new_zeros(1, 1, t, t)
            attn = attn.masked_fill(causal, min_v)
            keep = attention_mask.to(dtype=torch.bool).view(1, 1, 1, t)
            attn = attn.masked_fill(~keep, min_v)
            hidden = inputs_embeds
            rope = self.inner.rotary_emb(hidden, pos)
            for layer in self.inner.layers[:n_layers]:
                hidden = layer(
                    hidden,
                    attention_mask=attn,
                    position_ids=pos,
                    past_key_values=None,
                    cache_position=cache_pos,
                    position_embeddings=rope,
                )
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
            return self.inner.norm(hidden)

    return _Stack()


class FullSeqRunner:
    """Python keeps the prefix; every decode re-runs the growing sequence on GPU."""

    def __init__(self, compiled):
        self.compiled = compiled
        self.prefix = None

    def reset(self):
        self.prefix = None

    def step(self, embeds):
        import numpy as np
        import torch

        q = int(embeds.shape[1])
        piece = embeds.detach().float().contiguous()
        self.prefix = piece if self.prefix is None else torch.cat([self.prefix, piece], dim=1)
        arr = np.ascontiguousarray(self.prefix.cpu().numpy())
        hidden = torch.from_numpy(
            np.ascontiguousarray(self.compiled(arr, mask_np(None, arr.shape[1], 0))[0])
        )
        return hidden[:, -q:, :]


def causal_step_module(inner, n_layers, with_past):
    """Kept for tests. Do not export this: DynamicCache does not trace."""
    import torch.nn as nn

    class _Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds, attention_mask, *past):
            kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if with_past:
                kwargs["past_key_values"] = unflatten_kv(past, n_layers)
            out = self.inner(**kwargs)
            return (out.last_hidden_state, *flatten_kv(out.past_key_values))

    return _Step()


def mask_np(mask, q_len, past_len):
    """2-D [1, past+q] keep-mask for a causal step. 4-D additive masks: 0 = keep."""
    import numpy as np
    import torch

    total = int(past_len) + int(q_len)
    if mask is None:
        return np.ones((1, total), dtype=np.int64)
    t = mask.detach() if hasattr(mask, "detach") else torch.as_tensor(mask)
    if t.dim() == 2:
        return np.ascontiguousarray(t.long().cpu().numpy())
    sl = min(total, int(t.shape[-1]))
    row = t[0, 0, 0, :sl]
    if row.dtype.is_floating_point:
        keep = (row == 0)
    else:
        keep = row.bool()
    out = keep.long().cpu().numpy().reshape(1, -1)
    if out.shape[1] < total:
        out = np.concatenate([out, np.ones((1, total - out.shape[1]), dtype=np.int64)], 1)
    return np.ascontiguousarray(out)


def write_static_kv(cache, flat):
    """Copy OV present K/V (used prefix) back into a transformers StaticCache."""
    n = len(flat) // 2
    for i in range(n):
        k, v = flat[2 * i], flat[2 * i + 1]
        layer = cache.layers[i]
        sl = k.shape[-2]
        keys = getattr(layer, "keys", None)
        vals = getattr(layer, "values", None)
        if keys is None:
            keys = layer.key_cache
            vals = layer.value_cache
        keys[..., :sl, :].copy_(k)
        vals[..., :sl, :].copy_(v)


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
