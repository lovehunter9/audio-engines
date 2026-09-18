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


def _dynamize_time(ov_model, ranks=(2, 3, 4)):
    """Prefill embeds are [B,T,H]; cached K/V are [B,kv,T,D]. T must grow.

    FireRed decode is always q=patch_size (4). Dynamizing that axis freezes the
    q×q causal tail into a Select that no longer matches at runtime. Leave rank
    3 static there; still dynamize the mask (rank 2) and past K/V (rank 4).
    """
    mapping = {}
    for inp in ov_model.inputs:
        shape = inp.get_partial_shape()
        if shape.rank.is_dynamic:
            continue
        rank = shape.rank.get_length()
        if rank not in ranks:
            continue
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


def compile_causal(mod, example, xml, stamp, device, dynamize_ranks=(2, 3, 4)):
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
            ov_model = _dynamize_time(
                ov.convert_model(mod, example_input=example), dynamize_ranks
            )
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
    """Layer stack only. Official backbone.forward builds an HF causal mask that
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
            wdtype = next(self.inner.parameters()).dtype
            hidden = inputs_embeds.to(dtype=wdtype)
            t = hidden.shape[1]
            pos = torch.arange(t, device=hidden.device).unsqueeze(0)
            cache_pos = pos.reshape(-1)
            causal = torch.triu(
                torch.ones(t, t, dtype=torch.bool, device=hidden.device), 1
            )
            min_v = torch.finfo(hidden.dtype).min
            attn = hidden.new_zeros(1, 1, t, t)
            attn = attn.masked_fill(causal, min_v)
            keep = attention_mask.to(dtype=torch.bool).view(1, 1, 1, t)
            attn = attn.masked_fill(~keep, min_v)
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
            return self.inner.norm(hidden).to(dtype=inputs_embeds.dtype)

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


def causal_attn_bias(hidden, q_len, past_len):
    """Additive mask: query i sees keys 0 .. past_len+i. Needed for FireRed q=4.

    Prefill is a T×T triu (intel8 exported this and dynamized T). Decode keeps
    every past key and applies the same triu only on the new q×q block.
    Do not build the mask with arange-compare: OpenVINO freezes that Select
    at the example length and then 500s on a longer prompt.
    """
    import torch

    min_v = torch.finfo(hidden.dtype).min
    attn = hidden.new_zeros(1, 1, q_len, past_len + q_len)
    tail = torch.triu(
        torch.ones(q_len, q_len, dtype=torch.bool, device=hidden.device), 1
    )
    if past_len == 0:
        return attn.masked_fill(tail, min_v)
    attn[:, :, :, past_len:] = attn[:, :, :, past_len:].masked_fill(tail, min_v)
    return attn


def _rotate_half(x):
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, slen, head_dim)
        .reshape(b, n_kv * n_rep, slen, head_dim)
    )


def causal_kv_module(inner, with_past):
    """One transformer step with tensor K/V. Never calls official forward or Cache."""
    import torch
    import torch.nn as nn

    if not all(hasattr(inner, n) for n in ("layers", "norm", "rotary_emb")):
        raise RuntimeError("backbone missing layers/norm/rotary_emb for kv export")
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        cfg._attn_implementation = "eager"
    n_layers = int(getattr(cfg, "num_hidden_layers", len(inner.layers)))

    class _Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds, attention_mask, *past):
            wdtype = next(self.inner.parameters()).dtype
            hidden = inputs_embeds.to(dtype=wdtype)
            q_len = hidden.shape[1]
            past_len = 0
            if with_past:
                past_len = past[0].shape[-2]
            total = past_len + q_len
            pos = torch.arange(
                past_len, total, device=hidden.device
            ).unsqueeze(0)
            attn = causal_attn_bias(hidden, q_len, past_len)
            keep = attention_mask.to(dtype=torch.bool).view(1, 1, 1, total)
            attn = attn.masked_fill(~keep, torch.finfo(hidden.dtype).min)
            rope = self.inner.rotary_emb(hidden, pos)
            present = []
            for i, layer in enumerate(self.inner.layers[:n_layers]):
                pk = pv = None
                if with_past:
                    pk = past[2 * i].to(dtype=wdtype)
                    pv = past[2 * i + 1].to(dtype=wdtype)
                hidden, nk, nv = _layer_kv(layer, hidden, rope, attn, pk, pv)
                present.extend([nk, nv])
            hidden = self.inner.norm(hidden).to(dtype=inputs_embeds.dtype)
            out_kv = [t.to(dtype=inputs_embeds.dtype) for t in present]
            return (hidden, *out_kv)

    return _Step()


def _layer_kv(layer, hidden, rope, attn_mask, pk, pv):
    """input_ln → qkv/rope/cat → eager attn → o_proj → mlp. Tensor cache only."""
    import torch

    attn = layer.self_attn
    residual = hidden
    h = layer.input_layernorm(hidden)
    b, t, _ = h.shape
    head_dim = int(attn.head_dim)
    n_q = attn.q_proj.out_features // head_dim
    n_kv = attn.k_proj.out_features // head_dim
    q = attn.q_proj(h).view(b, t, n_q, head_dim)
    k = attn.k_proj(h).view(b, t, n_kv, head_dim)
    v = attn.v_proj(h).view(b, t, n_kv, head_dim)
    if getattr(attn, "q_norm", None) is not None:
        q = attn.q_norm(q)
    if getattr(attn, "k_norm", None) is not None:
        k = attn.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = rope
    q, k = _apply_rope(q, k, cos, sin)
    if pk is not None:
        k = torch.cat([pk, k], dim=2)
        v = torch.cat([pv, v], dim=2)
    n_rep = n_q // n_kv
    k_rep = _repeat_kv(k, n_rep)
    v_rep = _repeat_kv(v, n_rep)
    scale = float(getattr(attn, "scaling", head_dim ** -0.5))
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * scale
    scores = scores + attn_mask
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(probs, v_rep).transpose(1, 2).contiguous().view(b, t, n_q * head_dim)
    h = residual + attn.o_proj(out)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, k, v


class KvRunner:
    """Prefill once, then decode with tensor K/V. No full-seq rerun."""

    def __init__(self, prefill, decode, n_layers):
        self.prefill = prefill
        self.decode = decode
        self.n_layers = n_layers
        self.kv = None

    def reset(self):
        self.kv = None

    def seed_kv(self, cache):
        """Copy an official HF cache into numpy K/V after a CPU prefill."""
        import numpy as np

        flat = flatten_kv(cache)
        if len(flat) != 2 * self.n_layers:
            raise RuntimeError(
                "seed_kv expected %d tensors, got %d"
                % (2 * self.n_layers, len(flat))
            )
        self.kv = [
            np.ascontiguousarray(t.detach().float().cpu().numpy()) for t in flat
        ]

    @property
    def prefix_len(self):
        if self.kv is None:
            return 0
        return int(self.kv[0].shape[-2])

    def step(self, embeds):
        import numpy as np
        import torch

        q = int(embeds.shape[1])
        arr = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        if self.kv is None:
            out = self.prefill(arr, mask_np(None, q, 0))
        else:
            out = self.decode(arr, mask_np(None, q, self.prefix_len), *self.kv)
        hidden = torch.from_numpy(np.ascontiguousarray(out[0]))
        self.kv = [np.ascontiguousarray(out[i]) for i in range(1, 1 + 2 * self.n_layers)]
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
