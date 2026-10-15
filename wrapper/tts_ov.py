# Shared OpenVINO compile for TTS steps. Official Python still samples; each transformer step runs on GPU.
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


def release_parameters(mod, label):
    """Drop torch weight storage once those weights are in an OpenVINO IR.

    The next stage's convert_model allocates another copy. Leaving the compiled
    stage's parameters resident is what pushed Breeze over the cgroup after the
    backbone graph was already on GPU.
    """
    import gc

    if mod is None:
        return 0
    seen = set()
    nbytes = 0
    for p in list(mod.parameters()):
        storage = p.data.untyped_storage()
        key = storage.data_ptr()
        if key not in seen and p.data.numel():
            seen.add(key)
            nbytes += storage.nbytes()
        p.data = p.data.new_empty(0)
    gc.collect()
    trimmed = False
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        trimmed = True
    except OSError:
        pass
    log.info(
        "released %s torch params bytes=%d storages=%d trim=%s rss_kib=%s",
        label, nbytes, len(seen), trimmed, _rss_kib(),
    )
    return nbytes


def _rss_kib():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return line.split()[1]
    except OSError:
        return "?"
    return "?"


def _device_props(device, precision, latency=False):
    """GPU compute precision. CPU stays f32: that plugin rejects f16 and undefined KV.

    f16 is the iGPU path. LATENCY keeps a single stream so the codec graph is
    not a second resident copy of the same weights. The correct depth frame
    stays f32 and still wants that schedule.
    """
    cpu = (device or "").strip().upper().startswith("CPU")
    prec = "f32" if cpu else (precision or "f32")
    props = {"INFERENCE_PRECISION_HINT": prec}
    if not cpu:
        props["KV_CACHE_PRECISION"] = "undefined"
        if prec == "f16" or latency:
            props["PERFORMANCE_HINT"] = "LATENCY"
    return props, prec


def _compile_ir(xml, device, config=None):
    """Compile a saved IR. mmap of an intact IR on the hostPath raises SIGBUS."""
    import openvino as ov

    from .ov_compile_lock import gpu_compile

    core = ov.Core()
    core.set_property({"ENABLE_MMAP": False})
    props = dict(config or {})
    props["ENABLE_MMAP"] = False
    with gpu_compile(device):
        return core.compile_model(xml, device, props)


def _freeze_for_export(mod):
    """OpenVINO refuses a weight that still requires grad when the trace treats it as a constant."""
    if not hasattr(mod, "parameters"):
        return
    for p in mod.parameters():
        p.requires_grad_(False)


def compile_module(mod, example, xml, stamp, device, config=None, precision="f32",
                    compile_graph=True):
    """Export a torch nn.Module once, compile on GPU, return a callable(np)->np.

    The callable's .compiled is the ov.CompiledModel so a caller can keep one
    InferRequest instead of rebuilding the feed every step.
    """
    import numpy as np
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        if not isinstance(example, (tuple, list)):
            example = (example,)
        if hasattr(mod, "eval"):
            mod.eval()
        _freeze_for_export(mod)
        log.info("exporting %s example=%s", xml, [tuple(t.shape) for t in example])
        try:
            with torch.inference_mode():
                ov_model = ov.convert_model(mod, example_input=example)
        except Exception as e:
            log.warning("convert_model failed (%s); jit.trace then convert", e)
            traced = torch.jit.trace(mod, example, strict=False, check_trace=False)
            with torch.inference_mode():
                ov_model = ov.convert_model(traced)
        ov.save_model(ov_model, xml)
        open(stamp, "w").close()
    if not compile_graph:
        log.info("found %s; GPU compile deferred", xml)
        return None
    if config is None:
        config, prec = _device_props(device, precision)
    else:
        prec = (config or {}).get("INFERENCE_PRECISION_HINT", precision)
    compiled = _compile_ir(xml, device, config)
    log.info("compiled %s on %s inference_precision=%s", xml, device, prec)

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    run.compiled = compiled
    return run


def compile_static(mod, example, xml, stamp, device, precision="f32", compile_graph=True):
    """Like compile_module, but pin every input to the example shape.

    convert_model leaves codec T dynamic. Upsample residuals then Add a
    frozen length against a runtime length (intel11/intel20). Codec always
    pads to example_t, so the IR must be that same static T.
    """
    import numpy as np
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        if not isinstance(example, (tuple, list)):
            example = (example,)
        if hasattr(mod, "eval"):
            mod.eval()
        _freeze_for_export(mod)
        log.info("exporting static %s example=%s", xml, [tuple(t.shape) for t in example])
        try:
            with torch.inference_mode():
                ov_model = ov.convert_model(mod, example_input=example)
        except Exception as e:
            log.warning("convert_model failed (%s); jit.trace then convert", e)
            traced = torch.jit.trace(mod, example, strict=False, check_trace=False)
            with torch.inference_mode():
                ov_model = ov.convert_model(traced)
        mapping = {}
        for inp, t in zip(ov_model.inputs, example):
            mapping[inp.get_any_name()] = [int(x) for x in t.shape]
        ov_model.reshape(mapping)
        ov_model.validate_nodes_and_infer_types()
        log.info("froze %s inputs=%s", xml, mapping)
        ov.save_model(ov_model, xml)
        open(stamp, "w").close()
    if not compile_graph:
        log.info("exported static %s; GPU compile deferred", xml)
        return None
    props, prec = _device_props(device, precision)
    compiled = _compile_ir(xml, device, props)
    log.info("compiled static %s on %s inference_precision=%s", xml, device, prec)

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
        if rank == 1:
            shape[0] = -1
        elif rank == 2 or rank == 3:
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


def compile_causal(mod, example, xml, stamp, device, dynamize_ranks=(2, 3, 4),
                   compile_graph=True, precision="f32", latency=False):
    """Dynamize the time axis, then compile. compile_graph=False writes IR only so torch weights can be dropped first."""
    import numpy as np
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    wrote = not (os.path.isfile(xml) and os.path.isfile(stamp))
    if wrote:
        if not isinstance(example, (tuple, list)):
            example = (example,)
        log.info("exporting %s example=%s", xml, [tuple(t.shape) for t in example])
        mod.eval()
        _freeze_for_export(mod)
        with torch.inference_mode():
            ov_model = _dynamize_time(
                ov.convert_model(mod, example_input=example), dynamize_ranks
            )
        ov.save_model(ov_model, xml)
        open(stamp, "w").close()
        del ov_model
    if not compile_graph:
        log.info("%s %s; GPU compile deferred", "exported" if wrote else "found", xml)
        return None
    import gc
    gc.collect()
    # GPU accepts "undefined" and then keeps the IR dtype. CPU rejects it.
    props, prec = _device_props(device, precision, latency=latency)
    compiled = _compile_ir(xml, device, props)
    log.info(
        "compiled %s on %s inference_precision=%s performance_hint=%s",
        xml, device, prec, props.get("PERFORMANCE_HINT", ""),
    )

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    run.compiled = compiled
    return run


def compile_dyn_last(mod, example, xml, stamp, device, precision="f32", compile_graph=True):
    """Compile a chunk module whose last axis grows (codec conv time)."""
    import numpy as np
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        if not isinstance(example, (tuple, list)):
            example = (example,)
        log.info("exporting %s example=%s", xml, [tuple(t.shape) for t in example])
        mod.eval()
        _freeze_for_export(mod)
        with torch.inference_mode():
            ov_model = ov.convert_model(mod, example_input=example)
        mapping = {}
        for inp in ov_model.inputs:
            shape = inp.get_partial_shape()
            if shape.rank.is_dynamic:
                continue
            rank = shape.rank.get_length()
            if rank < 2:
                continue
            shape[rank - 1] = -1
            try:
                mapping[inp.any_name] = shape
            except Exception:
                mapping[inp] = shape
        if mapping:
            ov_model.reshape(mapping)
        ov.save_model(ov_model, xml)
        open(stamp, "w").close()
    if not compile_graph:
        log.info("exported %s; GPU compile deferred", xml)
        return None
    props, prec = _device_props(device, precision)
    compiled = _compile_ir(xml, device, props)
    log.info("compiled %s on %s inference_precision=%s", xml, device, prec)

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    run.compiled = compiled
    return run


def causal_attn_bias(hidden, q_len, past_len):
    """Additive causal mask via triu. An arange-compare Select freezes at the export length."""
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


def causal_kv_module(inner, with_past, external_position=False):
    """One transformer step with tensor K/V. Never calls official forward or Cache.

    external_position puts position_ids on the decode signature, in front of K/V.
    Deriving the id from the example cache length leaves every new frame at the
    export position. Decode callers pass the real id; a static prefill does not.
    """
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

        def forward(self, inputs_embeds, attention_mask, *rest):
            # Use the linear dtype, not rotary inv_freq: that buffer is float32 and would cast every activation.
            if external_position:
                position_ids, past = rest[0], rest[1:]
            else:
                position_ids, past = None, rest
            wdtype = self.inner.layers[0].self_attn.q_proj.weight.dtype
            hidden = inputs_embeds.to(dtype=wdtype)
            q_len = hidden.shape[1]
            past_len = 0
            if with_past:
                past_len = past[0].shape[-2]
            total = past_len + q_len
            if position_ids is None:
                pos = torch.arange(
                    past_len, total, device=hidden.device
                ).unsqueeze(0)
            else:
                pos = position_ids
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


def _proj_native(linear, x):
    """Run one linear in its own weight dtype and return x's dtype."""
    if linear.weight.dtype == x.dtype:
        return linear(x)
    return linear(x.to(dtype=linear.weight.dtype)).to(dtype=x.dtype)


def _mlp_wide_f16(mlp, x):
    """f16 on each projection that was cast. The product stays in x's dtype."""
    gate = _proj_native(mlp.gate_proj, x)
    up = _proj_native(mlp.up_proj, x)
    return _proj_native(mlp.down_proj, mlp.act_fn(gate) * up)


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
    q = _proj_native(attn.q_proj, h).view(b, t, n_q, head_dim)
    k = _proj_native(attn.k_proj, h).view(b, t, n_kv, head_dim)
    v = _proj_native(attn.v_proj, h).view(b, t, n_kv, head_dim)
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
    h = residual + _proj_native(attn.o_proj, out)
    x = layer.post_attention_layernorm(h)
    mlp = layer.mlp
    if (
        mlp.gate_proj.weight.dtype == x.dtype
        and mlp.up_proj.weight.dtype == x.dtype
        and mlp.down_proj.weight.dtype == x.dtype
    ):
        x = mlp(x)
    else:
        x = _mlp_wide_f16(mlp, x)
    h = h + x
    return h, k, v


def _scatter_slot(past, value, slot):
    """Write new keys into a fixed time axis. slot is one index per new token.

    The cache length stays max_t. A Python arange of the example width would
    freeze that width, so the indices come in as a tensor.
    """
    idx = slot.reshape(1, 1, -1, 1).expand(
        value.shape[0], value.shape[1], value.shape[2], value.shape[3]
    )
    return past.scatter(2, idx, value)


def _layer_fixed(layer, hidden, rope, attention_mask, slot, pk, pv):
    """Same qkv/rope/mlp as _layer_kv, but the cache is a fixed buffer, not a cat."""
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
    k = _scatter_slot(pk, k, slot)
    v = _scatter_slot(pv, v, slot)
    n_rep = n_q // n_kv
    k_rep = _repeat_kv(k, n_rep)
    v_rep = _repeat_kv(v, n_rep)
    scale = float(getattr(attn, "scaling", head_dim ** -0.5))
    scores = torch.matmul(q, k_rep.transpose(2, 3)) * scale
    # [1, q, max_t] to [1, 1, q, max_t]. q stays an input, not a constant from the export example.
    keep = attention_mask.to(dtype=torch.bool).unsqueeze(1)
    scores = scores.masked_fill(~keep, torch.finfo(hidden.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(probs, v_rep).transpose(1, 2).contiguous().view(b, t, n_q * head_dim)
    h = residual + attn.o_proj(out)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, k, v


def causal_fixed_slot_module(inner, max_t, lm_head=None):
    """Fixed cache, variable query length. One compiled graph holds the weights.

    Prefill feeds the whole prefix in one call. Decode feeds one token. A
    second backbone graph would be a second copy of the weights. lm_head, when
    passed, is that same graph's last matmul, not another compiled model.
    """
    import torch.nn as nn

    if not all(hasattr(inner, n) for n in ("layers", "norm", "rotary_emb")):
        raise RuntimeError("backbone missing layers/norm/rotary_emb for slot export")
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        cfg._attn_implementation = "eager"
    n_layers = int(getattr(cfg, "num_hidden_layers", len(inner.layers)))

    class _Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.lm_head = lm_head

        def forward(self, inputs_embeds, attention_mask, position_ids, slot, *past):
            wdtype = self.inner.layers[0].self_attn.q_proj.weight.dtype
            hidden = inputs_embeds.to(dtype=wdtype)
            rope = self.inner.rotary_emb(hidden, position_ids)
            present = []
            for i, layer in enumerate(self.inner.layers[:n_layers]):
                hidden, nk, nv = _layer_fixed(
                    layer, hidden, rope, attention_mask, slot,
                    past[2 * i].to(dtype=wdtype),
                    past[2 * i + 1].to(dtype=wdtype),
                )
                present.extend([nk, nv])
            hidden = self.inner.norm(hidden).to(dtype=inputs_embeds.dtype)
            out_kv = [t.to(dtype=inputs_embeds.dtype) for t in present]
            if self.lm_head is None:
                return (hidden, *out_kv)
            hdtype = self.lm_head.weight.dtype
            logits = self.lm_head(hidden[:, -1, :].to(dtype=hdtype))
            return (hidden, *out_kv, logits)

    return _Step()


class DeviceKvRunner:
    """Ping-pong InferRequests. Prefill and decode stay on device; only hidden returns to host."""

    def __init__(self, decode, n_layers, prefill=None, external_position=False):
        compiled = getattr(decode, "compiled", None)
        if compiled is None:
            raise RuntimeError("DeviceKvRunner needs compile_causal().compiled")
        self.compiled = compiled
        self.reqs = (compiled.create_infer_request(), compiled.create_infer_request())
        self.which = 0
        self.n_layers = n_layers
        self.kv = None
        self._prefix = 0
        self.external_position = bool(external_position)
        self.prefill = getattr(prefill, "compiled", None) if prefill is not None else None
        self.pre_req = self.prefill.create_infer_request() if self.prefill is not None else None

    def reset(self):
        self.kv = None
        self._prefix = 0
        self.which = 0

    @property
    def prefix_len(self):
        return int(self._prefix)

    def step(self, embeds):
        import numpy as np
        import openvino as ov
        import torch

        if self.kv is None:
            if self.pre_req is None:
                raise RuntimeError("DeviceKvRunner.step needs a prefill model")
            return self._prefill_step(embeds)
        q = int(embeds.shape[1])
        self._emb = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        self._mask = mask_np(None, q, self._prefix)
        req = self.reqs[self.which]
        req.set_input_tensor(0, ov.Tensor(self._emb))
        req.set_input_tensor(1, ov.Tensor(self._mask))
        kv_at = 2
        if self.external_position:
            # RoPE for this token is the cache length, not the export example.
            self._pos = np.arange(
                self._prefix, self._prefix + q, dtype=np.int64
            ).reshape(1, q)
            req.set_input_tensor(kv_at, ov.Tensor(self._pos))
            kv_at += 1
        for i, t in enumerate(self.kv):
            req.set_input_tensor(kv_at + i, t)
        req.infer()
        hidden = torch.from_numpy(np.array(req.get_output_tensor(0).data, copy=True))
        self.kv = [req.get_output_tensor(i) for i in range(1, 1 + 2 * self.n_layers)]
        self._prefix += q
        self.which ^= 1
        return hidden[:, -q:, :]

    def _prefill_step(self, embeds):
        import numpy as np
        import openvino as ov
        import torch

        q = int(embeds.shape[1])
        self._emb = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        self._mask = mask_np(None, q, 0)
        req = self.pre_req
        req.set_input_tensor(0, ov.Tensor(self._emb))
        req.set_input_tensor(1, ov.Tensor(self._mask))
        req.infer()
        hidden = torch.from_numpy(np.array(req.get_output_tensor(0).data, copy=True))
        self.kv = [req.get_output_tensor(i) for i in range(1, 1 + 2 * self.n_layers)]
        self._prefix = q
        return hidden[:, -q:, :]


class FixedSlotRunner:
    """One InferRequest pair, one static cache. K/V never change shape.

    Growing a stateful cache between tokens is what made the step time wander,
    and a second compiled backbone is a second copy of the weights.
    """

    def __init__(self, decode, n_layers, max_t, n_kv, head_dim):
        compiled = getattr(decode, "compiled", None)
        if compiled is None:
            raise RuntimeError("FixedSlotRunner needs compile_causal().compiled")
        self.compiled = compiled
        self.reqs = (
            compiled.create_infer_request(),
            compiled.create_infer_request(),
        )
        self.n_layers = int(n_layers)
        self.max_t = int(max_t)
        self.n_kv = int(n_kv)
        self.head_dim = int(head_dim)
        self.which = 0
        self.slot = 0
        self.kv = None
        self.reset()

    def reset(self):
        import numpy as np
        import openvino as ov

        zeros = np.zeros((1, self.n_kv, self.max_t, self.head_dim), dtype=np.float32)
        self.kv = [ov.Tensor(np.array(zeros, copy=True)) for _ in range(2 * self.n_layers)]
        self.slot = 0
        self.which = 0

    def step(self, embeds):
        return self._infer(embeds)

    def _infer(self, embeds):
        import numpy as np
        import openvino as ov
        import torch

        q = int(embeds.shape[1])
        if q < 1:
            raise RuntimeError("fixed slot infer needs at least one token")
        if self.slot + q > self.max_t:
            raise RuntimeError(
                "fixed slot %d+%d exceeds %d" % (self.slot, q, self.max_t)
            )
        self._emb = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        self._mask = np.zeros((1, q, self.max_t), dtype=np.int64)
        for i in range(q):
            self._mask[0, i, : self.slot + i + 1] = 1
        self._pos = np.arange(self.slot, self.slot + q, dtype=np.int64).reshape(1, q)
        self._slot = np.arange(self.slot, self.slot + q, dtype=np.int64)
        req = self.reqs[self.which]
        req.set_input_tensor(0, ov.Tensor(self._emb))
        req.set_input_tensor(1, ov.Tensor(self._mask))
        req.set_input_tensor(2, ov.Tensor(self._pos))
        req.set_input_tensor(3, ov.Tensor(self._slot))
        for i, t in enumerate(self.kv):
            req.set_input_tensor(4 + i, t)
        req.infer()
        hidden = torch.from_numpy(np.array(req.get_output_tensor(0).data, copy=True))
        n_kv = 2 * self.n_layers
        self.kv = [req.get_output_tensor(i) for i in range(1, 1 + n_kv)]
        self.logits = None
        if len(self.compiled.outputs) > 1 + n_kv:
            self.logits = torch.from_numpy(
                np.array(req.get_output_tensor(1 + n_kv).data, copy=True)
            )
        self.slot += q
        self.which ^= 1
        return hidden

    def prefill(self, embeds):
        return self._infer(embeds)


def fused_depth_frame(inner, head, n_codebooks, vocab, codebook_size, backbone_proj=None):
    """One audio frame: prefill plus every codebook step, one OpenVINO infer.

    Fifteen separate depth infers stay near 0.6s a frame; the time is the
    per-call sync. One uniform row per codebook keeps the next codebook in
    the same infer. The draw is Gumbel-max, the official multinomial
    distribution. A boolean CDF reduce compiled to a constant and the codec
    rendered that as noise.
    """
    import torch
    import torch.nn as nn

    if not all(hasattr(inner, n) for n in ("layers", "norm", "rotary_emb")):
        raise RuntimeError("depth model missing layers/norm/rotary_emb")
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        cfg._attn_implementation = "eager"
    n_layers = int(getattr(cfg, "num_hidden_layers", len(inner.layers)))
    n_tokens = int(n_codebooks) - 1
    max_k = min(1024, int(vocab))
    if n_tokens < 1:
        raise RuntimeError("fused depth frame needs codebooks, got %d" % n_codebooks)

    class _Frame(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.head = head
            self.backbone_proj = backbone_proj

        def _open_frame(self, backbone_h, first_id):
            # Embed [0, first_code], overwrite slot 0 with the backbone hidden, then project.
            first = first_id.reshape(-1).to(dtype=torch.long)
            ids = torch.stack([torch.zeros_like(first), first], dim=1)
            emb = self.inner.embed_tokens(ids)
            bh = backbone_h
            if bh.dim() == 1:
                bh = bh.unsqueeze(0)
            if bh.dim() == 3:
                bh = bh[:, 0]
            if self.backbone_proj is not None:
                bh = self.backbone_proj(bh.to(dtype=self.backbone_proj.weight.dtype))
            emb = emb.clone()
            emb[:, 0, :] = bh.to(dtype=emb.dtype)
            proj = self.inner.inputs_embeds_projector
            return proj(emb.to(dtype=proj.weight.dtype))

        def _pick(self, logits, u, temperature, top_k, top_p, do_sample):
            scores = logits[0, 0].float()
            if int(codebook_size) < int(vocab):
                scores = scores.clone()
                scores[int(codebook_size):int(vocab)] = float("-inf")
            scaled = scores / temperature.reshape(()).float()
            k_lim = top_k.reshape(()).to(dtype=torch.long)
            k_lim = torch.where(k_lim > 0, k_lim, torch.full_like(k_lim, max_k))
            vals, idx = torch.topk(scaled, max_k)
            keep = torch.arange(max_k, device=scaled.device) < k_lim
            vals = torch.where(keep, vals, torch.full_like(vals, float("-inf")))
            filtered = torch.full_like(scaled, float("-inf")).scatter(0, idx, vals)
            order = torch.argsort(filtered, descending=True)
            sorted_logits = filtered[order]
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p.reshape(()).float()
            remove = torch.cat([
                torch.zeros(1, dtype=torch.bool, device=remove.device), remove[:-1],
            ])
            sorted_logits = torch.where(
                remove, torch.full_like(sorted_logits, float("-inf")), sorted_logits,
            )
            filtered = torch.full_like(filtered, float("-inf")).scatter(
                0, order, sorted_logits,
            )
            # Gumbel-max. A boolean CDF reduce compiled to a constant and the codec rendered noise.
            uu = u.float().reshape(-1).clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uu))
            sampled = torch.argmax((filtered + gumbel).float()).to(dtype=torch.long)
            greedy = torch.argmax(scores).to(dtype=torch.long)
            take = do_sample.reshape(()).to(dtype=torch.long) != 0
            tok = torch.where(take, sampled, greedy)
            return tok.clamp(0, int(vocab) - 1)

        def _stack(self, hidden, past):
            wdtype = self.inner.layers[0].self_attn.q_proj.weight.dtype
            hidden = hidden.to(dtype=wdtype)
            q_len = hidden.shape[1]
            past_len = 0 if past is None else int(past[0].shape[-2])
            pos = torch.arange(past_len, past_len + q_len, device=hidden.device).unsqueeze(0)
            attn = causal_attn_bias(hidden, q_len, past_len)
            rope = self.inner.rotary_emb(hidden, pos)
            present = []
            for i, layer in enumerate(self.inner.layers[:n_layers]):
                pk = pv = None
                if past is not None:
                    pk = past[2 * i].to(dtype=wdtype)
                    pv = past[2 * i + 1].to(dtype=wdtype)
                hidden, nk, nv = _layer_kv(layer, hidden, rope, attn, pk, pv)
                present.extend([nk, nv])
            hidden = self.inner.norm(hidden).to(dtype=hidden.dtype)
            return hidden, present

        def forward(self, prefill_embeds, temperature, top_k, top_p, do_sample, uniform):
            # The two-token prefix stays f32 torch. Folding that gather into this graph changed the codes.
            hidden, past = self._stack(prefill_embeds, None)
            hdtype = self.head.weight.dtype
            tokens = []
            tok = self._pick(
                self.head(
                    hidden[:, 1:, :].to(dtype=hdtype),
                    cache_position=torch.tensor([1]),
                ),
                uniform[0], temperature, top_k, top_p, do_sample,
            )
            tokens.append(tok)
            for cb_idx in range(1, n_tokens):
                ids = (tok + cb_idx * int(vocab)).view(1, 1).clamp(
                    0, int(n_codebooks) * int(vocab) - 1,
                )
                emb = self.inner.embed_tokens(ids)
                proj = self.inner.inputs_embeds_projector
                emb = proj(emb.to(dtype=proj.weight.dtype))
                hidden, past = self._stack(emb, past)
                pos = torch.tensor([1 + cb_idx])
                tok = self._pick(
                    self.head(hidden[:, -1:, :].to(dtype=hdtype), cache_position=pos),
                    uniform[cb_idx], temperature, top_k, top_p, do_sample,
                )
                tokens.append(tok)
            return torch.stack(tokens)

    return _Frame()


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
