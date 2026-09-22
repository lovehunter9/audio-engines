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


def compile_module(mod, example, xml, stamp, device, config=None):
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
        for p in getattr(mod, "parameters", lambda: ())():
            p.requires_grad_(False)
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
    core = ov.Core()
    compiled = core.compile_model(xml, device, config or {})
    log.info("compiled %s on %s", xml, device)

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    run.compiled = compiled
    return run


def compile_static(mod, example, xml, stamp, device):
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
        for p in getattr(mod, "parameters", lambda: ())():
            p.requires_grad_(False)
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
    core = ov.Core()
    compiled = core.compile_model(xml, device)
    log.info("compiled static %s on %s", xml, device)

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


def patch_stateful_kv(ov_model, kv_from=2):
    """Intel FireRedTTS2 helper: hide K/V as InferRequest VariableState.

    Official notebook calls apply_make_stateful_transformation so decode does
    not bounce 28×2 tensors through Python every token.
    """
    import numpy as np
    import openvino.opset13 as opset13
    from openvino._offline_transformations import apply_make_stateful_transformation

    if len(ov_model.inputs) < 4:
        raise RuntimeError("stateful decode needs embeds, mask, and K/V")
    # kv_from skips non-cache inputs that sit in front of K/V (embeds, mask,
    # and a static-slot position). Default 2 is the original decode signature.
    for i, inp in enumerate(ov_model.inputs[kv_from:]):
        tensor = inp.get_tensor()
        if not tensor.get_names():
            tensor.add_names({"past_kv_%d" % i})
    kv_in = [inp.get_any_name() for inp in ov_model.inputs[kv_from:]]
    # convert_model leaves tuple outputs unnamed. get_any_name then throws
    # "Attempt to get a name for a Tensor without names".
    kv_out = []
    for i, out in enumerate(ov_model.outputs[1:]):
        tensor = out.get_tensor()
        if not tensor.get_names():
            tensor.add_names({"present_kv_%d" % i})
        kv_out.append(out.get_any_name())
    if len(kv_in) != len(kv_out):
        raise RuntimeError("stateful kv in/out %d vs %d" % (len(kv_in), len(kv_out)))
    import openvino as ov

    batch = ov_model.inputs[0].get_partial_shape()[0]
    beam_idx = opset13.parameter(
        name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([batch])
    )
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    for name in kv_in:
        port = ov_model.input(name)
        consumers = list(port.get_target_inputs())
        gather = opset13.gather(port, beam_idx, opset13.constant(0))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()
    apply_make_stateful_transformation(ov_model, dict(zip(kv_in, kv_out)))
    embeds = ov_model.inputs[0]
    batch_g = opset13.gather(
        opset13.shape_of(embeds, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        dims = [d.min_length for d in list(op.get_output_partial_shape(0))]
        dims[0] = batch_g
        parts = []
        for dim in dims:
            if isinstance(dim, int):
                parts.append(opset13.constant(np.array([max(dim, 0)], dtype=np.int64)))
            else:
                parts.append(dim)
        shape = opset13.concat(parts, axis=0)
        op.set_arguments([
            opset13.broadcast(
                opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape
            )
        ])
    ov_model.validate_nodes_and_infer_types()
    return ov_model


def compile_causal(mod, example, xml, stamp, device, dynamize_ranks=(2, 3, 4),
                   stateful=False, kv_from=2):
    """Like compile_module, but dynamize the time axis before save."""
    import numpy as np
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        if not isinstance(example, (tuple, list)):
            example = (example,)
        log.info("exporting %s example=%s stateful=%s", xml, [tuple(t.shape) for t in example], stateful)
        mod.eval()
        with torch.inference_mode():
            ov_model = _dynamize_time(
                ov.convert_model(mod, example_input=example), dynamize_ranks
            )
        if stateful:
            ov_model = patch_stateful_kv(ov_model, kv_from=kv_from)
        ov.save_model(ov_model, xml)
        open(stamp, "w").close()
        del ov_model
    import gc
    gc.collect()
    core = ov.Core()
    compiled = core.compile_model(
        xml,
        device,
        {
            "INFERENCE_PRECISION_HINT": "f32",
            "KV_CACHE_PRECISION": "undefined",
        },
    )
    log.info("compiled %s on %s inference_precision=f32 stateful=%s", xml, device, stateful)

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    run.compiled = compiled
    return run


def compile_dyn_last(mod, example, xml, stamp, device):
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
    core = ov.Core()
    compiled = core.compile_model(
        xml, device, {"INFERENCE_PRECISION_HINT": "f32"}
    )
    log.info("compiled %s on %s inference_precision=f32", xml, device)

    def run(*arrays):
        feed = {}
        for i, arr in enumerate(arrays):
            key = compiled.inputs[i]
            feed[key] = np.ascontiguousarray(arr)
        return compiled(feed)

    run.compiled = compiled
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
            # rotary inv_freq is float32 even when the linears are bf16.
            # Taking the first parameter casts every activation to float and
            # the next linear refuses the pair.
            wdtype = self.inner.layers[0].self_attn.q_proj.weight.dtype
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


def _layer_slot(layer, hidden, rope, attn_mask, pk, pv, position):
    """One new token written into a fixed-length cache. Shape never changes."""
    import torch

    attn = layer.self_attn
    residual = hidden
    h = layer.input_layernorm(hidden)
    b, t, _ = h.shape
    if t != 1:
        raise RuntimeError("static slot decode is one token, got %d" % t)
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
    pk = pk.to(dtype=k.dtype)
    pv = pv.to(dtype=v.dtype)
    index = position.to(dtype=torch.long).reshape(1, 1, 1, 1).expand(b, n_kv, 1, head_dim)
    k = pk.scatter(2, index, k)
    v = pv.scatter(2, index, v)
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


def static_slot_kv_module(inner, slots):
    """Decode one token into a fixed cache. OpenVINO then keeps that cache as state.

    A growing stateful cache reallocated between steps and the wall time
    wandered off the infer timer. The slot count never changes, so the
    compiled kernel stays put.
    """
    import torch
    import torch.nn as nn

    if not all(hasattr(inner, n) for n in ("layers", "norm", "rotary_emb")):
        raise RuntimeError("backbone missing layers/norm/rotary_emb for slot export")
    cfg = getattr(inner, "config", None)
    if cfg is not None:
        cfg._attn_implementation = "eager"
    n_layers = int(getattr(cfg, "num_hidden_layers", len(inner.layers)))
    width = int(slots)

    class _Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds, attention_mask, position, *past):
            wdtype = self.inner.layers[0].self_attn.q_proj.weight.dtype
            hidden = inputs_embeds.to(dtype=wdtype)
            pos = position.reshape(1, 1).to(dtype=torch.long)
            attn = hidden.new_zeros(1, 1, 1, width)
            keep = attention_mask.to(dtype=torch.bool).view(1, 1, 1, width)
            attn = attn.masked_fill(~keep, torch.finfo(hidden.dtype).min)
            rope = self.inner.rotary_emb(hidden, pos)
            present = []
            for i, layer in enumerate(self.inner.layers[:n_layers]):
                hidden, nk, nv = _layer_slot(
                    layer, hidden, rope, attn,
                    past[2 * i], past[2 * i + 1], position,
                )
                present.extend([nk, nv])
            hidden = self.inner.norm(hidden).to(dtype=inputs_embeds.dtype)
            out_kv = [t.to(dtype=inputs_embeds.dtype) for t in present]
            return (hidden, *out_kv)

    return _Step()


class KvRunner:
    """Prefill once, then decode with tensor K/V. No full-seq rerun.

    Copies every K/V to host numpy each token. Fine for tests / Breeze until
    DeviceKvRunner covers prefill too.
    """

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


class DeviceKvRunner:
    """Decode with ping-pong InferRequests. Only hidden comes back to host.

    intel12 fed 28×2 numpy K/V every token and lost to official eager (RTF 26
    vs 16). Keep present K/V as the next request's input tensors so the
    device-side buffers stay put. Prefill stays official; seed_kv uploads once.
    """

    def __init__(self, decode, n_layers, prefill=None):
        compiled = getattr(decode, "compiled", None)
        if compiled is None:
            raise RuntimeError("DeviceKvRunner needs compile_causal().compiled")
        self.compiled = compiled
        self.reqs = (compiled.create_infer_request(), compiled.create_infer_request())
        self.which = 0
        self.n_layers = n_layers
        self.kv = None
        self._prefix = 0
        self.prefill = getattr(prefill, "compiled", None) if prefill is not None else None
        self.pre_req = self.prefill.create_infer_request() if self.prefill is not None else None

    def reset(self):
        self.kv = None
        self._prefix = 0
        self.which = 0

    def seed_kv(self, cache):
        import numpy as np
        import openvino as ov

        flat = flatten_kv(cache)
        if len(flat) != 2 * self.n_layers:
            raise RuntimeError(
                "seed_kv expected %d tensors, got %d"
                % (2 * self.n_layers, len(flat))
            )
        self._host = []
        self.kv = []
        for t in flat:
            arr = np.ascontiguousarray(t.detach().float().cpu().numpy())
            self._host.append(arr)
            self.kv.append(ov.Tensor(arr))
        self._prefix = int(self._host[0].shape[-2])

    @property
    def prefix_len(self):
        return int(self._prefix)

    def step(self, embeds):
        import numpy as np
        import openvino as ov
        import torch

        if self.kv is None:
            if self.pre_req is None:
                raise RuntimeError("DeviceKvRunner.step needs seed_kv or a prefill model")
            return self._prefill_step(embeds)
        q = int(embeds.shape[1])
        self._emb = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        self._mask = mask_np(None, q, self._prefix)
        req = self.reqs[self.which]
        req.set_input_tensor(0, ov.Tensor(self._emb))
        req.set_input_tensor(1, ov.Tensor(self._mask))
        for i, t in enumerate(self.kv):
            req.set_input_tensor(2 + i, t)
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


def fused_depth_frame(inner, head, n_codebooks, vocab, codebook_size):
    """One audio frame: prefill plus every codebook step, one OpenVINO infer.

    Intel-tts13 spent ~0.59s per frame on 15 separate depth infers.
    Intel-tts14 kept that cost with a static cache, so the time is the
    per-call sync, not the changing sequence length. Sampling stays in
    the graph and uses a uniform drawn by the caller, so the next
    codebook can stay in the same infer.
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
            probs = torch.softmax(filtered, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            sampled = torch.sum(cdf < u.reshape(()).float()).to(dtype=torch.long)
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
            hidden, past = self._stack(prefill_embeds, None)
            head_w = self.head.weight
            tokens = []
            tok = self._pick(
                self.head(
                    hidden[:, 1:, :].to(dtype=head_w.dtype),
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
                    self.head(hidden[:, -1:, :].to(dtype=head_w.dtype), cache_position=pos),
                    uniform[cb_idx], temperature, top_k, top_p, do_sample,
                )
                tokens.append(tok)
            return torch.stack(tokens)

    return _Frame()


class StatefulKvRunner:
    """One InferRequest; K/V stay in VariableState. Official prefill then seed_kv."""

    def __init__(self, decode, n_layers, static_position=False):
        compiled = getattr(decode, "compiled", None)
        if compiled is None:
            raise RuntimeError("StatefulKvRunner needs compile_causal().compiled")
        self.compiled = compiled
        self.req = compiled.create_infer_request()
        self.n_layers = n_layers
        self.static_position = bool(static_position)
        self.slots = 0
        self._prefix = 0
        self._ready = False

    def reset(self):
        self.req.reset_state()
        self._prefix = 0
        self._ready = False

    def seed_kv(self, cache):
        self.seed_flat(flatten_kv(cache))

    def seed_flat(self, flat, prefix=None):
        """Write already-flat K/V (torch or numpy, our [B,kv,T,D] layout) into state.

        prefix is the real token count when the tensors are padded out to a
        static slot count. The padded shape is not the next write index.
        """
        import numpy as np
        import openvino as ov

        states = self.req.query_state()
        if len(states) != len(flat):
            raise RuntimeError(
                "state %d vs kv %d" % (len(states), len(flat))
            )
        width = None
        for st, t in zip(states, flat):
            if hasattr(t, "detach"):
                arr = np.ascontiguousarray(t.detach().float().cpu().numpy())
            else:
                arr = np.ascontiguousarray(np.asarray(t, dtype=np.float32))
            st.state = ov.Tensor(arr)
            if width is None:
                width = int(arr.shape[-2])
        self.slots = int(width or 0)
        self._prefix = int(self.slots if prefix is None else prefix)
        self._ready = True

    @property
    def prefix_len(self):
        return int(self._prefix)

    def step(self, embeds):
        import numpy as np
        import openvino as ov
        import torch

        if not self._ready:
            raise RuntimeError("StatefulKvRunner.step needs seed_kv")
        q = int(embeds.shape[1])
        self._emb = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        self._beam = np.zeros((1,), dtype=np.int32)
        req = self.req
        req.set_input_tensor(0, ov.Tensor(self._emb))
        if self.static_position:
            pos = int(self._prefix)
            if pos >= self.slots:
                raise RuntimeError(
                    "breeze ov decode position %d exceeds static %d" % (pos, self.slots)
                )
            self._mask = np.zeros((1, self.slots), dtype=np.int64)
            self._mask[0, : pos + 1] = 1
            self._pos = np.array([pos], dtype=np.int64)
            req.set_input_tensor(1, ov.Tensor(self._mask))
            req.set_input_tensor(2, ov.Tensor(self._pos))
            req.set_input_tensor(3, ov.Tensor(self._beam))
        else:
            self._mask = mask_np(None, q, self._prefix)
            req.set_input_tensor(1, ov.Tensor(self._mask))
            req.set_input_tensor(2, ov.Tensor(self._beam))
        req.infer()
        hidden = torch.from_numpy(np.array(req.get_output_tensor(0).data, copy=True))
        self._prefix += q
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
