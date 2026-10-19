# One OpenVINO infer for a frame's residual codes.
#
# qwen_tts asks the code predictor for num_code_groups-1 tokens, and the stock
# generate() pays one compiled launch per token. The predictor is five layers
# and the cache never grows past sixteen positions, so the launches dominate.
# This unrolls that loop. The draw is Gumbel-max on the same filtered logits
# the official top-k / top-p / temperature path uses. The uniforms come from
# numpy: torch.rand here would eat the talker's generator.
import os

import numpy as np

log = None  # set by the installer; tests call the pure helpers without tts


def residual_steps(config):
    """How many residual codes one frame asks for. Sliding layers are refused."""
    kinds = list(getattr(config, "layer_types", None) or [])
    if any(kind != "full_attention" for kind in kinds):
        raise RuntimeError(
            "code predictor frame only unrolls full attention, got %s" % kinds)
    steps = int(config.num_code_groups) - 1
    if steps < 1:
        raise RuntimeError("code predictor frame needs residual steps, got %s" % steps)
    return steps


def _rotate_half(x):
    import torch

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


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


def _proj(linear, x):
    if linear.weight.dtype == x.dtype:
        return linear(x)
    return linear(x.to(dtype=linear.weight.dtype)).to(dtype=x.dtype)


def _causal_bias(hidden, q_len, past_len):
    """Additive causal mask. triu, not an arange compare: the compare freezes at the traced length."""
    import torch

    neg = torch.finfo(hidden.dtype).min
    attn = hidden.new_zeros(1, 1, q_len, past_len + q_len)
    upper = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=hidden.device), 1)
    if past_len == 0:
        return attn.masked_fill(upper, neg)
    attn[:, :, :, past_len:] = attn[:, :, :, past_len:].masked_fill(upper, neg)
    return attn


def _layer(layer, hidden, rope, attn_mask, pk, pv):
    import torch

    attn = layer.self_attn
    residual = hidden
    h = layer.input_layernorm(hidden)
    b, t, _ = h.shape
    head_dim = int(attn.head_dim)
    n_q = attn.q_proj.out_features // head_dim
    n_kv = attn.k_proj.out_features // head_dim
    q = _proj(attn.q_proj, h).view(b, t, n_q, head_dim)
    k = _proj(attn.k_proj, h).view(b, t, n_kv, head_dim)
    v = _proj(attn.v_proj, h).view(b, t, n_kv, head_dim)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = rope
    q, k = _apply_rope(q, k, cos, sin)
    if pk is not None:
        k = torch.cat((pk, k), dim=2)
        v = torch.cat((pv, v), dim=2)
    scale = float(getattr(attn, "scaling", head_dim ** -0.5))
    scores = torch.matmul(q, _repeat_kv(k, n_q // n_kv).transpose(2, 3)) * scale
    scores = scores + attn_mask
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    out = torch.matmul(probs, _repeat_kv(v, n_q // n_kv)).transpose(1, 2).contiguous()
    out = out.view(b, t, n_q * head_dim)
    h = residual + _proj(attn.o_proj, out)
    x = layer.post_attention_layernorm(h)
    h = h + layer.mlp(x)
    return h, k, v


def _pick(logits, uniform, temperature, top_k, top_p, do_sample, vocab):
    """Last-position token. Gumbel-max matches multinomial; a CDF compare compiled to a constant."""
    import torch

    scores = logits.reshape(-1, logits.shape[-1])[-1].float()
    scaled = scores / temperature.reshape(()).float()
    width = int(vocab)
    k_lim = top_k.reshape(()).to(dtype=torch.long)
    k_lim = torch.where(k_lim > 0, k_lim, torch.full_like(k_lim, width))
    vals, idx = torch.topk(scaled, width)
    keep = torch.arange(width, device=scaled.device) < k_lim
    vals = torch.where(keep, vals, torch.full_like(vals, float("-inf")))
    filtered = torch.full_like(scaled, float("-inf")).scatter(0, idx, vals)
    order = torch.argsort(filtered, descending=True)
    sorted_logits = filtered[order]
    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > top_p.reshape(()).float()
    remove = torch.cat((
        torch.zeros(1, dtype=torch.bool, device=remove.device), remove[:-1],
    ))
    sorted_logits = torch.where(remove, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
    filtered = torch.full_like(filtered, float("-inf")).scatter(0, order, sorted_logits)
    draw = uniform.float().reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(draw))
    sampled = torch.argmax((filtered + gumbel).float()).to(dtype=torch.long)
    greedy = torch.argmax(scores).to(dtype=torch.long)
    take = do_sample.reshape(()).to(dtype=torch.long) != 0
    return torch.where(take, sampled, greedy).clamp(0, width - 1)


def predictor_frame(predictor, steps, vocab):
    """Prefill of the two prompt rows, then one residual code per step. Weights stay this module's."""
    import torch.nn as nn

    n_layers = len(predictor.model.layers)

    class _Frame(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = predictor

        def _stack(self, hidden, past):
            import torch

            inner = self.predictor.model
            wdtype = inner.layers[0].self_attn.q_proj.weight.dtype
            hidden = hidden.to(dtype=wdtype)
            q_len = hidden.shape[1]
            past_len = 0 if past is None else int(past[0].shape[-2])
            # The position is the cache length. An arange of the export example would pin every frame there.
            pos = torch.arange(past_len, past_len + q_len, device=hidden.device).view(1, q_len)
            rope = inner.rotary_emb(hidden, pos)
            bias = _causal_bias(hidden, q_len, past_len)
            present = []
            for i, layer in enumerate(inner.layers[:n_layers]):
                pk = pv = None
                if past is not None:
                    pk = past[2 * i].to(dtype=wdtype)
                    pv = past[2 * i + 1].to(dtype=wdtype)
                hidden, nk, nv = _layer(layer, hidden, rope, bias, pk, pv)
                present.extend((nk, nv))
            hidden = inner.norm(hidden)
            return hidden, present

        def forward(self, prefill, temperature, top_k, top_p, do_sample, uniform):
            import torch

            proj = self.predictor.small_to_mtp_projection
            hidden, past = self._stack(proj(prefill), None)
            tokens = []
            tok = _pick(
                self.predictor.lm_head[0](hidden[:, -1:, :]),
                uniform[0], temperature, top_k, top_p, do_sample, vocab,
            )
            tokens.append(tok)
            tables = self.predictor.model.codec_embedding
            for step in range(1, steps):
                emb = proj(tables[step - 1](tok.view(1, 1)))
                hidden, past = self._stack(emb, past)
                tok = _pick(
                    self.predictor.lm_head[step](hidden[:, -1:, :]),
                    uniform[step], temperature, top_k, top_p, do_sample, vocab,
                )
                tokens.append(tok)
            return torch.stack(tokens)

    return _Frame()


def _state(model_dir):
    import json
    from pathlib import Path

    root = Path(model_dir)
    prefix = "talker.code_predictor."
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted({weight_map[key] for key in weight_map if key.startswith(prefix)})
        files = [root / name for name in names]
    else:
        files = sorted(root.glob("*.safetensors"))
    if not files:
        raise RuntimeError("code predictor frame found no safetensors under %s" % root)
    from safetensors import safe_open

    out = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    out[key[len(prefix):]] = handle.get_tensor(key)
    if not out:
        raise RuntimeError("checkpoint %s has no %s weights" % (root, prefix))
    return out


def _fresh_predictor(ov, model_dir):
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    )

    talker = ov.model.talker
    fresh = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        talker.code_predictor.config, talker.config)
    fresh.load_state_dict(_state(model_dir), strict=True)
    fresh.eval()
    for module in fresh.modules():
        for param in module.parameters(recurse=False):
            param.requires_grad_(False)
    return fresh


def _compile(xml, device, cache):
    import openvino as ov

    from ..ov_compile_lock import gpu_compile

    core = ov.Core()
    core.set_property({"ENABLE_MMAP": False})
    props = {
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_MMAP": False,
        "CACHE_DIR": cache,
    }
    with gpu_compile(device):
        return core.compile_model(xml, device, props)


def _export(fresh, steps, vocab, hidden, xml):
    import openvino as ov
    import torch

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    frame = predictor_frame(fresh, steps, vocab).eval()
    example = (
        torch.zeros(1, 2, hidden, dtype=torch.float32),
        torch.tensor(0.9, dtype=torch.float32),
        torch.tensor(50, dtype=torch.int64),
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(1, dtype=torch.int64),
        torch.full((steps, vocab), 0.5, dtype=torch.float32),
    )
    log.info("exporting code predictor frame %s", xml)
    with torch.inference_mode():
        model = ov.convert_model(frame, example_input=example)
    ov.save_model(model, xml)
    del model, frame


class _Result:
    def __init__(self, sequences):
        self.sequences = sequences


def _generate(orig, compiled, steps, vocab, hidden):
    import time
    import torch

    seen = {"n": 0}

    def generate(inputs_embeds=None, max_new_tokens=None, do_sample=True,
                 top_k=50, top_p=1.0, temperature=0.9, **kwargs):
        embeds = inputs_embeds
        fallback = (
            embeds is None
            or int(embeds.shape[0]) != 1
            or int(embeds.shape[1]) != 2
            or int(embeds.shape[-1]) != hidden
            or int(max_new_tokens or steps) != steps
        )
        if fallback:
            return orig(
                inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
                do_sample=do_sample, top_k=top_k, top_p=top_p,
                temperature=temperature, **kwargs)
        sample = True if do_sample is None else bool(do_sample)
        temp = 0.9 if temperature is None else float(temperature)
        if temp <= 0:
            sample = False
            temp = 1.0
        k = 50 if top_k is None else int(top_k)
        p = 1.0 if top_p is None else float(top_p)
        uniform = np.random.random((steps, vocab)).astype(np.float32)
        feed = [
            np.ascontiguousarray(embeds.detach().float().cpu().numpy()),
            np.array(temp, dtype=np.float32),
            np.array(k, dtype=np.int64),
            np.array(p, dtype=np.float32),
            np.array(1 if sample else 0, dtype=np.int64),
            np.ascontiguousarray(uniform),
        ]
        t0 = time.perf_counter()
        tokens = np.array(compiled(feed)[0]).reshape(-1)
        if seen["n"] == 0:
            log.info(
                "code predictor frame %.3fs tokens=%d",
                time.perf_counter() - t0, int(tokens.shape[0]))
        seen["n"] += 1
        sequences = torch.from_numpy(np.ascontiguousarray(tokens.astype(np.int64))).view(1, -1)
        return _Result(sequences)

    return generate


_STAMP_FP = ".qwen3tts-predictor-frame-v1"
_STAMP_INT8 = ".qwen3tts-predictor-frame-v2"


def install(ov, model_dir, device, cache, compress=None):
    """Replace code_predictor.generate with one compiled frame. The fifteen launches are the thing this removes."""
    global log
    from . import tts
    log = tts.log

    import gc

    predictor = ov.model.talker.code_predictor
    steps = residual_steps(predictor.config)
    vocab = int(predictor.config.vocab_size)
    hidden = int(ov.model.talker.config.hidden_size)
    if compress is None:
        xml_name, stamp_name = "qwen3tts-predictor-frame-v1.xml", _STAMP_FP
    else:
        xml_name, stamp_name = "qwen3tts-predictor-frame-v2.xml", _STAMP_INT8
    xml = os.path.join(cache, xml_name)
    stamp = os.path.join(cache, stamp_name)
    if not (os.path.isfile(xml) and os.path.isfile(stamp)):
        fresh = _fresh_predictor(ov, model_dir)
        try:
            _export(fresh, steps, vocab, hidden, xml)
        finally:
            del fresh
            gc.collect()
        if compress is not None:
            compress(xml)
        open(stamp, "w").close()
    compiled = _compile(xml, device, cache)
    orig = predictor.generate
    predictor.generate = _generate(orig, compiled, steps, vocab, hidden)
    log.info(
        "code predictor frame on OpenVINO %s %s steps=%d vocab=%d",
        device, "int8" if compress is not None else "fp32", steps, vocab)
