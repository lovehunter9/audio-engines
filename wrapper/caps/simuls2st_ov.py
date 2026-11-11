"""SimulS2ST on the Intel GPU. The official sampling loop stays in Python.

VoiceBox and Vocos are FP32. The thinker and talker stacks are one FP16 graph
each: a second FP32 copy of the 7B weights does not fit the 32Gi engine limit.
Prefill and decode share that graph, and the KV cache stays on the GPU. Host
tensors the official loop reads are FP32. The audio encoder stays on CPU. Its
packed-chunk mask is a Python loop over the input lengths, so one graph would
freeze the example's chunk count.
"""
import logging
import os

log = logging.getLogger("audio-simuls2st")

_STAMP = ".ov-simuls2st-llm-v1"
_MAX_T = 4096


def is_ov():
    return (os.environ.get("AUDIO_BASE") or "").strip() == "simuls2stov"


def install(agent, snap, device):
    """Compile the waveform path and both transformer stacks. Fail if a graph misses the GPU."""
    import torch

    inference = agent.inference
    voicebox = inference.voicebox
    vocoder = inference.vocoder
    for mod in (voicebox, vocoder, getattr(inference, "mel_model", None)):
        if mod is not None and hasattr(mod, "float"):
            mod.float()
    agent.dtype = torch.float32
    agent.model_dtype = torch.float32
    _estimator(voicebox, snap, device)
    _vocoder(vocoder, int(voicebox.mel_dim), snap, device)

    thinker = inference.base_model.thinker
    talker = inference.base_model.talker
    thinker.half()
    _stack(thinker.model, snap, device, "simuls2st_thinker")
    _linear(thinker.lm_head, snap, device, "simuls2st_thinker_head")
    talker.half()
    _stack(talker.model, snap, device, "simuls2st_talker")
    _linear(talker.codec_head, snap, device, "simuls2st_talker_head")
    _linear(talker.thinker_to_talker_proj, snap, device, "simuls2st_talker_proj")
    log.info(
        "simuls2st OV on %s: waveform fp32, thinker and talker stacks fp16; "
        "audio encoder stays on CPU",
        device,
    )


def _estimator(voicebox, snap, device):
    import torch

    from .. import tts_ov

    step = voicebox.diff_estimator
    mel = int(voicebox.mel_dim)
    hidden = int(voicebox.hidden_size)
    x = torch.zeros(1, 8, mel, dtype=torch.float32)
    t = torch.zeros(1, dtype=torch.float32)
    cond = torch.zeros(1, 8, hidden, dtype=torch.float32)
    mask = torch.ones(1, 8, dtype=torch.float32)
    _, xml, stamp = tts_ov.ir_paths(snap, "simuls2st_diff", ".ov-simuls2st-v1")
    compiled = tts_ov.compile_module(step, (x, t, cond, mask), xml, stamp, device)
    if compiled is None:
        raise RuntimeError("VoiceBox diff_estimator did not compile on %s" % device)

    def forward(x_in, t_in, cond_in, mask_in, *args, **kwargs):
        import numpy as np

        arrays = [
            np.ascontiguousarray(item.detach().float().cpu().numpy())
            for item in (x_in, t_in, cond_in, mask_in)
        ]
        out = compiled(*arrays)[0]
        return torch.from_numpy(np.ascontiguousarray(out))

    step.forward = forward
    tts_ov.release_parameters(step, "simuls2st diff_estimator")
    log.info("VoiceBox diff_estimator on OpenVINO %s", device)


def _vocoder(vocoder, mel_dim, snap, device):
    import torch

    from .. import tts_ov

    example = torch.zeros(1, mel_dim, 16, dtype=torch.float32)
    _, xml, stamp = tts_ov.ir_paths(snap, "simuls2st_vocos", ".ov-simuls2st-v1")
    compiled = tts_ov.compile_module(vocoder, example, xml, stamp, device)
    if compiled is None:
        raise RuntimeError("Vocos did not compile on %s" % device)

    def forward(mel, *args, **kwargs):
        import numpy as np

        arr = np.ascontiguousarray(mel.detach().float().cpu().numpy())
        out = compiled(arr)[0]
        return torch.from_numpy(np.ascontiguousarray(out))

    vocoder.forward = forward
    tts_ov.release_parameters(vocoder, "simuls2st vocos")
    log.info("Vocos on OpenVINO %s", device)


def _stack(model, snap, device, name):
    """One graph for prefill and decode. A second runner serves use_cache=False."""
    import torch

    from .. import tts_ov

    if not all(hasattr(model, attr) for attr in ("layers", "norm", "rotary_emb", "embed_tokens")):
        raise RuntimeError("%s is not a Qwen2.5-Omni text stack" % name)
    cfg = model.config
    cfg._attn_implementation = "eager"
    n_layers, n_kv, head_dim = tts_ov.kv_meta(cfg)
    hidden = int(cfg.hidden_size)
    sections = _mrope_sections(cfg)
    example_q = 8
    embeds = torch.zeros(1, example_q, hidden, dtype=torch.float16)
    mask = torch.ones(1, example_q, _MAX_T, dtype=torch.long)
    position = torch.arange(example_q, dtype=torch.long).view(1, -1).expand(3, -1).contiguous()
    slot = torch.arange(example_q, dtype=torch.long)
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, _MAX_T, head_dim, dtype=torch.float16))
        past.append(torch.zeros(1, n_kv, _MAX_T, head_dim, dtype=torch.float16))
    _, xml, stamp = tts_ov.ir_paths(snap, name, _STAMP)
    tts_ov.compile_causal(
        _slot_step(model, sections),
        (embeds, mask, position, slot, *past), xml, stamp, device,
        dynamize_ranks=(1, 2, 3), compile_graph=False, precision="f16",
    )
    _release_stack(model, name)
    compiled = tts_ov.compile_causal(
        _slot_step(model, sections),
        (embeds, mask, position, slot, *past), xml, stamp, device,
        dynamize_ranks=(1, 2, 3), precision="f16",
    )
    if compiled is None:
        raise RuntimeError("%s did not compile on %s" % (name, device))
    live = _SlotRunner(compiled, n_layers, n_kv, head_dim)
    once = _SlotRunner(compiled, n_layers, n_kv, head_dim)
    model.forward = _bound_forward(model, live, once)
    log.info("%s on OpenVINO %s (%d layers, one graph)", name, device, n_layers)


def _linear(mod, snap, device, name):
    import torch

    from .. import tts_ov

    if not isinstance(mod, torch.nn.Linear):
        raise RuntimeError("%s is not a Linear" % name)
    hidden = int(mod.in_features)
    example = torch.zeros(1, 8, hidden, dtype=mod.weight.dtype)
    _, xml, stamp = tts_ov.ir_paths(snap, name, _STAMP)
    tts_ov.compile_causal(
        mod, example, xml, stamp, device,
        dynamize_ranks=(3,), compile_graph=False, precision="f16",
    )
    tts_ov.release_parameters(mod, name)
    compiled = tts_ov.compile_causal(
        mod, example, xml, stamp, device,
        dynamize_ranks=(3,), precision="f16",
    )
    if compiled is None:
        raise RuntimeError("%s did not compile on %s" % (name, device))

    def forward(x):
        import numpy as np

        arr = np.ascontiguousarray(x.detach().half().cpu().numpy())
        out = compiled(arr)[0]
        return torch.from_numpy(np.ascontiguousarray(out)).float()

    mod.forward = forward
    log.info("%s on OpenVINO %s", name, device)


def _release_stack(model, label):
    from .. import tts_ov

    for i, layer in enumerate(model.layers):
        tts_ov.release_parameters(layer, "%s layer %d" % (label, i))
    tts_ov.release_parameters(model.norm, label + " norm")
    tts_ov.release_parameters(model.rotary_emb, label + " rope")


def _mrope_sections(cfg):
    scaling = getattr(cfg, "rope_scaling", None) or {}
    raw = scaling.get("mrope_section")
    if not raw:
        raise RuntimeError("text config has no mrope_section")
    return tuple(int(item) for item in raw)


def _bound_forward(model, live, once):
    def forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if inputs_embeds is None:
            if input_ids is None:
                raise RuntimeError("thinker/talker forward needs input_ids or inputs_embeds")
            inputs_embeds = model.embed_tokens(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise RuntimeError(
                "simuls2st ov stack is batch 1; got %s" % (tuple(inputs_embeds.shape),)
            )
        q = int(inputs_embeds.shape[1])
        cached = use_cache is None or bool(use_cache)
        runner = live if cached else once
        seen = _cache_len(past_key_values) if cached else 0
        if seen == 0:
            runner.reset()
        elif seen != runner.slot:
            raise RuntimeError(
                "kv length %d does not match the OpenVINO cache %d" % (seen, runner.slot)
            )
        position = _position(position_ids, seen, q)
        hidden = runner.step(inputs_embeds, position, attention_mask)
        past_out = _seen_cache(runner.slot) if cached else None
        if return_dict is False:
            return (hidden, past_out) if cached else (hidden,)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_out)

    return forward


def _cache_len(past):
    if past is None:
        return 0
    get = getattr(past, "get_seq_length", None)
    if get is None:
        return 0
    try:
        return int(get())
    except TypeError:
        return int(get(0))


def _position(position_ids, seen, q):
    import torch

    if position_ids is None:
        pos = torch.arange(seen, seen + q, dtype=torch.long)
        return pos.view(1, -1).expand(3, -1).contiguous()
    t = position_ids.detach().long()
    if t.dim() == 3:
        t = t[:, 0, :]
    elif t.dim() == 2 and not (t.shape[0] == 3 and t.shape[-1] == q):
        t = t[0].view(1, -1).expand(3, -1)
    if tuple(t.shape) != (3, q):
        raise RuntimeError("position_ids shape %s does not match q=%d" % (tuple(t.shape), q))
    return t.contiguous()


def _seen_cache(n):
    from transformers.cache_utils import DynamicCache

    class _Seen(DynamicCache):
        def __init__(self, length):
            super().__init__()
            self._n = int(length)
            self._seen_tokens = int(length)

        def get_seq_length(self, layer_idx=0):
            return self._n

        def batch_repeat_interleave(self, repeats):
            if int(repeats) != 1:
                raise RuntimeError("simuls2st OpenVINO cache is batch 1; num_beams must be 1")

        def batch_select_indices(self, indices):
            width = int(indices.shape[0]) if hasattr(indices, "shape") else 1
            if width != 1:
                raise RuntimeError("simuls2st OpenVINO cache is batch 1")

    return _Seen(n)


def _slot_step(inner, sections):
    import torch
    import torch.nn as nn

    from .. import tts_ov

    n_layers = len(inner.layers)

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds, attention_mask, position, slot, *past):
            layer0 = self.inner.layers[0]
            wdtype = layer0.self_attn.q_proj.weight.dtype
            hidden = inputs_embeds.to(dtype=wdtype)
            pos = position.view(3, 1, -1)
            cos, sin = self.inner.rotary_emb(hidden, pos)
            present = []
            for i, layer in enumerate(self.inner.layers[:n_layers]):
                hidden, nk, nv = _layer_slot(
                    layer, hidden, cos, sin, attention_mask, slot,
                    past[2 * i].to(dtype=wdtype),
                    past[2 * i + 1].to(dtype=wdtype),
                    sections,
                )
                present.extend([nk, nv])
            hidden = self.inner.norm(hidden).to(dtype=inputs_embeds.dtype)
            out_kv = [item.to(dtype=inputs_embeds.dtype) for item in present]
            return (hidden, *out_kv)

    return Step()


def _layer_slot(layer, hidden, cos, sin, attention_mask, slot, pk, pv, sections):
    import torch

    from .. import tts_ov

    attn = layer.self_attn
    residual = hidden
    h = layer.input_layernorm(hidden)
    b, t, _ = h.shape
    head_dim = int(getattr(attn, "head_dim", h.shape[-1] // attn.num_heads))
    n_q = attn.q_proj.out_features // head_dim
    n_kv = attn.k_proj.out_features // head_dim
    q = attn.q_proj(h).view(b, t, n_q, head_dim).transpose(1, 2)
    k = attn.k_proj(h).view(b, t, n_kv, head_dim).transpose(1, 2)
    v = attn.v_proj(h).view(b, t, n_kv, head_dim).transpose(1, 2)
    q, k = _mrope(q, k, cos, sin, sections)
    k = tts_ov._scatter_slot(pk, k, slot)
    v = tts_ov._scatter_slot(pv, v, slot)
    n_rep = n_q // n_kv
    scale = head_dim ** -0.5
    scores = torch.matmul(q, tts_ov._repeat_kv(k, n_rep).transpose(2, 3)) * scale
    keep = attention_mask.to(dtype=torch.bool).unsqueeze(1)
    scores = scores.masked_fill(~keep, torch.finfo(hidden.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    out = torch.matmul(probs, tts_ov._repeat_kv(v, n_rep))
    out = out.transpose(1, 2).contiguous().view(b, t, n_q * head_dim)
    h = residual + attn.o_proj(out)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, k, v


def _mrope(q, k, cos, sin, sections):
    import torch

    parts = sections * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(parts, dim=-1))], dim=-1).unsqueeze(1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(parts, dim=-1))], dim=-1).unsqueeze(1)

    def rotate(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


class _SlotRunner(object):
    def __init__(self, compiled_fn, n_layers, n_kv, head_dim):
        compiled = compiled_fn.compiled
        self.reqs = (compiled.create_infer_request(), compiled.create_infer_request())
        self.n_layers = int(n_layers)
        self.n_kv = int(n_kv)
        self.head_dim = int(head_dim)
        self.max_t = _MAX_T
        self.reset()

    def _zeros(self):
        import numpy as np
        import openvino as ov

        zeros = np.zeros((1, self.n_kv, self.max_t, self.head_dim), dtype=np.float16)
        return [ov.Tensor(np.array(zeros, copy=True)) for _ in range(2 * self.n_layers)]

    def reset(self):
        self.kv = self._zeros()
        self.slot = 0
        self.which = 0

    def step(self, embeds, position, attn_mask):
        import numpy as np
        import openvino as ov
        import torch

        q = int(embeds.shape[1])
        if q < 1:
            raise RuntimeError("simuls2st ov stack needs at least one token")
        if self.slot + q > self.max_t:
            raise RuntimeError(
                "simuls2st ov cache %d+%d exceeds %d" % (self.slot, q, self.max_t)
            )
        emb = np.ascontiguousarray(embeds.detach().half().cpu().numpy())
        pos = np.ascontiguousarray(position.detach().long().cpu().numpy())
        mask = np.zeros((1, q, self.max_t), dtype=np.int64)
        for i in range(q):
            mask[0, i, : self.slot + i + 1] = 1
        keep = _keep_row(attn_mask, self.slot + q)
        if keep is not None:
            mask[0, :, : keep.shape[0]] *= keep.reshape(1, -1)
        slot = np.arange(self.slot, self.slot + q, dtype=np.int64)
        req = self.reqs[self.which]
        req.set_input_tensor(0, ov.Tensor(emb))
        req.set_input_tensor(1, ov.Tensor(mask))
        req.set_input_tensor(2, ov.Tensor(pos))
        req.set_input_tensor(3, ov.Tensor(slot))
        for i, tensor in enumerate(self.kv):
            req.set_input_tensor(4 + i, tensor)
        req.infer()
        hidden = torch.from_numpy(np.array(req.get_output_tensor(0).data, copy=True)).float()
        self.kv = [req.get_output_tensor(i) for i in range(1, 1 + 2 * self.n_layers)]
        self.which ^= 1
        self.slot += q
        return hidden


def _keep_row(attn_mask, total):
    import numpy as np

    if attn_mask is None or not hasattr(attn_mask, "detach"):
        return None
    row = attn_mask.detach()
    if row.dim() != 2 or int(row.shape[-1]) < total:
        return None
    row = row[0, :total]
    if row.dtype.is_floating_point:
        row = row != 0
    return np.ascontiguousarray(row.long().cpu().numpy())
