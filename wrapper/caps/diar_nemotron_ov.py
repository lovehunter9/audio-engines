# Intel image only. The CUDA image never sets AUDIO_BASE=nemotron-ov, so this module is not imported.
# Streaming still owns the FIFO, speaker cache and arrival order. OpenVINO replaces three calls:
# conv pre_encode, the acoustic Conformer (bypass_pre_encode) plus encoder_proj, and the rope
# transformer. Upsample and the speaker head stay in torch: upsample changes the time axis, and
# the head is a linear on those frames.
import logging
import os

log = logging.getLogger("audio-diar-nemotron-ov")

_STAMP = "nemotron-ov3"
_BASE = "nemotron-ov"


def wants():
    return (os.environ.get("AUDIO_BASE") or "").strip() == _BASE


def device():
    # Local copy of the Intel rule. Importing ovutil would claim --device, which this cap already closed.
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel"):
        return "GPU"
    raw = (os.environ.get("REQUIRED_GPU_MEMORY") or "").strip()
    if raw and raw != "0":
        return "GPU"
    return "CPU"


def install(model):
    """Export and compile the three IRs, then replace those methods on this instance only."""
    import openvino as ov
    import torch

    dev = device()
    cache = os.path.join(os.environ.get("HF_HOME") or "/cache/hf", "openvino-nemotron")
    os.makedirs(cache, exist_ok=True)
    core = ov.Core()
    feat = _feat_dim(model)
    log.info("OpenVINO device=%s feat=%d cache=%s", dev, feat, cache)

    class Pre(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder
            self.fn = model._call_pre_encode

        def forward(self, features, lengths):
            return self.fn(features, lengths)

    class Acoustic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder
            proj = model.sortformer_modules.encoder_proj
            self.proj = proj if proj is not None else torch.nn.Identity()
            self.has_proj = proj is not None

        def forward(self, emb, length):
            # forward_internal builds a FlexAttention block mask. That call does not trace
            # (unordered_map::at). Same blocks, RoPE and norms; attention is the matmul form.
            hidden, hidden_len = _acoustic_hidden(self.encoder, emb, length)
            hidden = hidden.transpose(1, 2)
            if self.has_proj:
                hidden = self.proj(hidden)
            return hidden, hidden_len

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.transformer_encoder
            self.head = model.sortformer_modules

        def forward(self, states, lengths):
            mask = self.head.length_to_mask(lengths, states.shape[1])
            return self.encoder(encoder_states=states, encoder_mask=mask)

    pre = _compile(core, Pre(), cache, "pre", dev, _zeros(feat, 160))
    with torch.inference_mode():
        emb, elen = model._call_pre_encode(*_zeros(feat, 160))
    tail = int(getattr(model.encoder, "causal_tail_len", 0) or 0)
    if tail:
        raise RuntimeError(
            "OpenVINO acoustic export expects causal_tail_len 0, got %s" % tail)
    # RoPE slices a cos/sin buffer. Extend it before tracing so a longer stream still fits.
    model.encoder.update_max_seq_length(4096, emb.device)
    acoustic = _compile(core, Acoustic(), cache, "acoustic", dev, (emb, elen))
    transformer = None
    if model.transformer_encoder is not None:
        with torch.inference_mode():
            states, slen = Acoustic()(emb, elen)
        transformer = _compile(
            core, Transformer(), cache, "transformer", dev, (states, slen))

    def call_pre(features, lengths):
        feats, lens = _run(pre, features, lengths)
        return feats, lens.to(dtype=torch.int64)

    def frontend(processed_signal, processed_signal_length, bypass_pre_encode=False):
        if not bypass_pre_encode:
            processed_signal, processed_signal_length = call_pre(
                processed_signal, processed_signal_length)
        feats, lens = _run(acoustic, processed_signal, processed_signal_length)
        return feats, lens.to(dtype=torch.int64)

    def forward_infer(emb_seq, emb_seq_length, return_logits=False):
        if transformer is None:
            trans = emb_seq
        else:
            trans = _run(transformer, emb_seq, emb_seq_length)[0]
        sm = model.sortformer_modules
        encoder_mask = sm.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans = sm.upsample_hidden(trans)
        if model.high_resolution:
            output_mask = encoder_mask.repeat_interleave(model.upsample_factor, dim=1)
        else:
            output_mask = encoder_mask
        logits = sm.forward_speaker_logits(trans)
        preds = torch.sigmoid(logits) * output_mask.unsqueeze(-1)
        if return_logits:
            return preds, logits, sm.forward_activity_logits(trans)
        return preds

    model._call_pre_encode = call_pre
    model.frontend_encoder = frontend
    model.forward_infer = forward_infer
    log.info("OpenVINO encoders installed on %s", dev)
    return dev


def _math_attn(attn, x, length, causal):
    import torch

    batches, steps, _ = x.shape
    heads, dim = attn.n_heads, attn.head_dim
    qkv = attn.w_qkv(x).view(batches, steps, 3, heads, dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    if attn.qk_norm:
        query = attn.q_norm(query).to(dtype=value.dtype)
        key = attn.k_norm(key).to(dtype=value.dtype)
    if attn.rope is not None:
        query, key = attn.rope(query, key)
    scores = torch.matmul(query, key.transpose(-1, -2)) * (dim ** -0.5)
    index = torch.arange(steps, device=x.device)
    valid = index.unsqueeze(0) < length.to(dtype=torch.int64).unsqueeze(1)
    fill = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~valid[:, None, None, :], fill)
    if causal:
        allow = index.unsqueeze(1) >= index.unsqueeze(0)
        scores = scores.masked_fill(~allow.view(1, 1, steps, steps), fill)
    weights = torch.softmax(scores, dim=-1)
    mixed = torch.matmul(weights, value)
    mixed = mixed.transpose(1, 2).contiguous().view(batches, steps, attn.d_model)
    return attn.out_proj(mixed)


def _acoustic_hidden(encoder, emb, length):
    import torch

    hidden = emb
    if encoder.self_attention_model == "rope":
        if encoder.xscale:
            hidden = hidden * encoder.xscale
        hidden = encoder.dropout_pre_encoder(hidden)
    hidden = encoder.embed_norm(hidden)
    causal = encoder.attn_mode == "causal"
    for layer in encoder.layers:
        attended = _math_attn(layer.attn, layer.norm1(hidden), length, causal)
        hidden = hidden + layer.drop(attended)
        hidden = hidden + layer.drop(layer.ffn(layer.norm2(hidden)))
    hidden = encoder.final_norm(hidden)
    if encoder.out_proj is not None:
        hidden = encoder.out_proj(hidden)
    return hidden.transpose(1, 2), length.to(dtype=torch.int64)


def _feat_dim(model):
    for obj in (model.encoder, getattr(model, "preprocessor", None)):
        if obj is None:
            continue
        for name in ("_feat_in", "feat_in"):
            val = getattr(obj, name, None)
            if isinstance(val, int) and val > 0:
                return val
    return 128


def _zeros(feat, steps):
    import torch

    return torch.zeros(1, steps, feat), torch.tensor([steps], dtype=torch.int64)


def _compile(core, module, cache, name, dev, example):
    import openvino as ov
    import torch

    xml = os.path.join(cache, "%s.xml" % name)
    stamp = os.path.join(cache, "%s.stamp" % name)
    if not (os.path.isfile(xml) and _read(stamp) == _STAMP):
        module.eval()
        log.info("exporting %s", name)
        try:
            with torch.inference_mode():
                exported = ov.convert_model(module, example_input=example)
        except Exception as e:
            raise RuntimeError(
                "OpenVINO export of Nemotron %s failed: %s" % (name, e)) from e
        _dynamic(exported)
        ov.save_model(exported, xml)
        with open(stamp, "w", encoding="utf-8") as fh:
            fh.write(_STAMP)
    return core.compile_model(xml, dev, {"CACHE_DIR": cache})


def _dynamic(exported):
    # reshape wants {input: PartialShape}. A list is one shape, so a PartialShape in that list
    # is read as a dimension and rejected. Batch stays 1; only the time axis is dynamic.
    import openvino as ov

    shapes = {}
    for port in exported.inputs:
        rank = len(port.partial_shape)
        dims = []
        for i in range(rank):
            dim = port.partial_shape[i]
            if rank >= 2 and i == 1:
                dims.append(-1)
            else:
                dims.append(-1 if not dim.is_static else dim.get_length())
        shapes[port.get_any_name()] = ov.PartialShape(dims)
    exported.reshape(shapes)


def _run(compiled, *tensors):
    import numpy as np
    import torch

    feeds = {port: tensor.detach().cpu().contiguous().numpy()
             for port, tensor in zip(compiled.inputs, tensors)}
    result = compiled(feeds)
    like = tensors[0]
    outs = []
    for port in compiled.outputs:
        out = torch.from_numpy(np.ascontiguousarray(np.array(result[port])))
        if out.dtype.is_floating_point and like.dtype.is_floating_point:
            out = out.to(dtype=like.dtype)
        outs.append(out)
    return outs


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""
