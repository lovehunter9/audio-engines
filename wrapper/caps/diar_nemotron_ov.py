# Intel image only. OpenVINO runs pre-encode, the acoustic encoder, the speaker head and mel; the speaker cache stays on the host with BLAS pinned to one thread.
import logging
import os
import time

log = logging.getLogger("audio-diar-nemotron-ov")

_STAMP = "nemotron-ov5"
# Acoustic only. Bumping this re-exports the 31-layer encoder and leaves pre on ov5.
_ACOUSTIC_STAMP = "nemotron-ov9-sdpa"
# Separate from _STAMP so a head-graph change does not re-export the acoustic encoder.
_HEAD_STAMP = "nemotron-ov7-head"
_POOL_STAMP = "nemotron-ov7-pool"
_MEL_STAMP = "nemotron-ov10-mel"
_BASE = "nemotron-ov"
_clock = {}
_bucket = "ov"
_blas_limit = None


def _one_blas_thread():
    # Kept alive: threadpoolctl restores the old count when this object is collected.
    global _blas_limit
    if _blas_limit is None:
        from threadpoolctl import threadpool_limits

        _blas_limit = threadpool_limits(limits=1, user_api="blas")
    return _blas_limit


def wants():
    return (os.environ.get("AUDIO_BASE") or "").strip() == _BASE


def reset_clock():
    _clock.clear()


def add(bucket, dt):
    slot = _clock.setdefault(bucket, [0.0, 0])
    slot[0] += dt
    slot[1] += 1


def summary():
    order = ("decode", "mel", "pre_h2d", "pre_infer", "pre_d2h",
             "acoustic_h2d", "acoustic_infer", "acoustic_d2h",
             "transformer_infer", "head_h2d", "head_infer", "head_d2h", "head",
             "pool_h2d", "pool_infer", "pool_d2h", "pool_hit", "pool_ov",
             "concat", "concat_pad", "mask", "update", "update_async",
             "compress", "silence", "scores", "gather", "topk", "host_fallback",
             "mel_fallback", "step")
    parts = []
    for name in order:
        slot = _clock.get(name)
        if slot:
            parts.append("%s=%.2fs/%d" % (name, slot[0], slot[1]))
    step = _clock.get("step", [0.0])[0]
    inside = sum(_clock.get(name, [0.0])[0] for name in (
        "pre_h2d", "pre_infer", "pre_d2h", "acoustic_h2d", "acoustic_infer",
        "acoustic_d2h", "transformer_infer", "head_h2d", "head_infer", "head_d2h", "head",
        "pool_h2d", "pool_infer", "pool_d2h",
        "concat", "concat_pad", "mask", "update", "update_async"))
    parts.append("cache=%.2fs" % (step - inside))
    return " ".join(parts)


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
    """Compile the OpenVINO graphs and install them on this instance."""
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
            # FlexAttention's block mask does not trace, so this is scaled-dot-product with the same RoPE and norms.
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

    class SpeakerHead(torch.nn.Module):
        # Same ops as forward_infer's torch tail. Eval mode makes dropout a no-op, matching inference.
        def __init__(self):
            super().__init__()
            self.sm = model.sortformer_modules
            self.high_resolution = bool(model.high_resolution)
            self.upsample_factor = int(model.upsample_factor)

        def forward(self, emb, length):
            import torch.nn.functional as F

            mask = self.sm.length_to_mask(length, emb.shape[1])
            hidden = self.sm.upsample_hidden(emb)
            if self.high_resolution:
                mask = mask.repeat_interleave(self.upsample_factor, dim=1)
            logits = self.sm.forward_speaker_logits(hidden)
            preds = torch.sigmoid(logits) * mask.unsqueeze(-1)
            # Same windowed mean as SortformerModules.downsample_preds (no lengths passed).
            factor = self.upsample_factor
            if factor > 1:
                pooled = F.avg_pool1d(
                    preds.transpose(1, 2), kernel_size=factor, stride=factor,
                    ceil_mode=True, count_include_pad=False).transpose(1, 2)
            else:
                pooled = preds
            return preds, pooled

    pre = _compile(core, Pre(), cache, "pre", dev, _zeros(feat, 160))
    with torch.inference_mode():
        emb, elen = model._call_pre_encode(*_zeros(feat, 160))
    tail = int(getattr(model.encoder, "causal_tail_len", 0) or 0)
    if tail:
        raise RuntimeError(
            "OpenVINO acoustic export expects causal_tail_len 0, got %s" % tail)
    # RoPE slices a cos/sin buffer. Extend it before tracing so a longer stream still fits.
    model.encoder.update_max_seq_length(4096, emb.device)
    acoustic = _compile(
        core, Acoustic(), cache, "acoustic", dev, (emb, elen), stamp=_ACOUSTIC_STAMP)
    transformer = None
    if model.transformer_encoder is not None:
        with torch.inference_mode():
            states, slen = Acoustic()(emb, elen)
        transformer = _compile(
            core, Transformer(), cache, "transformer", dev, (states, slen))
    head = _compile(
        core, SpeakerHead(), cache, "head", dev,
        _zeros(int(model.sortformer_modules.hidden_size), 8),
        hint="f32", stamp=_HEAD_STAMP)
    # The head graph already pooled. This second graph covers a downsample of a different tensor.
    class Pool(torch.nn.Module):
        def forward(self, preds):
            import torch.nn.functional as F

            return F.avg_pool1d(
                preds.transpose(1, 2), kernel_size=8, stride=8,
                ceil_mode=True, count_include_pad=False).transpose(1, 2)

    pool = _compile(
        core, Pool(), cache, "pool", dev, (torch.zeros(1, 32, 8),),
        hint="f32", stamp=_POOL_STAMP)
    pooled_for = [None, None, 0]

    def _same_tensor(a, b):
        return a is b or (a.shape == b.shape and a.data_ptr() == b.data_ptr())

    def downsample_preds(preds, downsample_factor, lengths=None):
        factor = int(downsample_factor)
        if (lengths is None and pooled_for[0] is not None and factor == pooled_for[2]
                and _same_tensor(preds, pooled_for[0])):
            add("pool_hit", 0.0)
            return pooled_for[1]
        if lengths is None and factor == 8 and int(preds.shape[1]) % 8 == 0:
            global _bucket
            _bucket = "pool"
            add("pool_ov", 0.0)
            return _run(pool, preds)[0]
        raise RuntimeError(
            "Nemotron downsample has no OpenVINO path for factor=%s lengths=%s frames=%s"
            % (factor, lengths is not None, tuple(preds.shape)))

    model.sortformer_modules.downsample_preds = downsample_preds

    def call_pre(features, lengths):
        global _bucket
        _bucket = "pre"
        feats, lens = _run(pre, features, lengths)
        return feats, lens.to(dtype=torch.int64)

    def frontend(processed_signal, processed_signal_length, bypass_pre_encode=False):
        global _bucket
        if not bypass_pre_encode:
            processed_signal, processed_signal_length = call_pre(
                processed_signal, processed_signal_length)
        _bucket = "acoustic"
        feats, lens = _run(acoustic, processed_signal, processed_signal_length)
        return feats, lens.to(dtype=torch.int64)

    def forward_infer(emb_seq, emb_seq_length, return_logits=False):
        global _bucket
        if transformer is not None:
            _bucket = "transformer"
            emb_seq = _run(transformer, emb_seq, emb_seq_length)[0]
        if return_logits:
            # The exclusive-logits path is not the timed one. Keep its torch tail untouched.
            t0 = time.perf_counter()
            sm = model.sortformer_modules
            encoder_mask = sm.length_to_mask(emb_seq_length, emb_seq.shape[1])
            trans = sm.upsample_hidden(emb_seq)
            if model.high_resolution:
                output_mask = encoder_mask.repeat_interleave(model.upsample_factor, dim=1)
            else:
                output_mask = encoder_mask
            logits = sm.forward_speaker_logits(trans)
            preds = torch.sigmoid(logits) * output_mask.unsqueeze(-1)
            add("head", time.perf_counter() - t0)
            return preds, logits, sm.forward_activity_logits(trans)
        _bucket = "head"
        preds, pooled = _run(head, emb_seq, emb_seq_length)
        pooled_for[0] = preds
        pooled_for[1] = pooled
        pooled_for[2] = int(model.upsample_factor)
        return preds

    model._call_pre_encode = call_pre
    model.frontend_encoder = frontend
    model.forward_infer = forward_infer

    # One BLAS thread: the default pool, one thread per core, stretched the next feature copy by about a second.
    import math
    import numpy as np

    _one_blas_thread()
    sm = model.sortformer_modules
    width = int(model.encoder.d_model)
    n_spk = int(sm.n_spk)
    thr = np.float32(sm.pred_score_threshold)
    half_log = np.float32(math.log(0.5))

    def _host_tensor(tensor, last):
        return (torch.is_tensor(tensor) and tensor.ndim == 3 and tensor.dtype == torch.float32
                and tensor.device.type == "cpu" and int(tensor.shape[0]) == 1
                and int(tensor.shape[1]) > 0 and int(tensor.shape[2]) == last)

    raw_cat = torch.cat

    def host_cat(tensors, dim=0, out=None):
        seq = tuple(tensors)
        head = seq[0] if seq else None
        if (out is None and dim == 1 and len(seq) in (2, 3) and head is not None
                and head.ndim == 3 and head.dtype == torch.float32
                and head.device.type == "cpu"):
            last = int(head.shape[2])
            if last in (width, n_spk) and all(_host_tensor(tensor, last) for tensor in seq):
                parts = []
                for tensor in seq:
                    tensor = tensor.detach()
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    parts.append(tensor.numpy())
                return torch.from_numpy(np.concatenate(parts, axis=1))
            add("host_fallback", 0.0)
        if out is None:
            return raw_cat(seq, dim=dim)
        return raw_cat(seq, dim=dim, out=out)

    torch.cat = host_cat

    raw_scores = sm._get_log_pred_scores

    def log_scores(preds):
        if not _host_tensor(preds, n_spk):
            add("host_fallback", 0.0)
            return raw_scores(preds)
        x = preds.detach().contiguous().numpy()
        floor = thr
        log_probs = np.log(np.maximum(x, floor))
        other = np.log(np.maximum(np.float32(1.0) - x, floor))
        out = log_probs - other + other.sum(axis=2, keepdims=True) - half_log
        return torch.from_numpy(out)

    raw_gather = sm._gather_spkcache_and_preds

    def gather_cache(emb_seq, preds, topk_indices, is_disabled, mean_sil_emb):
        if not (_host_tensor(emb_seq, width) and _host_tensor(preds, n_spk)
                and torch.is_tensor(topk_indices) and topk_indices.ndim == 2
                and topk_indices.device.type == "cpu"
                and int(topk_indices.shape[0]) == 1 and int(topk_indices.shape[1]) > 0
                and torch.is_tensor(is_disabled) and is_disabled.device.type == "cpu"
                and torch.is_tensor(mean_sil_emb) and mean_sil_emb.ndim == 2
                and mean_sil_emb.dtype == torch.float32 and mean_sil_emb.device.type == "cpu"
                and int(mean_sil_emb.shape[-1]) == width):
            add("host_fallback", 0.0)
            return raw_gather(emb_seq, preds, topk_indices, is_disabled, mean_sil_emb)
        e = emb_seq.detach().contiguous().numpy()
        p = preds.detach().contiguous().numpy()
        ix = topk_indices.detach().to(dtype=torch.int64).contiguous().numpy()
        dis = is_disabled.detach().to(dtype=torch.bool).contiguous().numpy()
        sil = mean_sil_emb.detach().contiguous().numpy()
        index = ix[:, :, None]
        gathered = np.take_along_axis(e, index, axis=1)
        silence = np.broadcast_to(sil[:, None, :], gathered.shape)
        mask = dis[:, :, None]
        gathered = np.where(mask, silence, gathered)
        picked = np.take_along_axis(p, index, axis=1)
        picked = np.where(mask, np.float32(0.0), picked)
        return (torch.from_numpy(np.ascontiguousarray(gathered)),
                torch.from_numpy(np.ascontiguousarray(picked)))

    sm._get_log_pred_scores = log_scores
    sm._gather_spkcache_and_preds = gather_cache

    def _watch(bucket, fn):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                add(bucket, time.perf_counter() - t0)
        return wrapped

    watched = model.sortformer_modules
    for bucket, name in (
        ("concat", "concat_embs"),
        ("concat_pad", "concat_and_pad"),
        ("mask", "apply_mask_to_preds"),
        ("update", "streaming_update"),
        ("update_async", "streaming_update_async"),
        ("compress", "_compress_spkcache"),
        ("silence", "_get_silence_profile"),
        ("scores", "_get_log_pred_scores"),
        ("gather", "_gather_spkcache_and_preds"),
        ("topk", "_get_topk_indices"),
    ):
        if hasattr(watched, name):
            setattr(watched, name, _watch(bucket, getattr(watched, name)))

    # Same featurizer the torch path calls. FP32: this preprocessor loses accuracy in half precision.
    raw_pre = model.preprocessor

    class MelGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.featurizer = raw_pre.featurizer

        def forward(self, signal, length):
            return self.featurizer(signal, length)

    mel_graph = _compile(
        core, MelGraph(), cache, "mel", dev,
        (torch.zeros(1, 16000), torch.tensor([16000], dtype=torch.int64)),
        hint="f32", stamp=_MEL_STAMP)

    class _MelCall(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.normalize = getattr(raw_pre, "normalize", "NA")

        def forward(self, input_signal, length):
            global _bucket
            _bucket = "mel"
            try:
                feats, lens = _run(mel_graph, input_signal, length)
            except Exception:
                add("mel_fallback", 0.0)
                log.warning("OpenVINO mel fell back to torch", exc_info=True)
                return raw_pre(input_signal=input_signal, length=length)
            return feats, lens.to(dtype=torch.int64)

    model.preprocessor = _MelCall()

    log.info("OpenVINO encoders installed on %s", dev)
    return dev


def _math_attn(attn, x, length, causal):
    # Scaled-dot-product keeps the score matrix inside one GPU kernel.
    import torch
    import torch.nn.functional as F

    batches, steps, _ = x.shape
    heads, dim = attn.n_heads, attn.head_dim
    qkv = attn.w_qkv(x).view(batches, steps, 3, heads, dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    if attn.qk_norm:
        query = attn.q_norm(query).to(dtype=value.dtype)
        key = attn.k_norm(key).to(dtype=value.dtype)
    if attn.rope is not None:
        query, key = attn.rope(query, key)
    index = torch.arange(steps, device=x.device)
    valid = index.unsqueeze(0) < length.to(dtype=torch.int64).unsqueeze(1)
    allow = valid[:, None, None, :]
    if causal:
        causal_ok = index.unsqueeze(1) >= index.unsqueeze(0)
        allow = allow & causal_ok.view(1, 1, steps, steps)
    fill = torch.finfo(query.dtype).min
    mask = torch.zeros(allow.shape, dtype=query.dtype, device=x.device)
    mask = mask.masked_fill(~allow, fill)
    mixed = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
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


def _compile(core, module, cache, name, dev, example, hint="f16", stamp=None):
    import openvino as ov
    import torch

    stamp = stamp or _STAMP
    xml = os.path.join(cache, "%s.xml" % name)
    stamp_path = os.path.join(cache, "%s.stamp" % name)
    if not (os.path.isfile(xml) and _read(stamp_path) == stamp):
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
        with open(stamp_path, "w", encoding="utf-8") as fh:
            fh.write(stamp)
    cfg = {"CACHE_DIR": cache}
    if hint:
        cfg["INFERENCE_PRECISION_HINT"] = hint
    return core.compile_model(xml, dev, cfg)


def _dynamic(exported):
    # Batch stays 1 and only the time axis is dynamic. A list of shapes is read as one dimension.
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

    t0 = time.perf_counter()
    feeds = {port: tensor.detach().cpu().contiguous().numpy()
             for port, tensor in zip(compiled.inputs, tensors)}
    t1 = time.perf_counter()
    result = compiled(feeds)
    t2 = time.perf_counter()
    like = tensors[0]
    outs = []
    for port in compiled.outputs:
        out = torch.from_numpy(np.ascontiguousarray(np.array(result[port])))
        if out.dtype.is_floating_point and like.dtype.is_floating_point:
            out = out.to(dtype=like.dtype)
        outs.append(out)
    t3 = time.perf_counter()
    add(_bucket + "_h2d", t1 - t0)
    add(_bucket + "_infer", t2 - t1)
    add(_bucket + "_d2h", t3 - t2)
    return outs


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""
