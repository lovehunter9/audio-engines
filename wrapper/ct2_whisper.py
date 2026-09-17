# Turn a Systran / faster-whisper CT2 snapshot into a transformers Whisper dir
# using only files already on disk. Never hits the Hub for a second repo.
import json
import logging
import os
import shutil
import struct

log = logging.getLogger("audio-ct2-whisper")

_TOKEN_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "normalizer.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "generation_config.json", "config.json",
)

# CTranslate2 DataType order (include/ctranslate2/types.h).
_CT2_DTYPES = {
    0: "float32",
    1: "int8",
    2: "int16",
    3: "int32",
    4: "float16",
    5: "bfloat16",
}


def is_ct2(path):
    return os.path.isfile(os.path.join(path, "model.bin"))


def is_transformers(path):
    return any(os.path.isfile(os.path.join(path, n)) for n in (
        "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))


def is_whisper_ir(path):
    # WhisperPipeline compiles the decoder and sets beam_idx. A stateless
    # `--task automatic-speech-recognition` export 500s with
    # "Port for tensor name beam_idx was not found."
    if not path or not os.path.isdir(path):
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    if not (os.path.isfile(enc) and os.path.isfile(dec)):
        return False
    try:
        with open(dec, "rb") as f:
            head = f.read(1048576)
    except OSError:
        return False
    return b"beam_idx" in head


def _copy_sidecar(src, dest):
    os.makedirs(dest, exist_ok=True)
    for name in _TOKEN_FILES:
        a, b = os.path.join(src, name), os.path.join(dest, name)
        if os.path.isfile(a) and not os.path.isfile(b):
            shutil.copy2(a, b)


def _read_ct2_string(fh):
    raw = fh.read(2)
    if len(raw) != 2:
        raise RuntimeError("truncated CT2 string length")
    n = struct.unpack("<H", raw)[0]
    buf = fh.read(n)
    if len(buf) != n:
        raise RuntimeError("truncated CT2 string")
    return buf.split(b"\0", 1)[0].decode("utf-8")


def _bf16_to_f32(u16):
    import numpy as np

    bits = u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def _dequant_int8(state):
    import numpy as np

    out = {}
    for name, arr in state.items():
        if name.endswith("_scale") or name.endswith("_zero"):
            continue
        scale = state.get(name + "_scale")
        if arr.dtype == np.int8 and scale is not None:
            w = arr.astype(np.float32)
            s = scale.astype(np.float32)
            if s.shape == () or s.size == 1:
                w = w / s.reshape(())
            elif s.shape[0] == w.shape[0]:
                w = w / s.reshape([s.shape[0]] + [1] * (w.ndim - 1))
            else:
                w = w / s
            out[name] = np.ascontiguousarray(w)
        else:
            out[name] = arr
    return out


def _read_ct2_bin(path):
    """Read model.bin the way model_spec.py writes it. No libctranslate2."""
    import numpy as np

    with open(path, "rb") as fh:
        head = fh.read(4)
        if len(head) != 4:
            raise RuntimeError("%s is not a CTranslate2 model.bin" % path)
        version = struct.unpack("<I", head)[0]
        if version < 4 or version > 6:
            raise RuntimeError("%s binary version %d is not a readable CT2 dump" % (path, version))
        spec = _read_ct2_string(fh)
        rev_raw = fh.read(4)
        nvar_raw = fh.read(4)
        if len(rev_raw) != 4 or len(nvar_raw) != 4:
            raise RuntimeError("%s header is truncated" % path)
        revision = struct.unpack("<I", rev_raw)[0]
        nvar = struct.unpack("<I", nvar_raw)[0]
        state = {}
        for _ in range(nvar):
            name = _read_ct2_string(fh)
            rank_b = fh.read(1)
            if len(rank_b) != 1:
                raise RuntimeError("truncated CT2 rank in %s" % path)
            rank = rank_b[0]
            dim_raw = fh.read(4 * rank) if rank else b""
            if rank and len(dim_raw) != 4 * rank:
                raise RuntimeError("truncated CT2 shape for %s" % name)
            dims = struct.unpack("<%dI" % rank, dim_raw) if rank else ()
            tid_b = fh.read(1)
            nb_raw = fh.read(4)
            if len(tid_b) != 1 or len(nb_raw) != 4:
                raise RuntimeError("truncated CT2 dtype for %s" % name)
            type_id = tid_b[0]
            nbytes = struct.unpack("<I", nb_raw)[0]
            raw = fh.read(nbytes)
            if len(raw) != nbytes:
                raise RuntimeError("truncated CT2 tensor %s in %s" % (name, path))
            dt = _CT2_DTYPES.get(type_id)
            if dt is None:
                raise RuntimeError("CT2 tensor %s has unknown type_id %d" % (name, type_id))
            if dt == "bfloat16":
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(dims)
                arr = np.ascontiguousarray(_bf16_to_f32(arr))
            elif dt == "int8":
                arr = np.array(np.frombuffer(raw, dtype=np.int8).reshape(dims), copy=True)
            else:
                arr = np.ascontiguousarray(
                    np.frombuffer(raw, dtype=np.dtype(dt)).reshape(dims).astype(np.float32, copy=False))
            state[name] = arr
        alias_head = fh.read(4)
        if len(alias_head) == 4:
            nalias = struct.unpack("<I", alias_head)[0]
            for _ in range(nalias):
                alias = _read_ct2_string(fh)
                src = _read_ct2_string(fh)
                if src in state and alias not in state:
                    state[alias] = state[src]
                for suf in ("_scale", "_zero"):
                    a2, s2 = alias + suf, src + suf
                    if s2 in state and a2 not in state:
                        state[a2] = state[s2]
    if not state:
        raise RuntimeError("dumped no tensors from %s" % path)
    log.info("read %d CT2 tensors from %s (spec=%s rev=%s v=%s)",
             len(state), path, spec, revision, version)
    return _dequant_int8(state)


def _dump_ct2_state_dict(src):
    # Parse model.bin in Python. The Intel CT2 wheel SIGSEGVs inside
    # ctranslate2.models.Whisper(...) even with device="cpu".
    path = os.path.join(src, "model.bin")
    if not os.path.isfile(path):
        raise RuntimeError("%s has no model.bin" % src)
    state = _read_ct2_bin(path)
    log.info("dumped %d CT2 tensors from %s", len(state), src)
    return state


def _split_rows(t, n):
    """Inverse of CTranslate2 fuse_linear: concatenate along the output axis."""
    if t is None:
        return [None] * n
    if t.shape[0] % n:
        raise RuntimeError("cannot split %s into %d row blocks" % (t.shape, n))
    w = t.shape[0] // n
    return [t[i * w:(i + 1) * w] for i in range(n)]


def _to_hf_names(ct2_state, config):
    # Inverse of CTranslate2 WhisperLoader. Self-attn stores fused QKV in
    # linear[0] and out in linear[1]. Cross-attn stores Q / fused KV / out.
    n_enc = int(config.get("encoder_layers") or config.get("num_hidden_layers") or 32)
    n_dec = int(config.get("decoder_layers") or config.get("num_hidden_layers") or 32)
    out = {}

    def take(src, dst, permute=None):
        if src not in ct2_state:
            return
        t = ct2_state[src]
        if permute is not None:
            t = t.transpose(permute)
        out[dst] = t

    def put(dst, t):
        if t is not None:
            out[dst] = t

    def lin(prefix, kind, i, suffix):
        # CT2 visit_spec writes list i as name_i (linear_0), not linear_layers/0.
        for pat in ("%s/%s/linear_%d/%s", "%s/%s/linear_layers/%d/%s"):
            t = ct2_state.get(pat % (prefix, kind, i, suffix))
            if t is not None:
                return t
        return None

    def attn(prefix, dest, kind):
        # kind: self (2 linears) or cross (3 linears). A 4-way dump is leftover
        # from the first converter and is only kept so the tiny unit dump still maps.
        w0, w1 = lin(prefix, kind, 0, "weight"), lin(prefix, kind, 1, "weight")
        w2, w3 = lin(prefix, kind, 2, "weight"), lin(prefix, kind, 3, "weight")
        b0, b1 = lin(prefix, kind, 0, "bias"), lin(prefix, kind, 1, "bias")
        b2, b3 = lin(prefix, kind, 2, "bias"), lin(prefix, kind, 3, "bias")
        q = dest + (".self_attn" if kind == "self_attention" else ".encoder_attn")
        if w3 is not None:
            put(q + ".q_proj.weight", w0)
            put(q + ".q_proj.bias", b0)
            put(q + ".k_proj.weight", w1)
            put(q + ".k_proj.bias", b1)
            put(q + ".v_proj.weight", w2)
            put(q + ".v_proj.bias", b2)
            put(q + ".out_proj.weight", w3)
            put(q + ".out_proj.bias", b3)
            return
        if kind == "self_attention" and w0 is not None and w0.shape[0] % 3 == 0:
            qw, kw, vw = _split_rows(w0, 3)
            qb, kb, vb = _split_rows(b0, 3)
            put(q + ".q_proj.weight", qw)
            put(q + ".q_proj.bias", qb)
            put(q + ".k_proj.weight", kw)
            put(q + ".k_proj.bias", kb)
            put(q + ".v_proj.weight", vw)
            put(q + ".v_proj.bias", vb)
            put(q + ".out_proj.weight", w1)
            put(q + ".out_proj.bias", b1)
            return
        if kind == "attention" and w1 is not None and w1.shape[0] % 2 == 0:
            put(q + ".q_proj.weight", w0)
            put(q + ".q_proj.bias", b0)
            kw, vw = _split_rows(w1, 2)
            kb, vb = _split_rows(b1, 2)
            put(q + ".k_proj.weight", kw)
            put(q + ".k_proj.bias", kb)
            put(q + ".v_proj.weight", vw)
            put(q + ".v_proj.bias", vb)
            put(q + ".out_proj.weight", w2)
            put(q + ".out_proj.bias", b2)
            return
        put(q + ".q_proj.weight", w0)
        put(q + ".q_proj.bias", b0)
        put(q + ".out_proj.weight", w1)
        put(q + ".out_proj.bias", b1)

    take("encoder/conv1/weight", "model.encoder.conv1.weight")
    take("encoder/conv1/bias", "model.encoder.conv1.bias")
    take("encoder/conv2/weight", "model.encoder.conv2.weight")
    take("encoder/conv2/bias", "model.encoder.conv2.bias")
    take("encoder/position_encodings", "model.encoder.embed_positions.weight")
    take("encoder/position_encodings/encodings", "model.encoder.embed_positions.weight")
    take("encoder/layer_norm/gamma", "model.encoder.layer_norm.weight")
    take("encoder/layer_norm/beta", "model.encoder.layer_norm.bias")
    for i in range(n_enc):
        p = "encoder/layer_%d" % i
        q = "model.encoder.layers.%d" % i
        attn(p, q, "self_attention")
        take("%s/ffn/linear_0/weight" % p, "%s.fc1.weight" % q)
        take("%s/ffn/linear_0/bias" % p, "%s.fc1.bias" % q)
        take("%s/ffn/linear_1/weight" % p, "%s.fc2.weight" % q)
        take("%s/ffn/linear_1/bias" % p, "%s.fc2.bias" % q)
        take("%s/self_attention/layer_norm/gamma" % p, "%s.self_attn_layer_norm.weight" % q)
        take("%s/self_attention/layer_norm/beta" % p, "%s.self_attn_layer_norm.bias" % q)
        take("%s/ffn/layer_norm/gamma" % p, "%s.final_layer_norm.weight" % q)
        take("%s/ffn/layer_norm/beta" % p, "%s.final_layer_norm.bias" % q)
    take("decoder/embeddings", "model.decoder.embed_tokens.weight")
    take("decoder/position_encodings", "model.decoder.embed_positions.weight")
    take("decoder/position_encodings/encodings", "model.decoder.embed_positions.weight")
    take("decoder/layer_norm/gamma", "model.decoder.layer_norm.weight")
    take("decoder/layer_norm/beta", "model.decoder.layer_norm.bias")
    for i in range(n_dec):
        p = "decoder/layer_%d" % i
        q = "model.decoder.layers.%d" % i
        attn(p, q, "self_attention")
        attn(p, q, "attention")
        take("%s/ffn/linear_0/weight" % p, "%s.fc1.weight" % q)
        take("%s/ffn/linear_0/bias" % p, "%s.fc1.bias" % q)
        take("%s/ffn/linear_1/weight" % p, "%s.fc2.weight" % q)
        take("%s/ffn/linear_1/bias" % p, "%s.fc2.bias" % q)
        take("%s/self_attention/layer_norm/gamma" % p, "%s.self_attn_layer_norm.weight" % q)
        take("%s/self_attention/layer_norm/beta" % p, "%s.self_attn_layer_norm.bias" % q)
        take("%s/attention/layer_norm/gamma" % p, "%s.encoder_attn_layer_norm.weight" % q)
        take("%s/attention/layer_norm/beta" % p, "%s.encoder_attn_layer_norm.bias" % q)
        take("%s/ffn/layer_norm/gamma" % p, "%s.final_layer_norm.weight" % q)
        take("%s/ffn/layer_norm/beta" % p, "%s.final_layer_norm.bias" % q)
    take("decoder/projection/weight", "proj_out.weight")
    if "proj_out.weight" not in out and "model.decoder.embed_tokens.weight" in out:
        out["proj_out.weight"] = out["model.decoder.embed_tokens.weight"]
    if len(out) < 8:
        raise RuntimeError(
            "CT2 name map matched %d tensors (names like %s); converter needs a fuller dump"
            % (len(out), list(ct2_state)[:8]))
    return out


def _count_layers(state, prefix):
    n = 0
    keys = (
        "%s/layer_%d/self_attention/linear_0/weight",
        "%s/layer_%d/self_attention/linear_layers/0/weight",
    )
    while any((k % (prefix, n)) in state for k in keys):
        n += 1
    return n


def _whisper_hf_config(state):
    # Systran config.json is CT2 (alignment_heads / suppress_ids), not transformers.
    embed = state.get("decoder/embeddings")
    enc_pos = state.get("encoder/position_encodings")
    dec_pos = state.get("decoder/position_encodings")
    conv1 = state.get("encoder/conv1/weight")
    fc1 = state.get("encoder/layer_0/ffn/linear_0/weight")
    d_model = int(embed.shape[-1]) if embed is not None else 1280
    vocab = int(embed.shape[0]) if embed is not None else 51866
    n_enc = _count_layers(state, "encoder") or 32
    n_dec = _count_layers(state, "decoder") or 32
    n_mels = int(conv1.shape[1]) if conv1 is not None and getattr(conv1, "ndim", 0) == 3 else 128
    n_heads = max(1, d_model // 64)
    ffn = int(fc1.shape[0]) if fc1 is not None else d_model * 4
    return {
        "model_type": "whisper",
        "architectures": ["WhisperForConditionalGeneration"],
        "activation_function": "gelu",
        "d_model": d_model,
        "encoder_layers": n_enc,
        "decoder_layers": n_dec,
        "num_hidden_layers": n_enc,
        "encoder_attention_heads": n_heads,
        "decoder_attention_heads": n_heads,
        "encoder_ffn_dim": ffn,
        "decoder_ffn_dim": ffn,
        "max_source_positions": int(enc_pos.shape[0]) if enc_pos is not None else 1500,
        "max_target_positions": int(dec_pos.shape[0]) if dec_pos is not None else 448,
        "num_mel_bins": n_mels,
        "vocab_size": vocab,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "decoder_layerdrop": 0.0,
        "init_std": 0.02,
        "scale_embedding": False,
        "use_cache": True,
        "is_encoder_decoder": True,
        "bos_token_id": 50257,
        "eos_token_id": 50257,
        "pad_token_id": 50256,
        "decoder_start_token_id": 50258,
        "max_length": 448,
        "begin_suppress_tokens": [220, 50257],
    }


# openai/whisper-large-v3. GenAI WhisperPipeline::generate reads this map
# and 500s multilingual models when it is missing.
_WHISPER_LANG_TO_ID = {
    "<|af|>": 50327, "<|am|>": 50334, "<|ar|>": 50272, "<|as|>": 50350,
    "<|az|>": 50304, "<|ba|>": 50355, "<|be|>": 50330, "<|bg|>": 50292,
    "<|bn|>": 50302, "<|bo|>": 50347, "<|br|>": 50309, "<|bs|>": 50315,
    "<|ca|>": 50270, "<|cs|>": 50283, "<|cy|>": 50297, "<|da|>": 50285,
    "<|de|>": 50261, "<|el|>": 50281, "<|en|>": 50259, "<|es|>": 50262,
    "<|et|>": 50307, "<|eu|>": 50310, "<|fa|>": 50300, "<|fi|>": 50277,
    "<|fo|>": 50338, "<|fr|>": 50265, "<|gl|>": 50319, "<|gu|>": 50333,
    "<|haw|>": 50352, "<|ha|>": 50354, "<|he|>": 50279, "<|hi|>": 50276,
    "<|hr|>": 50291, "<|ht|>": 50339, "<|hu|>": 50286, "<|hy|>": 50312,
    "<|id|>": 50275, "<|is|>": 50311, "<|it|>": 50274, "<|ja|>": 50266,
    "<|jw|>": 50356, "<|ka|>": 50329, "<|kk|>": 50316, "<|km|>": 50323,
    "<|kn|>": 50306, "<|ko|>": 50264, "<|la|>": 50294, "<|lb|>": 50345,
    "<|ln|>": 50353, "<|lo|>": 50336, "<|lt|>": 50293, "<|lv|>": 50301,
    "<|mg|>": 50349, "<|mi|>": 50295, "<|mk|>": 50308, "<|ml|>": 50296,
    "<|mn|>": 50314, "<|mr|>": 50320, "<|ms|>": 50282, "<|mt|>": 50343,
    "<|my|>": 50346, "<|ne|>": 50313, "<|nl|>": 50271, "<|nn|>": 50342,
    "<|no|>": 50288, "<|oc|>": 50328, "<|pa|>": 50321, "<|pl|>": 50269,
    "<|ps|>": 50340, "<|pt|>": 50267, "<|ro|>": 50284, "<|ru|>": 50263,
    "<|sa|>": 50344, "<|sd|>": 50332, "<|si|>": 50322, "<|sk|>": 50298,
    "<|sl|>": 50305, "<|sn|>": 50324, "<|so|>": 50326, "<|sq|>": 50317,
    "<|sr|>": 50303, "<|su|>": 50357, "<|sv|>": 50273, "<|sw|>": 50318,
    "<|ta|>": 50287, "<|te|>": 50299, "<|tg|>": 50331, "<|th|>": 50289,
    "<|tk|>": 50341, "<|tl|>": 50348, "<|tr|>": 50268, "<|tt|>": 50351,
    "<|uk|>": 50280, "<|ur|>": 50290, "<|uz|>": 50337, "<|vi|>": 50278,
    "<|yi|>": 50335, "<|yo|>": 50325, "<|yue|>": 50358, "<|zh|>": 50260,
}
_WHISPER_TASK_TO_ID = {"transcribe": 50360, "translate": 50359}


def _whisper_generation_config():
    return {
        "bos_token_id": 50257,
        "decoder_start_token_id": 50258,
        "eos_token_id": 50257,
        "pad_token_id": 50257,
        "is_multilingual": True,
        "lang_to_id": _WHISPER_LANG_TO_ID,
        "task_to_id": _WHISPER_TASK_TO_ID,
        "no_timestamps_token_id": 50364,
        "max_length": 448,
        "begin_suppress_tokens": [220, 50257],
        "forced_decoder_ids": [[1, None], [2, 50360]],
        # openai/whisper-large-v3 generation_config.json
        "suppress_tokens": [
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61,
            62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918,
            922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536,
            3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428,
            10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265,
            15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
            26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
            47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363,
        ],
    }


def _hf_config_ok(path):
    cfg = os.path.join(path, "config.json")
    try:
        with open(cfg, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    if data.get("model_type") != "whisper":
        return False
    return _generation_config_ok(path)


def _generation_config_ok(path):
    gen = os.path.join(path, "generation_config.json")
    try:
        with open(gen, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    return (bool(data.get("lang_to_id")) and bool(data.get("task_to_id"))
            and bool(data.get("suppress_tokens")))


def ensure_whisper_generation_config(dest):
    """Write lang_to_id into dest and dest/openvino (PVC reuse keeps the IR)."""
    if not dest or not os.path.isdir(dest):
        return
    data = _whisper_generation_config()
    targets = [dest]
    nested = os.path.join(dest, "openvino")
    if os.path.isdir(nested):
        targets.append(nested)
    for path in targets:
        if _generation_config_ok(path):
            continue
        with open(os.path.join(path, "generation_config.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")


def _write_whisper_hf_config(dest, state):
    os.makedirs(dest, exist_ok=True)
    cfg = _whisper_hf_config(state)
    with open(os.path.join(dest, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    pre = os.path.join(dest, "preprocessor_config.json")
    if not os.path.isfile(pre):
        with open(pre, "w", encoding="utf-8") as fh:
            json.dump({
                "chunk_length": 30,
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": cfg["num_mel_bins"],
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_side": "right",
                "padding_value": 0.0,
                "processor_class": "WhisperProcessor",
                "return_attention_mask": False,
                "sampling_rate": 16000,
            }, fh, indent=2)
            fh.write("\n")
    ensure_whisper_generation_config(dest)


def _repair_hf_dir(dest, src):
    _copy_sidecar(src, dest)
    _write_whisper_hf_config(dest, {})


def to_transformers_dir(src, dest):
    """src is the llm-init snapshot. dest is written next to it. No Hub."""
    if is_transformers(src):
        return src
    if not is_ct2(src):
        raise RuntimeError("%s is neither transformers nor CTranslate2 Whisper" % src)
    marker = os.path.join(dest, ".hf-from-ct2")
    stamp = os.path.join(dest, ".hf-from-ct2-v3")
    if is_transformers(dest) and os.path.isfile(stamp) and _hf_config_ok(dest):
        ensure_whisper_generation_config(dest)
        return dest
    if is_transformers(dest) and os.path.isfile(marker) and not os.path.isfile(stamp):
        log.info("dropping fused-QKV CT2->HF dir %s", dest)
        shutil.rmtree(dest)
    elif is_transformers(dest):
        if not _hf_config_ok(dest):
            _repair_hf_dir(dest, src)
        if _hf_config_ok(dest):
            ensure_whisper_generation_config(dest)
            open(marker, "w").close()
            open(stamp, "w").close()
            return dest
    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        raise RuntimeError("CT2 snapshot %s has no config.json" % src)
    with open(cfg_path, encoding="utf-8") as fh:
        config = json.load(fh)
    raw = _dump_ct2_state_dict(src)
    log.info("CT2 names sample %s", list(raw)[:24])
    state = _to_hf_names(raw, {
        "encoder_layers": _count_layers(raw, "encoder") or config.get("encoder_layers") or 32,
        "decoder_layers": _count_layers(raw, "decoder") or config.get("decoder_layers") or 32,
    })
    os.makedirs(dest, exist_ok=True)
    _copy_sidecar(src, dest)
    from safetensors.numpy import save_file

    save_file(state, os.path.join(dest, "model.safetensors"))
    _write_whisper_hf_config(dest, raw)
    open(marker, "w").close()
    open(stamp, "w").close()
    log.info("rebuilt transformers Whisper at %s from CT2 %s", dest, src)
    return dest
