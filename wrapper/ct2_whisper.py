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


def _to_hf_names(ct2_state, config):
    # Inverse of CTranslate2's WhisperConverter name map (encoder/decoder layers).
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

    take("encoder/conv1/weight", "model.encoder.conv1.weight")
    take("encoder/conv1/bias", "model.encoder.conv1.bias")
    take("encoder/conv2/weight", "model.encoder.conv2.weight")
    take("encoder/conv2/bias", "model.encoder.conv2.bias")
    take("encoder/position_encodings", "model.encoder.embed_positions.weight")
    take("encoder/layer_norm/gamma", "model.encoder.layer_norm.weight")
    take("encoder/layer_norm/beta", "model.encoder.layer_norm.bias")
    for i in range(n_enc):
        p = "encoder/layer_%d" % i
        q = "model.encoder.layers.%d" % i
        take("%s/self_attention/linear_layers/0/weight" % p, "%s.self_attn.q_proj.weight" % q)
        take("%s/self_attention/linear_layers/0/bias" % p, "%s.self_attn.q_proj.bias" % q)
        take("%s/self_attention/linear_layers/1/weight" % p, "%s.self_attn.k_proj.weight" % q)
        take("%s/self_attention/linear_layers/1/bias" % p, "%s.self_attn.k_proj.bias" % q)
        take("%s/self_attention/linear_layers/2/weight" % p, "%s.self_attn.v_proj.weight" % q)
        take("%s/self_attention/linear_layers/2/bias" % p, "%s.self_attn.v_proj.bias" % q)
        take("%s/self_attention/linear_layers/3/weight" % p, "%s.self_attn.out_proj.weight" % q)
        take("%s/self_attention/linear_layers/3/bias" % p, "%s.self_attn.out_proj.bias" % q)
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
    take("decoder/layer_norm/gamma", "model.decoder.layer_norm.weight")
    take("decoder/layer_norm/beta", "model.decoder.layer_norm.bias")
    for i in range(n_dec):
        p = "decoder/layer_%d" % i
        q = "model.decoder.layers.%d" % i
        take("%s/self_attention/linear_layers/0/weight" % p, "%s.self_attn.q_proj.weight" % q)
        take("%s/self_attention/linear_layers/0/bias" % p, "%s.self_attn.q_proj.bias" % q)
        take("%s/self_attention/linear_layers/1/weight" % p, "%s.self_attn.k_proj.weight" % q)
        take("%s/self_attention/linear_layers/1/bias" % p, "%s.self_attn.k_proj.bias" % q)
        take("%s/self_attention/linear_layers/2/weight" % p, "%s.self_attn.v_proj.weight" % q)
        take("%s/self_attention/linear_layers/2/bias" % p, "%s.self_attn.v_proj.bias" % q)
        take("%s/self_attention/linear_layers/3/weight" % p, "%s.self_attn.out_proj.weight" % q)
        take("%s/self_attention/linear_layers/3/bias" % p, "%s.self_attn.out_proj.bias" % q)
        take("%s/attention/linear_layers/0/weight" % p, "%s.encoder_attn.q_proj.weight" % q)
        take("%s/attention/linear_layers/0/bias" % p, "%s.encoder_attn.q_proj.bias" % q)
        take("%s/attention/linear_layers/1/weight" % p, "%s.encoder_attn.k_proj.weight" % q)
        take("%s/attention/linear_layers/1/bias" % p, "%s.encoder_attn.k_proj.bias" % q)
        take("%s/attention/linear_layers/2/weight" % p, "%s.encoder_attn.v_proj.weight" % q)
        take("%s/attention/linear_layers/2/bias" % p, "%s.encoder_attn.v_proj.bias" % q)
        take("%s/attention/linear_layers/3/weight" % p, "%s.encoder_attn.out_proj.weight" % q)
        take("%s/attention/linear_layers/3/bias" % p, "%s.encoder_attn.out_proj.bias" % q)
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
    if len(out) < 8:
        raise RuntimeError(
            "CT2 name map matched %d tensors (names like %s); converter needs a fuller dump"
            % (len(out), list(ct2_state)[:8]))
    return out


def to_transformers_dir(src, dest):
    """src is the llm-init snapshot. dest is written next to it. No Hub."""
    if is_transformers(src):
        return src
    if not is_ct2(src):
        raise RuntimeError("%s is neither transformers nor CTranslate2 Whisper" % src)
    marker = os.path.join(dest, ".hf-from-ct2")
    if os.path.isfile(marker) and is_transformers(dest):
        return dest
    cfg_path = os.path.join(src, "config.json")
    if not os.path.isfile(cfg_path):
        raise RuntimeError("CT2 snapshot %s has no config.json" % src)
    with open(cfg_path, encoding="utf-8") as fh:
        config = json.load(fh)
    state = _to_hf_names(_dump_ct2_state_dict(src), config)
    os.makedirs(dest, exist_ok=True)
    _copy_sidecar(src, dest)
    from safetensors.numpy import save_file

    save_file(state, os.path.join(dest, "model.safetensors"))
    open(marker, "w").close()
    log.info("rebuilt transformers Whisper at %s from CT2 %s", dest, src)
    return dest
