# Turn a Systran / faster-whisper CT2 snapshot into a transformers Whisper dir
# using only files already on disk. Never hits the Hub for a second repo.
import json
import logging
import os
import shutil

log = logging.getLogger("audio-ct2-whisper")

_TOKEN_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "normalizer.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "generation_config.json", "config.json",
)


def is_ct2(path):
    return os.path.isfile(os.path.join(path, "model.bin"))


def is_transformers(path):
    return any(os.path.isfile(os.path.join(path, n)) for n in (
        "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))


def is_whisper_ir(path):
    if not path or not os.path.isdir(path):
        return False
    enc = os.path.join(path, "openvino_encoder_model.xml")
    dec = os.path.join(path, "openvino_decoder_model.xml")
    return os.path.isfile(enc) and os.path.isfile(dec)


def _copy_sidecar(src, dest):
    os.makedirs(dest, exist_ok=True)
    for name in _TOKEN_FILES:
        a, b = os.path.join(src, name), os.path.join(dest, name)
        if os.path.isfile(a) and not os.path.isfile(b):
            shutil.copy2(a, b)


def _dump_ct2_state_dict(src):
    # ctranslate2 is a weight reader here, not the inference device.
    import ctranslate2

    model = ctranslate2.models.Whisper(src, device="cpu", compute_type="float32")
    raw = getattr(model, "model", None) or getattr(model, "_model", None)
    getter = None
    if raw is not None:
        getter = getattr(raw, "get_variable", None) or getattr(raw, "get_variable_if_exists", None)
    if getter is None:
        getter = getattr(model, "get_variable", None)
    if getter is None:
        raise RuntimeError(
            "this ctranslate2 build cannot dump Whisper weights from model.bin; "
            "refusing to fetch another HuggingFace repo")

    names = getattr(raw, "variable_names", None) or getattr(model, "variable_names", None)
    if callable(names):
        names = names()
    if not names:
        raise RuntimeError(
            "ctranslate2 Whisper has no variable_names; cannot rebuild transformers weights")

    import numpy as np

    state = {}
    for name in names:
        try:
            val = getter(name)
        except Exception:
            continue
        if val is None:
            continue
        arr = val if hasattr(val, "shape") else None
        try:
            arr = val.numpy() if hasattr(val, "numpy") else (val.to_numpy()
                                                             if hasattr(val, "to_numpy") else arr)
        except Exception:
            arr = None
        if arr is None:
            continue
        state[name] = np.ascontiguousarray(arr)
    if not state:
        raise RuntimeError("dumped no tensors from %s/model.bin" % src)
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
