"""OpenVINO Qwen3-ASR group inference; CUDA never imports this at runtime."""
import os
import time

# Encoder is 25 Hz / 400 frames per 16 s; only groups shorter than 4 s get that floor (9s×10 must not pad to T=400).
MIN_INTEL_GPU_AUDIO_SAMPLES = 256000
MIN_INTEL_GPU_ENCODER_FRAMES = 400
# Only groups shorter than this get the 16s / 400-frame floor.
MIN_GROUP_FLOOR_SAMPLES = 64000
AUDIO_PAD = "<|audio_pad|>"
# Traced encoder view is n_window*2. A remainder cannot become [1,128,N,100].
MEL_WINDOW = 100
# Official Qwen3-ASR hardcodes these; the exported IR has no generation_config.
EOS_TOKEN_ID = 151643
IM_END_TOKEN_ID = 151645


def pad_mel_time(features, window=MEL_WINDOW):
    """Right-pad mel time so it divides the traced encoder view.

    Eager pads a short tail up to n_window*2. The export kept the view and
    dropped that pad, so [1,128,1301] cannot become [1,128,13,100].
    """
    import numpy as np

    window = int(window)
    if features is None or window <= 1:
        return features
    arr = np.asarray(features)
    if arr.ndim < 1:
        return features
    t = int(arr.shape[-1])
    if t <= 0 or t % window == 0:
        return np.ascontiguousarray(arr)
    pad = window - (t % window)
    out = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)])
    return np.ascontiguousarray(out)


def extend_audio_tokens(prompt, n):
    """Official GenAI: one <|audio_pad|> becomes T pads matching encoder T."""
    if AUDIO_PAD not in prompt:
        raise RuntimeError("extend_audio_tokens: %s missing from prompt" % AUDIO_PAD)
    return prompt.replace(AUDIO_PAD, AUDIO_PAD * int(n), 1)


def last_logits(logits, batch):
    """Prefill returns [B,S,V]; decode steps return [B,1,V] or [B,V]."""
    import numpy as np

    a = np.asarray(logits)
    if a.ndim == 3:
        a = a[:, -1, :]
    elif a.ndim != 2:
        raise RuntimeError("decoder logits %s want [B,S,V] or [B,V]" % (a.shape,))
    if int(a.shape[0]) != int(batch):
        raise RuntimeError("decoder logits %s want batch %d" % (a.shape, batch))
    return a


def stack_encoder_hiddens(hiddens, min_t=MIN_INTEL_GPU_ENCODER_FRAMES):
    """[1, T, H] (or [T, H]) clips -> [B, max(min_t, longest), H], zero-padded."""
    import numpy as np

    arrs = []
    for h in hiddens:
        a = np.asarray(h, dtype=np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            raise RuntimeError("encoder hidden states must be [1,T,H] or [T,H], got %s"
                               % (a.shape,))
        arrs.append(a)
    max_t = max(max(a.shape[0] for a in arrs), int(min_t))
    hidden = arrs[0].shape[1]
    out = np.zeros((len(arrs), max_t, hidden), dtype=np.float32)
    for i, a in enumerate(arrs):
        if a.shape[1] != hidden:
            raise RuntimeError("encoder hidden dim %d != %d" % (a.shape[1], hidden))
        out[i, :a.shape[0]] = a
    return out


def keep_encoder_hidden_batch(model):
    """Bypass the flatten Unsqueeze that copies clip 0 onto every row.

    Exported decoder does [B,T,H] -> Reshape[-1,H] -> Unsqueeze[1,B*T,H].
    GatherElements along time then reads clip 0. Feed [B,T,H] through instead.
    """
    hidden = 0
    for inp in model.inputs:
        if inp.any_name != "encoder_hidden_states":
            continue
        shape = inp.partial_shape
        if len(shape) == 3 and shape[2].is_static:
            hidden = int(shape[2].get_length())
    if hidden <= 0:
        return 0
    try:
        ops = list(model.get_ops())
    except Exception:
        ops = list(model.get_ordered_ops())
    rewired = 0
    for op in ops:
        if op.get_type_name() != "Unsqueeze":
            continue
        out_shape = op.get_output_partial_shape(0)
        if out_shape.rank.is_dynamic or len(out_shape) != 3:
            continue
        if not out_shape[0].is_static or int(out_shape[0].get_length()) != 1:
            continue
        if not out_shape[2].is_static or int(out_shape[2].get_length()) != hidden:
            continue
        try:
            reshape = op.input(0).get_source_output().get_node()
            src = reshape.input(0).get_source_output().get_node()
        except Exception:
            continue
        if reshape is None or reshape.get_type_name() != "Reshape":
            continue
        if src is None:
            continue
        src_shape = src.get_output_partial_shape(0)
        if src_shape.rank.is_dynamic or len(src_shape) != 3:
            continue
        if not src_shape[2].is_static or int(src_shape[2].get_length()) != hidden:
            continue
        for consumer in op.output(0).get_target_inputs():
            consumer.replace_source_output(src.output(0))
        rewired += 1
    if rewired:
        model.validate_nodes_and_infer_types()
    return rewired


def keep_last_token_logits(model):
    """Cut lm_head to the last position before the vocab MatMul.

    Official IR is Multiply[B,S,2048] x Convert[151936,2048] -> logits[B,S,V].
    One infer at B=10, S=225 is ~1.4 GiB of logits and OOM-killed the 16Gi
    chart. Attention still sees the full prompt; only the projection shrinks
    to [B,1,V]. A decode step already has S=1, so the slice is a no-op.
    Windowed prefill on the with-past graph reset KV and looped the text.
    """
    import numpy as np
    from openvino import opset13 as opset

    rewired = 0
    for out in model.outputs:
        name = out.any_name or ""
        if name and "logits" not in name:
            continue
        mm = out.node.input(0).get_source_output().get_node()
        if mm.get_type_name() != "MatMul":
            continue
        hidden = mm.input(0).get_source_output()
        shape = hidden.get_partial_shape()
        if shape.rank.is_dynamic or len(shape) != 3:
            continue
        start = opset.constant(np.array([-1], dtype=np.int64))
        stop = opset.constant(np.array([2147483647], dtype=np.int64))
        step = opset.constant(np.array([1], dtype=np.int64))
        axes = opset.constant(np.array([1], dtype=np.int64))
        last = opset.slice(hidden, start, stop, step, axes)
        mm.input(0).replace_source_output(last.output(0))
        rewired += 1
    if rewired:
        model.validate_nodes_and_infer_types()
    return rewired


def _dynamize_batch(model):
    import openvino as ov

    shapes = {}
    for inp in model.inputs:
        shape = ov.PartialShape(inp.partial_shape)
        name = inp.any_name
        if name in ("encoder_hidden_states", "input_ids", "beam_idx",
                    "input_features", "attention_mask") and len(shape) >= 1:
            shape[0] = ov.Dimension.dynamic()
        shapes[name] = shape
    if shapes:
        model.reshape(shapes)


def _tensor(arr):
    import numpy as np
    import openvino as ov

    return ov.Tensor(np.ascontiguousarray(arr))


def _input_names(compiled):
    return [inp.any_name for inp in compiled.inputs]


def _output_data(request, compiled):
    for out in compiled.outputs:
        name = out.any_name
        if "logits" in name or name == "":
            try:
                return request.get_tensor(out).data
            except Exception:
                continue
    return request.get_output_tensor(0).data


class Engine:
    def __init__(self, encoder, decoder, processor, device):
        self.encoder = encoder
        self.decoder = decoder
        self.processor = processor
        self.device = device
        self._enc_names = _input_names(encoder)
        self._dec_names = _input_names(decoder)

    @classmethod
    def load(cls, ir_dir, src_dir, device, cache_dir):
        import openvino as ov

        os.makedirs(cache_dir, exist_ok=True)
        # mmap of an intact IR on the hostPath raises SIGBUS (exit 135); read the file instead.
        props = {"CACHE_DIR": cache_dir, "ENABLE_MMAP": False}
        core = ov.Core()
        core.set_property({"ENABLE_MMAP": False})
        enc_m = core.read_model(os.path.join(ir_dir, "openvino_encoder_model.xml"))
        try:
            _dynamize_batch(enc_m)
        except Exception:
            pass
        # iGPU [B,F,T] matched serial wall; THROUGHPUT + in-flight B=1 is the overlap.
        enc_props = dict(props)
        enc_props["CACHE_DIR"] = os.path.join(cache_dir, "enc-throughput")
        os.makedirs(enc_props["CACHE_DIR"], exist_ok=True)
        try:
            from openvino import properties as ovprops
            enc_props[ovprops.hint.performance_mode] = ovprops.hint.PerformanceMode.THROUGHPUT
            enc_props[ovprops.hint.num_requests] = 8
        except Exception:
            enc_props["PERFORMANCE_HINT"] = "THROUGHPUT"
            enc_props["PERFORMANCE_HINT_NUM_REQUESTS"] = "8"
        encoder = core.compile_model(enc_m, device, enc_props)
        dec_m = core.read_model(os.path.join(ir_dir, "openvino_decoder_model.xml"))
        _dynamize_batch(dec_m)
        if keep_encoder_hidden_batch(dec_m) <= 0:
            raise RuntimeError(
                "Qwen3-ASR decoder IR is missing the encoder flatten Unsqueeze; "
                "refusing a compile that would copy clip 0")
        if keep_last_token_logits(dec_m) <= 0:
            raise RuntimeError(
                "Qwen3-ASR decoder IR is missing the lm_head MatMul; "
                "refusing a compile that materializes [B,S,V] logits")
        decoder = core.compile_model(dec_m, device, props)
        processor = _load_processor(src_dir)
        return cls(encoder, decoder, processor, device)

    def generate_many(self, clips, language=None, context="", max_new_tokens=64):
        import numpy as np
        from qwen_asr.inference.utils import parse_asr_output

        wavs = [np.asarray(c, dtype=np.float32).reshape(-1) for c in clips]
        if not wavs:
            return []
        # 16 s / 400-frame floors are for 1s-class T crashes, not for 9 s.
        short = max(len(w) for w in wavs) < MIN_GROUP_FLOOR_SAMPLES
        if short:
            floor = MIN_INTEL_GPU_AUDIO_SAMPLES
            wavs = [np.pad(w, (0, floor - len(w))) if len(w) < floor else w for w in wavs]
        max_n = max(len(w) for w in wavs)
        padded = [np.pad(w, (0, max_n - len(w))) if len(w) < max_n else w for w in wavs]
        t0 = time.monotonic()
        hidden = self._encode(padded, min_t=(MIN_INTEL_GPU_ENCODER_FRAMES if short else 0))
        encode_s = time.monotonic() - t0
        prompt = extend_audio_tokens(_text_prompt(context or "", language), hidden.shape[1])
        t1 = time.monotonic()
        ids = self._prompt_ids([prompt] * len(padded))
        prompt_s = time.monotonic() - t1
        t2 = time.monotonic()
        raws = self._decode(ids, hidden, max_new_tokens)
        decode_s = time.monotonic() - t2
        print("[ov-asr] clips=%d encode=%.3fs prompt=%.3fs decode=%.3fs T=%d"
              % (len(padded), encode_s, prompt_s, decode_s, hidden.shape[1]), flush=True)
        out = []
        for raw in raws:
            lang, text = parse_asr_output(raw or "", user_language=language)
            out.append(((text or "").strip(), (lang or language or "").strip()))
        return out

    def _encode(self, wavs, min_t=MIN_INTEL_GPU_ENCODER_FRAMES):
        import numpy as np

        t0 = time.monotonic()
        feats = self.processor.feature_extractor(
            list(wavs), sampling_rate=16000, return_tensors="np", padding=True
        )
        arr = np.asarray(feats["input_features"], dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if int(arr.shape[0]) != len(wavs):
            raise RuntimeError("feature batch %s != %d clips" % (arr.shape, len(wavs)))
        feats_s = time.monotonic() - t0
        t1 = time.monotonic()
        parts = self._infer_encoder_many(arr)
        infer_s = time.monotonic() - t1
        print("[ov-asr] feats=%.3fs infer=%.3fs n=%d" % (feats_s, infer_s, len(wavs)),
              flush=True)
        return stack_encoder_hiddens(parts, min_t=min_t)

    def _infer_encoder_many(self, features):
        import numpy as np

        # In-flight B=1; a stacked [B,F,T] infer matched serial wall and raised the peak.
        name = "input_features" if "input_features" in self._enc_names else self._enc_names[0]
        n = int(features.shape[0])
        reqs = []
        for i in range(n):
            clip = pad_mel_time(np.ascontiguousarray(features[i:i + 1]))
            req = self.encoder.create_infer_request()
            req.set_tensor(name, _tensor(clip))
            if "attention_mask" in self._enc_names:
                mask = np.ones(clip.shape[:1] + clip.shape[-1:], dtype=np.int64)
                try:
                    req.set_tensor("attention_mask", _tensor(mask))
                except Exception:
                    pass
            if hasattr(req, "start_async"):
                req.start_async()
            else:
                req.infer()
            reqs.append(req)
        parts = []
        for req in reqs:
            if hasattr(req, "wait"):
                req.wait()
            parts.append(req.get_output_tensor(0).data.copy())
        return parts

    def _infer_encoder(self, features):
        return self._infer_encoder_many(features)[0]

    def _prompt_ids(self, prompts):
        import numpy as np

        # Text only; a second processor pass on wavs padded a multimodal prompt the with-past decoder could not keep aligned.
        tok = getattr(self.processor, "tokenizer", self.processor)
        packed = tok(list(prompts), return_tensors="np", padding=True)
        ids = np.asarray(packed["input_ids"])
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        return ids.astype("int64")

    def _decode(self, input_ids, hidden, max_new_tokens):
        import numpy as np

        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.ndim == 2:
            hidden = hidden[None, ...]
        batch = hidden.shape[0]
        if int(input_ids.shape[0]) != batch:
            raise RuntimeError("prompt batch %s != hidden %s"
                               % (input_ids.shape, hidden.shape))
        # Fresh request every group: reset_state poisoned Arc shapes; keep batch size B for the whole decode.
        req = self.decoder.create_infer_request()
        if "encoder_hidden_states" in self._dec_names:
            req.set_tensor("encoder_hidden_states", _tensor(hidden))
        tok = getattr(self.processor, "tokenizer", self.processor)
        stop = {EOS_TOKEN_ID, IM_END_TOKEN_ID}
        eos = getattr(tok, "eos_token_id", None)
        if eos is not None:
            stop.add(int(eos))
        stop_id = EOS_TOKEN_ID
        beams = np.arange(batch, dtype=np.int32)
        last = self._step(req, input_ids, beams, batch)
        generated = [[] for _ in range(batch)]
        finished = [False] * batch
        for _ in range(int(max_new_tokens)):
            nxt = np.argmax(last, axis=-1).astype("int64")
            for i in range(batch):
                if finished[i]:
                    nxt[i] = stop_id
                    continue
                generated[i].append(int(nxt[i]))
                if int(nxt[i]) in stop:
                    finished[i] = True
            if all(finished):
                break
            last = self._step(req, nxt.reshape(batch, 1), beams, batch)
        texts = []
        for row in generated:
            texts.append(tok.decode(row, skip_special_tokens=True) if row else "")
        return texts

    def _step(self, req, input_ids, beams, batch):
        import numpy as np

        ids = np.ascontiguousarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids.reshape(batch, -1)
        if ids.shape[0] != batch:
            raise RuntimeError("input_ids %s want batch %d" % (ids.shape, batch))
        req.set_tensor("input_ids", _tensor(ids))
        if "beam_idx" in self._dec_names:
            req.set_tensor("beam_idx", _tensor(np.ascontiguousarray(beams[:batch])))
        req.infer()
        # Copy only [B,V]. The full [B,S,V] tensor is what killed B=10.
        return np.ascontiguousarray(last_logits(_output_data(req, self.decoder), batch))


def _load_processor(src_dir):
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRProcessor
    from transformers import AutoConfig, AutoProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    return AutoProcessor.from_pretrained(src_dir, fix_mistral_regex=True)


def _text_prompt(context, language):
    """Official Qwen3ASR::build_text_prompt. apply_chat_template is not this."""
    prompt = (
        "<|im_start|>system\n" + (context or "") +
        "<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language:
        prompt += "language %s<asr_text>" % language
    return prompt
