# Breeze TTS 2 in-process. Official PyTorch: clone (ref+transcript), design, voice direction.
import logging
import os
from pathlib import Path

from . import tts_el, tts_long

log = logging.getLogger("audio-breeze")

# Measured here: 12.5 frames a second, Chinese at 0.71 text and 3.35 audio tokens a char.
_FPS = 12.5
_TOK_PER_CHAR = 0.71
_GEN_PER_CHAR = 3.35
_HEADROOM = 0.9
_SPECIALS = 8
_LIMIT_CEILING = 375  # the longest read seen to come back whole
_LIMIT_FLOOR = 80
# Speaker boost re-reads the tail of the slice before: its frames plus its text.
_CTX_SECONDS = 10.0
# infer.py's --seed default, and set_deterministic's own signature default.
_UPSTREAM_SEED = 42
_CTX_TOKENS = int(_CTX_SECONDS * _FPS) + 40


def _fallback_attn(requested, impl):
    return tts_el.fallback_attn(requested, impl)


def _rewrite_flash_attn(requested):
    tts_el.rewrite_flash_attn(requested)


def speak_limit(prompt_tokens=0, reserve=0):
    """Chars one generate can finish. max_seq_len covers prompt and output both, and
    the model walks off the end in silence rather than raising, so leave 10% behind."""
    seq = int(tts_el.MAX_SEQ_LEN)
    gen = int(tts_el.MAX_NEW_TOKENS)
    room = seq - int(prompt_tokens) - int(reserve) - _SPECIALS
    limit = int(_HEADROOM * room / (_TOK_PER_CHAR + _GEN_PER_CHAR))
    if gen > 0:
        limit = min(limit, int(gen / _GEN_PER_CHAR))
    return min(limit, _LIMIT_CEILING)


def _no_room(limit):
    raise ValueError(
        "Reference audio and instruction leave room for only %d characters a slice, "
        "under the %d needed to read at all, inside max_seq_len=%d. Use a shorter "
        "reference or a shorter instruction." % (max(0, limit), _LIMIT_FLOOR,
                                                 int(tts_el.MAX_SEQ_LEN)))


def split_speak(text, limit=None):
    """The shared splitter, with Breeze's own budget when the caller names none."""
    return tts_long.split_speak(text, speak_limit() if limit is None else int(limit))


_CTX_TEMPLATE = None


def _ctx_template():
    """ref_edit_tata has one audio slot. Boost needs two: the reference pair, the tail
    of the slice before, then what to read now. The library has no field for it."""
    global _CTX_TEMPLATE
    if _CTX_TEMPLATE is not None:
        return _CTX_TEMPLATE
    from breeze_infer.templates import INSTRUCTION_BOS, INSTRUCTION_EOS, TemplateSpec

    def prefix(req):
        sp = req.get("speaker") or ""
        return sp if sp.startswith("[") else ("[%s]" % sp if sp else "")

    def heard(path):
        return {"type": "audio", "audio_path": str(path),
                "append_eos": True, "drop_last_frame": False}

    def pairs(req):
        p = prefix(req)
        return [{"type": "text", "text": p + req["ref_text"]},
                heard(req["ref_audio_path"]),
                {"type": "text", "text": p + req["ctx_text"]},
                heard(req["ctx_audio_path"])]

    def positive(req):
        return pairs(req) + [{"type": "text", "text": "%s%s%s%s%s" % (
            prefix(req), INSTRUCTION_BOS, req["instruction"],
            INSTRUCTION_EOS, req["text"])}]

    def negative(req):
        return pairs(req) + [{"type": "text", "text": prefix(req) + req["text"]}]

    _CTX_TEMPLATE = TemplateSpec(
        name="ref_edit_tata_ctx",
        required_fields=("text", "instruction", "ref_audio_path", "ref_text",
                         "ctx_audio_path", "ctx_text"),
        build_segments=positive,
        build_negative_segments=negative,
    )
    return _CTX_TEMPLATE


def _too_long(n):
    raise ValueError(
        "Input is too long: piece has %d characters but max_seq_len=%d. "
        "Use shorter text or shorter reference audio." % (n, int(tts_el.MAX_SEQ_LEN)))


class BreezeBackend:
    sample_rate = 24000
    uses_shared_pack = True
    # Card instruction is display / pace copy. Only a request-level instruction is Voice Direction.
    card_instruction_is_direction = False

    def __init__(self, runtime, tokenizer, audio_tokenizer, model):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.model = model
        sr = getattr(runtime, "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def native_presets(self):
        return []

    def presets(self):
        return tts_el.premade_cards(self, "breeze")

    def _iter_generate(self, text, instruction, ref_path=None, ref_text=None, cfg=None,
                       ctx=None, seed=None, carry=None):
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs

        request = {
            "id": "req",
            "text": text,
            "instruction": instruction or "Speak clearly and naturally.",
            "speaker": "S0",
        }
        template = get_template("tts_instruction")
        if ref_path:
            request["ref_audio_path"] = str(ref_path)
            request["ref_text"] = (ref_text or "").strip()
            template = get_template("ref_edit_tata")
            if carry:
                request["ctx_audio_path"] = str(carry[0])
                request["ctx_text"] = carry[1]
                template = _ctx_template()
        # Unspecified seed means infer.py's own default, not "leave the last request's RNG".
        if seed is None:
            seed = tts_el.SEED
        set_all_seeds(int(_UPSTREAM_SEED if seed is None else seed))
        inputs = prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            template,
            guidance_scale=float(cfg if cfg is not None else tts_el.CFG_SCALE),
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        tts_el.job_tick(ctx)
        n = 0
        # A Python exception through the CUDA codec loop SIGSEGVs the process.
        for chunk in self.runtime.iter_audio_chunks(inputs, request_id="req"):
            audio = tts_long.collect(getattr(chunk, "audio", chunk))
            if not len(audio):
                continue
            n += 1
            yield audio, self.sample_rate
        if n == 0:
            raise RuntimeError("Breeze TTS 2 produced no audio")

    def _generate(self, text, instruction, ref_path=None, ref_text=None, cfg=None,
                  ctx=None, seed=None, carry=None):
        import numpy as np

        waves = [w for w, _ in self._iter_generate(
            text, instruction, ref_path=ref_path, ref_text=ref_text, cfg=cfg,
            ctx=ctx, seed=seed, carry=carry)]
        return np.concatenate(waves), self.sample_rate

    def _text_tokens(self, s):
        s = (s or "").strip()
        if not s:
            return 0
        # The budget is an estimate anyway; count characters if no tokenizer is at hand.
        if self.tokenizer is None:
            return int(len(s) * _TOK_PER_CHAR) + 1
        return len(self.tokenizer(s, add_special_tokens=True)["input_ids"])

    def _prompt_tokens(self, seconds, ref_text, instruction):
        """Everything the prompt costs before the slice being read is added to it."""
        return (int(round(float(seconds) * _FPS)) + 1
                + self._text_tokens(ref_text) + self._text_tokens(instruction))

    def _carry(self, path, text, waves):
        """Freeze the tail of a finished slice for the next one to hear."""
        import numpy as np
        import soundfile as sf

        tail_text, tail = tts_long.tail_pair(text, np.concatenate(waves), self.sample_rate,
                                             _CTX_SECONDS)
        if not len(tail) or not tail_text:
            return None
        sf.write(path, tail, self.sample_rate, format="WAV", subtype="PCM_16")
        return path, tail_text

    def iter_clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        import soundfile as sf
        import tempfile

        # Voice Direction: ref + transcript + instruction at CFG 4; bare clone keeps CFG 1.
        direction = str(kw.get("instruction") or "").strip()
        base_cfg = tts_el.DESIGN_CFG if direction else tts_el.CFG_SCALE
        # No speed in the written direction: the tempo filter already sets the rate.
        kn = tts_el.resolve_settings(kw.get("settings"), base_cfg,
                                     direction or "Speak clearly and naturally.", text=text,
                                     include_speed_direction=False)
        wav = tts_long.collect(prompt_audio)
        rate = int(prompt_sr or self.sample_rate)
        reserve = _CTX_TOKENS if kn.context else 0
        fixed = self._prompt_tokens(len(wav) / float(rate), prompt_text, kn.instruction)
        limit = speak_limit(fixed, reserve)
        log.info("breeze budget: prompt=%d reserve=%d limit=%d of max_seq_len=%d",
                 fixed, reserve, limit, int(tts_el.MAX_SEQ_LEN))
        if limit < _LIMIT_FLOOR:
            _no_room(limit)
        parts = split_speak(text, limit)
        if not parts:
            parts = [text]
        for part in parts:
            if len(part) > limit:
                _too_long(len(part))
        with tempfile.NamedTemporaryFile(prefix="breeze-ref-", suffix=".wav", delete=False) as fh:
            path = fh.name
        carry_path = path + ".carry.wav"
        carry = None
        # One tempo filter for the whole reading; a filter per chunk would seam every 80 ms.
        tempo = tts_long.TempoStream(self.sample_rate, kn.speed) if kw.get("pace_each") else None
        try:
            sf.write(path, wav, rate, format="WAV", subtype="PCM_16")
            ctx = kw.get("ctx")
            total = len(parts)
            for i, part in enumerate(parts):
                tts_el.job_tick(ctx, i, total)
                if tempo is not None:
                    if i:
                        gap_n = max(0, int(tts_long.pause_ms(text) / 1000.0 * self.sample_rate))
                        if gap_n:
                            import numpy as np
                            for paced in tempo.write(np.zeros(gap_n, dtype="float32"),
                                                     self.sample_rate):
                                yield paced, self.sample_rate
                    n = 0
                    said = []
                    for audio, sr in self._iter_generate(
                            part, kn.instruction, ref_path=path, ref_text=prompt_text,
                            cfg=kn.cfg, ctx=ctx, seed=kn.seed, carry=carry):
                        wave = tts_long.collect(audio)
                        said.append(wave)
                        if i and n == 0:
                            wave = tts_long.fade_head(
                                wave, max(1, int(tts_long.JOIN_FADE_MS / 1000.0 * sr)))
                        n += 1
                        for paced in tempo.write(wave, sr):
                            yield paced, self.sample_rate
                    if n == 0:
                        raise RuntimeError("Breeze TTS 2 produced no audio")
                    if kn.context and i + 1 < total:
                        carry = self._carry(carry_path, part, said)
                else:
                    audio, sr = self._generate(part, kn.instruction,
                                               ref_path=path, ref_text=prompt_text,
                                               cfg=kn.cfg, ctx=ctx, seed=kn.seed,
                                               carry=carry)
                    wave = tts_long.collect(audio)
                    if kn.context and i + 1 < total:
                        carry = self._carry(carry_path, part, [wave])
                    yield tts_long.pace(wave, sr, kn.speed)
                tts_el.job_tick(ctx, i + 1, total)
            if tempo is not None:
                for paced in tempo.finish():
                    yield paced, self.sample_rate
        finally:
            if tempo is not None:
                tempo.abort()
            tts_long.vram(log, "clone")
            for gone in (path, carry_path):
                try:
                    os.unlink(gone)
                except OSError:
                    pass

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        waves, sr = [], self.sample_rate
        kw = dict(kw)
        kw["pace_each"] = False
        for wave, sr in self.iter_clone(text, prompt_audio, prompt_sr, prompt_text, **kw):
            waves.append(wave)
        return tts_long.join(waves, sr, gap_ms=tts_long.pause_ms(text))

    def stream_gap(self, text, sr):
        # Intra-part codec chunks already include the part pause from iter_clone.
        return None

    def stream_next_slice(self, wave, sr):
        return wave

    def design(self, instruction, text, **kw):
        ctx = kw.get("ctx")
        # Same reason as the clone path: the tempo filter below is the one that sets the rate.
        kn = tts_el.resolve_settings(kw.get("settings"), tts_el.DESIGN_CFG,
                                     instruction, text=text,
                                     include_speed_direction=False,
                                     seed_jitter=int(kw.get("seed_jitter") or 0))
        tts_el.job_tick(ctx, 0, 1, stage="design")
        audio, sr = self._generate(text, kn.instruction or instruction,
                                   cfg=kn.cfg, ctx=ctx, seed=kn.seed)
        audio, sr = tts_long.pace(audio, sr, kn.speed)
        tts_el.job_tick(ctx, 1, 1, stage="design")
        return audio, sr, {}


def _is_ov():
    from .. import tts_ov

    return tts_ov.is_breeze_ov()


def _breeze_snapshot_missing(path):
    """Files the tokenizer and the weight load both require.

    llm-init can publish the download sentinel while the snapshot still only
    has the small sidecars. AutoTokenizer then builds GemmaTokenizerFast with
    tokenizer_file=None, sentencepiece is not in this image, and
    convert_slow_tokenizer dies on vocab_file.endswith.
    """
    import json

    missing = []
    tok = path / "tokenizer.json"
    if not tok.is_file() or tok.stat().st_size < 8 * 1024 * 1024:
        missing.append("tokenizer.json")
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        missing.append("model.safetensors.index.json")
        return missing
    try:
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
    except (OSError, json.JSONDecodeError):
        missing.append("model.safetensors.index.json")
        return missing
    shards = sorted(set(weight_map.values()))
    if not shards:
        missing.append("model.safetensors.index.json")
        return missing
    for name in shards:
        shard = path / name
        if not shard.is_file() or shard.stat().st_size < 100 * 1024 * 1024:
            missing.append(name)
    return missing


def _wait_breeze_snapshot(path, timeout_s=1800):
    """Block in the load thread until the snapshot can actually be tokenized."""
    import time

    deadline = time.monotonic() + timeout_s
    last = 0.0
    while True:
        missing = _breeze_snapshot_missing(path)
        if not missing:
            log.info("breeze snapshot ready at %s", path)
            return
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                "Breeze snapshot %s still missing %s; refusing to load the tokenizer"
                % (path, ", ".join(missing))
            )
        now = time.monotonic()
        if now - last >= 15:
            log.info("breeze snapshot incomplete, waiting for %s", ", ".join(missing))
            last = now
        time.sleep(2)


def _register_breeze_tokenizer():
    """BreezeConfig is not in Transformers' tokenizer table.

    The checkpoint tokenizer is GemmaTokenizerFast. AutoTokenizer.from_pretrained
    only sees that when tokenizer_config.json is read. If it falls through to the
    model config, TOKENIZER_MAPPING raises KeyError: 'BreezeConfig' and the
    process stays up returning 503.
    """
    from transformers import AutoTokenizer, GemmaTokenizerFast
    from models.breeze_config import BreezeConfig

    AutoTokenizer.register(
        BreezeConfig,
        fast_tokenizer_class=GemmaTokenizerFast,
        exist_ok=True,
    )


def _load():
    attn = tts_el.ATTN or "eager"
    _rewrite_flash_attn(attn)
    from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    path = Path(tts_el.model_path())
    _wait_breeze_snapshot(path)
    _register_breeze_tokenizer()
    if _is_ov():
        from .. import tts_ov

        device = tts_ov.require_gpu()
        tts_ov.allow_breeze_fast_on_cpu()
        log.info("loading Breeze TTS 2 OpenVINO from %s (attn=%s, ov=%s)", path, attn, device)
        tokenizer, model, audio_tokenizer = load_runtime(
            path, device="cpu", attn_implementation=attn,
        )
        _install_breeze_ov(model, path, device, audio_tokenizer)
    else:
        log.info("loading Breeze TTS 2 from %s (attn=%s)", path, attn)
        tokenizer, model, audio_tokenizer = load_runtime(
            path, device=resolve_device(), attn_implementation=attn,
        )
    update_generation_config_for_breeze(model)
    config = FastStreamingConfig(
        max_new_tokens=int(tts_el.MAX_NEW_TOKENS),
        max_seq_len=int(tts_el.MAX_SEQ_LEN),
        fast_all=tts_el.FAST_ALL,
        repetition_penalty=1.1,
    )
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)
    tts_long.vram(log, "loaded")
    return BreezeBackend(runtime, tokenizer, audio_tokenizer, model)


def _install_breeze_ov(model, path, device, audio_tokenizer=None):
    """Official five stages all go to OpenVINO: text, backbone prefill/decode, depth, codec."""
    import torch

    from models.cudagraph.backbone_graph import BackboneGraph

    from .. import tts_ov

    backbone = getattr(model, "backbone_model", None)
    if backbone is None:
        raise RuntimeError("Breeze model has no backbone_model to export")
    cfg = getattr(backbone, "config", None) or model.config
    n_layers, n_kv, head_dim = tts_ov.kv_meta(cfg)
    hidden = int(cfg.hidden_size)
    src = str(path)
    example_t = 16
    embeds_pre = torch.zeros(1, example_t, hidden, dtype=torch.float32)
    embeds_dec = torch.zeros(1, 1, hidden, dtype=torch.float32)
    mask_pre = torch.ones(1, example_t, dtype=torch.long)
    mask_dec = torch.ones(1, example_t + 1, dtype=torch.long)
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
    _, pre_xml, pre_stamp = tts_ov.ir_paths(src, "breeze_backbone_prefill", ".ov-breeze-v4")
    _, dec_xml, dec_stamp = tts_ov.ir_paths(src, "breeze_backbone_decode", ".ov-breeze-v4")
    prefill = tts_ov.compile_causal(
        tts_ov.causal_kv_module(backbone, False),
        (embeds_pre, mask_pre), pre_xml, pre_stamp, device,
    )
    import gc
    gc.collect()
    decode = tts_ov.compile_causal(
        tts_ov.causal_kv_module(backbone, True),
        (embeds_dec, mask_dec, *past), dec_xml, dec_stamp, device,
    )
    runner = tts_ov.DeviceKvRunner(decode, n_layers, prefill=prefill)
    orig_bb = backbone.forward

    def bb_forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None and args:
            embeds = args[0]
        past_in = kwargs.get("past_key_values")
        if embeds is None:
            return orig_bb(*args, **kwargs)
        if embeds.shape[0] != 1:
            raise RuntimeError(
                "breeze ov backbone is compiled for batch=1 (cfg_scale=1); got %s"
                % (tuple(embeds.shape),)
            )
        if past_in is None:
            runner.reset()
        hidden = runner.step(embeds)
        return type("BBOut", (), {
            "last_hidden_state": hidden,
            "past_key_values": past_in,
        })()

    backbone.forward = bb_forward

    def prefill_kv(self, past_key_values):
        seq_len = runner.prefix_len
        if seq_len <= 0:
            raise RuntimeError("breeze ov prefill_kv before a kv prefill")
        if seq_len > self.max_seq_len:
            raise RuntimeError(
                "Input too long: prefill has %d tokens but max_seq_len=%d."
                % (seq_len, self.max_seq_len)
            )
        self._prefill_len = seq_len
        return seq_len

    def decode_step(self):
        if self.batch_size != 1:
            raise RuntimeError(
                "breeze ov BackboneGraph is compiled for batch=1; got %d" % self.batch_size
            )
        inputs_embeds = self.embed_tokens(self.input_ids_buf)
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=self.attn_mask,
            past_key_values=self.static_cache,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            use_cache=True,
        )
        self.hidden_buf.copy_(out.last_hidden_state.to(self.hidden_buf.dtype))
        logits = self.lm_head(self.hidden_buf[:, -1, :].float())
        self.logits_buf.copy_(logits)
        self.cfg_logits_buf.copy_(self.logits_buf[: self.half])

    prefill_kv._ov = True
    decode_step._ov = True
    BackboneGraph.prefill_kv = prefill_kv
    BackboneGraph._decode_step = decode_step
    log.info(
        "breeze backbone prefill+decode device-KV on OpenVINO %s layers=%d",
        device, n_layers,
    )
    # Layer stack is in the OV blob. embed_tokens / lm_head stay for the Python loop.
    tts_ov.release_parameters(getattr(backbone, "layers", None), "breeze backbone layers")
    _install_breeze_text_ov(model, path, device)
    _install_breeze_depth_ov(model, path, device)
    _install_breeze_codec_ov(audio_tokenizer, path, device)


def _install_breeze_text_ov(model, path, device):
    """Official stage 1: TextEncoderGraphCache is CUDA-only. Same call site, OV IR."""
    import numpy as np
    import torch
    import torch.nn as nn

    from .. import tts_ov

    enc = getattr(model, "text_encoder", None)
    if enc is None:
        raise RuntimeError("Breeze model has no text_encoder")
    t_fixed = 256

    class _Text(nn.Module):
        def forward(self, input_ids, attention_mask, position_ids):
            return enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
            ).last_hidden_state

    ids = torch.zeros(1, t_fixed, dtype=torch.long)
    mask = torch.ones(1, t_fixed, dtype=torch.long)
    pos = torch.arange(t_fixed, dtype=torch.long).unsqueeze(0)
    _, xml, stamp = tts_ov.ir_paths(str(path), "breeze_text_encoder", ".ov-breeze-text-v1")
    compiled = tts_ov.compile_module(_Text(), (ids, mask, pos), xml, stamp, device)
    orig = model._batched_text_encoder_forward
    proj = getattr(model, "text_encoder_proj", None)
    if proj is not None and hasattr(proj, "parameters"):
        wdtype = next(proj.parameters()).dtype
    else:
        wdtype = next(enc.parameters()).dtype

    def _batched_text_encoder_forward(self, segments, output_hidden_states=False):
        if output_hidden_states or not segments:
            return orig(segments, output_hidden_states=output_hidden_states)
        hidden = []
        for seg in segments:
            length = int(seg.shape[0])
            if length <= 0:
                raise RuntimeError("breeze ov text encoder got an empty segment")
            if length > t_fixed:
                raise RuntimeError(
                    "breeze ov text encoder is compiled for T=%d; got %d"
                    % (t_fixed, length)
                )
            pad = torch.zeros(1, t_fixed, dtype=seg.dtype)
            attn = torch.zeros(1, t_fixed, dtype=torch.long)
            pos_ids = torch.zeros(1, t_fixed, dtype=torch.long)
            pad[0, :length] = seg.detach().cpu()
            attn[0, :length] = 1
            pos_ids[0, :length] = torch.arange(length)
            out = compiled(
                np.ascontiguousarray(pad.numpy()),
                np.ascontiguousarray(attn.numpy()),
                np.ascontiguousarray(pos_ids.numpy()),
            )[0]
            hidden.append(
                torch.from_numpy(np.ascontiguousarray(out[0, :length])).to(dtype=wdtype)
            )
        return hidden, []

    model._batched_text_encoder_forward = _batched_text_encoder_forward.__get__(
        model, type(model)
    )
    tts_ov.release_parameters(enc, "breeze text encoder")
    log.info("breeze text encoder on OpenVINO %s static_t=%d", device, t_fixed)


def _install_breeze_depth_ov(model, path, device):
    """Official DepthDecoderGraph loop stays; layer stack runs on GPU like backbone."""
    import torch

    from models.cudagraph.depth_decoder_graph import DepthDecoderGraph

    from .. import tts_ov

    depth = getattr(model, "depth_decoder", None)
    inner = getattr(depth, "model", None) if depth is not None else None
    if inner is None:
        raise RuntimeError("Breeze model has no depth_decoder.model")
    cfg = getattr(model.config, "depth_decoder_config", None) or inner.config
    n_layers, n_kv, head_dim = tts_ov.kv_meta(cfg)
    hidden = int(cfg.hidden_size)
    src = str(path)
    embeds_pre = torch.zeros(1, 2, hidden, dtype=torch.float32)
    embeds_dec = torch.zeros(1, 1, hidden, dtype=torch.float32)
    mask_pre = torch.ones(1, 2, dtype=torch.long)
    mask_dec = torch.ones(1, 3, dtype=torch.long)
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, 2, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, n_kv, 2, head_dim, dtype=torch.float32))
    _, pre_xml, pre_stamp = tts_ov.ir_paths(src, "breeze_depth_prefill", ".ov-breeze-depth-v1")
    _, dec_xml, dec_stamp = tts_ov.ir_paths(src, "breeze_depth_decode", ".ov-breeze-depth-v1")
    prefill = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, False),
        (embeds_pre, mask_pre), pre_xml, pre_stamp, device,
    )
    decode = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, True),
        (embeds_dec, mask_dec, *past), dec_xml, dec_stamp, device,
    )
    runner = tts_ov.DeviceKvRunner(decode, n_layers, prefill=prefill)

    def _full_loop(self):
        if int(getattr(self, "batch_size", 1)) != 1:
            raise RuntimeError(
                "breeze ov depth is compiled for batch=1; got %s" % self.batch_size
            )
        self.prefill_input_ids[:, 0] = 0
        self.prefill_input_ids[:, 1] = self.first_cb_token_buf
        prefill_embeds = self.embed_tokens(self.prefill_input_ids)
        backbone_h = self.backbone_hidden_buf
        if self.backbone_hidden_state_projector is not None:
            backbone_h = self.backbone_hidden_state_projector(backbone_h)
        prefill_embeds[:, 0] = backbone_h
        prefill_embeds = self.inputs_embeds_projector(prefill_embeds)
        runner.reset()
        hidden_states = runner.step(prefill_embeds)
        first_logits = self.codebooks_head(
            hidden_states[:, 1:, :].float(),
            cache_position=self.head_prefill_pos,
        )
        if self.debug_logits is not None:
            self.debug_logits[0].copy_(first_logits[:, 0, :])
        self._cfg_sample(first_logits)
        self._tok_buf.clamp_(0, self.vocab_size - 1)
        self.output_tokens[:, 0] = self._tok_buf
        for cb_idx in range(1, self.num_decode_codebooks):
            offset_tok = self._tok_buf + self.codebook_offsets[cb_idx]
            emb = self.embed_tokens(
                offset_tok.unsqueeze(1).clamp_(
                    0, self.num_codebooks * self.vocab_size - 1
                )
            )
            emb = self.inputs_embeds_projector(emb)
            hidden_states = runner.step(emb)
            cache_pos = self.decode_cache_positions[cb_idx - 1]
            logits = self.codebooks_head(hidden_states.float(), cache_position=cache_pos)
            if self.debug_logits is not None:
                self.debug_logits[cb_idx].copy_(logits[:, 0, :])
            self._cfg_sample(logits)
            self._tok_buf.clamp_(0, self.vocab_size - 1)
            self.output_tokens[:, cb_idx] = self._tok_buf

    DepthDecoderGraph._full_loop = _full_loop
    log.info("breeze depth prefill+decode device-KV on OpenVINO %s layers=%d", device, n_layers)
    tts_ov.release_parameters(getattr(inner, "layers", None), "breeze depth layers")


def _install_breeze_codec_ov(audio_tokenizer, path, device):
    """Official codec stage is codes→wav (quantizer, pre_conv, pre_transformer, tail).

    intel11 dynamized T=2 and Add died. intel13 only compiled the conv tail, leaving
    pre_transformer on CPU. intel22 froze T=32 and re-ran the pad every 2-frame chunk.
    Official chunk is T=2; freeze that and stop left-padding to 32.
    """
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from models.stream_runtime.stream.lane import ExecutionLane

    from .. import tts_ov

    if audio_tokenizer is None or getattr(audio_tokenizer, "model", None) is None:
        raise RuntimeError("Breeze audio_tokenizer.model is required for codec OV")
    dec = audio_tokenizer.model.decoder
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    pt = dec.pre_transformer
    if hasattr(pt, "config"):
        pt.config.use_cache = False
        pt.config._attn_implementation = "eager"
    n_q = int(getattr(dec.config, "num_quantizers", 16))
    example_t = 2

    class _Quant(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.quantizer = decoder.quantizer

        def forward(self, codes):
            return self.quantizer.decode(codes)

    class _Pre(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.pre_conv = decoder.pre_conv
            self.pre_transformer = decoder.pre_transformer

        def forward(self, h):
            import torch

            h = self.pre_conv(h).transpose(1, 2)
            pt = self.pre_transformer
            h = pt.input_proj(h)
            t_len = int(h.shape[1])
            pos = torch.arange(t_len, device=h.device).unsqueeze(0)
            rope = pt.rotary_emb(h, pos)
            attn = tts_ov.causal_attn_bias(h, t_len, 0)
            for layer in pt.layers:
                h = layer(
                    h,
                    attention_mask=attn,
                    position_ids=pos,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=pos.squeeze(0),
                    position_embeddings=rope,
                )
            return pt.output_proj(pt.norm(h))

    class _Tail(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.upsample = decoder.upsample
            self.tail = decoder.decoder

        def forward(self, h):
            for blocks in self.upsample:
                for block in blocks:
                    h = block(h)
            wav = h
            for block in self.tail:
                wav = block(wav)
            return wav.clamp(min=-1, max=1)

    example = torch.zeros(1, n_q, example_t, dtype=torch.long)
    with torch.inference_mode():
        ex_q = dec.quantizer.decode(example)
        ex_pre = dec.pre_conv(ex_q).transpose(1, 2)
        ex_h = dec.pre_transformer(inputs_embeds=ex_pre, use_cache=False).last_hidden_state
    src = str(path)
    _, q_xml, q_stamp = tts_ov.ir_paths(src, "breeze_codec_quant", ".ov-breeze-codec-quant-v9")
    _, p_xml, p_stamp = tts_ov.ir_paths(src, "breeze_codec_pre", ".ov-breeze-codec-pre-v9")
    _, t_xml, t_stamp = tts_ov.ir_paths(src, "breeze_codec_tail", ".ov-breeze-codec-tail-v9")
    compiled_q = tts_ov.compile_static(_Quant(dec), example, q_xml, q_stamp, device)
    compiled_pre = tts_ov.compile_static(_Pre(dec), ex_q, p_xml, p_stamp, device)
    compiled_tail = tts_ov.compile_static(
        _Tail(dec), ex_h.permute(0, 2, 1).contiguous(), t_xml, t_stamp, device
    )
    # Speak-time MultiRequestStreamRuntime calls decoder(dummy_codes) to learn
    # samples_per_code. That is the torch codebook. Measure the official lengths
    # while the weights still exist, then point decoder.forward at OpenVINO
    # before release_parameters empties the embedding.
    wav_lens = {}
    with torch.inference_mode():
        for t_len in (1, example_t):
            official = dec(torch.zeros(1, n_q, t_len, dtype=torch.long))
            if isinstance(official, (tuple, list)):
                official = official[0]
            wav_lens[t_len] = int(official.shape[-1])
    log.info("breeze codec official wav lengths %s", wav_lens)
    times = {"quant": 0.0, "pre": 0.0, "tail": 0.0, "n": 0}

    def codes_to_wav(codes):
        t_len = int(codes.shape[-1])
        if t_len < 1 or t_len > example_t:
            raise RuntimeError(
                "breeze codec ov chunk t=%s is outside 1..%s" % (t_len, example_t)
            )
        if t_len < example_t:
            padded = F.pad(codes, (example_t - t_len, 0))
        else:
            padded = codes
        codes_np = np.ascontiguousarray(padded.detach().cpu().numpy())
        t0 = time.perf_counter()
        try:
            h = compiled_q(codes_np)[0]
        except Exception:
            log.exception("codec quant in=%s", getattr(codes_np, "shape", None))
            raise
        times["quant"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            h = compiled_pre(np.ascontiguousarray(h))[0]
        except Exception:
            log.exception("codec pre in=%s", getattr(h, "shape", None))
            raise
        times["pre"] += time.perf_counter() - t0
        h = np.ascontiguousarray(np.transpose(h, (0, 2, 1)))
        t0 = time.perf_counter()
        try:
            wav = torch.from_numpy(np.ascontiguousarray(compiled_tail(h)[0]))
        except Exception:
            log.exception("codec tail in=%s", getattr(h, "shape", None))
            raise
        times["tail"] += time.perf_counter() - t0
        want = wav_lens[t_len]
        if int(wav.shape[-1]) < want:
            raise RuntimeError(
                "breeze codec ov wav %s shorter than official t=%s len=%s"
                % (tuple(wav.shape), t_len, want)
            )
        return wav[..., -want:].to(dtype=torch.float32)

    with torch.inference_mode():
        for t_len, official_len in wav_lens.items():
            got = int(codes_to_wav(torch.zeros(1, n_q, t_len, dtype=torch.long)).shape[-1])
            if got != official_len:
                raise RuntimeError(
                    "breeze codec ov length t=%s got=%s official=%s"
                    % (t_len, got, official_len)
                )

    def forward(codes):
        if not hasattr(codes, "detach"):
            raise RuntimeError("breeze codec ov expected a tensor, got %s" % type(codes).__name__)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if int(codes.shape[0]) != 1:
            raise RuntimeError(
                "breeze codec ov is compiled for batch=1; got %s" % (tuple(codes.shape),)
            )
        return codes_to_wav(codes.detach())

    dec.forward = forward
    for part in ("quantizer", "pre_conv", "pre_transformer", "upsample", "decoder"):
        tts_ov.release_parameters(getattr(dec, part, None), "breeze codec " + part)

    def run_step(self, codes_chunk, step_idx):
        codes = codes_chunk.detach()
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        wav = codes_to_wav(codes)
        times["n"] += 1
        if times["n"] == 1 or times["n"] % 5 == 0:
            log.info(
                "breeze codec step=%d n=%d quant=%.3fs pre=%.3fs tail=%.3fs",
                int(step_idx), times["n"], times["quant"], times["pre"], times["tail"],
            )
        return wav

    ExecutionLane.run_step = run_step
    log.info(
        "breeze codec codes-to-wav on OpenVINO %s upsample=%d static_t=%d n_q=%d",
        device, upsample, example_t, n_q,
    )


def build_app(supports):
    return tts_el.build_app(supports, module="breeze")


def run(supports):
    tts_el.run(supports, module="breeze", watchdog="Breeze TTS 2",
               load=_load, sample_rate=24000)
