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


def _load():
    attn = tts_el.ATTN or "eager"
    _rewrite_flash_attn(attn)
    from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    path = Path(tts_el.model_path())
    if _is_ov():
        from .. import tts_ov

        device = tts_ov.require_gpu()
        tts_ov.allow_breeze_fast_on_cpu()
        log.info("loading Breeze TTS 2 OpenVINO from %s (attn=%s, ov=%s)", path, attn, device)
        tokenizer, model, audio_tokenizer = load_runtime(
            path, device="cpu", attn_implementation=attn,
        )
        _install_breeze_ov(model, path, device)
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


def _install_breeze_ov(model, path, device):
    """Compile the backbone that FastBreezeStreamingRuntime actually calls.

    Official iter_audio_chunks never hits model.depth_decoder.forward. Prefill
    is backbone_model(...); every later frame is BackboneGraph._decode_step
    (embed → backbone → lm_head). Those two go to GPU. Export failure is fatal.
    """
    import numpy as np
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
    _, pre_xml, pre_stamp = tts_ov.ir_paths(src, "breeze_backbone_prefill", ".ov-breeze-v2")
    _, dec_xml, dec_stamp = tts_ov.ir_paths(src, "breeze_backbone_decode", ".ov-breeze-v2")
    prefill = tts_ov.compile_causal(
        tts_ov.causal_step_module(backbone, n_layers, False),
        (embeds_pre, mask_pre), pre_xml, pre_stamp, device,
    )
    decode = tts_ov.compile_causal(
        tts_ov.causal_step_module(backbone, n_layers, True),
        (embeds_dec, mask_dec, *past), dec_xml, dec_stamp, device,
    )

    def _as_out(hidden_np, kv_np, fallback_past=None):
        hidden = torch.from_numpy(np.ascontiguousarray(hidden_np))
        kv = [torch.from_numpy(np.ascontiguousarray(x)) for x in kv_np]
        past_out = tts_ov.unflatten_kv(kv, n_layers) if kv else fallback_past
        return type("BBOut", (), {"last_hidden_state": hidden, "past_key_values": past_out})()

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
        arr = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
        attn = kwargs.get("attention_mask")
        if past_in is None:
            mask = tts_ov.mask_np(attn, arr.shape[1], 0)
            out = prefill(arr, mask)
            return _as_out(out[0], list(out)[1:])
        flat = []
        used = None
        if kwargs.get("cache_position") is not None:
            used = int(kwargs["cache_position"].reshape(-1)[0])
        for t in tts_ov.flatten_kv(past_in):
            # StaticCache is allocated to max_seq; feed the used prefix only.
            sl = t.shape[-2]
            take = sl if used is None else max(1, min(used, sl))
            flat.append(np.ascontiguousarray(t[..., :take, :].detach().float().cpu().numpy()))
        past_len = flat[0].shape[-2]
        mask = tts_ov.mask_np(attn, arr.shape[1], past_len)
        out = decode(arr, mask, *flat)
        hidden_np, kv_np = out[0], list(out)[1:]
        if hasattr(past_in, "layers"):
            kv_t = [torch.from_numpy(np.ascontiguousarray(x)) for x in kv_np]
            tts_ov.write_static_kv(past_in, kv_t)
            return _as_out(hidden_np, [], fallback_past=past_in)
        return _as_out(hidden_np, kv_np)

    backbone.forward = bb_forward

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

    decode_step._ov = True
    BackboneGraph._decode_step = decode_step
    log.info(
        "breeze backbone prefill+decode on OpenVINO %s layers=%d (depth loop stays official)",
        device, n_layers,
    )


def build_app(supports):
    return tts_el.build_app(supports, module="breeze")


def run(supports):
    tts_el.run(supports, module="breeze", watchdog="Breeze TTS 2",
               load=_load, sample_rate=24000)
