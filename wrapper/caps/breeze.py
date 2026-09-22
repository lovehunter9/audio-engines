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
    audio = path / "audio_tokenizer" / "model.safetensors"
    if not audio.is_file() or audio.stat().st_size < 100 * 1024 * 1024:
        missing.append("audio_tokenizer/model.safetensors")
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
    if _is_ov():
        from .. import tts_ov

        # Sentinel, tokenizer register, and the snapshot wait are Intel-only.
        # The CUDA branch below is load_runtime on the resolved device.
        _wait_breeze_snapshot(path)
        _register_breeze_tokenizer()
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


def _cache_backbone_suppress_mask():
    """Build the reserved-codec mask once.

    Official sample_logits turns range(codebook, vocab) into a fresh Python
    list on every backbone token. That vocab is the text vocab, about 128k,
    and the copy was sitting outside the infer timer.
    """
    import torch
    import models.fast_streaming as fast_streaming
    from models.cudagraph import sampling

    orig = sampling.sample_logits
    cached = {}

    def sample_logits(logits, *args, suppress_tokens=None, **kwargs):
        if suppress_tokens:
            width = int(logits.shape[-1])
            mask = cached.get(width)
            if mask is None:
                mask = torch.zeros(width, dtype=torch.bool)
                mask[list(suppress_tokens)] = True
                cached[width] = mask
            kwargs["suppress_mask"] = mask.to(device=logits.device)
            suppress_tokens = None
        return orig(logits, *args, suppress_tokens=suppress_tokens, **kwargs)

    sampling.sample_logits = sample_logits
    fast_streaming.sample_logits = sample_logits


def _cache_prompt_encode():
    """Reuse codec codes when the reference wav bytes have not changed.

    iter_clone writes a new temp path every request, so a path cache never hits.
    prepare_inputs encodes that file again between the budget log and prefill.
    """
    import hashlib
    import time

    import breeze_infer.templates as tmpl

    original = tmpl.encode_prompt_audio
    cache = {}

    def cached(audio_tokenizer, audio_path):
        with open(audio_path, "rb") as fh:
            raw = fh.read()
        key = hashlib.sha1(raw).hexdigest()
        hit = cache.get(key)
        if hit is not None:
            log.info("breeze prompt encode cache hit bytes=%d", len(raw))
            return hit.clone()
        t0 = time.perf_counter()
        codes = original(audio_tokenizer, audio_path)
        kept = codes.detach().cpu().contiguous()
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[key] = kept
        log.info(
            "breeze prompt encode cache miss bytes=%d %.3fs shape=%s",
            len(raw), time.perf_counter() - t0, tuple(kept.shape),
        )
        return kept.clone()

    tmpl.encode_prompt_audio = cached


def _install_breeze_ov(model, path, device, audio_tokenizer=None):
    """Official five stages all go to OpenVINO: text, backbone prefill/decode, depth, codec."""
    _cache_backbone_suppress_mask()
    _cache_prompt_encode()
    import torch

    from models.cudagraph.backbone_graph import BackboneGraph

    from .. import tts_ov

    backbone = getattr(model, "backbone_model", None)
    if backbone is None:
        raise RuntimeError("Breeze model has no backbone_model to export")
    cfg = getattr(backbone, "config", None) or model.config
    n_layers, n_kv, head_dim = tts_ov.kv_meta(cfg)
    hidden = int(cfg.hidden_size)
    # A long reference prefill overflowed in bf16. Decode used to be traced
    # before this cast, so it kept bf16 weights and then read the f32 prefill
    # cache: the first frame could be a real phone, and every later frame was
    # the other network. Both exports are f32. CUDA never reaches here.
    backbone.float()
    src = str(path)
    example_t = 16
    prefill_t = 320
    embeds_pre = torch.zeros(1, prefill_t, hidden, dtype=torch.float32)
    embeds_dec = torch.zeros(1, 1, hidden, dtype=torch.float32)
    mask_pre = torch.ones(1, prefill_t, dtype=torch.long)
    mask_dec = torch.ones(1, example_t + 1, dtype=torch.long)
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
    # v11 still invented RoPE from the example cache. v12 takes the position
    # the official loop already wrote (prefill_len + step).
    _, dec_xml, dec_stamp = tts_ov.ir_paths(src, "breeze_backbone_decode", ".ov-breeze-v12")
    pos_dec = torch.zeros(1, 1, dtype=torch.long)
    decode = tts_ov.compile_causal(
        tts_ov.causal_kv_module(backbone, True, external_position=True),
        (embeds_dec, mask_dec, pos_dec, *past), dec_xml, dec_stamp, device,
        stateful=True, kv_from=3,
    )
    _, pre_xml, pre_stamp = tts_ov.ir_paths(src, "breeze_backbone_prefill", ".ov-breeze-v7")
    prefill = tts_ov.compile_causal(
        tts_ov.causal_kv_module(backbone, False),
        (embeds_pre, mask_pre), pre_xml, pre_stamp, device,
        dynamize_ranks=(),
    )
    import gc
    gc.collect()
    runner = tts_ov.StatefulKvRunner(decode, n_layers)
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
            import numpy as np
            import time
            real = int(embeds.shape[1])
            if real > prefill_t:
                raise RuntimeError(
                    "breeze ov prefill length %d exceeds static %d" % (real, prefill_t)
                )
            t0 = time.perf_counter()
            buf = np.zeros((1, prefill_t, hidden), dtype=np.float32)
            buf[:, :real] = np.ascontiguousarray(embeds.detach().float().cpu().numpy())
            mask = np.zeros((1, prefill_t), dtype=np.int64)
            mask[:, :real] = 1
            out = prefill(buf, mask)
            # Do not assign to `hidden`: that name is the closure width, and
            # any assignment makes the zeros() above an unbound local.
            states = torch.from_numpy(np.ascontiguousarray(out[0]))[:, :real]
            kv = []
            for i in range(1, 1 + 2 * n_layers):
                arr = np.array(out[i], copy=True)
                kv.append(np.ascontiguousarray(arr[:, :, :real, :]))
            runner.seed_flat(kv)
            log.info("breeze prefill tokens=%d %.3fs", real, time.perf_counter() - t0)
        else:
            pos = kwargs.get("position_ids")
            if pos is None:
                raise RuntimeError("breeze ov decode has no position_ids")
            states = runner.step(embeds, pos.detach())
        if states.dtype != embeds.dtype:
            states = states.to(dtype=embeds.dtype)
        return type("BBOut", (), {
            "last_hidden_state": states,
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

    step_t = {"n": 0, "embed": 0.0, "ov": 0.0, "head": 0.0}

    def decode_step(self):
        import time

        if self.batch_size != 1:
            raise RuntimeError(
                "breeze ov BackboneGraph is compiled for batch=1; got %d" % self.batch_size
            )
        t0 = time.perf_counter()
        inputs_embeds = self.embed_tokens(self.input_ids_buf)
        t1 = time.perf_counter()
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=self.attn_mask,
            past_key_values=self.static_cache,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            use_cache=True,
        )
        t2 = time.perf_counter()
        self.hidden_buf.copy_(out.last_hidden_state.to(self.hidden_buf.dtype))
        logits = self.lm_head(
            self.hidden_buf[:, -1, :].to(dtype=self.lm_head.weight.dtype)
        )
        self.logits_buf.copy_(logits)
        self.cfg_logits_buf.copy_(self.logits_buf[: self.half])
        t3 = time.perf_counter()
        step_t["n"] += 1
        step_t["embed"] += t1 - t0
        step_t["ov"] += t2 - t1
        step_t["head"] += t3 - t2
        if step_t["n"] % 8 == 0:
            log.info(
                "breeze backbone steps=%d embed=%.3fs ov=%.3fs head=%.3fs",
                step_t["n"], step_t["embed"], step_t["ov"], step_t["head"],
            )

    prefill_kv._ov = True
    decode_step._ov = True
    BackboneGraph.prefill_kv = prefill_kv
    BackboneGraph._decode_step = decode_step
    log.info(
        "breeze backbone prefill+decode stateful-KV on OpenVINO %s layers=%d",
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
    enc.float()
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
    _, xml, stamp = tts_ov.ir_paths(str(path), "breeze_text_encoder", ".ov-breeze-text-v2")
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
    """Leave depth-codebook sampling on the official loop.

    The fused OpenVINO frame drew every codebook with its own inverse CDF.
    The official codec of those ids still transcribed as a grunt, so the ids
    were not speech. Weights stay for the model's own sampler. CUDA never
    calls this.

    backbone.float() (f32 prefill export) also floats the tied codebook
    embedding. The depth projector stays bf16, and the official loop then
    dies in inputs_embeds_projector: float != BFloat16. Put that shared
    embedding back on the projector dtype. The projector itself is not cast.
    """
    depth = getattr(model, "depth_decoder", None)
    inner = getattr(depth, "model", None) if depth is not None else None
    if inner is None:
        raise RuntimeError("Breeze model has no depth_decoder.model")
    head = getattr(depth, "codebooks_head", None)
    if head is None or getattr(head, "weight", None) is None:
        raise RuntimeError("Breeze depth decoder has no codebooks_head")
    proj = getattr(inner, "inputs_embeds_projector", None)
    if proj is None or getattr(proj, "weight", None) is None:
        raise RuntimeError("Breeze depth decoder has no inputs_embeds_projector")
    emb = getattr(inner, "embed_tokens", None)
    if emb is None or getattr(emb, "weight", None) is None:
        raise RuntimeError("Breeze depth decoder has no embed_tokens")
    wdtype = proj.weight.dtype
    if emb.weight.dtype != wdtype:
        emb.weight.data = emb.weight.data.to(dtype=wdtype)
        log.info(
            "breeze depth codebook embedding restored to %s for the official loop",
            wdtype,
        )
    log.info(
        "breeze depth stays on the official codebook loop path=%s device=%s",
        path, device,
    )
    return depth


def _install_breeze_codec_ov(audio_tokenizer, path, device):
    """Leave codes→wav on the official streaming decoder.

    A stateless OpenVINO forward of each 2-frame chunk zeroed the causal left
    context. The joined waveform jumped at every chunk and ASR returned a
    grunt, not the text. The streaming lane already carries that cache.
    Decoder weights stay put so the lane can run. CUDA never calls this.
    """
    if audio_tokenizer is None or getattr(audio_tokenizer, "model", None) is None:
        raise RuntimeError("Breeze audio_tokenizer.model is required for the codec")
    dec = getattr(audio_tokenizer.model, "decoder", None)
    if dec is None:
        raise RuntimeError("Breeze audio tokenizer has no decoder")
    log.info(
        "breeze codec stays on the official streaming decoder path=%s device=%s",
        path, device,
    )
    return dec


def build_app(supports):
    return tts_el.build_app(supports, module="breeze")


def run(supports):
    tts_el.run(supports, module="breeze", watchdog="Breeze TTS 2",
               load=_load, sample_rate=24000)
