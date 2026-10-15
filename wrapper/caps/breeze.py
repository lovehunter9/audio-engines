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
        self.runtime._ov_wav = None
        ov_codec = bool(getattr(self.runtime, "_ov_codec_on", False))
        # A Python exception through the CUDA codec loop SIGSEGVs the process.
        for chunk in self.runtime.iter_audio_chunks(inputs, request_id="req"):
            if ov_codec:
                n += 1
                continue
            audio = tts_long.collect(getattr(chunk, "audio", chunk))
            if not len(audio):
                continue
            n += 1
            yield audio, self.sample_rate
        wav = getattr(self.runtime, "_ov_wav", None)
        if ov_codec:
            if wav is None or not len(wav):
                raise RuntimeError("Breeze OpenVINO codec produced no audio")
            yield wav, self.sample_rate
            return
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


# GPU compiles run only after every torch weight copy has been released.
_OV_AFTER_EXPORT = []


def _untie_depth_codebooks(model, backbone):
    """Depth codebook embedding shares storage with the backbone table.

    Releasing the backbone table after its export empties that storage.
    The depth trace then sees a 0-d weight and dies in embedding.
    """
    import torch.nn as nn

    depth = getattr(getattr(model, "depth_decoder", None), "model", None)
    depth_emb = getattr(depth, "embed_tokens", None) if depth is not None else None
    audio = getattr(getattr(backbone, "embed_tokens", None), "embed_audio_tokens", None)
    if depth_emb is None or audio is None or getattr(depth_emb, "weight", None) is None:
        return
    if depth_emb.weight.data_ptr() != audio.weight.data_ptr():
        return
    depth_emb.weight = nn.Parameter(audio.weight.detach().clone(), requires_grad=False)
    log.info(
        "untied depth codebook embedding shape=%s dtype=%s",
        tuple(depth_emb.weight.shape), depth_emb.weight.dtype,
    )


def _install_breeze_embed_ov(backbone, path, device):
    """One compiled lookup for codebook ids. Prefill and decode both call it.

    The slot graph stays one transformer. This table is not a second backbone.
    """
    import torch

    from .. import tts_ov

    emb = getattr(backbone, "embed_tokens", None)
    if emb is None:
        raise RuntimeError("Breeze backbone has no embed_tokens")
    n_cb = int(emb.audio_tokens_offsets.numel())
    _, xml, stamp = tts_ov.ir_paths(str(path), "breeze_embed", ".ov-breeze-embed-v1")
    example = torch.zeros(1, 8, n_cb, dtype=torch.long)
    tts_ov.compile_causal(
        emb, example, xml, stamp, device,
        dynamize_ranks=(3,), compile_graph=False, precision="f16",
    )
    tts_ov.release_parameters(emb, "breeze backbone embed")
    emb._ov_embed = (example, xml, stamp, device)

    def _bind():
        compiled = tts_ov.compile_causal(
            emb, example, xml, stamp, device,
            dynamize_ranks=(3,), precision="f16",
        )
        _bind_embed(emb, compiled)

    emb._ov_embed_bind = _bind
    log.info("breeze backbone embed exported codebooks=%d", n_cb)


def _bind_embed(emb, compiled):
    def forward(_self, input_ids):
        import numpy as np
        import torch as th

        arr = np.ascontiguousarray(input_ids.detach().cpu().numpy())
        out = compiled(arr)[0]
        return th.from_numpy(np.ascontiguousarray(out))

    emb.forward = forward.__get__(emb, type(emb))
    log.info("breeze backbone embed on OpenVINO codebooks=%d", int(emb.audio_tokens_offsets.numel()))


def _install_breeze_ov(model, path, device, audio_tokenizer=None):
    """Text, backbone, depth, and codec on OpenVINO. Backbone has one graph.

    Replaying the 320-wide prefix on every new token was about 1.5s a step.
    Walking that prefix one token at a time was still about 28s. A second
    compiled backbone is another copy of the weights and OOM'd the iGPU
    machine. This one graph writes the whole prefix in one call, then one
    token per decode step. Token i is slot i, and the cache shape stays 320.
    """
    _cache_backbone_suppress_mask()
    _cache_prompt_encode()
    import torch

    from models.cudagraph.backbone_graph import BackboneGraph

    from .. import tts_ov

    backbone = getattr(model, "backbone_model", None)
    if backbone is None:
        raise RuntimeError("Breeze model has no backbone_model to export")
    cfg = getattr(backbone, "config", None) or model.config
    hidden = int(cfg.hidden_size)
    n_layers, n_kv, head_dim = tts_ov.kv_meta(cfg)
    # A long reference prefill overflowed in bf16. CUDA never reaches here.
    backbone.float()
    src = str(path)
    max_t = 320
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, max_t, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, n_kv, max_t, head_dim, dtype=torch.float32))
    # Trace a short query, then let q vary. q=1 would bake a one-token scatter.
    example_q = 8
    embeds = torch.zeros(1, example_q, hidden, dtype=torch.float32)
    mask = torch.ones(1, example_q, max_t, dtype=torch.long)
    pos = torch.arange(example_q, dtype=torch.long).unsqueeze(0)
    slot = torch.arange(example_q, dtype=torch.long)
    # f16 logits from the slot graph, and an f16 codebook gather, both changed
    # the sampled tokens. The transcript matched when this head and this table
    # stayed f32 torch. They are one matmul and one gather, not a second backbone.
    lm_head = model.lm_head.float()
    _untie_depth_codebooks(model, backbone)
    _, xml, stamp = tts_ov.ir_paths(src, "breeze_backbone_slot", ".ov-breeze-slot-v3")
    example = (embeds, mask, pos, slot, *past)
    # Export while the weights are still loaded, drop them, then compile once.
    tts_ov.compile_causal(
        tts_ov.causal_fixed_slot_module(backbone, max_t, lm_head),
        example, xml, stamp, device,
        dynamize_ranks=(1, 2, 3),
        compile_graph=False,
        precision="f16",
    )
    tts_ov.release_parameters(getattr(backbone, "layers", None), "breeze backbone layers")
    tts_ov.release_parameters(getattr(backbone, "norm", None), "breeze backbone norm")
    _install_breeze_text_ov(model, path, device)
    _install_breeze_depth_ov(model, path, device)
    _install_breeze_codec_ov(audio_tokenizer, path, device)
    compiled = tts_ov.compile_causal(
        tts_ov.causal_fixed_slot_module(backbone, max_t, lm_head),
        example, xml, stamp, device,
        dynamize_ranks=(1, 2, 3),
        precision="f16",
    )
    runner = tts_ov.FixedSlotRunner(compiled, n_layers, max_t, n_kv, head_dim)
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
            import time
            n = int(embeds.shape[1])
            if n > max_t:
                raise RuntimeError(
                    "breeze ov prefix length %d exceeds static %d" % (n, max_t)
                )
            runner.reset()
            t0 = time.perf_counter()
            states = runner.prefill(embeds.detach())
            log.info(
                "breeze prefill tokens=%d %.3fs",
                n, time.perf_counter() - t0,
            )
        else:
            if int(embeds.shape[1]) != 1:
                raise RuntimeError(
                    "breeze ov continuation expects one new token, got %s"
                    % (tuple(embeds.shape),)
                )
            if runner.slot <= 0:
                raise RuntimeError("breeze ov continuation before a prefill")
            if runner.slot >= max_t:
                raise RuntimeError(
                    "breeze ov slot %d exceeds %d" % (runner.slot, max_t)
                )
            states = runner.step(embeds.detach())
        if states.dtype != embeds.dtype:
            states = states.to(dtype=embeds.dtype)
        return type("BBOut", (), {
            "last_hidden_state": states,
            "past_key_values": past_in,
        })()

    backbone.forward = bb_forward

    def prefill_kv(self, past_key_values):
        seq_len = int(runner.slot)
        if seq_len <= 0:
            raise RuntimeError("breeze ov prefill_kv before a prefill")
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
        "breeze backbone one slot graph on OpenVINO %s layers=%d",
        device, n_layers,
    )
    for _bind in list(_OV_AFTER_EXPORT):
        _bind()
    _OV_AFTER_EXPORT.clear()


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

    proj = getattr(model, "text_encoder_proj", None)
    feature_idx = getattr(model, "text_encoder_feature_layer_idx", (-1,))
    if feature_idx != (-1,):
        raise RuntimeError(
            "breeze text encoder feature layers %s are not in the OpenVINO text graph"
            % (feature_idx,)
        )
    if getattr(model, "text_encoder_layer_projs", None) is not None:
        raise RuntimeError(
            "breeze text_encoder_layer_projs is still a torch module; it has to be in the text graph"
        )

    class _Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            if proj is not None:
                self.proj = proj

        def forward(self, input_ids, attention_mask, position_ids):
            hs = self.enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
            ).last_hidden_state
            # Official _project_segments runs this after the encoder. Fold it
            # in so the linear is not a second CPU matmul.
            layer = getattr(self, "proj", None)
            if layer is None:
                return hs
            if isinstance(layer, (nn.Linear, nn.Sequential)):
                return layer(hs)
            return layer(inputs_embeds=hs, position_ids=position_ids)

    ids = torch.zeros(1, t_fixed, dtype=torch.long)
    mask = torch.ones(1, t_fixed, dtype=torch.long)
    pos = torch.arange(t_fixed, dtype=torch.long).unsqueeze(0)
    _, xml, stamp = tts_ov.ir_paths(str(path), "breeze_text_encoder", ".ov-breeze-text-v3")
    # enc.float() makes the hidden state float32. The projection stays bf16
    # unless it is cast too, and Linear then dies: float != BFloat16.
    # Keep the original dtype for the backbone; the graph itself is f16.
    if proj is not None and hasattr(proj, "parameters"):
        try:
            wdtype = next(proj.parameters()).dtype
        except StopIteration:
            wdtype = next(enc.parameters()).dtype
    else:
        wdtype = next(enc.parameters()).dtype
    if proj is not None and hasattr(proj, "float"):
        proj.float()
    tts_ov.compile_module(
        _Text(), (ids, mask, pos), xml, stamp, device,
        precision="f16", compile_graph=False,
    )
    tts_ov.release_parameters(enc, "breeze text encoder")
    tts_ov.release_parameters(proj, "breeze text encoder proj")
    tts_ov.release_parameters(
        getattr(model, "embed_text_tokens", None), "breeze unused text token embed",
    )

    def _bind_text():
        compiled = tts_ov.compile_module(
            _Text(), (ids, mask, pos), xml, stamp, device, precision="f16",
        )
        orig = model._batched_text_encoder_forward

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

        def _project_segments(self, segment_lengths, seg_hidden_states, seg_layer_hidden_states, is_separate):
            # text_encoder_proj already ran inside the OpenVINO text graph.
            if self.text_encoder_layer_projs is not None:
                raise RuntimeError("breeze text layer projs are not on OpenVINO")
            parts = []
            for seg_hs in seg_hidden_states:
                if seg_hs.dim() == 3:
                    seg_hs = seg_hs.squeeze(0)
                parts.append(seg_hs)
            if not parts:
                raise RuntimeError("breeze ov text projection got no segments")
            return torch.cat(parts, dim=0), None

        model._project_segments = _project_segments.__get__(model, type(model))
        log.info("breeze text encoder on OpenVINO %s static_t=%d", device, t_fixed)

    _OV_AFTER_EXPORT.append(_bind_text)


def _install_breeze_depth_ov(model, path, device):
    """One OpenVINO infer per frame. Sampling stays the official multinomial.

    Fifteen separate depth calls stay near 0.6s a frame, and a stateful
    one-token step did not get faster or reach EOS. The frame graph unrolls
    the official two-token prefill and each later codebook. The draw is
    Gumbel-max. CUDA never calls this.

    backbone.float() also floats the tied codebook embedding. The depth
    projector stays bf16, and the official embed then dies in
    inputs_embeds_projector. Put that shared embedding back on the projector
    dtype before the frame copies it. The projector itself is not cast.
    """
    import time

    import numpy as np
    import torch

    from models.cudagraph.depth_decoder_graph import DepthDecoderGraph

    from .. import tts_ov

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
    cfg = inner.config
    hidden = int(cfg.hidden_size)
    n_layers, _, _ = tts_ov.kv_meta(cfg)
    n_codebooks = int(
        getattr(head, "num_codebooks", 0) or getattr(cfg, "num_codebooks", 0) or 0
    )
    vocab = int(head.weight.shape[-1])
    codec_cfg = getattr(getattr(model, "config", None), "codec_config", None)
    codebook_size = int(getattr(codec_cfg, "codebook_size", vocab) or vocab)
    n_tokens = n_codebooks - 1
    if n_tokens < 1:
        raise RuntimeError("breeze depth frame needs codebooks, got %d" % n_codebooks)
    # Intel-tts52 put layers[6:] in f16 and the wav matched the all-f16 file.
    # The first half staying f32 did not change a sample. Only the last
    # three layers' wide projections move this time.
    inner.layers.float()
    inner.norm.float()
    for layer in inner.layers[-3:]:
        layer.mlp.gate_proj.half()
        layer.mlp.up_proj.half()
    _, frame_xml, frame_stamp = tts_ov.ir_paths(
        str(path), "breeze_depth_frame", ".ov-breeze-depth-frame-v13",
    )
    frame_mod = tts_ov.fused_depth_frame(
        inner, head, n_codebooks, vocab, codebook_size,
    )
    frame_example = (
        torch.zeros(1, 2, hidden, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
        torch.ones(1, dtype=torch.long),
        torch.ones(1, dtype=torch.float32),
        torch.ones(1, dtype=torch.long),
        torch.full((n_tokens, vocab), 0.5, dtype=torch.float32),
    )
    tts_ov.compile_causal(
        frame_mod, frame_example, frame_xml, frame_stamp, device,
        dynamize_ranks=(), compile_graph=False, precision="f16", pin_f32=True, latency=True,
    )
    tts_ov.release_parameters(inner.layers, "breeze depth layers")
    tts_ov.release_parameters(inner.norm, "breeze depth norm")
    tts_ov.release_parameters(head, "breeze depth head")

    def _bind_depth():
        frame = tts_ov.compile_causal(
            frame_mod, frame_example, frame_xml, frame_stamp, device,
            dynamize_ranks=(), precision="f16", pin_f32=True, latency=True,
        )

        def full_loop(self):
            t0 = time.perf_counter()
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
            seq = self.inputs_embeds_projector(prefill_embeds)
            if seq.shape[0] != 1 or int(seq.shape[1]) != 2:
                raise RuntimeError(
                    "breeze ov depth prefill is [1, 2, H]; got %s" % (tuple(seq.shape),)
                )
            temp = np.ascontiguousarray(
                self.temperature_buf[:1, 0].detach().float().cpu().numpy()
            )
            topk = np.ascontiguousarray(
                self.top_k_buf[:1, 0].detach().cpu().numpy().astype(np.int64)
            )
            topp = np.ascontiguousarray(
                self.top_p_buf[:1, 0].detach().float().cpu().numpy()
            )
            dos = np.ascontiguousarray(
                self.do_sample_buf[:1].detach().cpu().numpy().astype(np.int64)
            )
            uniform = np.ascontiguousarray(
                torch.rand(n_tokens, vocab).numpy().astype(np.float32)
            )
            embeds = np.ascontiguousarray(seq.detach().float().cpu().numpy())
            toks = np.array(
                frame(embeds, temp, topk, topp, dos, uniform)[0]
            ).reshape(-1)
            if int(toks.shape[0]) != int(self.num_decode_codebooks):
                raise RuntimeError(
                    "breeze depth frame returned %d tokens, expected %d"
                    % (int(toks.shape[0]), int(self.num_decode_codebooks))
                )
            for i, tok in enumerate(toks):
                self.output_tokens[:, i] = int(tok)
            log.info(
                "breeze depth frame codebooks=%d %.3fs",
                int(toks.shape[0]), time.perf_counter() - t0,
            )

        DepthDecoderGraph._full_loop = full_loop
        log.info(
            "breeze depth frame on OpenVINO %s layers=%d codebooks=%d",
            device, n_layers, n_codebooks,
        )

    _OV_AFTER_EXPORT.append(_bind_depth)
    return depth


def _codec_trace_mask(
    config, input_embeds, attention_mask, cache_position, past_key_values,
    position_ids=None, sliding_window=None, **kwargs,
):
    """Causal mask as a plain tensor.

    transformers builds this with vmap. torch.jit.trace dies there with
    unordered_map::at, so the codec never exports. Eager wants 0 / -inf.
    SDPA wants True where a position is kept. No cache: this graph is one
    static window.
    """
    import torch

    if past_key_values is not None:
        raise RuntimeError("codec trace mask does not take a kv cache")
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    q = cache_position
    kv = torch.arange(q.shape[0], device=q.device)
    keep = kv.view(1, 1, 1, -1) <= q.view(1, 1, -1, 1)
    if sliding_window is not None:
        keep = keep & (kv.view(1, 1, 1, -1) > q.view(1, 1, -1, 1) - int(sliding_window))
    impl = getattr(config, "_attn_implementation", "eager")
    batches = input_embeds.shape[0]
    if impl == "sdpa":
        return keep.expand(batches, 1, q.shape[0], kv.shape[0])
    if impl not in ("eager", "flex_attention"):
        raise RuntimeError("codec attention implementation %s cannot be traced" % impl)
    neg = torch.tensor(torch.finfo(input_embeds.dtype).min, dtype=input_embeds.dtype, device=q.device)
    zero = torch.zeros((), dtype=input_embeds.dtype, device=q.device)
    return torch.where(keep, zero, neg).expand(batches, 1, q.shape[0], kv.shape[0])


def _patch_codec_trace_masks():
    import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as dec_mod
    import transformers.models.mimi.modeling_mimi as enc_mod

    saved = (
        dec_mod.create_causal_mask,
        dec_mod.create_sliding_window_causal_mask,
        enc_mod.create_causal_mask,
    )

    def causal(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None, **kwargs):
        return _codec_trace_mask(
            config, input_embeds, attention_mask, cache_position, past_key_values,
            position_ids=position_ids, sliding_window=None,
        )

    def sliding(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None, **kwargs):
        return _codec_trace_mask(
            config, input_embeds, attention_mask, cache_position, past_key_values,
            position_ids=position_ids, sliding_window=getattr(config, "sliding_window", None),
        )

    dec_mod.create_causal_mask = causal
    dec_mod.create_sliding_window_causal_mask = sliding
    enc_mod.create_causal_mask = causal
    log.info("codec trace masks are plain tensors")
    return dec_mod, enc_mod, saved


def _restore_codec_trace_masks(patched):
    dec_mod, enc_mod, saved = patched
    dec_mod.create_causal_mask = saved[0]
    dec_mod.create_sliding_window_causal_mask = saved[1]
    enc_mod.create_causal_mask = saved[2]


def _install_breeze_codec_ov(audio_tokenizer, path, device):
    """Encoder and decoder on one OpenVINO graph each.

    Official chunked_decode feeds the previous codes back in as left context
    and trims that prefix off the waveform. A stateless 2-frame forward zeroed
    that context and ASR heard a grunt. The window here is the same idea:
    static T, real codes on the left, right pad the causal tail.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from models.fast_streaming import FastBreezeStreamingRuntime

    from .. import tts_ov

    if audio_tokenizer is None or getattr(audio_tokenizer, "model", None) is None:
        raise RuntimeError("Breeze audio_tokenizer.model is required for the codec")
    tok_model = audio_tokenizer.model
    dec = getattr(tok_model, "decoder", None)
    enc = getattr(tok_model, "encoder", None)
    if dec is None or enc is None:
        raise RuntimeError("Breeze audio tokenizer needs encoder and decoder")
    # Same cast as the text encoder. A bf16 weight against a float32 wave
    # dies in the first Linear during the trace.
    dec.float()
    enc.float()
    n_q = int(dec.config.num_quantizers)
    valid_q = int(getattr(tok_model, "encoder_valid_num_quantizers", n_q))
    upsample = int(dec.total_upsample)
    static_t = 64
    left_context = 25
    src = str(path)

    class _Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.dec = dec

        def forward(self, codes):
            return self.dec(codes)

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, wave):
            out = self.enc.encode(input_values=wave.unsqueeze(1), return_dict=True)
            codes = out.audio_codes[:, :valid_q]
            return codes

    _, dec_xml, dec_stamp = tts_ov.ir_paths(src, "breeze_codec_dec", ".ov-breeze-codec-dec-v2")
    _, enc_xml, enc_stamp = tts_ov.ir_paths(src, "breeze_codec_enc", ".ov-breeze-codec-enc-v2")
    dec_example = torch.zeros(1, n_q, static_t, dtype=torch.long)
    enc_example = torch.zeros(1, 16000, dtype=torch.float32)
    patched = _patch_codec_trace_masks()
    try:
        tts_ov.compile_static(
            _Dec(), dec_example, dec_xml, dec_stamp, device,
            precision="f16", compile_graph=False,
        )
        tts_ov.compile_dyn_last(
            _Enc(), enc_example, enc_xml, enc_stamp, device,
            precision="f16", compile_graph=False,
        )
    finally:
        _restore_codec_trace_masks(patched)
    downsample = int(tok_model.get_encode_downsample_rate())
    tts_ov.release_parameters(tok_model, "breeze audio tokenizer")

    def _bind_codec():
        compiled_dec = tts_ov.compile_static(
            _Dec(), dec_example, dec_xml, dec_stamp, device, precision="f16",
        )
        compiled_enc = tts_ov.compile_dyn_last(
            _Enc(), enc_example, enc_xml, enc_stamp, device, precision="f16",
        )

        def _decode_window(codes_bqt):
            """codes [1, Q, T] -> wav [samples]. Left context is real codes."""
            total = int(codes_bqt.shape[-1])
            pieces = []
            start = 0
            while start < total:
                context = min(left_context, start)
                take = static_t - context
                end = min(start + take, total)
                begin = end - (context + (end - start))
                window = np.ascontiguousarray(codes_bqt[:, :, begin:end])
                pad = static_t - int(window.shape[-1])
                if pad > 0:
                    window = np.pad(window, ((0, 0), (0, 0), (0, pad)))
                elif pad < 0:
                    raise RuntimeError(
                        "breeze codec window %d exceeds static %d" % (window.shape[-1], static_t)
                    )
                wav = np.array(compiled_dec(window)[0])
                per = int(wav.shape[-1]) // static_t
                if per < 1:
                    raise RuntimeError(
                        "breeze codec decode returned %s, static_t=%d upsample=%d"
                        % (wav.shape, static_t, upsample)
                    )
                a0 = context * per
                a1 = a0 + (end - start) * per
                pieces.append(np.ascontiguousarray(wav[..., a0:a1]).reshape(-1))
                start = end
            if not pieces:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(pieces).astype(np.float32, copy=False)

        class _OvCodec:
            def __init__(self):
                self.codes = []

            def open_request(self, req_id, reset=True, is_first_decode=None):
                self.codes = []

            def close_request(self, req_id):
                if not self.codes:
                    return
                cat = torch.cat(
                    [c.detach().to(dtype=torch.long).cpu() for c in self.codes], dim=-1,
                )
                # Official frames arrive [1, Q, T]. decoder.forward wants [1, Q, T].
                if cat.dim() != 3 or int(cat.shape[1]) != n_q:
                    raise RuntimeError(
                        "breeze codec codes are [1, %d, T], got %s" % (n_q, tuple(cat.shape))
                    )
                import time
                t0 = time.perf_counter()
                wav = _decode_window(cat.numpy())
                log.info(
                    "breeze codec decode frames=%d samples=%d %.3fs",
                    int(cat.shape[-1]), int(wav.shape[0]), time.perf_counter() - t0,
                )
                self._runtime._ov_wav = wav

            def decode_request_chunk(self, req_id, codes_chunk, reset=False):
                if reset:
                    self.codes = []
                self.codes.append(codes_chunk.detach())
                n = int(codes_chunk.shape[-1]) * upsample
                return torch.zeros(1, 1, n)

        holder = _OvCodec()

        def _codec(self):
            if getattr(self, "_ov_codec", None) is None:
                holder._runtime = self
                self._ov_codec = holder
            return self._ov_codec

        FastBreezeStreamingRuntime._ov_codec_on = True
        FastBreezeStreamingRuntime._codec = _codec
        orig_encode = audio_tokenizer.encode

        def encode(audios, sr=None, return_dict=True):
            wavs = audio_tokenizer._normalize_audio_inputs(audios, sr=sr)
            inputs = audio_tokenizer.feature_extractor(
                raw_audio=wavs,
                sampling_rate=int(audio_tokenizer.feature_extractor.sampling_rate),
                return_tensors="pt",
            )
            wave = inputs["input_values"]
            if wave.dim() == 3:
                wave = wave.squeeze(1)
            mask = inputs["padding_mask"]
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            import time
            t0 = time.perf_counter()
            raw = np.array(
                compiled_enc(np.ascontiguousarray(wave.detach().float().cpu().numpy()))[0]
            )
            log.info(
                "breeze codec encode samples=%d %.3fs shape=%s",
                int(wave.shape[-1]), time.perf_counter() - t0, tuple(raw.shape),
            )
            # raw is [B, Q, T_codes]. Official encode returns a list of [T, Q].
            out = []
            for i in range(int(raw.shape[0])):
                summed = int(mask[i].sum())
                valid = -(-summed // downsample)
                code = raw[i, :, :valid]
                out.append(torch.from_numpy(np.ascontiguousarray(code.transpose(1, 0))))
            return {"audio_codes": out}

        audio_tokenizer.encode = encode
        audio_tokenizer._encode_cpu = orig_encode
        log.info(
            "breeze codec on OpenVINO %s quantizers=%d static_t=%d left_context=%d upsample=%d",
            device, n_q, static_t, left_context, upsample,
        )

    _OV_AFTER_EXPORT.append(_bind_codec)
    return dec


def build_app(supports):
    return tts_el.build_app(supports, module="breeze")


def run(supports):
    tts_el.run(supports, module="breeze", watchdog="Breeze TTS 2",
               load=_load, sample_rate=24000)
