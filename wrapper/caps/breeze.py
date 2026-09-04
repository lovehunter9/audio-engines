# Breeze TTS 2 in-process. Official PyTorch: clone (ref+transcript), design, voice direction.
import logging
import os
import re
from pathlib import Path

from . import tts_el

log = logging.getLogger("audio-breeze")

# Official 8 inline events live in the speak text. Do not cut inside them.
_EVENT_RE = re.compile(
    r"\[(?:笑|咳嗽|清嗓子|叹气)\]"
    r"|\((?:laugh|cough|clears throat|sigh)\)"
)
# CJK stops need no following space; EN .!? need space/end, and "." is not a cut after a digit (3.14) or a title (Mr.).
_SENT_END = re.compile(
    r"\.\.\.|…"
    r"|[。．！？；]"
    r"|(?<!\d)[.!?](?=\s|$|[\"'”’）)\]])"
)
_CLAUSE_END = re.compile(r"[，、]|[；;](?=\s|$)|(?<!\d),(?=\s)")
_EN_ABBR = re.compile(
    r"(?:^|[\s(\[（])(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.$",
    re.I,
)
# Head fade is only anti-click into the pause. The last slice is never trimmed.
_JOIN_FADE_MS = 16.0
_PAUSE_MS_ZH = 520.0
_PAUSE_MS_EN = 280.0
_TRIM_KEEP_MS = 80.0
_TRIM_THRESH = 0.008
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
_CTX_TOKENS = int(_CTX_SECONDS * _FPS) + 40
_BREAK_RE = re.compile(r"[。．！？；;，、!?,]\s*")

def _fallback_attn(requested, impl):
    return tts_el.fallback_attn(requested, impl)


def _rewrite_flash_attn(requested):
    tts_el.rewrite_flash_attn(requested)


def _collect(audio):
    import numpy as np

    a = np.asarray(audio, dtype="float32")
    if a.ndim > 1:
        a = a.reshape(-1)
    return np.clip(a, -1.0, 1.0)


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


def _hold_events(text):
    held = []

    def keep(m):
        held.append(m.group(0))
        return "\x00%d\x00" % (len(held) - 1)

    return _EVENT_RE.sub(keep, text), held


def _unhold(text, held):
    if not held:
        return text
    return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)


def _skip_en_abbr(text, match):
    """Do not treat 'Mr. Smith' as two sentences. CJK marks never hit this."""
    return match.group(0) == "." and bool(_EN_ABBR.search(text[:match.end()]))


def _cut(text, ender, skip=None):
    """Keep the delimiter on the left piece. Empty parts are dropped."""
    parts, start = [], 0
    for m in ender.finditer(text):
        if skip and skip(text, m):
            continue
        piece = text[start:m.end()]
        if piece:
            parts.append(piece)
        start = m.end()
    tail = text[start:]
    if tail:
        parts.append(tail)
    return parts


def _hard_cut(piece, limit):
    """A stretch with nothing to break on. Prefer a space in reach over the count."""
    out = []
    while len(piece) > limit:
        at = piece[:limit].rfind(" ")
        if at < limit // 2:
            at = limit
        out.append(piece[:at].strip())
        piece = piece[at:].lstrip()
    out.append(piece.strip())
    return [p for p in out if p]


def split_speak(text, limit=None):
    """Newlines always split. Sentence then clause only when a piece is over budget."""
    limit = speak_limit() if limit is None else int(limit)
    raw = text or ""
    if not raw.strip():
        return [raw] if raw else []
    protected, held = _hold_events(raw)
    out = []
    for line in re.split(r"\n+", protected):
        if not line.strip():
            continue
        if len(_unhold(line, held)) <= limit:
            piece = _unhold(line, held).strip()
            if piece:
                out.append(piece)
            continue
        for sent in _cut(line, _SENT_END, skip=_skip_en_abbr) or [line]:
            if len(_unhold(sent, held)) <= limit:
                piece = _unhold(sent, held).strip()
                if piece:
                    out.append(piece)
                continue
            for cl in _cut(sent, _CLAUSE_END) or [sent]:
                piece = _unhold(cl, held).strip()
                if not piece:
                    continue
                # Counting characters is the last resort: punctuation carries the prosody.
                if len(piece) <= limit:
                    out.append(piece)
                else:
                    out.extend(_hard_cut(piece, limit))
    return out


def _pause_ms(text):
    if any("\u4e00" <= ch <= "\u9fff" for ch in text or ""):
        return _PAUSE_MS_ZH
    return _PAUSE_MS_EN


def _snap(text, at, span):
    """First break at or after `at`. Only forward, so the tail can come in under the
    seconds asked for but never over them, which is what the budget was reserved for."""
    m = _BREAK_RE.search(text, at, min(len(text), at + span))
    return m.end() if m else at


def _tail_pair(text, wave, sr, seconds=_CTX_SECONDS):
    """The tail of a finished slice with the text that goes with it. The text is cut by
    the share of the audio it covers, moved to a break, and the audio cut to match."""
    import numpy as np

    w = np.asarray(wave, dtype="float32").reshape(-1)
    total = len(w) / float(sr or 1)
    if not len(w) or not (text or "").strip() or total <= seconds:
        return (text or "").strip(), w
    want = len(text) * (seconds / total)
    at = _snap(text, int(len(text) - want), max(4, int(want * 0.3)))
    if at <= 0 or at >= len(text):
        return text.strip(), w
    cut = max(int(len(w) * (float(at) / len(text))), len(w) - int(seconds * (sr or 1)))
    return text[at:].strip(), w[cut:]


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


def _fade_head(wave, fade):
    import numpy as np

    n = min(int(fade), len(wave))
    if n <= 0:
        return wave
    out = np.array(wave, dtype="float32", copy=True)
    out[:n] *= np.linspace(0.0, 1.0, n, dtype="float32")
    return out


def _trim_tail(wave, sr, thresh=_TRIM_THRESH, keep_ms=_TRIM_KEEP_MS):
    """Drop trailing silence only; keep keep_ms after the last voiced sample so the last syllable stays."""
    import numpy as np

    w = np.asarray(wave, dtype="float32").reshape(-1)
    if not len(w):
        return w
    keep = max(0, int(float(keep_ms) / 1000.0 * sr))
    loud = np.flatnonzero(np.abs(w) >= thresh)
    if not len(loud):
        return w[:max(1, keep)]
    return w[:min(len(w), int(loud[-1]) + 1 + keep)]


def _too_long(n):
    raise ValueError(
        "Input is too long: piece has %d characters but max_seq_len=%d. "
        "Use shorter text or shorter reference audio." % (n, int(tts_el.MAX_SEQ_LEN)))


def _join(waves, sr, fade_ms=_JOIN_FADE_MS, pause_ms=_PAUSE_MS_ZH):
    import numpy as np

    raw = [np.asarray(w, dtype="float32").reshape(-1) for w in waves]
    last = len(raw) - 1
    parts = []
    for i, w in enumerate(raw):
        parts.append(w if i == last else _trim_tail(w, sr))
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros(0, dtype="float32"), sr
    if len(parts) == 1:
        return parts[0], sr
    fade = max(1, int(float(fade_ms) / 1000.0 * sr))
    gap = np.zeros(max(0, int(float(pause_ms) / 1000.0 * sr)), dtype="float32")
    out = parts[0]
    for wave in parts[1:]:
        out = np.concatenate([out, gap, _fade_head(wave, fade)])
    return np.clip(out, -1.0, 1.0), sr


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
        set_all_seeds(int(seed if seed is not None else tts_el.SEED))
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
            audio = _collect(getattr(chunk, "audio", chunk))
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

        tail_text, tail = _tail_pair(text, np.concatenate(waves), self.sample_rate)
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
        kn = tts_el.resolve_settings(kw.get("settings"), base_cfg,
                                     direction or "Speak clearly and naturally.", text=text)
        wav = _collect(prompt_audio)
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
        try:
            sf.write(path, wav, rate, format="WAV", subtype="PCM_16")
            ctx = kw.get("ctx")
            pace = bool(kw.get("pace_each"))
            total = len(parts)
            for i, part in enumerate(parts):
                tts_el.job_tick(ctx, i, total)
                if pace:
                    if i:
                        gap_n = max(0, int(_pause_ms(text) / 1000.0 * self.sample_rate))
                        if gap_n:
                            import numpy as np
                            yield self._pace(np.zeros(gap_n, dtype="float32"),
                                             self.sample_rate, kn.speed)
                    n = 0
                    said = []
                    for audio, sr in self._iter_generate(
                            part, kn.instruction, ref_path=path, ref_text=prompt_text,
                            cfg=kn.cfg, ctx=ctx, seed=kn.seed, carry=carry):
                        wave = _collect(audio)
                        said.append(wave)
                        if i and n == 0:
                            wave = _fade_head(wave, max(1, int(_JOIN_FADE_MS / 1000.0 * sr)))
                        n += 1
                        yield self._pace(wave, sr, kn.speed)
                    if n == 0:
                        raise RuntimeError("Breeze TTS 2 produced no audio")
                    if kn.context and i + 1 < total:
                        carry = self._carry(carry_path, part, said)
                else:
                    audio, sr = self._generate(part, kn.instruction,
                                               ref_path=path, ref_text=prompt_text,
                                               cfg=kn.cfg, ctx=ctx, seed=kn.seed,
                                               carry=carry)
                    wave = _collect(audio)
                    if kn.context and i + 1 < total:
                        carry = self._carry(carry_path, part, [wave])
                    yield self._pace(wave, sr, kn.speed)
                tts_el.job_tick(ctx, i + 1, total)
        finally:
            _vram("clone")
            for gone in (path, carry_path):
                try:
                    os.unlink(gone)
                except OSError:
                    pass

    def _pace(self, wave, sr, speed):
        import numpy as np

        factor = max(0.25, min(4.0, float(speed if speed is not None else 1.0)))
        w = _collect(wave)
        if abs(factor - 1.0) < 0.02 or not len(w):
            return w, sr
        n = max(1, int(round(len(w) / factor)))
        if n == len(w):
            return w, sr
        x = np.linspace(0.0, float(len(w) - 1), n)
        return np.interp(x, np.arange(len(w), dtype="float64"), w).astype("float32"), sr

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        waves, sr = [], self.sample_rate
        kw = dict(kw)
        kw["pace_each"] = False
        for wave, sr in self.iter_clone(text, prompt_audio, prompt_sr, prompt_text, **kw):
            waves.append(wave)
        return _join(waves, sr, pause_ms=_pause_ms(text))

    def stream_gap(self, text, sr):
        # Intra-part codec chunks already include the part pause from iter_clone.
        return None

    def stream_next_slice(self, wave, sr):
        return wave

    def design(self, instruction, text, **kw):
        ctx = kw.get("ctx")
        kn = tts_el.resolve_settings(kw.get("settings"), tts_el.DESIGN_CFG,
                                     instruction, text=text)
        tts_el.job_tick(ctx, 0, 1, stage="design")
        audio, sr = self._generate(text, kn.instruction or instruction,
                                   cfg=kn.cfg, ctx=ctx, seed=kn.seed)
        audio, sr = self._pace(audio, sr, kn.speed)
        tts_el.job_tick(ctx, 1, 1, stage="design")
        return audio, sr, {}


def _vram(tag):
    """This process's own high-water mark, then reset it. On a timeslice card the whole
    card's used is every tenant's sum, so ours is the only figure we can attribute."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        log.info("breeze vram %s: peak=%.0f MiB now=%.0f MiB reserved=%.0f MiB", tag,
                 torch.cuda.max_memory_allocated() / 2 ** 20,
                 torch.cuda.memory_allocated() / 2 ** 20,
                 torch.cuda.memory_reserved() / 2 ** 20)
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _load():
    attn = tts_el.ATTN or "eager"
    _rewrite_flash_attn(attn)
    from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    path = Path(tts_el.model_path())
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
    _vram("loaded")
    return BreezeBackend(runtime, tokenizer, audio_tokenizer, model)


def build_app(supports):
    return tts_el.build_app(supports, module="breeze")


def run(supports):
    tts_el.run(supports, module="breeze", watchdog="Breeze TTS 2",
               load=_load, sample_rate=24000)
