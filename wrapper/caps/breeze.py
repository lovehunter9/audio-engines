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


def speak_limit():
    """Chars one clone generate can finish. 1500 new tokens ≈ 120s; do not wait for 2048."""
    seq = int(tts_el.MAX_SEQ_LEN)
    gen = int(tts_el.MAX_NEW_TOKENS)
    if gen <= 0:
        return max(64, seq)
    return max(64, min(seq, gen // 4))


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
                if piece:
                    out.append(piece)
    return out


def _pause_ms(text):
    if any("\u4e00" <= ch <= "\u9fff" for ch in text or ""):
        return _PAUSE_MS_ZH
    return _PAUSE_MS_EN


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

    def _generate(self, text, instruction, ref_path=None, ref_text=None, cfg=None, ctx=None):
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs

        request = {
            "id": "req",
            "text": text,
            "instruction": instruction or "Speak clearly and naturally.",
            "speaker": "S0",
        }
        template_name = "tts_instruction"
        if ref_path:
            request["ref_audio_path"] = str(ref_path)
            request["ref_text"] = (ref_text or "").strip()
            template_name = "ref_edit_tata"
        set_all_seeds(int(tts_el.SEED))
        inputs = prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            get_template(template_name),
            guidance_scale=float(cfg if cfg is not None else tts_el.CFG_SCALE),
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        chunks = []
        for chunk in self.runtime.iter_audio_chunks(inputs, request_id="req"):
            tts_el.job_tick(ctx)
            audio = getattr(chunk, "audio", chunk)
            chunks.append(_collect(audio))
        if not chunks:
            raise RuntimeError("Breeze TTS 2 produced no audio")
        import numpy as np
        return np.concatenate(chunks), self.sample_rate

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        import soundfile as sf
        import tempfile

        # Voice Direction: ref + transcript + instruction at CFG 4; bare clone keeps CFG 1.
        direction = str(kw.get("instruction") or "").strip()
        if direction:
            instruction, cfg = direction, tts_el.DESIGN_CFG
        else:
            instruction, cfg = "Speak clearly and naturally.", tts_el.CFG_SCALE
        parts = split_speak(text)
        if not parts:
            parts = [text]
        limit = speak_limit()
        for part in parts:
            if len(part) > limit:
                _too_long(len(part))
        wav = _collect(prompt_audio)
        with tempfile.NamedTemporaryFile(prefix="breeze-ref-", suffix=".wav", delete=False) as fh:
            path = fh.name
        try:
            sf.write(path, wav, int(prompt_sr), format="WAV", subtype="PCM_16")
            waves, sr = [], self.sample_rate
            ctx = kw.get("ctx")
            total = len(parts)
            for i, part in enumerate(parts):
                tts_el.job_tick(ctx, i, total)
                audio, sr = self._generate(part, instruction,
                                           ref_path=path, ref_text=prompt_text, cfg=cfg, ctx=ctx)
                waves.append(_collect(audio))
                tts_el.job_tick(ctx, i + 1, total)
            return _join(waves, sr, pause_ms=_pause_ms(text))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def design(self, instruction, text, **kw):
        ctx = kw.get("ctx")
        tts_el.job_tick(ctx, 0, 1, stage="design")
        audio, sr = self._generate(text, instruction, cfg=tts_el.DESIGN_CFG, ctx=ctx)
        tts_el.job_tick(ctx, 1, 1, stage="design")
        return audio, sr, {}


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
    return BreezeBackend(runtime, tokenizer, audio_tokenizer, model)


def build_app(supports):
    return tts_el.build_app(supports, module="breeze")


def run(supports):
    tts_el.run(supports, module="breeze", watchdog="Breeze TTS 2",
               load=_load, sample_rate=24000)
