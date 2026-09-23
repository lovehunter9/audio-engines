# FireRedTTS3-Instruct in-process (clone + design + speak); Base is never loaded.
import logging
import re

from . import tts_el, tts_long

log = logging.getLogger("audio-firered")

# Speed is ffmpeg atempo; generate_acoustic_edit stays off the speak path.
_SPEED_MIN = 0.25
_SPEED_MAX = 4.0
# Speaker boost re-reads this much of the slice before, matching Breeze.
_CTX_SECONDS = 10.0
# generate_tts often stops early; do not claim more text than this audio could hold.
_CHARS_PER_SEC = 6.0
# Only re-queue leftover text when even a fast reading could not have finished it.
_FAST_CHARS_PER_SEC = 10.0
_MIN_OVERLAP = 8
# Official 80 can glue the next paragraph; unstick on 。 / newline only, not commas.
_SENT_END = "。．！？!?"
_SENT_RE = re.compile(r".+?[%s]|.+$" % re.escape(_SENT_END), re.S)
# FireRed joins at 80 ms; the Breeze 520 ms window is too long here.
_SLICE_GAP_MS = 80.0


def _pause_ms(prev, text):
    return _SLICE_GAP_MS


def _unstick(text):
    """Keep official / sentence pieces. Only split a glued next sentence or paragraph."""
    out = []
    for line in re.split(r"\n+", text or ""):
        line = line.strip()
        if not line:
            continue
        bits = [p.strip() for p in _SENT_RE.findall(line) if p and p.strip()]
        out.extend(bits or [line])
    return out


def _strip_ctx_overlap(ctx_text, text):
    """Drop a shared suffix/prefix pair so the next slice does not speak the tail twice.

    Never return empty: a wiped slice is a skipped sentence. Short matches are
    punctuation, not a real overlap."""
    ctx_text = (ctx_text or "").strip()
    text = (text or "").strip()
    if not ctx_text or not text:
        return text
    max_n = min(len(ctx_text), len(text))
    for n in range(max_n, _MIN_OVERLAP - 1, -1):
        if ctx_text[-n:] == text[:n]:
            rest = text[n:].lstrip()
            return rest if rest else text
    return text


def _spoken_text(text, wave, sr):
    """Only the prefix this wave could have covered. A short clip must not carry the rest."""
    import numpy as np

    text = (text or "").strip()
    w = np.asarray(wave, dtype="float32").reshape(-1)
    if not text or not len(w):
        return ""
    n = min(len(text), max(0, int(len(w) / float(sr or 1) * _CHARS_PER_SEC)))
    return text[:n]


def _unsaid(text, wave, sr):
    """Suffix generate_tts never reached. Snap to the last comma or stop in the covered prefix."""
    import numpy as np

    text = (text or "").strip()
    w = np.asarray(wave, dtype="float32").reshape(-1)
    if not text or not len(w):
        return ""
    covered = int(len(w) / float(sr or 1) * _FAST_CHARS_PER_SEC)
    if covered >= max(0, len(text) - 2):
        return ""
    prefix = text[:max(1, covered)]
    cut = 0
    for i, ch in enumerate(prefix):
        if ch in "，、" + _SENT_END:
            cut = i + 1
    rest = (text[cut:] if cut else text[covered:]).lstrip()
    if not rest or rest == text or len(rest) < _MIN_OVERLAP:
        return ""
    return rest


def _queue_rest(sents, i, origin, speak, wave, sr):
    rest = _unsaid(speak, wave, sr)
    if rest and rest != speak and len(sents) < origin * 2 + 6:
        sents.insert(i + 1, rest)
    return len(sents)


def _as_wave(audio, sr):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    a = np.asarray(audio, dtype="float32")
    if a.ndim == 2:
        a = a[0] if a.shape[0] <= 4 else a[:, 0]
    return np.clip(a.reshape(-1), -1.0, 1.0), int(sr)


def _as_torch(audio):
    import numpy as np
    import torch

    t = torch.as_tensor(np.asarray(audio, dtype="float32"), dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def clamp_speed(x):
    return max(_SPEED_MIN, min(_SPEED_MAX, float(x)))


def tempo_for(speed):
    """FireRed reads at a machine-gun pace of its own, so --speak-speed sets the baseline
    the model needs to sound normal, and the caller's EL speed multiplies that."""
    return clamp_speed(float(tts_el.SPEAK_SPEED) * float(speed if speed is not None else 1.0))


def as_tts_triplet(out):
    """Official core.py unpacks 3 values; the Instruct backend generate_tts returns 2."""
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1], None
    return out


def patch_backend_tts_triplet():
    from fireredtts3.llm.fireredtts3_instruct import FireRedTTS3Instruct as Backend

    orig = Backend.generate_tts
    if getattr(orig, "_el_triplet", False):
        return orig

    def wrapped(self, *args, **kwargs):
        return as_tts_triplet(orig(self, *args, **kwargs))

    wrapped._el_triplet = True
    Backend.generate_tts = wrapped
    return wrapped


def _seed_kw(kn):
    """Absent means absent: the model then uses the seed its authors wrote for it."""
    return {} if kn.seed is None else {"seed": int(kn.seed)}


class FireRedBackend:
    sample_rate = 24000
    uses_shared_pack = True
    # Designs speak the approved sample; re-designing per reading is a different speaker.
    prefer_design_speak = False
    # A design blurb describes the speaker, not how to read this sentence.
    card_instruction_is_direction = False

    def __init__(self, model):
        self.model = model
        sr = getattr(getattr(model, "redae", None), "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def native_presets(self):
        return []

    def presets(self):
        return tts_el.premade_cards(self, "firered")

    def _sentences(self, text):
        """Keep official TN/80 split, then unstick a glued next sentence or paragraph."""
        apply = getattr(self.model, "_apply_frontend", None)
        chunks = []
        if apply is not None:
            _joined, _lang, parts = apply(text)
            chunks = [str(p).strip() for p in (parts or []) if p and str(p).strip()]
        if not chunks:
            return _unstick(text) or [text]
        out = []
        for p in chunks:
            out.extend(_unstick(p) or [p])
        return out or [text]

    def _join(self, waves, sr, text="", texts=None):
        # Official 50 ms fade. Do not insert the 520 ms Breeze sentence window.
        import numpy as np

        raw = [np.asarray(w, dtype="float32").reshape(-1) for w in waves]
        last = len(raw) - 1
        parts = []
        for i, w in enumerate(raw):
            wave = w if i == last else tts_long.trim_tail(w, sr)
            if not len(wave):
                continue
            prev = texts[i] if texts and i < len(texts) else ""
            parts.append((wave, prev))
        if not parts:
            return np.zeros(0, dtype="float32"), sr
        if len(parts) == 1:
            return parts[0][0], sr
        fade = max(1, int(tts_long.JOIN_FADE_MS / 1000.0 * sr))
        out = parts[0][0]
        for i, (wave, _prev) in enumerate(parts[1:], 1):
            gap_ms = _pause_ms(parts[i - 1][1], text)
            gap = np.zeros(max(0, int(gap_ms / 1000.0 * sr)), dtype="float32")
            out = np.concatenate([out, gap, tts_long.fade_head(wave, fade)])
        return np.clip(out, -1.0, 1.0), sr

    def _pace(self, audio, sr, speed):
        return tts_long.pace(audio, sr, tempo_for(speed))

    def _carry(self, text, wave, sr):
        """Keep the tail of what this wave actually covered, not the whole slice text."""
        spoken = _spoken_text(text, wave, sr)
        if not spoken:
            return None
        tail_text, tail = tts_long.tail_pair(spoken, wave, sr, _CTX_SECONDS)
        return (tail_text, tail, int(sr)) if len(tail) and tail_text else None

    def _ctx_prompt(self, audio, sr, text, carry):
        """SB stays on official generate_tts. Tail first, clean reference last.

        DiT only conditions on the last 8 latent frames. Putting the synthetic
        tail at the end made each slice continue from degraded audio (the metal).
        The backbone still sees the tail earlier in the prompt for continuity."""
        import numpy as np

        tail_text, tail, tail_sr = carry
        head = tts_long.collect(audio)
        sr_out = int(sr)
        if int(tail_sr) != sr_out:
            tail, _ = tts_el._resample(tts_long.collect(tail), int(tail_sr), sr_out)
        return (tail_text or "") + (text or ""), _as_torch(np.concatenate([tail, head])), sr_out

    def _speak(self, waves, i, total, text, tempo, prev=""):
        """One finished slice on its way out: silence trimmed, pause ahead of it, faded
        in, and through the one tempo filter that spans the whole reading."""
        import numpy as np

        wave, sr = waves
        if i + 1 < total:
            wave = tts_long.trim_tail(wave, sr)
        if i:
            gap = max(0, int(_pause_ms(prev, text) / 1000.0 * sr))
            if gap:
                for paced in tempo.write(np.zeros(gap, dtype="float32"), sr):
                    yield paced, self.sample_rate
            wave = tts_long.fade_head(wave, max(1, int(tts_long.JOIN_FADE_MS / 1000.0 * sr)))
        for paced in tempo.write(wave, sr):
            yield paced, self.sample_rate

    def iter_clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        # Split here so cancel/progress land between sentences; stream paces each slice.
        ctx = kw.get("ctx")
        direction = str(kw.get("instruction") or "").strip()
        if direction:
            # Clone has no instruction+reference mode; direction is ignored.
            log.warning("voice direction ignored on a cloned voice: %r", direction[:60])
        kn = tts_el.resolve_settings(kw.get("settings"), tts_el.INFERENCE_CFG,
                                     direction, text=text,
                                     include_speed_direction=False)
        already = hasattr(self.model, "_apply_frontend")
        sents = self._sentences(text)
        origin = len(sents)
        total = origin
        prompt = _as_torch(prompt_audio)
        extra = {"do_clean": False, "do_tn": False, "do_split": False} if already else {}
        pace_each = bool(kw.get("pace_each"))
        tempo = tts_long.TempoStream(self.sample_rate, tempo_for(kn.speed)) if pace_each else None
        carry = None
        spans = []
        prev = ""
        slice_out = kw.get("slice_out")
        try:
            i = 0
            while i < len(sents):
                sent = sents[i]
                total = len(sents)
                tts_el.job_tick(ctx, i, total)
                p_text, p_audio, p_sr = prompt_text or "", prompt, int(prompt_sr)
                speak = sent
                if carry is not None:
                    speak = _strip_ctx_overlap(carry[0], sent)
                    p_text, p_audio, p_sr = self._ctx_prompt(
                        prompt_audio, prompt_sr, prompt_text, carry)
                audio, sr, _ = as_tts_triplet(self.model.generate_tts(
                    prompt_text=p_text,
                    prompt_audio=p_audio,
                    prompt_audio_sr=p_sr,
                    text=speak,
                    n_timesteps=int(tts_el.N_TIMESTEPS),
                    inference_cfg=float(kn.cfg),
                    **_seed_kw(kn),
                    **extra,
                ))
                wave, sr_out = _as_wave(audio, sr)
                spans.append(len(wave) / float(sr_out or 1))
                total = _queue_rest(sents, i, origin, speak, wave, sr_out)
                if kn.context and i + 1 < total:
                    carry = self._carry(speak, wave, sr_out)
                if slice_out is not None:
                    slice_out.append(speak)
                if tempo is not None:
                    yield from self._speak((wave, sr_out), i, total, text, tempo, prev)
                else:
                    yield wave, sr_out
                prev = sent
                i += 1
                tts_el.job_tick(ctx, i, total)
            if tempo is not None:
                for paced in tempo.finish():
                    yield paced, self.sample_rate
        finally:
            if tempo is not None:
                tempo.abort()
            if spans:
                # generate_tts can stop at max_gen_steps with no signal that the sentence was cut.
                log.info("firered read %d slices, %.1fs total, longest %.1fs, boost=%s",
                         len(spans), sum(spans), max(spans), bool(kn.context))
            tts_long.vram(log, "clone")
        tts_el.job_tick(ctx, total, total)

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        waves, sr_out = [], self.sample_rate
        kw = dict(kw)
        kn = tts_el.resolve_settings(kw.get("settings"), tts_el.INFERENCE_CFG,
                                     str(kw.get("instruction") or ""), text=text,
                                     include_speed_direction=False)
        kw["pace_each"] = False
        slice_out = []
        kw["slice_out"] = slice_out
        for wave, sr_out in self.iter_clone(text, prompt_audio, prompt_sr, prompt_text, **kw):
            waves.append(wave)
        joined, sr_out = self._join(waves, sr_out, text, texts=slice_out)
        return self._pace(joined, sr_out, kn.speed)

    def iter_design(self, instruction, text, **kw):
        # Re-plan once and reuse it so later sentences keep 口音 / 语速 / 音色.
        ctx = kw.get("ctx")
        extra_out = kw.get("extra_out")
        kn = tts_el.resolve_settings(kw.get("settings"), tts_el.DESIGN_CFG,
                                     instruction, text=text, include_speed_direction=False,
                                     seed_jitter=int(kw.get("seed_jitter") or 0))
        speak_as = kn.instruction or instruction
        plan_out = None
        already = hasattr(self.model, "_apply_frontend")
        sents = self._sentences(text)
        origin = len(sents)
        total = origin
        extra = {"do_clean": False, "do_tn": False, "do_split": False} if already else {}
        pace_each = bool(kw.get("pace_each"))
        tempo = tts_long.TempoStream(self.sample_rate, tempo_for(kn.speed)) if pace_each else None
        slice_out = kw.get("slice_out")
        # Later design slices speak against the first slice; generate_tts cannot take instruction+reference.
        anchor = None
        carry = None
        prev = ""
        try:
            i = 0
            while i < len(sents):
                sent = sents[i]
                total = len(sents)
                tts_el.job_tick(ctx, i, total, stage="design")
                seg_plan = None
                speak = sent
                if kn.context and anchor is not None:
                    a_text, a_wave, a_sr = anchor
                    if carry is None:
                        p_text, p_audio, p_sr = a_text, _as_torch(a_wave), a_sr
                    else:
                        speak = _strip_ctx_overlap(carry[0], sent)
                        p_text, p_audio, p_sr = self._ctx_prompt(a_wave, a_sr, a_text, carry)
                    audio, sr, _ = as_tts_triplet(self.model.generate_tts(
                        prompt_text=p_text,
                        prompt_audio=p_audio,
                        prompt_audio_sr=p_sr,
                        text=speak,
                        n_timesteps=int(tts_el.N_TIMESTEPS),
                        inference_cfg=float(kn.cfg),
                        **_seed_kw(kn),
                        **extra,
                    ))
                else:
                    audio, sr, seg_plan = self.model.generate_voice_design(
                        instruction=speak_as,
                        text=sent,
                        n_timesteps=int(tts_el.N_TIMESTEPS),
                        inference_cfg=float(kn.cfg),
                        **_seed_kw(kn),
                        **extra,
                    )
                if seg_plan and plan_out is None:
                    plan_out = seg_plan
                    speak_as = seg_plan
                    if extra_out is not None:
                        extra_out.append({"plan": plan_out})
                wave, sr_out = _as_wave(audio, sr)
                total = _queue_rest(sents, i, origin, speak, wave, sr_out)
                if anchor is None:
                    anchor = (speak, wave, sr_out)
                elif kn.context and i + 1 < total:
                    carry = self._carry(speak, wave, sr_out)
                if slice_out is not None:
                    slice_out.append(speak)
                if tempo is not None:
                    yield from self._speak((wave, sr_out), i, total, text, tempo, prev)
                else:
                    yield wave, sr_out
                prev = sent
                i += 1
                tts_el.job_tick(ctx, i, total, stage="design")
            if tempo is not None:
                for paced in tempo.finish():
                    yield paced, self.sample_rate
        finally:
            if tempo is not None:
                tempo.abort()
            if total > 1:
                log.info("firered designed %d slices, anchored=%s", total, bool(kn.context))
            tts_long.vram(log, "design")
        tts_el.job_tick(ctx, total, total, stage="design")
        if extra_out is not None and not extra_out:
            extra_out.append({"plan": plan_out})

    def design(self, instruction, text, **kw):
        extra_box = []
        waves, sr_out = [], self.sample_rate
        kw = dict(kw)
        kn = tts_el.resolve_settings(kw.get("settings"), tts_el.DESIGN_CFG,
                                     instruction, text=text, include_speed_direction=False,
                                     seed_jitter=int(kw.get("seed_jitter") or 0))
        kw["pace_each"] = False
        kw["extra_out"] = extra_box
        slice_out = []
        kw["slice_out"] = slice_out
        for wave, sr_out in self.iter_design(instruction, text, **kw):
            waves.append(wave)
        wave, sr = self._join(waves, sr_out, text, texts=slice_out)
        wave, sr = self._pace(wave, sr, kn.speed)
        extra = extra_box[0] if extra_box else {"plan": None}
        return wave, sr, extra

    def stream_gap(self, text, sr):
        # Slice pauses are laid inside iter_clone / iter_design so they share the tempo filter.
        return None


def _is_ov():
    from .. import tts_ov

    return tts_ov.is_firered_ov()


def _firered_snapshot_missing(path):
    """Weight files FireRedTTS3Instruct opens. Config-only dirs are not enough."""
    from pathlib import Path

    path = Path(path)
    need = (
        ("redae/model.safetensors", 100 * 1024 * 1024),
        ("fireredtts3_instruct/model.safetensors", 100 * 1024 * 1024),
        ("text_tokenizer/tokenizer.json", 1024 * 1024),
    )
    missing = []
    for rel, floor in need:
        f = path / rel
        if not f.is_file() or f.stat().st_size < floor:
            missing.append(rel)
    return missing


def _wait_firered_snapshot(path, timeout_s=1800):
    """OV load only. CUDA calls FireRedTTS3Instruct directly."""
    import time

    deadline = time.monotonic() + timeout_s
    last = 0.0
    while True:
        missing = _firered_snapshot_missing(path)
        if not missing:
            log.info("firered snapshot ready at %s", path)
            return
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                "FireRed snapshot %s still missing %s; refusing to construct the model"
                % (path, ", ".join(missing))
            )
        now = time.monotonic()
        if now - last >= 15:
            log.info("firered snapshot incomplete, waiting for %s", ", ".join(missing))
            last = now
        time.sleep(2)


def _load():
    from fireredtts3.core import FireRedTTS3Instruct

    patch_backend_tts_triplet()
    tts_el.rewrite_flash_attn(tts_el.ATTN or "eager")
    path = tts_el.model_path()
    if _is_ov():
        from .. import tts_ov

        device = tts_ov.require_gpu()
        restore = tts_ov.force_cpu_torch_device()
        _wait_firered_snapshot(path)
        try:
            log.info("loading FireRedTTS3-Instruct OpenVINO from %s (ov=%s)", path, device)
            instruct = FireRedTTS3Instruct(
                path, use_wetext=tts_el.USE_WETEXT, use_llm_tn=tts_el.USE_LLM_TN,
            )
        finally:
            restore()
        _install_firered_ov(instruct, path, device)
    else:
        log.info("loading FireRedTTS3-Instruct from %s", path)
        instruct = FireRedTTS3Instruct(
            path, use_wetext=tts_el.USE_WETEXT, use_llm_tn=tts_el.USE_LLM_TN,
        )
    tts_long.vram(log, "loaded")
    return FireRedBackend(instruct)


def _install_firered_ov(instruct, path, device):
    """Official generate() loop stays. DiT + patch + AR prefill/decode on GPU."""
    import torch

    from .. import tts_ov

    core = getattr(instruct, "tts_core", None)
    if core is None:
        raise RuntimeError("FireRedTTS3Instruct has no tts_core")
    log.info("firered OV prefill + device-KV decode; DiT+patch on OpenVINO %s", device)
    dit = core.dit
    patch = core.patch_encoder
    hist = int(core.history_length)
    psize = int(core.patch_size)
    redae = int(core.redae_dim)
    hidden = int(core.config.dit_hidden_size)
    t_len = hist + psize
    x = torch.zeros(2, t_len, redae + hidden, dtype=torch.float32)
    t = torch.zeros(2, 1, 1, dtype=torch.float32)
    _, dit_xml, dit_stamp = tts_ov.ir_paths(path, "firered_dit", ".ov-firered-v1")
    compiled_dit = tts_ov.compile_module(dit, (x, t), dit_xml, dit_stamp, device)
    dit_req = compiled_dit.compiled.create_infer_request()
    dit_shape = (2, t_len, redae + hidden)
    orig_dit = dit.forward
    times = {"prefill": 0.0, "ar": 0.0, "dit": 0.0}
    core._ov_times = times

    def dit_forward(x_in=None, t_in=None, **kwargs):
        import time
        import numpy as np
        import openvino as ov

        if x_in is None:
            x_in = kwargs.get("x")
        if t_in is None:
            t_in = kwargs.get("t")
        if x_in is None or t_in is None:
            return orig_dit(x=x_in, t=t_in, **kwargs)
        t0 = time.perf_counter()
        xa = np.ascontiguousarray(x_in.detach().float().cpu().numpy())
        ta = np.ascontiguousarray(t_in.detach().float().cpu().numpy())
        # Official Euler stays in Python. One request avoids a new feed per step.
        if tuple(xa.shape) != dit_shape or ta.shape != (2, 1, 1):
            out = compiled_dit(xa, ta)[0]
        else:
            dit_req.set_input_tensor(0, ov.Tensor(xa))
            dit_req.set_input_tensor(1, ov.Tensor(ta))
            dit_req.infer()
            out = np.array(dit_req.get_output_tensor(0).data, copy=True)
        times["dit"] += time.perf_counter() - t0
        return torch.from_numpy(np.ascontiguousarray(out)).to(x_in.device)

    dit.forward = dit_forward
    lat = torch.zeros(1, psize, redae, dtype=torch.float32)
    _, pe_xml, pe_stamp = tts_ov.ir_paths(path, "firered_patch", ".ov-firered-v1")
    compiled_pe = tts_ov.compile_module(patch, lat, pe_xml, pe_stamp, device)
    orig_pe = patch.forward

    def pe_forward(latents, *args, **kwargs):
        import numpy as np

        if not hasattr(latents, "detach"):
            return orig_pe(latents, *args, **kwargs)
        arr = np.ascontiguousarray(latents.detach().float().cpu().numpy())
        out = compiled_pe(arr)[0]
        return torch.from_numpy(np.ascontiguousarray(out)).to(latents.device)

    patch.forward = pe_forward
    log.info("firered DiT + patch_encoder on OpenVINO %s", device)
    _install_firered_backbone(core, path, device)
    _install_firered_redae_cache(instruct, times)
    _install_firered_decoder_ov(instruct, path, device)
    _install_firered_decode_log(instruct, times)


def _firered_sliding_mask(hidden, window):
    """Additive mask: causal, and only the last `window` keys (eager Qwen3).

    kv_idx > q_idx - window matches transformers sliding_window_overlay.
    """
    import torch

    t = hidden.shape[1]
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=hidden.device), 1)
    q_idx = torch.arange(t, device=hidden.device).view(t, 1)
    kv_idx = torch.arange(t, device=hidden.device).view(1, t)
    blocked = causal | (kv_idx <= q_idx - window)
    attn = hidden.new_zeros(1, 1, t, t)
    return attn.masked_fill(blocked, torch.finfo(hidden.dtype).min)


def _install_firered_decoder_ov(instruct, path, device):
    """Put the RedAE decoder transformer on OpenVINO. ISTFT stays in torch.

    A short speak spent ~8s in this decode, after the AR loop. Qwen3Model.forward
    builds the sliding-window mask through the HF cache helpers and the trace
    dies with 'tuple index out of range'. Walk the layers with that same mask.
    Export failure keeps the official decoder and is logged; load still finishes.
    """
    import torch
    from torch import nn

    from .. import tts_ov

    redae = getattr(instruct, "redae", None)
    dec = getattr(redae, "decoder", None) if redae is not None else None
    qwen = getattr(dec, "qwen3", None) if dec is not None else None
    if qwen is None:
        raise RuntimeError("FireRed RedAE decoder has no qwen3")
    hidden = int(dec.qwen3_config.hidden_size)
    cfg = qwen.config
    cfg.use_cache = False
    cfg._attn_implementation = "eager"
    layer_types = [str(x) for x in (getattr(cfg, "layer_types", None) or [])]
    window = int(getattr(cfg, "sliding_window", 0) or 0)
    n_sliding = sum(x == "sliding_attention" for x in layer_types)
    log.info(
        "firered redae decoder layers=%d sliding=%d window=%s",
        len(layer_types), n_sliding, window,
    )

    class _Dec(nn.Module):
        def __init__(self, qwen):
            super().__init__()
            self.qwen = qwen

        def forward(self, embeds):
            wdtype = next(self.qwen.parameters()).dtype
            h = embeds.to(dtype=wdtype)
            t = h.shape[1]
            pos = torch.arange(t, device=h.device).unsqueeze(0)
            rope = self.qwen.rotary_emb(h, pos)
            full = h.new_zeros(1, 1, t, t)
            full = full.masked_fill(
                torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), 1),
                torch.finfo(h.dtype).min,
            )
            sliding = _firered_sliding_mask(h, window) if window else full
            for i, layer in enumerate(self.qwen.layers):
                kind = layer_types[i] if i < len(layer_types) else "full_attention"
                mask = sliding if kind == "sliding_attention" and window else full
                h = layer(
                    h,
                    attention_mask=mask,
                    position_ids=pos,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=rope,
                )
                if isinstance(h, (tuple, list)):
                    h = h[0]
            return self.qwen.norm(h).to(dtype=embeds.dtype)

    _, xml, stamp = tts_ov.ir_paths(
        str(path), "firered_redae_dec", ".ov-firered-dec-v2"
    )
    try:
        compiled = tts_ov.compile_causal(
            _Dec(qwen), torch.zeros(1, 32, hidden, dtype=torch.float32),
            xml, stamp, device,
            dynamize_ranks=(3,),
        )
    except Exception:
        log.exception("firered redae decoder stayed on torch")
        return
    orig_fwd = dec.forward

    def forward(xs):
        import numpy as np

        if not hasattr(xs, "detach"):
            return orig_fwd(xs)
        h = dec.in_proj(xs)
        h = h.reshape(h.shape[0], -1, hidden)
        arr = np.ascontiguousarray(h.detach().float().cpu().numpy())
        try:
            out = compiled(arr)[0]
        except Exception as e:
            log.error("firered redae decoder ov failed, torch fallback: %s", e)
            return orig_fwd(xs)
        return dec.istft_head(torch.from_numpy(np.ascontiguousarray(out)))

    dec.forward = forward
    log.info("firered redae decoder transformer on OpenVINO %s", device)


def _install_firered_decode_log(instruct, times):
    """RedAE decode sits after the AR loop and was not in the step timer."""
    import time

    redae = getattr(instruct, "redae", None)
    orig = getattr(redae, "decode", None) if redae is not None else None
    if orig is None:
        raise RuntimeError("FireRedTTS3Instruct has no redae.decode to time")

    def decode(latents, *args, **kwargs):
        t0 = time.perf_counter()
        out = orig(latents, *args, **kwargs)
        times["dec"] = time.perf_counter() - t0
        log.info(
            "firered ov slice prefill=%.3fs ar=%.3fs dit=%.3fs dec=%.3fs "
            "redae=%.3fs hits=%s",
            times.get("prefill", 0.0), times.get("ar", 0.0), times.get("dit", 0.0),
            times["dec"], times.get("redae", 0.0), times.get("redae_hits", 0),
        )
        return out

    redae.decode = decode


def _install_firered_redae_cache(instruct, times):
    """Same reference wav must not pay RedAE encode again on the warm speak."""
    import hashlib
    import time

    import numpy as np

    orig_tok = getattr(instruct, "_tokenize_audio", None)
    if orig_tok is None:
        raise RuntimeError("FireRedTTS3Instruct has no _tokenize_audio to cache")
    cache = {}
    times["redae"] = 0.0
    times["redae_hits"] = 0

    def _tokenize_audio(audio, audio_sr):
        t0 = time.perf_counter()
        if hasattr(audio, "detach"):
            arr = np.ascontiguousarray(audio.detach().cpu().float().numpy())
        else:
            arr = np.ascontiguousarray(np.asarray(audio, dtype="float32"))
        key = (int(audio_sr), tuple(int(x) for x in arr.shape),
               hashlib.sha1(arr.tobytes()).hexdigest())
        hit = key in cache
        if hit:
            latents = cache[key].clone()
            times["redae_hits"] += 1
        else:
            latents = orig_tok(audio, audio_sr)
            cache[key] = latents.detach()
        dt = time.perf_counter() - t0
        times["redae"] += dt
        log.info(
            "firered redae %s t=%.3fs total=%.3fs hits=%d",
            "hit" if hit else "miss", dt, times["redae"], times["redae_hits"],
        )
        return latents

    instruct._tokenize_audio = _tokenize_audio
    log.info("firered prompt RedAE encode cached on %s", type(instruct).__name__)


def _install_firered_backbone(core, path, device):
    """Replace _backbone_one_step: OV prefill seeds stateful decode (same KV layout)."""
    import torch

    from .. import tts_ov

    llm = getattr(core, "backbone_llm", None)
    inner = getattr(llm, "model", None) if llm is not None else None
    if inner is None:
        raise RuntimeError("FireRed tts_core has no backbone_llm.model")
    cfg = inner.config
    n_layers, n_kv, head_dim = tts_ov.kv_meta(cfg)
    hidden = int(cfg.hidden_size)
    q_patch = int(getattr(core, "patch_size", 4) or 4)
    example_t = 16
    src = str(path)
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))

    embeds_pre = torch.zeros(1, example_t, hidden, dtype=torch.float32)
    mask_pre = torch.ones(1, example_t, dtype=torch.long)
    _, pre_xml, pre_stamp = tts_ov.ir_paths(
        src, "firered_llm_prefill", ".ov-firered-prefill-v15"
    )
    prefill = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, False),
        (embeds_pre, mask_pre), pre_xml, pre_stamp, device,
        dynamize_ranks=(2, 3),
        stateful=False,
    )

    embeds_q1 = torch.zeros(1, 1, hidden, dtype=torch.float32)
    mask_q1 = torch.ones(1, example_t + 1, dtype=torch.long)
    # Position is an input. Tracing arange(past) would freeze every token at example_t.
    pos_q1 = torch.arange(example_t, example_t + 1, dtype=torch.long).unsqueeze(0)
    _, xml, stamp = tts_ov.ir_paths(
        src, "firered_llm_decode_q1", ".ov-firered-decode-q1-v16"
    )
    compiled_q1 = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, True, external_position=True),
        (embeds_q1, mask_q1, pos_q1, *past), xml, stamp, device,
        dynamize_ranks=(2, 4),
        stateful=False,
    )
    runner = tts_ov.DeviceKvRunner(
        compiled_q1, n_layers, prefill=prefill, external_position=True,
    )
    n_step = {"i": 0}
    times = core._ov_times

    def _log_step(embeds, hidden, prefix):
        last = hidden[:, -1].float()
        score = float("nan")
        std = float("nan")
        if torch.isfinite(last).all():
            score = float(torch.sigmoid(core.stop_head(last)).item())
            std = float(hidden.float().std())
        log.info(
            "firered ov step=%d q=%d prefix=%d stop=%.4f hidden_std=%.4f "
            "prefill=%.3fs ar=%.3fs dit=%.3fs redae=%.3fs redae_hits=%s",
            n_step["i"], int(embeds.shape[1]), prefix, score, std,
            times["prefill"], times["ar"], times["dit"],
            times.get("redae", 0.0), times.get("redae_hits", 0),
        )

    def _backbone_one_step(input_embeds, cache=None):
        import time

        if input_embeds.shape[0] != 1:
            raise RuntimeError(
                "firered ov backbone is compiled for batch=1; got %s"
                % (tuple(input_embeds.shape),)
            )
        if cache is None:
            runner.reset()
            times["prefill"] = times["ar"] = times["dit"] = 0.0
            n_step["i"] = 0
            t0 = time.perf_counter()
            hidden = runner.step(input_embeds)
            times["prefill"] += time.perf_counter() - t0
            n_step["i"] = 1
            _log_step(input_embeds, hidden, runner.prefix_len)
            return hidden, True
        q = int(input_embeds.shape[1])
        t0 = time.perf_counter()
        if q == 1:
            hidden = runner.step(input_embeds)
        else:
            parts = [runner.step(input_embeds[:, i:i + 1]) for i in range(q)]
            hidden = torch.cat(parts, dim=1)
        times["ar"] += time.perf_counter() - t0
        n_step["i"] += 1
        if n_step["i"] % 20 == 0:
            _log_step(input_embeds, hidden, runner.prefix_len)
        return hidden, True

    core._backbone_one_step = _backbone_one_step
    log.info(
        "firered OV prefill + device-KV decode q=1 (q=%d sliced) on OpenVINO %s",
        q_patch, device,
    )


def build_app(supports):
    return tts_el.build_app(supports, module="firered")


def run(supports):
    tts_el.run(supports, module="firered", watchdog="FireRedTTS3-Instruct",
               load=_load, sample_rate=24000)
