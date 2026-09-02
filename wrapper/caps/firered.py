# FireRedTTS3-Instruct in-process (clone + design + speak); Base is never loaded.
import logging
import re

from . import tts_el

log = logging.getLogger("audio-firered")

# Official generate_acoustic_edit: X in [0.5, 2.0], step 0.1.
_SPEED_MIN = 0.5
_SPEED_MAX = 2.0
_SLOW_RE = re.compile(r"很慢|非常慢|缓慢|very\s+slow|extremely\s+slow", re.I)
_FAST_RE = re.compile(r"很快|非常快|very\s+fast|extremely\s+fast", re.I)


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
    x = max(_SPEED_MIN, min(_SPEED_MAX, float(x)))
    return round(x * 10.0) / 10.0


def speed_for(instruction=""):
    """Map a design instruction onto official acoustic-edit speed (0.5–2.0)."""
    factor = clamp_speed(tts_el.SPEAK_SPEED)
    text = instruction or ""
    if _SLOW_RE.search(text):
        return min(factor, 0.6)
    if _FAST_RE.search(text):
        return 1.0
    return factor


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


class FireRedBackend:
    sample_rate = 24000
    uses_shared_pack = True
    # generate_tts has no instruction; design-identity cards keep generate_voice_design + plan, clones speak the wav.
    prefer_design_speak = True

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
        """Official core splits at token_max_n=80 and runs wetext. Keep that."""
        apply = getattr(self.model, "_apply_frontend", None)
        if apply is None:
            return [text]
        _joined, _lang, sentences = apply(text)
        return [s for s in (sentences or []) if s and str(s).strip()] or [text]

    def _join(self, waves, sr, fade_ms=50.0):
        import numpy as np

        if len(waves) == 1:
            return waves[0], sr
        try:
            import torch
            from fireredtts3.core import cross_fade

            out = _as_torch(waves[0])
            fade = int(fade_ms / 1000.0 * sr)
            for wave in waves[1:]:
                out = cross_fade(out, _as_torch(wave), fade)
            return _as_wave(out, sr)
        except Exception:
            return np.concatenate([np.asarray(w, dtype="float32").reshape(-1) for w in waves]), sr

    def _pace(self, audio, sr, instruction="", speed=None):
        factor = clamp_speed(speed) if speed is not None else speed_for(instruction)
        if abs(factor - 1.0) < 0.05:
            return audio, sr
        # Official example: "adjust the speed to 0.5x"; max_gen_steps is 400.
        paced, out_sr = self.model.generate_acoustic_edit(
            instruction="adjust the speed to {:.1f}x".format(factor),
            audio_in=_as_torch(audio),
            audio_in_sr=int(sr),
            n_timesteps=int(tts_el.N_TIMESTEPS),
            inference_cfg=1.2,
            seed=int(tts_el.SEED),
        )
        log.info("acoustic_edit speed %.1fx after synth", factor)
        return _as_wave(paced, out_sr)

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        # Split here so cancel/progress land between sentences; one acoustic-edit on the join.
        ctx = kw.get("ctx")
        already = hasattr(self.model, "_apply_frontend")
        sents = self._sentences(text)
        total = len(sents)
        prompt = _as_torch(prompt_audio)
        waves, sr_out = [], self.sample_rate
        extra = {"do_clean": False, "do_tn": False, "do_split": False} if already else {}
        for i, sent in enumerate(sents):
            tts_el.job_tick(ctx, i, total)
            audio, sr, _ = as_tts_triplet(self.model.generate_tts(
                prompt_text=prompt_text or "",
                prompt_audio=prompt,
                prompt_audio_sr=int(prompt_sr),
                text=sent,
                n_timesteps=int(tts_el.N_TIMESTEPS),
                inference_cfg=float(tts_el.INFERENCE_CFG),
                seed=int(tts_el.SEED),
                **extra,
            ))
            wave, sr_out = _as_wave(audio, sr)
            waves.append(wave)
            tts_el.job_tick(ctx, i + 1, total)
        tts_el.job_tick(ctx, total, total)
        return self._pace(*self._join(waves, sr_out),
                          instruction=str(kw.get("instruction") or ""),
                          speed=kw.get("speed"))

    def design(self, instruction, text, **kw):
        # Re-plan once and reuse it so later sentences keep 口音 / 语速 / 音色; official design keeps the original blurb.
        ctx = kw.get("ctx")
        speak_as = instruction
        plan_out, waves, sr_out = None, [], self.sample_rate
        already = hasattr(self.model, "_apply_frontend")
        sents = self._sentences(text)
        total = len(sents)
        extra = {"do_clean": False, "do_tn": False, "do_split": False} if already else {}
        for i, sent in enumerate(sents):
            tts_el.job_tick(ctx, i, total, stage="design")
            audio, sr, seg_plan = self.model.generate_voice_design(
                instruction=speak_as,
                text=sent,
                n_timesteps=int(tts_el.N_TIMESTEPS),
                inference_cfg=float(tts_el.DESIGN_CFG),
                seed=int(tts_el.SEED),
                **extra,
            )
            if seg_plan and plan_out is None:
                plan_out = seg_plan
                speak_as = seg_plan
            wave, sr_out = _as_wave(audio, sr)
            waves.append(wave)
            tts_el.job_tick(ctx, i + 1, total, stage="design")
        tts_el.job_tick(ctx, total, total, stage="design")
        wave, sr = self._join(waves, sr_out)
        wave, sr = self._pace(wave, sr, instruction, speed=kw.get("speed"))
        return wave, sr, {"plan": plan_out}


def _load():
    from fireredtts3.core import FireRedTTS3Instruct

    patch_backend_tts_triplet()
    tts_el.rewrite_flash_attn(tts_el.ATTN or "eager")
    path = tts_el.model_path()
    log.info("loading FireRedTTS3-Instruct from %s", path)
    instruct = FireRedTTS3Instruct(
        path, use_wetext=tts_el.USE_WETEXT, use_llm_tn=tts_el.USE_LLM_TN,
    )
    return FireRedBackend(instruct)


def build_app(supports):
    return tts_el.build_app(supports, module="firered")


def run(supports):
    tts_el.run(supports, module="firered", watchdog="FireRedTTS3-Instruct",
               load=_load, sample_rate=24000)
