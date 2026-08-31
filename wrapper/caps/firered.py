# FireRedTTS3-Instruct in-process (clone + design + speak); Base is never loaded.
import logging

from . import tts_el

log = logging.getLogger("audio-firered")

PRESETS = [
    {"voice_id": "fr3-warm-zh-f", "name": "Warm ZH Female", "category": "premade",
     "instruction": "一个年轻女性的温柔嗓音，语速稍慢，带一点俏皮。",
     "sample_text": "今天天气很好，我们一起去公园散步吧。",
     "description": "Young female, warm, slightly playful, unhurried."},
    {"voice_id": "fr3-calm-zh-m", "name": "Calm ZH Male", "category": "premade",
     "instruction": "一位沉稳的中年男性，声音低沉清晰，语速中等。",
     "sample_text": "各位同事，我们开始今天的会议。",
     "description": "Adult male, low, clear, measured."},
    {"voice_id": "fr3-clear-en-f", "name": "Clear EN Female", "category": "premade",
     "instruction": "A clear young female voice, warm and measured, with a slight smile.",
     "sample_text": "Welcome aboard. Your journey begins now.",
     "description": "Young female, clear, warm English."},
    {"voice_id": "fr3-warm-en-m", "name": "Warm EN Male", "category": "premade",
     "instruction": "A warm adult male narrator, calm and confident.",
     "sample_text": "It is good to hear your voice again after all this time.",
     "description": "Adult male narrator, warm English."},
]


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

    def __init__(self, model):
        self.model = model
        sr = getattr(getattr(model, "redae", None), "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def presets(self):
        return list(PRESETS)

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **_kw):
        audio, sr, _ = as_tts_triplet(self.model.generate_tts(
            prompt_text=prompt_text or "",
            prompt_audio=_as_torch(prompt_audio),
            prompt_audio_sr=int(prompt_sr),
            text=text,
            n_timesteps=int(tts_el.N_TIMESTEPS),
            inference_cfg=float(tts_el.INFERENCE_CFG),
            seed=int(tts_el.SEED),
        ))
        return _as_wave(audio, sr)

    def design(self, instruction, text, **_kw):
        audio, sr, plan = self.model.generate_voice_design(
            instruction=instruction,
            text=text,
            n_timesteps=int(tts_el.N_TIMESTEPS),
            inference_cfg=float(tts_el.DESIGN_CFG),
            seed=int(tts_el.SEED),
        )
        wave, sr = _as_wave(audio, sr)
        return wave, sr, {"plan": plan}


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
