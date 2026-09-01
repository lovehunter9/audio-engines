# Breeze TTS 2 in-process. Official PyTorch: clone (ref+transcript), design, voice direction.
import logging
import os
from pathlib import Path

from . import tts_el

log = logging.getLogger("audio-breeze")

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


class BreezeBackend:
    sample_rate = 24000
    uses_shared_pack = True

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

    def _generate(self, text, instruction, ref_path=None, ref_text=None, cfg=None):
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
        wav = _collect(prompt_audio)
        with tempfile.NamedTemporaryFile(prefix="breeze-ref-", suffix=".wav", delete=False) as fh:
            path = fh.name
        try:
            sf.write(path, wav, int(prompt_sr), format="WAV", subtype="PCM_16")
            return self._generate(text, instruction,
                                  ref_path=path, ref_text=prompt_text, cfg=cfg)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def design(self, instruction, text, **_kw):
        audio, sr = self._generate(text, instruction, cfg=tts_el.DESIGN_CFG)
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
