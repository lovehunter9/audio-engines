"""OpenVINO-only waveform fixes for streaming S2ST.

1. Drop vocoded prompt-mel prefix after Vocos (upstream streaming path never trims).
2. On the first chunk with empty turn history, disable prompt_mel so English source
   mel does not condition VoiceBox when the output mel cannot extend past the prompt.
"""
import logging
import types

log = logging.getLogger("audio-simuls2st")


def patch_prompt_mel_trim(agent):
    """Wrap inference.codes_to_mel / mel_to_audio on this agent's inference instance."""
    inf = agent.inference
    if getattr(inf, "_ov_prompt_mel_trim", False):
        return
    orig_c2m = inf.codes_to_mel
    orig_m2a = inf.mel_to_audio
    ctx = {"prompt_mel": None, "full_mel": None}

    import torch

    @torch.inference_mode()
    def codes_to_mel(*args, **kwargs):
        mel = orig_c2m(*args, **kwargs)
        ctx["full_mel"] = mel
        pm = kwargs.get("prompt_mel")
        if pm is None and len(args) >= 3:
            pm = args[2]
        ctx["prompt_mel"] = pm
        return mel

    @torch.inference_mode()
    def mel_to_audio(mel):
        audio = orig_m2a(mel)
        pm = ctx.get("prompt_mel")
        fm = ctx.get("full_mel")
        if pm is not None and fm is not None and fm.data_ptr() == mel.data_ptr():
            full_len = int(mel.shape[1])
            prompt_len = int(pm.shape[1])
            if full_len > prompt_len > 0:
                audio = inf._trim_prompt_prefix_audio(
                    audio=audio, full_mel=mel, prompt_mel=pm,
                )
            else:
                log.warning(
                    "skip prompt-mel trim: full_mel=%d prompt_mel=%d (no safe prefix to drop)",
                    full_len, prompt_len,
                )
        return audio

    inf.codes_to_mel = codes_to_mel
    inf.mel_to_audio = mel_to_audio
    inf._ov_prompt_mel_trim = True
    log.info(
        "OpenVINO S2ST: trim prompt-mel prefix after Vocos "
        "(matches offline _trim_prompt_prefix_audio)"
    )

    orig_syn = agent._synthesize_codes

    @torch.inference_mode()
    def synthesize_codes(self, generated_codes, states, current_audio_16k):
        saved = bool(self.use_prompt_mel)
        if saved and not (getattr(states, "turn_history", None) or []):
            self.use_prompt_mel = False
            log.debug("first S2ST chunk without history: disable prompt_mel")
        try:
            return orig_syn(generated_codes, states, current_audio_16k)
        finally:
            self.use_prompt_mel = saved

    agent._synthesize_codes = types.MethodType(synthesize_codes, agent)
    log.info("OpenVINO S2ST: disable prompt_mel on first chunk (empty turn_history)")
