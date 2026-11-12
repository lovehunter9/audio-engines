"""OpenVINO-only waveform fix: drop prompt-mel prefix from synthesized audio.

The upstream streaming S2ST agent vocodes the full mel (prompt + generated) but never
calls `_trim_prompt_prefix_audio`, unlike offline unpaired TTS. English source as
prompt_mel then bleeds into the start of the translated clip (heard as smeared onsets).
"""
import logging

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
