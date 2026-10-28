# Ming-3.0-tts in one process. `load_model` is the device seam; a knob not in `_ACCEPTED_KNOBS` is a 400.
import os
import threading

import numpy as np

from .tts_voice import Turn


class UnsupportedControls(Exception):
    status_code = 400

    def __init__(self, names):
        self.detail = "these controls are not wired yet: %s" % ", ".join(names)
        super().__init__(self.detail)


def _use_soundfile():
    """Route torchaudio.load through soundfile. TorchAudio 2.9 calls TorchCodec, which this image does not install."""
    try:
        import torch
        import torchaudio
    except ImportError:
        return
    import soundfile as sf

    if getattr(torchaudio.load, "_ming_soundfile", False):
        return

    def load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
        data, rate = sf.read(uri, dtype="float32", always_2d=True)
        start = max(0, int(frame_offset or 0))
        if start:
            data = data[start:]
        count = int(num_frames)
        if count >= 0:
            data = data[:count]
        wave = torch.from_numpy(np.ascontiguousarray(data))
        if channels_first:
            wave = wave.transpose(0, 1).contiguous()
        return wave, int(rate)

    load._ming_soundfile = True
    torchaudio.load = load


def load_model(repo):
    """Load the 0.5B checkpoint. CUDA uses cuda:0. AUDIO_BASE=mingov compiles the OpenVINO GPU graphs."""
    if (os.environ.get("AUDIO_BASE") or "").strip() == "mingov":
        from .ming_ov import load_model as load_ov
        return load_ov(repo)
    from ming_tts import MingTTS

    return MingTTS(repo, device="cuda:0", text_normalization=False)


# Controls this build applies. Add a name here and read it in `_generate` when that task accepts it.
_ACCEPTED_KNOBS = {
    "ip": (),
    "clone": (),
    "design": (),
}


def _audio(wave):
    if hasattr(wave, "detach"):
        wave = wave.detach().float().cpu().numpy()
    audio = np.asarray(wave, dtype="float32")
    if audio.ndim > 1:
        audio = audio.reshape(-1) if audio.shape[0] == 1 else audio.mean(axis=0)
    return np.clip(audio.reshape(-1), -1.0, 1.0)


# Upstream stops at 512 positions, so a half-minute reference decodes nothing and torch.cat crashes. The position table is 32768.
_DECODE_CACHE_LEN = 8192


def widen_decode_cache():
    try:
        import modeling_bailingmm as mm
    except ImportError:
        return
    if mm.MAX_CACHE_LEN < _DECODE_CACHE_LEN:
        mm.MAX_CACHE_LEN = _DECODE_CACHE_LEN


class MingBackend:
    def __init__(self, model):
        _use_soundfile()
        self.model = model
        self.sample_rate = int(getattr(model, "sample_rate", 0) or 0)
        self._lock = threading.Lock()

    def synthesize(self, turn):
        if not isinstance(turn, Turn):
            raise TypeError("synthesize expects a Turn")
        accepted = _ACCEPTED_KNOBS.get(turn.task)
        if accepted is None:
            raise ValueError("unknown task %r" % turn.task)
        refused = [name for name in sorted(turn.knobs) if name not in accepted]
        if refused:
            raise UnsupportedControls(refused)
        widen_decode_cache()
        try:
            with self._lock:
                wave = self._generate(turn)
        except ValueError as exc:
            if "non-empty list of Tensors" in str(exc):
                raise RuntimeError(
                    "the reference audio is too long for one pass; use a shorter clip"
                ) from exc
            raise
        if self.sample_rate <= 0:
            self.sample_rate = int(getattr(self.model, "sample_rate", 0) or 0)
        return _audio(wave), self.sample_rate

    def _generate(self, turn):
        lang = {}
        if turn.language_id:
            lang["language_id"] = turn.language_id
        if turn.task == "ip":
            return self.model.synthesize(
                task="IP", text=turn.text, controls={"ip": turn.ip},
                use_zero_spk_emb=True, **lang)
        if turn.task == "clone":
            if not turn.prompt_wav or not turn.prompt_text:
                raise ValueError("clone needs a reference wav and its transcript")
            return self.model.synthesize(
                task="TTS", text=turn.text, prompt_wav_path=turn.prompt_wav,
                prompt_text=turn.prompt_text, use_spk_emb=True, **lang)
        if turn.task == "design":
            if not turn.style:
                raise ValueError("design needs a style instruction")
            return self.model.synthesize(
                task="VOICE_DESIGN", text=turn.text, controls={"style": turn.style},
                use_zero_spk_emb=True, **lang)
        raise ValueError("unknown task %r" % turn.task)
