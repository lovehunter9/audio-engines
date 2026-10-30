# Ming-3.0-tts in one process. Knobs only move CFG and tempo; base CFG 2.0 keeps the model's own voice. CUDA and OpenVINO share this class.
import logging
import os
import tempfile
import threading

import numpy as np

from . import tts_el
from . import tts_long
from .ming_catalog import load_ip_catalog
from .ming_text import spoken_zh_numbers
from .tts_voice import Turn

log = logging.getLogger("ming")

# A clone clause that never stops used to fill the 1000-step minute with silence, so a long quiet piece is split.
_CLONE_CHARS = 40


def _steps_for(text):
    n = max(1, len((text or "").strip()))
    return min(1000, max(64, n * 14))


def _runaway(wave, sr, text):
    """True when this clip is far longer than its text and mostly quiet."""
    w = np.asarray(wave, dtype="float32").reshape(-1)
    if not len(w) or not sr:
        return True
    dur = len(w) / float(sr)
    expect = max(1.0, len((text or "").strip()) / 4.2)
    if dur < expect * 2.5:
        return False
    frame = max(1, int(0.05 * sr))
    n = len(w) // frame
    if n < 4:
        return False
    peaks = np.abs(w[:n * frame]).reshape(n, frame).max(axis=1)
    return float(np.mean(peaks < 0.015)) > 0.65


def _keep_voiced(wave, sr):
    """Drop a clip that is quiet all the way through. Keep the voiced span of a mixed one."""
    w = np.asarray(wave, dtype="float32").reshape(-1)
    if not len(w) or not sr:
        return w
    frame = max(1, int(0.05 * sr))
    n = len(w) // frame
    if n < 2:
        return w
    peaks = np.abs(w[:n * frame]).reshape(n, frame).max(axis=1)
    voiced = np.flatnonzero(peaks >= 0.015)
    if not len(voiced):
        return np.zeros(max(1, int(0.2 * sr)), dtype="float32")
    start = max(0, int(voiced[0]) * frame - int(0.05 * sr))
    end = min(len(w), int(voiced[-1] + 1) * frame + int(0.1 * sr))
    return w[start:end]


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
    sample_rate = 24000
    uses_shared_pack = True
    # A saved design is the baked sample. Do not redesign it on every reading.
    prefer_design_speak = False
    card_instruction_is_direction = False

    def __init__(self, model):
        _use_soundfile()
        self.model = model
        self.sample_rate = int(getattr(model, "sample_rate", 0) or 0) or self.sample_rate
        self._lock = threading.Lock()

    def native_presets(self):
        return load_ip_catalog()

    def presets(self):
        return tts_el.premade_cards(self, "ming")

    def _knobs(self, settings, text, instruction):
        # 2.0 is MingTTS.synthesize's own default. Style text is not appended.
        return tts_el.resolve_settings(
            settings, 2.0, instruction or "", text=text or "",
            include_speed_direction=False, include_style_direction=False)

    def _arm(self, turn, kn, sampling, instruction=None, xvec=False):
        turn.applied_cfg = kn.cfg
        turn.applied_sampling = dict(sampling or {})
        turn.applied_instruction = (instruction or "").strip() or None
        turn.applied_xvec = bool(xvec)

    def _call(self, turn, **kwargs):
        sampling = getattr(turn, "applied_sampling", None) or {}
        cfg = getattr(turn, "applied_cfg", None)
        if cfg is not None:
            kwargs["cfg"] = float(cfg)
        lang = str(sampling.get("language") or "").strip()
        if lang.lower() == "auto":
            lang = ""
        if not lang and turn.language_id:
            lang = turn.language_id
        if lang:
            kwargs["language_id"] = lang
        if sampling.get("temperature") is not None:
            kwargs["temperature"] = float(sampling["temperature"])
        if sampling.get("max_new_tokens") is not None:
            kwargs["max_decode_steps"] = int(sampling["max_new_tokens"])
        return self.model.synthesize(**kwargs)

    def _paced(self, audio, sr, speed):
        return tts_long.pace(audio, sr, speed)

    def _seed(self, kn):
        if kn.seed is None:
            return
        try:
            import torch
            torch.manual_seed(int(kn.seed))
        except Exception:
            return

    def _prompt_path(self, audio, sr):
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="ming-ref-")
        os.close(fd)
        sf.write(path, np.asarray(audio, dtype="float32").reshape(-1), int(sr))
        return path

    def speak_native(self, meta, text, instruction=None, ctx=None, settings=None, sampling=None):
        # Task IP builds its own caption and raises if instruction= is also set.
        kn = self._knobs(settings, text, "")
        vid = str((meta or {}).get("ip") or (meta or {}).get("voice_id") or "")
        labels = (meta or {}).get("labels") or {}
        lang = str(labels.get("language") or "")
        self._seed(kn)
        turn = Turn(task="ip", text=text, ip=vid, language_id=lang)
        self._arm(turn, kn, sampling)
        audio, sr = self.synthesize(turn)
        return self._paced(audio, sr, kn.speed)

    def _clone_once(self, path, text, transcript, xvec, user, kn, sampling):
        piece_sampling = dict(sampling)
        if piece_sampling.get("max_new_tokens") is None:
            piece_sampling["max_new_tokens"] = _steps_for(text)
        turn = Turn(task="clone", text=text, prompt_wav=path,
                    prompt_text="" if xvec else transcript)
        self._arm(turn, kn, piece_sampling, instruction=user, xvec=xvec)
        return self.synthesize(turn)

    def _clone_piece(self, path, text, transcript, xvec, user, kn, sampling):
        audio, sr = self._clone_once(path, text, transcript, xvec, user, kn, sampling)
        if not _runaway(audio, sr, text):
            return audio, sr
        smaller = tts_long.split_speak(text, max(12, len(text) // 2))
        log.warning("clone piece was long and quiet (%d chars, %.1fs); splitting",
                    len(text), len(audio) / float(sr or 1))
        if len(smaller) <= 1:
            return _keep_voiced(audio, sr), sr
        parts = []
        for bit in smaller:
            bit_audio, sr = self._clone_once(
                path, bit, transcript, xvec, user, kn, sampling)
            if _runaway(bit_audio, sr, bit):
                log.warning("clone piece still quiet after split (%d chars)", len(bit))
                bit_audio = _keep_voiced(bit_audio, sr)
            parts.append(bit_audio)
        return tts_long.join(parts, sr, gap_ms=80)

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        user = (kw.get("instruction") or "").strip()
        kn = self._knobs(kw.get("settings"), text, "")
        self._seed(kn)
        sampling = dict(kw.get("sampling") or {})
        transcript = (prompt_text or "").strip()
        xvec = bool(sampling.get("x_vector_only")) or not transcript
        path = self._prompt_path(prompt_audio, prompt_sr)
        try:
            pieces = tts_long.split_speak(text, _CLONE_CHARS) or [(text or "").strip()]
            waves = []
            sr = self.sample_rate
            for piece in pieces:
                audio, sr = self._clone_piece(
                    path, piece, transcript, xvec, user, kn, sampling)
                waves.append(audio)
            if len(waves) == 1:
                audio = waves[0]
            else:
                # Same reference on every piece. The previous piece is not the next prompt.
                audio, sr = tts_long.join(waves, sr, gap_ms=80)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return self._paced(audio, sr, kn.speed)

    def design(self, instruction, text, **kw):
        kn = self._knobs(kw.get("settings"), text, "")
        self._seed(kn)
        style = (instruction or "").strip()
        turn = Turn(task="design", text=text, style=style)
        self._arm(turn, kn, kw.get("sampling"))
        audio, sr = self.synthesize(turn)
        paced, sr = self._paced(audio, sr, kn.speed)
        return paced, sr, {}

    def synthesize(self, turn):
        if not isinstance(turn, Turn):
            raise TypeError("synthesize expects a Turn")
        if turn.task not in ("ip", "clone", "design"):
            raise ValueError("unknown task %r" % turn.task)
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
        # The checkpoint misreads raw digits. 22号楼 has to arrive as 二十二号楼.
        turn.text = spoken_zh_numbers(turn.text)
        if turn.task == "ip":
            return self._call(
                turn, task="IP", text=turn.text, controls={"ip": turn.ip},
                use_zero_spk_emb=True)
        if turn.task == "clone":
            if not turn.prompt_wav:
                raise ValueError("clone needs a reference wav")
            kwargs = {
                "task": "TTS", "text": turn.text, "prompt_wav_path": turn.prompt_wav,
                "use_spk_emb": True,
            }
            # No transcript: keep the speaker embedding and drop the acoustic prompt.
            if getattr(turn, "applied_xvec", False) or not turn.prompt_text:
                kwargs["prompt_text"] = None
            else:
                kwargs["prompt_text"] = turn.prompt_text
            instr = getattr(turn, "applied_instruction", None)
            if instr:
                kwargs["instruction"] = instr
            return self._call(turn, **kwargs)
        if turn.task == "design":
            if not turn.style:
                raise ValueError("design needs a style instruction")
            return self._call(
                turn, task="VOICE_DESIGN", text=turn.text,
                controls={"style": turn.style}, use_zero_spk_emb=True)
        raise ValueError("unknown task %r" % turn.task)
