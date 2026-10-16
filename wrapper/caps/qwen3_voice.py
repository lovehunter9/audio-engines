# One Qwen3-TTS instance: Base stays loaded, VoiceDesign loads for a design call, CustomVoice
# only bakes the nine official speakers once. Those nine live in this backend's own directory.
# zh-f / zh-m / en-f / en-m stay in the shared pack every TTS reads.
import gc
import logging
import os
from pathlib import Path

from . import tts_el
from . import tts_long

log = logging.getLogger("audio-qwen3-voice")

BASE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DESIGN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# voice_id, model language, the sentence CustomVoice speaks (and the clone transcript),
# the official description, labels. Ids are the checkpoint's speaker names.
SPEAKERS = (
    ("Vivian", "Chinese", "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
     "Bright, slightly edgy young female voice.",
     {"language": "zh", "gender": "female", "origin": "qwen"}),
    ("Serena", "Chinese", "今天的风很轻，我们慢慢走就好。",
     "Warm, gentle young female voice.",
     {"language": "zh", "gender": "female", "origin": "qwen"}),
    ("Uncle_Fu", "Chinese", "这件事啊，急不得，想清楚了再做。",
     "Seasoned male voice with a low, mellow timbre.",
     {"language": "zh", "gender": "male", "origin": "qwen"}),
    ("Dylan", "Chinese", "得嘞，您说的这个我听明白了。",
     "Youthful Beijing male voice with a clear, natural timbre.",
     {"language": "zh", "accent": "beijing", "gender": "male", "origin": "qwen"}),
    ("Eric", "Chinese", "要得，这个事情我来给你办。",
     "Lively Chengdu male voice with a slightly husky brightness.",
     {"language": "zh", "accent": "sichuan", "gender": "male", "origin": "qwen"}),
    ("Ryan", "English", "She said she would be here by noon.",
     "Dynamic male voice with strong rhythmic drive.",
     {"language": "en", "gender": "male", "origin": "qwen"}),
    ("Aiden", "English", "It's a bright day, and the road ahead is clear.",
     "Sunny American male voice with a clear midrange.",
     {"language": "en", "gender": "male", "origin": "qwen"}),
    ("Ono_Anna", "Japanese", "ただいま。待っててくれて、ありがとう。",
     "Playful Japanese female voice with a light, nimble timbre.",
     {"language": "ja", "gender": "female", "origin": "qwen"}),
    ("Sohee", "Korean", "오늘 날씨가 참 좋네요. 같이 걸을까요.",
     "Warm Korean female voice with rich emotion.",
     {"language": "ko", "gender": "female", "origin": "qwen"}),
)


def native_dir():
    """Private premade pack. Not voices/, so FireRed and Breeze never seed these ids."""
    return Path(tts_el.VOICE_DIR) / ".qwen-native"


def _mono(audio):
    import numpy as np

    if isinstance(audio, (list, tuple)):
        audio = audio[0] if len(audio) == 1 else np.concatenate(
            [np.asarray(a, dtype="float32").reshape(-1) for a in audio])
    return np.clip(np.asarray(audio, dtype="float32").reshape(-1), -1.0, 1.0)


def _pair(out):
    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError("qwen tts returned %s, expected (audio, sample_rate)" % type(out).__name__)
    audio, sr = out
    return _mono(audio), int(sr)


def _snapshot(repo):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"),
                             token=tts_el.HF_TOKEN)


def _load_faster(repo):
    from faster_qwen3_tts import FasterQwen3TTS

    tts_el.rewrite_flash_attn(tts_el.ATTN or "sdpa")
    path = _snapshot(repo)
    log.info("loading %s from %s", repo, path)
    return FasterQwen3TTS.from_pretrained(
        path, device="cuda", attn_implementation=tts_el.ATTN or "sdpa",
        max_seq_len=int(tts_el.MAX_SEQ_LEN), local_files_only=True)


def _write_speaker(slot, speaker, text, description, labels, audio, sr):
    import array
    import json
    import wave

    slot.mkdir(parents=True, exist_ok=True)
    frames = array.array("h", (
        max(-32767, min(32767, int(float(x) * 32767.0))) for x in audio))
    with wave.open(str(slot / "prompt.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(frames.tobytes())
    (slot / "prompt.txt").write_text(text)
    meta = {
        "voice_id": speaker,
        "name": speaker,
        "category": "premade",
        "source": "clone",
        "description": description,
        "transcript": text,
        "labels": labels,
    }
    (slot / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _baked(slot, text):
    wav, txt = slot / "prompt.wav", slot / "prompt.txt"
    if not wav.is_file() or not txt.is_file():
        return False
    return " ".join(txt.read_text().split()) == " ".join(text.split())


def bake_native(dest, generate=None):
    """Write the nine official speakers. `generate` stands in for CustomVoice in tests."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    missing = [row for row in SPEAKERS if not _baked(dest / row[0], row[2])]
    if not missing:
        return dest
    own = generate is None
    model = None
    if own:
        model = _load_faster(CUSTOM_REPO)

        def generate(text, speaker, language):
            return model.generate_custom_voice(text=text, speaker=speaker, language=language)

    try:
        for speaker, language, text, description, labels in missing:
            audio, sr = _pair(generate(text, speaker, language))
            _write_speaker(dest / speaker, speaker, text, description, labels, audio, sr)
            log.info("baked official speaker %s (%d Hz, %.2fs)", speaker, sr, len(audio) / float(sr or 1))
    finally:
        if own and model is not None:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    return dest


class QwenVoiceBackend:
    sample_rate = 24000
    uses_shared_pack = True
    # A saved design is the approved sample, spoken by Base. Do not re-design each reading.
    prefer_design_speak = False
    # Shared-card blurbs are display copy. A request instruction is the only Voice Direction.
    card_instruction_is_direction = False

    def __init__(self, model, native):
        self.model = model
        self._native_dir = Path(native)
        self._design = None
        sr = getattr(model, "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def native_voices_dir(self):
        return self._native_dir

    def native_presets(self):
        return tts_el.load_voice_cards(self._native_dir, category="premade")

    def presets(self):
        return tts_el.premade_cards(self, "qwen3_voice")

    def _drop_design(self):
        model = self._design
        self._design = None
        if model is None:
            return
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _design_model(self):
        if self._design is None:
            self._design = _load_faster(DESIGN_REPO)
        return self._design

    def _speed(self, settings, text):
        kn = tts_el.resolve_settings(settings, 1.0, "", text=text, include_speed_direction=False)
        return kn.speed

    def _prompt_path(self, audio, sr):
        import tempfile

        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="qwen-ref-")
        os.close(fd)
        sf.write(path, _mono(audio), int(sr))
        return path

    def _generate_clone(self, text, prompt_audio, prompt_sr, prompt_text, instruction):
        self._drop_design()
        path = self._prompt_path(prompt_audio, prompt_sr)
        try:
            kw = {"instruct": instruction} if instruction else {}
            return _pair(self.model.generate_voice_clone(
                text=text, language="Auto", ref_audio=path,
                ref_text=prompt_text or "", xvec_only=False, **kw))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        audio, sr = self._generate_clone(
            text, prompt_audio, prompt_sr, prompt_text, str(kw.get("instruction") or "").strip())
        return tts_long.pace(audio, sr, self._speed(kw.get("settings"), text))

    def iter_clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        audio, sr = self.clone(text, prompt_audio, prompt_sr, prompt_text, **kw)
        tts_el.job_tick(kw.get("ctx"), 1, 1)
        yield audio, sr

    def _generate_design(self, instruction, text):
        model = self._design_model()
        try:
            return _pair(model.generate_voice_design(
                text=text, language="Auto", instruct=instruction))
        finally:
            self._drop_design()

    def design(self, instruction, text, **kw):
        audio, sr = self._generate_design(instruction, text)
        paced, sr = tts_long.pace(audio, sr, self._speed(kw.get("settings"), text))
        return paced, sr, {}

    def iter_design(self, instruction, text, **kw):
        audio, sr, _extra = self.design(instruction, text, **kw)
        tts_el.job_tick(kw.get("ctx"), 1, 1, stage="design")
        yield audio, sr

    def stream_gap(self, text, sr):
        return None


def _warmup(model):
    warm = getattr(model, "warmup", None)
    if not callable(warm):
        return
    try:
        warm(prefill_len=100)
    except Exception as e:
        log.warning("CUDA graph warmup failed: %s", e)


def _load():
    baked = bake_native(native_dir())
    model = _load_faster(BASE_REPO)
    _warmup(model)
    return QwenVoiceBackend(model, baked)


def build_app(supports):
    return tts_el.build_app(supports, module="qwen3_voice", openai_alias=True)


def run(supports):
    tts_el.run(supports, module="qwen3_voice", watchdog="Qwen3-TTS",
               load=_load, sample_rate=24000)
