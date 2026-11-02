# One Qwen3-TTS process. CUDA keeps Base loaded. OpenVINO runs each checkpoint in a child process so the GPU pool is freed on exit. Nine official speakers are private cards. zh-f/zh-m/en-f/en-m stay in the shared pack.
import gc
import logging
import os
import threading
import time
from pathlib import Path

from . import tts_el
from . import tts_long

log = logging.getLogger("audio-qwen3-voice")

BASE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CUSTOM_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DESIGN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# voice_id, CustomVoice language, the sentence it speaks (also the clone transcript), the official description, labels.
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


def _stream_item(item):
    """generate_*_streaming yields (chunk, sample_rate, timing). A bare pair is accepted too."""
    if isinstance(item, tuple) and len(item) >= 2:
        return _mono(item[0]), int(item[1])
    raise RuntimeError("qwen stream yielded %s, expected (audio, sample_rate, ...)" % type(item).__name__)


def _chunk_size(kw):
    """Discrete Qwen's --chunk-size default is 8 codec steps. A request may override it."""
    raw = kw.get("chunk_size")
    if raw is None:
        raw = (kw.get("sampling") or {}).get("chunk_size")
    try:
        n = int(raw) if raw not in (None, "") else 8
    except (TypeError, ValueError):
        n = 8
    return max(1, n)


def _snapshot(repo):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"),
                             token=tts_el.HF_TOKEN)


def _load_ov(repo):
    from . import tts_ov

    path = _snapshot(repo)
    log.info("loading %s from %s on OpenVINO", repo, path)
    model, _speakers, _custom, _design = tts_ov.load_checkpoint(path, repo)
    return model


def _load_faster(repo):
    if (os.environ.get("AUDIO_BASE") or "").strip() == "qwen3voiceov":
        return _load_ov(repo)
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


# The checkpoint's own sampling. A default knob set must land here and must not pass top_p.
_MODEL_TEMPERATURE = 0.9
# Tests replace these. Production OpenVINO speaks in a child process.
_OV_GENERATE_HOOK = None
_OV_STREAM_HOOK = None


def _official_speaker(prompt_text):
    """The nine baked sentences are the official speakers. Anything else is a real clone."""
    norm = " ".join(str(prompt_text or "").split())
    if not norm:
        return None
    for speaker, _language, text, _description, _labels in SPEAKERS:
        if " ".join(text.split()) == norm:
            return speaker
    return None


def _cfg_sampling(cfg, base):
    """Higher CFG than this task's base is tighter sampling. The base itself is a no-op."""
    base = float(base) or 1.0
    ratio = float(cfg) / base
    temp = max(0.15, min(1.5, round(_MODEL_TEMPERATURE / ratio, 3)))
    if ratio <= 1.001:
        return temp, None
    top_p = round(max(0.5, min(0.98, 1.0 - 0.2 * (ratio - 1.0))), 3)
    return temp, top_p


def _sampling(settings, text, instruction, seed_jitter=0, base_cfg=None, sampling=None):
    """Five knobs share Breeze/FireRed's CFG, then temperature and top_p. Style stays in the CFG, and a request number replaces the mapped one."""
    base = float(tts_el.INFERENCE_CFG if base_cfg is None else base_cfg)
    kn = tts_el.resolve_settings(
        settings, base, instruction or "", text=text or "",
        include_speed_direction=False, include_style_direction=False,
        seed_jitter=int(seed_jitter or 0))
    temp, top_p = _cfg_sampling(kn.cfg, base)
    gen = {"temperature": temp}
    if top_p is not None:
        gen["top_p"] = top_p
    for key in ("temperature", "top_k", "top_p", "max_new_tokens", "repetition_penalty"):
        if sampling and sampling.get(key) is not None:
            gen[key] = sampling[key]
    if kn.instruction:
        gen["instruct"] = kn.instruction
    language = str((sampling or {}).get("language") or "Auto").strip() or "Auto"
    return gen, kn.speed, kn.seed, language


def _is_ov():
    return (os.environ.get("AUDIO_BASE") or "").strip() == "qwen3voiceov"


def _ov_generate(repo, method, kwargs):
    """Load one checkpoint in a child and exit. Dropping a compiled model in-process does not return the Arc pool."""
    if _OV_GENERATE_HOOK is not None:
        return _OV_GENERATE_HOOK(repo, method, dict(kwargs))
    import json
    import subprocess
    import sys
    import tempfile

    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="qwen-ov-"))
    out = tmp / "audio.npy"
    job = {"repo": repo, "method": method, "kwargs": kwargs, "out": str(out)}
    path = tmp / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "wrapper.caps.qwen3_ov_job", str(path)],
        capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "openvino child failed").strip()
        raise RuntimeError(detail[-2000:])
    audio = np.load(out)
    sr_path = out.with_suffix(".sr")
    sr = int(sr_path.read_text(encoding="utf-8")) if sr_path.is_file() else 24000
    return audio, int(sr)


def _read_follow(fh, proc, n):
    """Read n bytes from a file the child is still writing."""
    buf = b""
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if chunk:
            buf += chunk
            continue
        if proc.poll() is not None:
            chunk = fh.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
            continue
        time.sleep(0.02)
    return buf


def _ov_stream(repo, method, kwargs):
    """Yield chunks while the child is still decoding. The child exits and returns the pool."""
    if _OV_STREAM_HOOK is not None:
        for audio, sr in _OV_STREAM_HOOK(repo, method, dict(kwargs)):
            wave = _mono(audio)
            if len(wave):
                yield wave, int(sr)
        return
    import json
    import struct
    import subprocess
    import sys
    import tempfile

    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="qwen-ov-stream-"))
    frames_path = tmp / "frames.bin"
    frames_path.write_bytes(b"")
    path = tmp / "job.json"
    path.write_text(json.dumps({
        "repo": repo, "method": method, "kwargs": kwargs, "stream": True,
        "frames": str(frames_path),
    }), encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "wrapper.caps.qwen3_ov_job", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err = []

    def _drain():
        err.append(proc.stderr.read() if proc.stderr is not None else b"")

    threading.Thread(target=_drain, daemon=True).start()
    fh = open(frames_path, "rb", buffering=0)
    try:
        while True:
            hdr = _read_follow(fh, proc, 8)
            if hdr is None:
                break
            sr, n = struct.unpack("<II", hdr)
            if n == 0:
                break
            raw = _read_follow(fh, proc, n * 4)
            if raw is None:
                break
            wave = np.frombuffer(raw, dtype="<f4").astype("float32", copy=True)
            if len(wave):
                yield wave, int(sr or 24000)
    finally:
        fh.close()
        rc = proc.wait(timeout=30)
    if rc != 0:
        detail = (err[0] if err else b"").decode("utf-8", "replace").strip()
        raise RuntimeError(detail[-2000:] or "openvino stream child failed")


def _ov_bake(dest):
    import json
    import subprocess
    import sys
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="qwen-ov-bake-"))
    path = tmp / "job.json"
    path.write_text(json.dumps({"method": "bake", "dest": str(dest)}), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "wrapper.caps.qwen3_ov_job", str(path)],
        capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "openvino bake failed").strip()
        raise RuntimeError(detail[-2000:])


def _apply_seed(seed):
    if seed is None:
        return
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class QwenVoiceBackend:
    sample_rate = 24000
    uses_shared_pack = True
    # A saved design is the approved sample, spoken by Base. Do not re-design each reading.
    prefer_design_speak = False
    # Shared-card blurbs are display copy. A request instruction is the only Voice Direction.
    card_instruction_is_direction = False
    # OpenAI stream and the socket emit codec chunks as they decode. Ming stays whole-then-cut.
    streams_while_generating = True
    # The socket is raw pcm. A container format is a POST, same as the discrete Qwen engine.
    socket_pcm_only = True

    def __init__(self, model, native):
        self.model = model
        self._native_dir = Path(native)
        self._design = None
        self._custom = None
        sr = getattr(model, "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)

    def native_voices_dir(self):
        return self._native_dir

    def native_presets(self):
        return tts_el.load_voice_cards(self._native_dir, category="premade")

    def presets(self):
        return tts_el.premade_cards(self, "qwen3_voice")

    def _drop_held(self, name):
        model = getattr(self, name)
        setattr(self, name, None)
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

    def _drop_design(self):
        self._drop_held("_design")

    def _drop_custom(self):
        self._drop_held("_custom")

    def _openvino(self):
        return (os.environ.get("AUDIO_BASE") or "").strip() == "qwen3voiceov"

    def _park_base(self):
        # A second fp32 checkpoint does not fit beside Base on the Arc quota.
        if not self._openvino() or self.model is None:
            return
        model = self.model
        self.model = None
        ov = getattr(model, "_ov", None)
        if ov is not None:
            model._ov = None
        del ov, model
        gc.collect()

    def _ensure_base(self):
        if self.model is not None:
            return
        self.model = _load_faster(BASE_REPO)

    def _design_model(self):
        self._drop_custom()
        self._park_base()
        if self._design is None:
            self._design = _load_faster(DESIGN_REPO)
        return self._design

    def _custom_model(self):
        self._drop_design()
        self._park_base()
        if self._custom is None:
            self._custom = _load_faster(CUSTOM_REPO)
        return self._custom

    def _prompt_path(self, audio, sr):
        import tempfile

        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="qwen-ref-")
        os.close(fd)
        sf.write(path, _mono(audio), int(sr))
        return path

    def _generate_custom(self, speaker, text, gen, seed, language):
        try:
            model = self._custom_model()
            _apply_seed(seed)
            return _pair(model.generate_custom_voice(
                text=text, speaker=speaker, language=language, **gen))
        finally:
            self._drop_custom()
            self._ensure_base()

    def _generate_clone(self, text, prompt_audio, prompt_sr, prompt_text, gen, seed,
                        language, xvec):
        self._drop_design()
        self._drop_custom()
        self._ensure_base()
        path = self._prompt_path(prompt_audio, prompt_sr)
        try:
            _apply_seed(seed)
            return _pair(self.model.generate_voice_clone(
                text=text, language=language, ref_audio=path,
                ref_text="" if xvec else (prompt_text or ""),
                xvec_only=bool(xvec), **gen))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        # One call for the whole text. Do not feed the previous sentence back in.
        instruction = str(kw.get("instruction") or "").strip()
        sampling = kw.get("sampling") or {}
        gen, speed, seed, language = _sampling(
            kw.get("settings"), text, instruction, sampling=sampling)
        if self._openvino():
            speaker = _official_speaker(prompt_text)
            if speaker:
                audio, sr = _ov_generate(CUSTOM_REPO, "generate_custom_voice", {
                    "text": text, "speaker": speaker, "language": language,
                    "seed": seed, **gen})
            else:
                xvec = bool(sampling.get("x_vector_only")) or not str(prompt_text or "").strip()
                path = self._prompt_path(prompt_audio, prompt_sr)
                try:
                    audio, sr = _ov_generate(BASE_REPO, "generate_voice_clone", {
                        "text": text, "language": language, "ref_audio": path,
                        "ref_text": "" if xvec else (prompt_text or ""),
                        "xvec_only": bool(xvec), "seed": seed, **gen})
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            return tts_long.pace(audio, sr, speed)
        speaker = _official_speaker(prompt_text)
        if speaker:
            audio, sr = self._generate_custom(speaker, text, gen, seed, language)
        else:
            xvec = bool(sampling.get("x_vector_only")) or not str(prompt_text or "").strip()
            audio, sr = self._generate_clone(
                text, prompt_audio, prompt_sr, prompt_text, gen, seed, language, xvec)
        return tts_long.pace(audio, sr, speed)

    def _pace_chunks(self, pieces, speed):
        tempo = None
        try:
            for chunk, sr in pieces:
                if tempo is None:
                    tempo = tts_long.TempoStream(int(sr), speed)
                for paced in tempo.write(chunk, int(sr)):
                    if len(paced):
                        yield paced, int(sr)
            if tempo is not None:
                for paced in tempo.finish():
                    if len(paced):
                        yield paced, int(tempo.sr)
        finally:
            if tempo is not None:
                tempo.abort()

    def _iter_loaded(self, model, method, call, seed, chunk_size):
        _apply_seed(seed)
        kwargs = dict(call)
        kwargs["chunk_size"] = int(chunk_size)
        fn = getattr(model, method + "_streaming", None)
        if not callable(fn):
            kwargs.pop("chunk_size", None)
            audio, sr = _pair(getattr(model, method)(**kwargs))
            if len(audio):
                yield audio, sr
            return
        for item in fn(**kwargs):
            audio, sr = _stream_item(item)
            if len(audio):
                yield audio, sr

    def _iter_clone_chunks(self, text, prompt_audio, prompt_sr, prompt_text, gen, seed,
                           language, sampling, chunk_size):
        # One generation for the whole text. Chunks are codec steps, not earlier sentences.
        if self._openvino():
            speaker = _official_speaker(prompt_text)
            if speaker:
                yield from _ov_stream(CUSTOM_REPO, "generate_custom_voice_streaming", {
                    "text": text, "speaker": speaker, "language": language,
                    "seed": seed, "chunk_size": chunk_size, **gen})
                return
            xvec = bool(sampling.get("x_vector_only")) or not str(prompt_text or "").strip()
            path = self._prompt_path(prompt_audio, prompt_sr)
            try:
                yield from _ov_stream(BASE_REPO, "generate_voice_clone_streaming", {
                    "text": text, "language": language, "ref_audio": path,
                    "ref_text": "" if xvec else (prompt_text or ""),
                    "xvec_only": bool(xvec), "seed": seed, "chunk_size": chunk_size, **gen})
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            return
        speaker = _official_speaker(prompt_text)
        if speaker:
            try:
                model = self._custom_model()
                yield from self._iter_loaded(model, "generate_custom_voice", {
                    "text": text, "speaker": speaker, "language": language, **gen,
                }, seed, chunk_size)
            finally:
                self._drop_custom()
                self._ensure_base()
            return
        xvec = bool(sampling.get("x_vector_only")) or not str(prompt_text or "").strip()
        self._drop_design()
        self._drop_custom()
        self._ensure_base()
        path = self._prompt_path(prompt_audio, prompt_sr)
        try:
            yield from self._iter_loaded(self.model, "generate_voice_clone", {
                "text": text, "language": language, "ref_audio": path,
                "ref_text": "" if xvec else (prompt_text or ""),
                "xvec_only": bool(xvec), **gen,
            }, seed, chunk_size)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def iter_clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        instruction = str(kw.get("instruction") or "").strip()
        sampling = kw.get("sampling") or {}
        gen, speed, seed, language = _sampling(
            kw.get("settings"), text, instruction, sampling=sampling)
        chunks = self._iter_clone_chunks(
            text, prompt_audio, prompt_sr, prompt_text, gen, seed, language, sampling,
            _chunk_size(kw))
        yield from self._pace_chunks(chunks, speed)
        tts_el.job_tick(kw.get("ctx"), 1, 1)

    def _generate_design(self, text, gen, seed, language):
        try:
            model = self._design_model()
            _apply_seed(seed)
            return _pair(model.generate_voice_design(
                text=text, language=language, **gen))
        finally:
            self._drop_design()
            self._ensure_base()

    def design(self, instruction, text, **kw):
        gen, speed, seed, language = _sampling(
            kw.get("settings"), text, instruction,
            seed_jitter=int(kw.get("seed_jitter") or 0),
            base_cfg=tts_el.DESIGN_CFG, sampling=kw.get("sampling"))
        if self._openvino():
            audio, sr = _ov_generate(DESIGN_REPO, "generate_voice_design", {
                "text": text, "language": language, "seed": seed, **gen})
            paced, sr = tts_long.pace(audio, sr, speed)
            return paced, sr, {}
        audio, sr = self._generate_design(text, gen, seed, language)
        paced, sr = tts_long.pace(audio, sr, speed)
        return paced, sr, {}

    def _iter_design_chunks(self, text, gen, seed, language, chunk_size):
        if self._openvino():
            yield from _ov_stream(DESIGN_REPO, "generate_voice_design_streaming", {
                "text": text, "language": language, "seed": seed,
                "chunk_size": chunk_size, **gen})
            return
        try:
            model = self._design_model()
            yield from self._iter_loaded(model, "generate_voice_design", {
                "text": text, "language": language, **gen,
            }, seed, chunk_size)
        finally:
            self._drop_design()
            self._ensure_base()

    def iter_design(self, instruction, text, **kw):
        gen, speed, seed, language = _sampling(
            kw.get("settings"), text, instruction,
            seed_jitter=int(kw.get("seed_jitter") or 0),
            base_cfg=tts_el.DESIGN_CFG, sampling=kw.get("sampling"))
        chunks = self._iter_design_chunks(text, gen, seed, language, _chunk_size(kw))
        yield from self._pace_chunks(chunks, speed)
        tts_el.job_tick(kw.get("ctx"), 1, 1, stage="design")


def _warmup(model):
    warm = getattr(model, "warmup", None)
    if not callable(warm):
        return
    try:
        warm(prefill_len=100)
    except Exception as e:
        log.warning("CUDA graph warmup failed: %s", e)


def _load():
    dest = native_dir()
    if _is_ov():
        # The parent must not compile a checkpoint. The Arc pool is not returned in-process.
        missing = [row for row in SPEAKERS if not _baked(dest / row[0], row[2])]
        if missing:
            _ov_bake(dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        return QwenVoiceBackend(None, dest)
    baked = bake_native(dest)
    model = _load_faster(BASE_REPO)
    _warmup(model)
    return QwenVoiceBackend(model, baked)


def build_app(supports):
    return tts_el.build_app(supports, module="qwen3_voice", openai_alias=True,
                            task_legacy=True)


def run(supports):
    tts_el.run(supports, module="qwen3_voice", watchdog="Qwen3-TTS",
               load=_load, sample_rate=24000, openai_alias=True, task_legacy=True)
