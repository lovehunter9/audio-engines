# ElevenLabs-shaped TTS: list / design / clone / speak on voice_id; premade slots freeze a design sample on first speak.
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .. import hfgate
from .. import tasks
from .. import watchdog
from ..audioio import decode, probe_seconds, seconds, wav_seconds
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-tts-el")

_runtime = Runtime(model=None, backend=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = ((_runtime.model_repo or "").split() or [""])[0]
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
VOICE_DIR = Path(str(_args.text("--voice-dir", "/data/voices") or "/data/voices"))
N_TIMESTEPS = int(_args.number("--n-timesteps", 10))
INFERENCE_CFG = _args.number("--inference-cfg", 2.0)
DESIGN_CFG = _args.number("--design-cfg", 1.2)
# FireRed generate_tts/design is machine-gun; acoustic_edit 0.5–2.0 step 0.1, chart default 0.7. Breeze ignores this.
SPEAK_SPEED = _args.number("--speak-speed", 0.7)
CFG_SCALE = _args.number("--cfg-scale", 1.0)
# No seed of ours: None leaves each model the seed its authors chose; --seed is for a specific run.
_seed_arg = str(_args.text("--seed", "") or "").strip()
SEED = int(float(_seed_arg)) if _seed_arg else None
ATTN = str(_args.text("--attn-implementation", "") or "")
DEFAULT_PREVIEW = str(_args.text("--preview-text", "") or "")
USE_WETEXT = _args.switch("--use-wetext", True)
USE_LLM_TN = _args.switch("--use-llm-tn", False)
FAST_ALL = _args.switch("--fast-all", False)
# 0 = off (FireRed). Breeze Chart turns these on to match BreezeBlue's product gates.
REF_MIN_SECONDS = _args.number("--ref-min-seconds", 0)
REF_MAX_SECONDS = _args.number("--ref-max-seconds", 0)
REF_MAX_MB = _args.number("--ref-max-mb", 0)
DESIGN_MIN_CHARS = int(_args.number("--design-min-chars", 0))
DESIGN_MAX_CHARS = int(_args.number("--design-max-chars", 0))
# Official infer.py. Breeze reads these; FireRed ignores them.
MAX_NEW_TOKENS = int(_args.number("--max-new-tokens", 1500) or 1500)
MAX_SEQ_LEN = int(_args.number("--max-seq-len", 2048) or 2048)
if MAX_NEW_TOKENS <= 0:
    MAX_NEW_TOKENS = 1500
if MAX_SEQ_LEN <= 0:
    MAX_SEQ_LEN = 2048

# Official code pins flash_attention_2; this image has no flash_attn, so rewrite those requests.
_FLASH_IMPLS = ("flash_attention_2", "flash_attention_3")
_FLASH_REWRITTEN = False


def fallback_attn(requested, impl):
    requested = (requested or "eager").strip() or "eager"
    if impl in _FLASH_IMPLS or (
        isinstance(impl, str) and impl.startswith("flash_attention")
    ):
        return requested
    return impl if impl is not None else requested


def rewrite_flash_attn(requested=None):
    global _FLASH_REWRITTEN
    if _FLASH_REWRITTEN:
        return
    requested = fallback_attn(requested or ATTN, None)
    from transformers.modeling_utils import PreTrainedModel

    orig = PreTrainedModel._check_and_adjust_attn_implementation

    def _checked(self, attn_implementation, is_init_check=False, **kwargs):
        impl = fallback_attn(requested, attn_implementation)
        if impl != attn_implementation:
            log.info(
                "asked for %s; this image has no flash_attn, using %s",
                attn_implementation,
                impl,
            )
            cfg = getattr(self, "config", None)
            if cfg is not None:
                try:
                    cfg._attn_implementation = impl
                except Exception:
                    pass
        return orig(self, impl, is_init_check, **kwargs)

    PreTrainedModel._check_and_adjust_attn_implementation = _checked
    _FLASH_REWRITTEN = True

BOOT_TIMEOUT_S = 1800.0
WARMUP_TIMEOUT_S = 300.0

_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),
    "ulaw": (None, None, "audio/basic"),
    "alaw": (None, None, "audio/PCMA"),
}
# Public FRP / LLM gateway resets around 10–25 MB (530). Speech mp3 stays under this.
_EDGE_SAFE_BYTES = 6 * 1024 * 1024

# EL voice_settings; the defaults are a no-op on ENGINE_ARGS.
_DEFAULT_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 1.0,
}

_state = _runtime.state
_gen_lock = threading.Lock()
# module name this process registered as (firered / breeze); set by run()/build_app.
_MODULE = {"name": ""}


def repo_id():
    """MODEL_SOURCE may carry llm-init --include flags; snapshot_download wants the hub id only."""
    raw = (os.environ.get("MODEL_SOURCE") or "").strip()
    if raw.startswith("hf://"):
        raw = raw[5:]
    return (raw.split() or [MODEL_REPO])[0]


def model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id(), local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _to_mono(audio):
    import numpy as np

    if isinstance(audio, (list, tuple)):
        audio = audio[0] if len(audio) == 1 else np.concatenate(
            [np.asarray(a).reshape(-1) for a in audio])
    a = np.asarray(audio, dtype="float32").reshape(-1)
    return np.clip(a, -1.0, 1.0)


def _pcm16(audio):
    return (_to_mono(audio) * 32767.0).astype("<i2").tobytes()


def _encode(audio, sr, fmt):
    if fmt == "pcm":
        return _pcm16(audio)
    import soundfile as sf

    container, subtype, _mime = _FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, _to_mono(audio), sr, format=container, subtype=subtype)
    return buf.getvalue()


def _wav_bytes(audio, sr):
    return _encode(audio, sr, "wav")


def _resample(audio, sr, want_sr):
    import numpy as np

    w = _to_mono(audio)
    src, dst = int(sr), int(want_sr or sr)
    if dst <= 0 or dst == src or not len(w):
        return w, src
    n = max(1, int(round(len(w) * float(dst) / float(src))))
    x = np.linspace(0.0, float(len(w) - 1), n)
    return np.interp(x, np.arange(len(w), dtype="float64"), w).astype("float32"), dst


def _ffmpeg_mp3(audio, sr, want_sr, kbps):
    pcm = _to_mono(audio).astype("<f4").tobytes()
    cmd = ["ffmpeg", "-y", "-nostdin",
           "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "pipe:0"]
    if want_sr and int(want_sr) != int(sr):
        cmd.extend(["-ar", str(int(want_sr))])
    cmd.extend(["-b:a", "%dk" % int(kbps), "-f", "mp3", "pipe:1"])
    try:
        r = subprocess.run(cmd, input=pcm, capture_output=True, check=False)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        log.warning("ffmpeg mp3 rc=%s: %s", r.returncode, (r.stderr or b"")[-200:])
    except OSError as e:
        log.warning("ffmpeg mp3 missing: %s", e)
    return None


def _encode_mp3_fit(audio, sr):
    """Speech mp3 sized to fit the public hop. 64 kbps is the ceiling; longer clips drop further."""
    dur = max(float(seconds(audio, sr) or 0.0), 0.25)
    kbps = min(64, max(24, int(_EDGE_SAFE_BYTES * 8 / dur / 1000)))
    body = _ffmpeg_mp3(audio, sr, sr, kbps)
    if body:
        return body
    return _encode(audio, sr, "mp3")


class OutSpec:
    __slots__ = ("kind", "sr", "bitrate", "token")

    def __init__(self, kind, sr=None, bitrate=None, token=""):
        self.kind = kind
        self.sr = sr
        self.bitrate = bitrate
        self.token = token or kind


# ElevenLabs' public tokens plus raw 24 kHz PCM for first-party live preview.
_FORMAT_TOKENS = {
    "mp3": ("mp3", 44100, 128),
    "mp3_44100_128": ("mp3", 44100, 128),
    "mp3_44100_192": ("mp3", 44100, 192),
    "wav_8000": ("wav", 8000, None),
    "wav_16000": ("wav", 16000, None),
    "wav_22050": ("wav", 22050, None),
    "wav_24000": ("wav", 24000, None),
    "wav_44100": ("wav", 44100, None),
    "wav_48000": ("wav", 48000, None),
    "pcm_24000": ("pcm", 24000, None),
}


def _output_format(raw, default="mp3_44100_128"):
    """ElevenLabs `mp3_44100_128` / `wav_44100` / raw `pcm_24000` → OutSpec."""
    if isinstance(raw, OutSpec):
        return raw
    token = str(raw or default).strip().lower()
    if not token:
        token = default
    if token in _FORMAT_TOKENS:
        kind, sr, br = _FORMAT_TOKENS[token]
        return OutSpec(kind, sr, br, token)
    raise HTTPException(
        status_code=400,
        detail="output_format must be one of %s" % ", ".join(
            k for k in _FORMAT_TOKENS if k != "mp3"))


def _as_spec(fmt):
    return fmt if isinstance(fmt, OutSpec) else _output_format(fmt)


def _ffmpeg_codec(audio, sr, spec):
    """One-shot ffmpeg for EL tokens soundfile cannot write (μ-law / A-law / bitrate opus)."""
    spec = _as_spec(spec)
    pcm = _to_mono(audio).astype("<f4").tobytes()
    cmd = ["ffmpeg", "-y", "-nostdin",
           "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "pipe:0"]
    if spec.sr and int(spec.sr) != int(sr):
        cmd.extend(["-ar", str(int(spec.sr))])
    if spec.kind == "ulaw":
        cmd.extend(["-f", "mulaw", "-acodec", "pcm_mulaw", "pipe:1"])
    elif spec.kind == "alaw":
        cmd.extend(["-f", "alaw", "-acodec", "pcm_alaw", "pipe:1"])
    elif spec.kind == "opus":
        cmd.extend(["-c:a", "libopus"])
        if spec.bitrate:
            cmd.extend(["-b:a", "%dk" % int(spec.bitrate)])
        cmd.extend(["-f", "ogg", "pipe:1"])
    else:
        return None
    try:
        r = subprocess.run(cmd, input=pcm, capture_output=True, check=False)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        log.warning("ffmpeg %s rc=%s: %s", spec.kind, r.returncode, (r.stderr or b"")[-200:])
    except OSError as e:
        log.warning("ffmpeg %s missing: %s", spec.kind, e)
    return None


def _encode_out(audio, sr, spec):
    spec = _as_spec(spec)
    want = spec.sr or sr
    if spec.kind == "mp3" and spec.bitrate:
        body = _ffmpeg_mp3(audio, sr, want, spec.bitrate)
        if body:
            return body
        audio, sr = _resample(audio, sr, want)
        return _encode(audio, sr, "mp3")
    if spec.kind in ("ulaw", "alaw") or (spec.kind == "opus" and spec.bitrate):
        body = _ffmpeg_codec(audio, sr, spec)
        if body:
            return body
    audio, sr = _resample(audio, sr, want)
    if spec.kind in ("ulaw", "alaw"):
        raise HTTPException(status_code=500, detail="ffmpeg cannot encode %s" % spec.kind)
    return _encode(audio, sr, spec.kind)


class _LiveMux:
    """One encoder for the whole /stream: per-slice MP3 headers make short reads tremble."""

    def __init__(self, spec, in_sr):
        self.spec = _as_spec(spec)
        self.in_sr = int(in_sr)
        self.proc = None
        self._q = queue.Queue()
        self._pcm = self.spec.kind == "pcm"
        # Logical audio accepted by the output encoder. Counting samples here
        # makes duration independent of the wire container and includes the
        # silence inserted between generated slices.
        self.output_duration_seconds = 0.0

    def start(self):
        if self._pcm:
            return
        cmd = ["ffmpeg", "-nostdin",
               "-f", "f32le", "-ar", str(self.in_sr), "-ac", "1", "-i", "pipe:0"]
        if self.spec.sr and int(self.spec.sr) != self.in_sr:
            cmd.extend(["-ar", str(int(self.spec.sr))])
        if self.spec.kind == "mp3":
            cmd.extend(["-b:a", "%dk" % int(self.spec.bitrate or 128), "-f", "mp3", "pipe:1"])
        elif self.spec.kind == "opus":
            cmd.extend(["-c:a", "libopus"])
            if self.spec.bitrate:
                cmd.extend(["-b:a", "%dk" % int(self.spec.bitrate)])
            cmd.extend(["-f", "ogg", "pipe:1"])
        elif self.spec.kind == "flac":
            cmd.extend(["-f", "flac", "pipe:1"])
        elif self.spec.kind == "ulaw":
            cmd.extend(["-ar", str(int(self.spec.sr or 8000)), "-f", "mulaw",
                        "-acodec", "pcm_mulaw", "pipe:1"])
        elif self.spec.kind == "alaw":
            cmd.extend(["-ar", str(int(self.spec.sr or 8000)), "-f", "alaw",
                        "-acodec", "pcm_alaw", "pipe:1"])
        else:
            self._pcm = False
            self._oneshot = True
            return
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as e:
            log.warning("live mux ffmpeg missing: %s", e)
            self.proc = None
            self._oneshot = True
            return
        threading.Thread(target=self._pump, daemon=True, name="el-mux").start()

    def _pump(self):
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                self._q.put(None)
                return
            self._q.put(chunk)

    def write(self, audio, sr):
        if getattr(self, "_oneshot", False):
            self._q.put(_encode_out(audio, sr, self.spec))
            duration = len(_to_mono(audio)) / float(sr)
            self.output_duration_seconds += duration
            return duration
        if self._pcm:
            a, out_sr = _resample(audio, sr, self.spec.sr or sr)
            self._q.put(_pcm16(a))
            duration = len(a) / float(out_sr)
            self.output_duration_seconds += duration
            return duration
        a, out_sr = _resample(audio, sr, self.in_sr)
        try:
            self.proc.stdin.write(a.astype("<f4").tobytes())
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return 0.0
        duration = len(a) / float(out_sr)
        self.output_duration_seconds += duration
        return duration

    def drain(self, wait=0.08):
        out = []
        deadline = time.time() + wait
        while True:
            left = deadline - time.time()
            try:
                item = self._q.get(timeout=left if left > 0 else 0)
            except queue.Empty:
                break
            if item is None:
                break
            out.append(item)
        return out

    def close(self):
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        out = []
        while True:
            try:
                item = self._q.get(timeout=2)
            except queue.Empty:
                break
            if item is None:
                break
            out.append(item)
        return out


def _encode_edge(audio, sr, fmt):
    """Encode `fmt`, but never hand the public hop a body over _EDGE_SAFE_BYTES."""
    spec = _as_spec(fmt)
    body = _encode_out(audio, sr, spec)
    if len(body) <= _EDGE_SAFE_BYTES:
        return body, spec.kind
    compact = _encode_mp3_fit(audio, sr)
    if compact and len(compact) < len(body):
        log.info("edge-safe %s %dB -> mp3 %dB", spec.token, len(body), len(compact))
        return compact, "mp3"
    return body, spec.kind


def _explain(e):
    name = type(e).__name__
    text = str(e).strip()
    if name == "NoBackendError" or "Format not recognised" in text:
        return ("could not decode the reference audio: no decoder for that container. "
                "Convert it to wav/mp3/flac and retry")
    return text or name


def _new_id(prefix):
    return "%s%s" % (prefix, uuid.uuid4().hex[:20])


_ID_RE = re.compile(r"\A(hi)[0-9a-f]{20}\Z")


def _named_id(raw, prefix):
    """A caller-chosen id, or None. The id becomes a directory name, so nothing
    that is not exactly what _new_id would have minted is allowed near the store."""
    name = str(raw or "").strip()
    if not name:
        return None
    if not _ID_RE.match(name) or not name.startswith(prefix):
        raise HTTPException(
            status_code=400,
            detail="history_item_id must look like %s + 20 hex digits" % prefix)
    return name


def voices_root():
    """Repo `voices/` in checkout; `/app/voices` once append-image copies it."""
    return Path(__file__).resolve().parents[2] / "voices"


def builtin_voices_dir(module=None):
    """One shared card tree. `module` is reserved; ids are not engine-prefixed."""
    root = voices_root()
    return root if root.is_dir() else None


def load_voice_cards(root, category=None):
    """Read `<dir>/<voice_id>/meta.json` cards. Same shape at repo and at --voice-dir."""
    root = Path(root) if root else None
    if root is None or not root.is_dir():
        return []
    cards = []
    for child in sorted(root.iterdir()):
        meta_path = child / "meta.json"
        if not child.is_dir() or child.name == "pending" or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        meta.setdefault("voice_id", child.name)
        if category and meta.get("category") != category:
            continue
        cards.append(meta)
    return cards


def builtin_presets(module):
    src = builtin_voices_dir(module)
    return load_voice_cards(src, category="premade") if src else []


def premade_cards(backend, module=None):
    """Model-native voices first, then our shared pack if the backend wants it."""
    rows, seen = [], set()
    native = getattr(backend, "native_presets", None)
    for card in list(native() if callable(native) else []) or []:
        if not isinstance(card, dict):
            continue
        vid = str(card.get("voice_id") or "").strip()
        if not vid or vid in seen:
            continue
        seen.add(vid)
        rows.append(dict(card))
    if backend is None or bool(getattr(backend, "uses_shared_pack", True)):
        for card in builtin_presets(module):
            vid = str(card.get("voice_id") or "").strip()
            if not vid or vid in seen:
                continue
            seen.add(vid)
            rows.append(dict(card))
    return rows


# Catalog fields only; identity is instruction (design) or prompt.wav+txt (clone).
_CARD_KEYS = ("name", "description", "instruction", "sample_text", "labels", "source",
              "transcript")


def _premade_dir(path):
    """True only for a premade card. Cloned / designed / unknown dirs stay."""
    path = Path(path)
    if not path.is_dir() or path.name == "pending":
        return False
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(meta, dict) and meta.get("category") == "premade"


def _read_json(path):
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


def _instruction_differs(want, have):
    """Present on one side only, or both present and not equal."""
    a, b = "instruction" in (want or {}), "instruction" in (have or {})
    if a != b:
        return True
    return a and want["instruction"] != have["instruction"]


def _norm_text(s):
    return " ".join((s or "").replace("\r\n", "\n").split())


def _read_text(path):
    path = Path(path)
    try:
        return path.read_text() if path.is_file() else None
    except OSError:
        return None


def _wav_fp(path):
    path = Path(path) if path is not None else None
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _card_source(card, pack=None, vid=None):
    """clone = factory ref wav+transcript; design = instruction freeze. Infer wav in pack."""
    raw = str((card or {}).get("source") or "").strip().lower()
    if raw in ("clone", "design"):
        return raw
    if pack is not None and vid and (Path(pack) / vid / "prompt.wav").is_file():
        return "clone"
    return "design"


def _replant_needed(target, card, pack):
    """Wipe only when the identity of that source kind changed."""
    source = _card_source(card, pack, target.name)
    have = _read_json(target / "meta.json") or {}
    if source == "clone":
        pack_root = Path(pack) / target.name if pack is not None else None
        want_fp = _wav_fp(pack_root / "prompt.wav") if pack_root is not None else None
        have_fp = _wav_fp(target / "prompt.wav")
        want_t = _norm_text(
            (_read_text(pack_root / "prompt.txt") if pack_root is not None else None)
            or card.get("transcript") or "")
        have_t = _norm_text(_read_text(target / "prompt.txt") or have.get("transcript") or "")
        if want_fp is None:
            return have.get("source") != "clone" or have_fp is not None
        return want_fp != have_fp or want_t != have_t
    return _instruction_differs(card, have)


def _write_card(target, card, pack=None):
    target.mkdir(parents=True, exist_ok=True)
    meta = dict(card)
    meta.setdefault("category", "premade")
    meta.setdefault("source", _card_source(card, pack, target.name))
    meta["voice_id"] = target.name
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _plant(target, card, pack):
    src = pack / target.name if pack is not None else None
    if src is not None and src.is_dir() and (src / "meta.json").is_file():
        shutil.copytree(src, target)
        return
    _write_card(target, card, pack)


def _patch_card(target, want, pack=None):
    """Copy catalog fields from the union card. Keep frozen / plan / wav."""
    have = _read_json(target / "meta.json") or {}
    for key in _CARD_KEYS:
        if key in want:
            have[key] = want[key]
        elif key in have:
            del have[key]
    have["voice_id"] = target.name
    have.setdefault("category", "premade")
    have["source"] = _card_source(want, pack, target.name)
    (target / "meta.json").write_text(json.dumps(have, ensure_ascii=False, indent=2))


def seed_builtins(module, dest, backend=None, cards=None, pack=None):
    """Reconcile premade cards with the current union. User clones / designs stay.

    Two premade kinds, compared differently:

    * design — identity is `instruction`. Change it → wipe (next speak redesigns).
    * clone — identity is `prompt.wav` + `prompt.txt` (sha256 + normalized text).
      Change the factory ref → wipe and copy the new pair. Changing name /
      instruction (Voice Direction) only patches meta.

    Kind is `source` on the card, or inferred from a wav in the pack.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pending").mkdir(parents=True, exist_ok=True)
    if cards is None:
        cards = premade_cards(backend, module)
    want = {}
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        vid = str(card.get("voice_id") or "").strip()
        if vid:
            want[vid] = card
    if pack is None:
        pack = builtin_voices_dir(module)
    else:
        pack = Path(pack)
    for child in list(dest.iterdir()):
        if not _premade_dir(child):
            continue
        if child.name not in want:
            shutil.rmtree(child)
    for vid, card in want.items():
        target = dest / vid
        if target.exists() and not _premade_dir(target):
            continue
        if not target.exists():
            _plant(target, card, pack)
            continue
        if _replant_needed(target, card, pack):
            shutil.rmtree(target)
            _plant(target, card, pack)
            continue
        _patch_card(target, card, pack)
    planted = {c["voice_id"]: c for c in load_voice_cards(dest, category="premade")}
    return [planted[vid] for vid in want if vid in planted]


class VoiceStore:
    """PVC-backed voice library. Premade slots come from the backend; clones/designs are files."""

    def __init__(self, root, presets):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "pending").mkdir(exist_ok=True)
        self.presets = {p["voice_id"]: dict(p) for p in presets}

    def _dir(self, vid):
        return self.root / vid

    def _read_meta(self, vid):
        path = self._dir(vid) / "meta.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def _write(self, vid, meta, wav_bytes=None, transcript=None):
        d = self._dir(vid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        if wav_bytes is not None:
            (d / "prompt.wav").write_bytes(wav_bytes)
        if transcript is not None:
            (d / "prompt.txt").write_text(transcript)

    def prompt(self, vid):
        wav = self._dir(vid) / "prompt.wav"
        txt = self._dir(vid) / "prompt.txt"
        if not wav.is_file():
            return None, None, None
        transcript = txt.read_text() if txt.is_file() else ""
        return wav, transcript, self._read_meta(vid)

    def freeze(self, vid, audio, sr, transcript, extra=None):
        meta = self._read_meta(vid) or dict(self.presets.get(vid) or {}, voice_id=vid)
        meta["frozen"] = True
        if extra:
            meta["extra"] = extra
            if extra.get("plan"):
                meta["plan"] = extra["plan"]
        self._write(vid, meta, wav_bytes=_wav_bytes(audio, sr), transcript=transcript)
        return meta

    def remember_plan(self, vid, plan):
        """Keep the official 12-item voice plan without replacing the frozen wav."""
        if not plan:
            return
        meta = self.get(vid)
        if meta is None:
            return
        extra = dict(meta.get("extra") or {})
        extra["plan"] = plan
        meta["plan"] = plan
        meta["extra"] = extra
        self._write(vid, meta)

    def add_clone(self, name, wav_bytes, transcript, description=""):
        vid = _new_id("clone")
        meta = {"voice_id": vid, "name": name or vid, "category": "cloned",
                "source": "clone", "description": description or "", "frozen": True}
        self._write(vid, meta, wav_bytes=wav_bytes, transcript=transcript)
        return meta

    def put_preview(self, instruction, audio, sr, text, extra=None):
        gid = _new_id("gen")
        pending = self.root / "pending" / gid
        pending.mkdir(parents=True, exist_ok=True)
        wav = _wav_bytes(audio, sr)
        (pending / "prompt.wav").write_bytes(wav)
        (pending / "prompt.txt").write_text(text or "")
        doc = {"generated_voice_id": gid, "instruction": instruction,
               "text": text or "", "extra": extra or {}, "sample_rate": sr}
        (pending / "meta.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2))
        return gid, wav, sr

    def promote_preview(self, generated_voice_id, name, description=""):
        pending = self.root / "pending" / generated_voice_id
        if not pending.is_dir():
            raise HTTPException(status_code=404,
                                detail="unknown generated_voice_id %r" % generated_voice_id)
        doc = json.loads((pending / "meta.json").read_text())
        vid = _new_id("des")
        wav = (pending / "prompt.wav").read_bytes()
        transcript = (pending / "prompt.txt").read_text() if (pending / "prompt.txt").is_file() else ""
        extra = doc.get("extra") or {}
        plan = extra.get("plan")
        meta = {"voice_id": vid, "name": name or vid, "category": "generated",
                "source": "design",
                "description": description or doc.get("instruction") or "",
                "instruction": doc.get("instruction") or "", "frozen": True,
                "generated_voice_id": generated_voice_id}
        if extra:
            meta["extra"] = extra
        if plan:
            meta["plan"] = plan
        self._write(vid, meta, wav_bytes=wav, transcript=transcript)
        return meta

    def get(self, vid):
        if vid in self.presets:
            stored = self._read_meta(vid) or {}
            out = dict(self.presets[vid])
            out.update(stored)
            out.setdefault("category", "premade")
            out["frozen"] = (self._dir(vid) / "prompt.wav").is_file()
            return out
        meta = self._read_meta(vid)
        if meta:
            return meta
        return None

    def list(self):
        seen, rows = set(), []
        for vid, preset in self.presets.items():
            rows.append(self.get(vid))
            seen.add(vid)
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name == "pending" or child.name in seen:
                continue
            meta = self._read_meta(child.name)
            if meta:
                rows.append(meta)
        return rows

    def _premade(self, vid, meta):
        return vid in self.presets or (meta or {}).get("category") == "premade"

    def delete(self, vid):
        meta = self.get(vid)
        if meta is None:
            raise HTTPException(status_code=404, detail="unknown voice_id %r" % vid)
        if self._premade(vid, meta):
            raise HTTPException(status_code=400,
                                detail="premade voice %r cannot be deleted" % vid)
        d = self._dir(vid)
        if d.is_dir():
            shutil.rmtree(d)
        return {"status": "ok"}

    def edit(self, vid, *, name, description=None, labels=None, wav_bytes=None,
             transcript=None):
        meta = self.get(vid)
        if meta is None:
            raise HTTPException(status_code=404, detail="unknown voice_id %r" % vid)
        if self._premade(vid, meta):
            raise HTTPException(status_code=400,
                                detail="premade voice %r cannot be edited" % vid)
        meta["name"] = name
        if description is not None:
            meta["description"] = description
        if labels is not None:
            meta["labels"] = labels
        _old_wav, old_tr, _stored = self.prompt(vid)
        if wav_bytes is not None:
            tr = (transcript or old_tr or "").strip()
            if not tr:
                raise HTTPException(
                    status_code=400,
                    detail="description or ref_text must be the exact transcript of the new reference")
            meta["frozen"] = True
            self._write(vid, meta, wav_bytes=wav_bytes, transcript=tr)
        else:
            self._write(vid, meta)
        return meta

    def settings(self, vid):
        meta = self.get(vid)
        if meta is None:
            raise HTTPException(status_code=404, detail="unknown voice_id %r" % vid)
        return _settings_of(meta)

    def edit_settings(self, vid, patch):
        """Premade cards can change settings; name/edit still refuses them."""
        meta = self.get(vid)
        if meta is None:
            raise HTTPException(status_code=404, detail="unknown voice_id %r" % vid)
        merged = _settings_of(meta)
        merged.update(_normalize_settings(patch, partial=True))
        stored = self._read_meta(vid)
        if stored is None:
            stored = dict(self.presets.get(vid) or {"voice_id": vid})
            stored["voice_id"] = vid
        stored["settings"] = merged
        self._write(vid, stored)
        return merged


def _history_dir(store=None):
    """Next to the voice store when writable, else /tmp: smoke tests never mount /data."""
    store = store if store is not None else _state.get("store")
    base = Path(getattr(store, "root", None) or VOICE_DIR)
    candidates = (base / "history", Path(tempfile.gettempdir()) / "el-history")
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    return candidates[-1]


class HistoryStore:
    """ElevenLabs GET /v1/history. Not /v1/tasks — that is llm-init's queue."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else _history_dir()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.root = Path(tempfile.gettempdir()) / "el-history"
            self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._chars_path = self.root / "chars.json"
        self._chars = 0
        try:
            self._chars = int(json.loads(self._chars_path.read_text()).get("n") or 0)
        except Exception:
            self._chars = 0

    def add(self, *, voice_id, voice_name, voice_category, text, settings,
            content_type, output_format, audio, source="TTS"):
        hid = _new_id("hi")
        n = len(text or "")
        with self._lock:
            frm, to = self._chars, self._chars + n
            self._chars = to
            self._chars_path.write_text(json.dumps({"n": to}))
            folder = self.root / hid
            folder.mkdir()
            doc = {
                "history_item_id": hid,
                "request_id": hid,
                "date_unix": int(time.time()),
                "character_count_change_from": frm,
                "character_count_change_to": to,
                "content_type": content_type,
                "state": "created",
                "voice_id": voice_id or "",
                "voice_name": voice_name or "",
                "voice_category": voice_category or "",
                "model_id": MODEL_NAME,
                "text": text or "",
                "settings": dict(settings or {}),
                "source": source,
                "output_format": output_format,
                "feedback": None,
            }
            (folder / "item.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            (folder / "audio.bin").write_bytes(audio)
        return doc

    def begin(self, *, voice_id, voice_name, voice_category, text, settings,
              content_type, output_format, source="TTS", hid=None):
        """Register a reading before it has audio: `processing`, with a file append() grows.

        A caller may name the reading itself, which is what lets it be stopped before
        the response that would have carried the name back has arrived. mkdir is the
        claim: two requests naming the same id, one wins and the other is told so.
        """
        hid = hid or _new_id("hi")
        n = len(text or "")
        with self._lock:
            frm, to = self._chars, self._chars + n
            self._chars = to
            self._chars_path.write_text(json.dumps({"n": to}))
            folder = self.root / hid
            folder.mkdir()
            doc = {
                "history_item_id": hid,
                "request_id": hid,
                "date_unix": int(time.time()),
                "character_count_change_from": frm,
                "character_count_change_to": to,
                "content_type": content_type,
                "state": "processing",
                "voice_id": voice_id or "",
                "voice_name": voice_name or "",
                "voice_category": voice_category or "",
                "model_id": MODEL_NAME,
                "text": text or "",
                "settings": dict(settings or {}),
                "source": source,
                "output_format": output_format,
                "feedback": None,
            }
            (folder / "item.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            (folder / "audio.bin").write_bytes(b"")
        return doc

    def append(self, hid, chunk):
        """Add the next bytes; a deleted reading took its folder, so dropping them is right."""
        if not chunk:
            return
        path = self.root / hid / "audio.bin"
        with self._lock:
            try:
                with open(path, "ab") as fh:
                    fh.write(chunk)
            except OSError:
                pass

    def finish(self, hid, state="created", output_duration_seconds=None):
        """Mark a reading complete, so a follower knows nothing more is coming."""
        with self._lock:
            path = self.root / hid / "item.json"
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if (state == "created" and output_duration_seconds is not None
                    and math.isfinite(float(output_duration_seconds))
                    and float(output_duration_seconds) >= 0):
                doc["output_duration_seconds"] = float(output_duration_seconds)
            else:
                doc.pop("output_duration_seconds", None)
            # Publish the complete metadata in one rename: readers must never
            # observe state=created before its final duration is present.
            doc["state"] = state
            pending = path.with_suffix(".json.pending")
            pending.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            os.replace(pending, path)
        return doc

    def size(self, hid):
        try:
            return (self.root / hid / "audio.bin").stat().st_size
        except OSError:
            return 0

    def read_at(self, hid, pos, n):
        try:
            with open(self.root / hid / "audio.bin", "rb") as fh:
                fh.seek(pos)
                return fh.read(max(0, n))
        except OSError:
            return b""

    def _ids_newest(self):
        rows = []
        for p in self.root.iterdir():
            if not p.is_dir():
                continue
            item = p / "item.json"
            if not item.is_file():
                continue
            rows.append((item.stat().st_mtime, p.name))
        rows.sort(reverse=True)
        return [name for _m, name in rows]

    def get(self, hid):
        p = self.root / hid / "item.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def audio_path(self, hid):
        p = self.root / hid / "audio.bin"
        return p if p.is_file() else None

    def delete(self, hid):
        folder = self.root / hid
        if not folder.is_dir():
            return False
        shutil.rmtree(folder, ignore_errors=True)
        return True

    def list(self, page_size=100, start_after=None, voice_id=None):
        ids = self._ids_newest()
        if start_after:
            try:
                ids = ids[ids.index(start_after) + 1:]
            except ValueError:
                pass
        cap = max(1, min(int(page_size or 100), 1000))
        history = []
        for hid in ids:
            doc = self.get(hid)
            if not doc:
                continue
            if voice_id and doc.get("voice_id") != voice_id:
                continue
            history.append(doc)
            if len(history) >= cap:
                break
        scanned = ids
        matched = 0
        for hid in scanned:
            doc = self.get(hid)
            if not doc:
                continue
            if voice_id and doc.get("voice_id") != voice_id:
                continue
            matched += 1
        has_more = matched > len(history)
        last = history[-1]["history_item_id"] if history else None
        out = {"history": history, "has_more": has_more, "last_history_item_id": last}
        if history:
            out["scanned_until"] = history[-1].get("date_unix")
        return out


def _store():
    store = _state.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail=_state.get("error") or "engine not ready")
    return store


def _history():
    h = _state.get("history")
    if h is None:
        h = HistoryStore(_history_dir())
        _state["history"] = h
    return h


def _history_audio_response(hid):
    doc = _history().get(hid)
    path = _history().audio_path(hid)
    if doc is None or path is None:
        raise HTTPException(status_code=404, detail="unknown history_item_id %r" % hid)
    mime = doc.get("content_type") or "application/octet-stream"
    return FileResponse(path, media_type=mime,
                        headers={"Connection": "close", "Accept-Ranges": "bytes"})


def _parse_range(header):
    """`bytes=N-` or `bytes=N-M`, the only forms a resume needs; None otherwise."""
    raw = (header or "").strip().lower()
    if not raw.startswith("bytes="):
        return None
    first, _, last = raw[6:].split(",")[0].strip().partition("-")
    if not first.isdigit():
        return None
    return int(first), int(last) if last.isdigit() else None


async def _wait_for_bytes(hid, pos, request, deadline):
    """The size once it grows past `pos`, None when it never will; looking counts as following."""
    store = _history()
    while True:
        live = _live_get(hid)
        if live is not None:
            # Asking for bytes through the item is a follower, which earns the longer grace.
            live.followed = True
            live.touch()
        size = store.size(hid)
        if size > pos:
            return size
        if (store.get(hid) or {}).get("state") != "processing":
            return None
        if time.time() >= deadline or await request.is_disconnected():
            return None
        await asyncio.sleep(_RESUME_POLL_SECONDS)


async def _follow_audio(hid, start, request, deadline):
    """Yield a reading from `start`, waiting on the engine while it speaks."""
    pos = start
    while True:
        size = await _wait_for_bytes(hid, pos, request, deadline)
        if size is None:
            return
        chunk = _history().read_at(hid, pos, size - pos)
        if not chunk:
            return
        pos += len(chunk)
        yield chunk


async def _history_audio_range(hid, doc, rng, request):
    """Serve from an offset, answering with whatever exists the moment there is any."""
    start, end = rng
    store = _history()
    deadline = time.time() + _RESUME_WAIT_SECONDS
    size = await _wait_for_bytes(hid, start, request, deadline)
    body = b""
    if size is not None:
        stop = size if end is None else min(size, end + 1)
        body = store.read_at(hid, start, stop - start)
    if not body:
        return Response(status_code=416,
                        headers={"Accept-Ranges": "bytes",
                                 "Content-Range": "bytes */%d" % store.size(hid)})
    # RFC 9110 spells the length of an unfinished reading `*`.
    done = (store.get(hid) or {}).get("state") != "processing"
    total = str(store.size(hid)) if done else "*"
    return Response(content=body, status_code=206,
                    media_type=doc.get("content_type") or "application/octet-stream",
                    headers={"Accept-Ranges": "bytes",
                             "Content-Range": "bytes %d-%d/%s"
                                              % (start, start + len(body) - 1, total)})


def _backend():
    b = _state.get("backend")
    if b is None:
        raise HTTPException(status_code=503, detail=_state.get("error") or "engine not ready")
    return b


def _require_ready():
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail=_state.get("error") or "engine not ready")


def _clamp01(x, lo=0.0, hi=1.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


def _normalize_settings(raw, partial=False):
    """Accept EL keys; ignore unknown extras. No 400 for leftover fields."""
    out = {} if partial else dict(_DEFAULT_SETTINGS)
    if not isinstance(raw, dict):
        return out
    if "stability" in raw:
        v = _clamp01(raw["stability"])
        if v is not None:
            out["stability"] = v
    if "similarity_boost" in raw:
        v = _clamp01(raw["similarity_boost"])
        if v is not None:
            out["similarity_boost"] = v
    if "style" in raw:
        v = _clamp01(raw["style"])
        if v is not None:
            out["style"] = v
    if "speed" in raw:
        v = _clamp01(raw["speed"], 0.25, 4.0)
        if v is not None:
            out["speed"] = v
    if "use_speaker_boost" in raw:
        out["use_speaker_boost"] = bool(raw["use_speaker_boost"])
    return out


def _settings_of(meta):
    out = dict(_DEFAULT_SETTINGS)
    raw = (meta or {}).get("settings")
    if isinstance(raw, dict):
        out.update(_normalize_settings(raw, partial=True))
    return out


def _merged_settings(meta, payload):
    out = _settings_of(meta)
    req = (payload or {}).get("voice_settings")
    if isinstance(req, dict):
        out.update(_normalize_settings(req, partial=True))
    return out


def _is_cjk(*parts):
    return any("\u4e00" <= ch <= "\u9fff" for part in parts for ch in (part or ""))


def _settings_direction(settings, instruction="", text="", include_speed=True):
    """Style / speed have no engine flag: wrap them into the speak instruction."""
    style = float((settings or {}).get("style") or 0.0)
    try:
        speed = float((settings or {}).get("speed"))
    except (TypeError, ValueError):
        speed = 1.0
    zh = _is_cjk(instruction, text)
    extra = []
    if style >= 0.7:
        extra.append("语气夸张，富有表现力。" if zh else "Deliver with strong expressive style.")
    elif style >= 0.35:
        extra.append("带一点语气和情绪。" if zh else "Speak with noticeable style and emotion.")
    if include_speed:
        if speed <= 0.85:
            extra.append("说得慢一点。" if zh else "Speak slowly.")
        elif speed >= 1.15:
            extra.append("说得快一点。" if zh else "Speak quickly.")
    base = (instruction or "").strip()
    return " ".join([base] + extra).strip()


class Knobs:
    __slots__ = ("cfg", "seed", "speed", "instruction", "context")

    def __init__(self, cfg, seed, speed, instruction, context=False):
        self.cfg = float(cfg)
        # None: do not pass a seed at all, so the model uses the one it ships with.
        self.seed = None if seed is None else int(seed)
        self.speed = float(speed)
        self.instruction = instruction or ""
        # Speaker boost also buys cross-slice context on backends that can carry it.
        self.context = bool(context)


def resolve_settings(settings, base_cfg, instruction="", text="", include_speed_direction=True,
                     seed_jitter=0):
    """EL knobs → CFG / seed / instruction / acoustic-edit; the defaults are a no-op."""
    s = dict(_DEFAULT_SETTINGS)
    if isinstance(settings, dict):
        s.update(_normalize_settings(settings, partial=True))
    sim = float(s["similarity_boost"])
    stab = float(s["stability"])
    style = float(s["style"])
    boost = bool(s["use_speaker_boost"])
    speed = float(s["speed"])
    cfg = float(base_cfg) * max(0.4, min(1.8, sim / 0.75))
    if not boost:
        cfg *= 0.85
    cfg *= (1.15 - 0.30 * stab)
    cfg *= (1.0 + 0.40 * style)
    cfg = round(max(0.3, min(8.0, cfg)), 4)
    # None means we pass no seed; the model keeps the constant its authors picked.
    if SEED is not None:
        seed = int(SEED)
    elif stab >= 0.5:
        seed = None
    else:
        import random
        seed = random.randint(0, 0x7FFFFFFF)
    # Alternate takes; take zero is untouched so a caller that never asks hears the same.
    if seed_jitter:
        import random
        anchor = seed if seed is not None else random.randint(0, 0x7FFFFFFF)
        seed = (anchor + int(seed_jitter) * 7919) & 0x7FFFFFFF
    return Knobs(cfg, seed, speed,
                 _settings_direction(s, instruction, text,
                                     include_speed=include_speed_direction), boost)


def _public_voice(meta):
    if not meta:
        return None
    return {
        "voice_id": meta["voice_id"],
        "name": meta.get("name") or meta["voice_id"],
        "category": meta.get("category") or "premade",
        "description": meta.get("description") or meta.get("instruction") or "",
        "labels": meta.get("labels") or {},
        "preview_url": None,
        "available_for_tiers": [],
        "high_quality_base_model_ids": [],
        "settings": _settings_of(meta),
    }


def _check_ref_audio(data):
    """Product gates from ENGINE_ARGS. 0 means off. Reject, do not trim."""
    raw = bytes(data or b"")
    if REF_MAX_MB > 0 and len(raw) > REF_MAX_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="reference audio is %.1f MiB; --ref-max-mb is %g" % (
                len(raw) / (1024 * 1024), REF_MAX_MB))
    if REF_MIN_SECONDS <= 0 and REF_MAX_SECONDS <= 0:
        return
    dur = wav_seconds(raw)
    if dur is None:
        dur = probe_seconds(raw)
    if dur is None:
        raise HTTPException(status_code=400,
                            detail="could not read reference audio duration")
    if REF_MIN_SECONDS > 0 and dur < REF_MIN_SECONDS:
        raise HTTPException(
            status_code=400,
            detail="reference audio is %ss; --ref-min-seconds is %g" % (dur, REF_MIN_SECONDS))
    if REF_MAX_SECONDS > 0 and dur > REF_MAX_SECONDS:
        raise HTTPException(
            status_code=400,
            detail="reference audio is %ss; --ref-max-seconds is %g" % (dur, REF_MAX_SECONDS))


def _check_design_text(instruction):
    text = (instruction or "").strip()
    n = len(text)
    if DESIGN_MIN_CHARS > 0 and n < DESIGN_MIN_CHARS:
        raise HTTPException(
            status_code=400,
            detail="voice_description is %d characters; --design-min-chars is %d" % (
                n, DESIGN_MIN_CHARS))
    if DESIGN_MAX_CHARS > 0 and n > DESIGN_MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail="voice_description is %d characters; --design-max-chars is %d" % (
                n, DESIGN_MAX_CHARS))


def _as_prompt_wav(data):
    """Upload bytes → real PCM WAV. add used to store mp4 as prompt.wav."""
    raw = bytes(data or b"")
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw
    import numpy as np
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(raw), always_2d=False)
        audio = np.asarray(audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return _wav_bytes(audio, int(sr))
    except Exception:
        pass
    try:
        wav, sr = decode(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=_explain(e)) from e
    audio = wav.detach().cpu().float().numpy() if hasattr(wav, "detach") else np.asarray(wav, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=-1)
    return _wav_bytes(audio, int(sr))


def _load_prompt_wav(path):
    import numpy as np
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), always_2d=False)
    except Exception:
        wav, sr = decode(str(path))
        audio = wav.detach().cpu().float().numpy() if hasattr(wav, "detach") else np.asarray(wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=-1)
        return np.asarray(audio, dtype="float32").reshape(-1), int(sr)
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def _ensure_frozen(vid, ctx=None):
    """Premade slots have no wav until first use: run design once and keep the sample."""
    store, backend = _store(), _backend()
    meta = store.get(vid)
    if meta is None:
        raise HTTPException(status_code=404, detail="unknown voice_id %r" % vid)
    wav_path, transcript, _stored = store.prompt(vid)
    if wav_path is not None:
        return meta, wav_path, transcript
    if (meta.get("source") or "").strip().lower() == "clone":
        raise HTTPException(status_code=400,
                            detail="premade clone voice %r is missing prompt.wav / prompt.txt"
                            % vid)
    instruction = meta.get("instruction") or ""
    sample = meta.get("sample_text") or _preview_text(instruction)
    if not instruction:
        raise HTTPException(status_code=400,
                            detail="voice %r has no frozen sample and no design instruction" % vid)
    audio, sr, extra = backend.design(instruction, sample, ctx=ctx)
    store.freeze(vid, audio, sr, sample, extra=extra)
    wav_path, transcript, _stored = store.prompt(vid)
    return store.get(vid), wav_path, transcript


_PREVIEW_TAKES_MAX = 3


def _preview_takes(payload):
    """How many voices to offer. Absent means one, which is what callers had."""
    raw = (payload or {}).get("takes")
    if raw is None:
        raw = (payload or {}).get("num_previews")
    if raw is None:
        return 1
    try:
        takes = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="takes must be a whole number")
    if takes < 1 or takes > _PREVIEW_TAKES_MAX:
        raise HTTPException(status_code=400,
                            detail="takes must be between 1 and %d" % _PREVIEW_TAKES_MAX)
    return takes


def _preview_text(instruction):
    if DEFAULT_PREVIEW.strip():
        return DEFAULT_PREVIEW.strip()
    # Match the instruction language so design models that require it do not mix scripts.
    if any("\u4e00" <= ch <= "\u9fff" for ch in instruction or ""):
        return "今天天气很好，我们一起去公园散步吧。"
    return "Welcome aboard. Your journey begins now."


def _design_instruction(meta):
    """Prefer the saved 12-item plan. Re-planning from the short blurb drops 口音."""
    meta = meta or {}
    plan = meta.get("plan") or (meta.get("extra") or {}).get("plan") or ""
    return (plan or meta.get("instruction") or "").strip()


def job_tick(ctx, done=None, total=None, stage="synthesize"):
    """See a cancel between two units, and publish done/total when the job has pieces."""
    if ctx is None:
        return
    ctx.checkpoint()
    if done is not None and total is not None:
        ctx.progress(stage=stage, done=int(done), total=int(total))


def _design_identity(meta):
    """True when the card's identity is an instruction / 12-item plan, not a factory wav."""
    meta = meta or {}
    src = str(meta.get("source") or "").strip().lower()
    cat = str(meta.get("category") or "").strip().lower()
    if src == "clone" or cat == "cloned":
        return False
    if src == "design" or cat == "generated":
        return True
    return cat == "premade"


def _speak_voice(vid, text, instruction=None, ctx=None, settings=None):
    backend = _backend()
    meta, wav_path, transcript = _ensure_frozen(vid, ctx=ctx)
    # FireRed design-identity stays on generate_voice_design; Breeze factory blurbs are not Voice Direction.
    if getattr(backend, "prefer_design_speak", False) and _design_identity(meta):
        speak_as = (instruction or _design_instruction(meta) or "").strip()
        if speak_as:
            with _gen_lock:
                audio, sr, extra = backend.design(speak_as, text, ctx=ctx, settings=settings)
            if extra and extra.get("plan") and not (meta or {}).get("plan"):
                _store().remember_plan(vid, extra["plan"])
            return audio, sr
    speak_as = (instruction or "").strip()
    if not speak_as and getattr(backend, "card_instruction_is_direction", True):
        speak_as = ((meta or {}).get("instruction") or "").strip()
    prompt, sr = _load_prompt_wav(wav_path)
    with _gen_lock:
        return backend.clone(text, prompt, sr, transcript or text,
                             instruction=speak_as, ctx=ctx, settings=settings)


def _speak_ref(text, wav_bytes, transcript, instruction=None, ctx=None, settings=None):
    import soundfile as sf

    backend = _backend()
    buf = io.BytesIO(wav_bytes)
    audio, sr = sf.read(buf, always_2d=False)
    import numpy as np
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    with _gen_lock:
        return backend.clone(text, audio, int(sr), transcript or "",
                             instruction=(instruction or "").strip(), ctx=ctx,
                             settings=settings)


def _speak_design(instruction, text, ctx=None, settings=None):
    backend = _backend()
    with _gen_lock:
        audio, sr, _extra = backend.design(instruction, text, ctx=ctx, settings=settings)
    return audio, sr


def _iter_speak_voice(vid, text, instruction=None, ctx=None, settings=None):
    """Yield (wave, sr) as each slice finishes. Caller holds no lock; we take _gen_lock."""
    backend = _backend()
    meta, wav_path, transcript = _ensure_frozen(vid, ctx=ctx)
    if getattr(backend, "prefer_design_speak", False) and _design_identity(meta):
        speak_as = (instruction or _design_instruction(meta) or "").strip()
        if speak_as:
            extra_box = []
            with _gen_lock:
                if hasattr(backend, "iter_design"):
                    for wave, sr in backend.iter_design(speak_as, text, ctx=ctx,
                                                        settings=settings, pace_each=True,
                                                        extra_out=extra_box):
                        if extra_box and extra_box[0].get("plan") and not (meta or {}).get("plan"):
                            _store().remember_plan(vid, extra_box[0]["plan"])
                            meta = dict(meta or {}, plan=extra_box[0]["plan"])
                        yield wave, sr
                else:
                    audio, sr, extra = backend.design(speak_as, text, ctx=ctx, settings=settings)
                    if extra and extra.get("plan") and not (meta or {}).get("plan"):
                        _store().remember_plan(vid, extra["plan"])
                    yield audio, sr
            return
    speak_as = (instruction or "").strip()
    if not speak_as and getattr(backend, "card_instruction_is_direction", True):
        speak_as = ((meta or {}).get("instruction") or "").strip()
    prompt, sr = _load_prompt_wav(wav_path)
    with _gen_lock:
        if hasattr(backend, "iter_clone"):
            yield from backend.iter_clone(text, prompt, sr, transcript or text,
                                          instruction=speak_as, ctx=ctx,
                                          settings=settings, pace_each=True)
        else:
            yield backend.clone(text, prompt, sr, transcript or text,
                                instruction=speak_as, ctx=ctx, settings=settings)


def _stream_gap(backend, text, sr):
    fn = getattr(backend, "stream_gap", None)
    if not callable(fn):
        return None
    gap = fn(text, sr)
    if gap is None:
        return None
    import numpy as np
    a = np.asarray(gap, dtype="float32").reshape(-1)
    return a if len(a) else None


def _stream_next(backend, wave, sr, index):
    if index <= 0:
        return wave
    fn = getattr(backend, "stream_next_slice", None)
    if callable(fn):
        return fn(wave, sr)
    return wave


def _raise_speak(e):
    if isinstance(e, HTTPException):
        raise e
    text = str(e).strip() or type(e).__name__
    if isinstance(e, ValueError) and "too long" in text.lower():
        raise HTTPException(status_code=400, detail=text)
    raise HTTPException(status_code=500, detail="speech synthesis failed: %s" % text)


# A wedged CUDA call holds the generation lock; only a stall this long trips a restart.
_STALL_SECONDS = 600.0

# The entrance cuts at five minutes, so a reading outlives its response and resumes with Range.
_RESUME_GRACE_SECONDS = 90.0
# A reader that came back for more has proved it is following; do not drop a long reading over one bad minute.
_RESUME_GRACE_FOLLOWED_SECONDS = 900.0
_RESUME_WAIT_SECONDS = 240.0
_RESUME_POLL_SECONDS = 0.1

_LIVE_STREAMS = {}
_LIVE_LOCK = threading.Lock()
_LIVE_WATCHDOG = []

# Remember a stop that arrived before the start request, or the later reading runs unattended.
_STOPPED_BEFORE_START = {}
_STOPPED_BEFORE_START_MAX = 512
_STOPPED_BEFORE_START_TTL = 900.0


def _remember_refusal(hid):
    now = time.time()
    with _LIVE_LOCK:
        for old, when in list(_STOPPED_BEFORE_START.items()):
            if now - when > _STOPPED_BEFORE_START_TTL:
                _STOPPED_BEFORE_START.pop(old, None)
        while len(_STOPPED_BEFORE_START) >= _STOPPED_BEFORE_START_MAX:
            _STOPPED_BEFORE_START.pop(next(iter(_STOPPED_BEFORE_START)), None)
        _STOPPED_BEFORE_START[hid] = now


def _was_refused(hid):
    with _LIVE_LOCK:
        if _STOPPED_BEFORE_START.pop(hid, None) is None:
            return False
    return True


class _LiveReading:
    """One reading in flight, and when a reader last looked at it."""

    def __init__(self, hid, stop):
        self.hid = hid
        self.stop = stop
        # The request that started the reading is its first reader.
        self.readers = 1
        self.last_seen = time.time()
        self.last_slice = time.time()
        self.slices = 0
        # Set once somebody comes back for bytes the first response did not carry.
        self.followed = False

    def grace(self):
        return _RESUME_GRACE_FOLLOWED_SECONDS if self.followed else _RESUME_GRACE_SECONDS

    def touch(self):
        self.last_seen = time.time()

    def slice_done(self):
        self.last_slice = time.time()
        self.slices += 1


def _live_begin(hid, stop):
    live = _LiveReading(hid, stop)
    with _LIVE_LOCK:
        _LIVE_STREAMS[hid] = live
        if not _LIVE_WATCHDOG:
            th = threading.Thread(target=_live_watch, daemon=True, name="el-resume-grace")
            _LIVE_WATCHDOG.append(th)
            th.start()
    return live


def _live_end(hid):
    with _LIVE_LOCK:
        _LIVE_STREAMS.pop(hid, None)


def _live_get(hid):
    with _LIVE_LOCK:
        return _LIVE_STREAMS.get(hid)


def cuda_accounting():
    """The allocator's own counters. No driver call, so this still answers when the driver
    is what stopped answering -- which is exactly when it is worth reading."""
    try:
        import torch

        if not torch.cuda.is_initialized():
            return "cuda not initialized"
        mib = 1 << 20
        return ("cuda allocated=%.0f MiB reserved=%.0f MiB peak=%.0f MiB"
                % (torch.cuda.memory_allocated() / mib,
                   torch.cuda.memory_reserved() / mib,
                   torch.cuda.max_memory_allocated() / mib))
    except Exception as e:
        return "cuda accounting unavailable: %s" % e


def thread_stacks():
    """Every thread's stack, for the one question a wedged engine cannot otherwise answer."""
    import traceback

    frames = sys._current_frames()
    out = []
    for th in threading.enumerate():
        frame = frames.get(th.ident)
        out.append("thread %s (id=%s, daemon=%s)" % (th.name, th.ident, th.daemon))
        out.extend("".join(traceback.format_stack(frame)).rstrip().splitlines()
                   if frame else ["  <no frame>"])
        out.append("")
    return "\n".join(out)


def _live_watch():
    """Cancel readings nobody came back for, and restart if one wedges the GPU."""
    while True:
        time.sleep(1.0)
        now = time.time()
        with _LIVE_LOCK:
            live_now = list(_LIVE_STREAMS.values())
        for live in live_now:
            # Still registered means produce() has not finished; a long silence here is a wedge.
            if now - live.last_slice > _STALL_SECONDS:
                log.error("reading %s produced nothing for %.0fs after %s slices; the "
                          "generation lock is wedged and only a restart frees it. %s. "
                          "Every thread's stack follows.",
                          live.hid, now - live.last_slice, live.slices, cuda_accounting())
                log.error("%s", thread_stacks())
                os._exit(watchdog.EXIT_CODE)
            if live.readers > 0 or live.stop.is_set():
                continue
            grace = live.grace()
            if now - live.last_seen > grace:
                log.info("no reader resumed %s for %.0fs (followed=%s); canceling GPU work",
                         live.hid, grace, live.followed)
                live.stop.set()


_Q_EMPTY = object()
_STREAM_DONE = object()


async def _qget(q, timeout):
    try:
        return await asyncio.to_thread(q.get, True, timeout)
    except queue.Empty:
        return _Q_EMPTY


async def _poll_q(q, stop, request, timeout=0.2, live=None):
    """Next queue item, None ends this response; leaving cancels only what cannot resume."""
    while True:
        if request is not None and await request.is_disconnected():
            if live is not None:
                live.touch()
                log.info("stream reader left %s; holding %.0fs for a resume",
                         live.hid, _RESUME_GRACE_SECONDS)
                return None
            if not stop.is_set():
                log.info("stream client disconnected; canceling GPU work")
            stop.set()
        if stop.is_set():
            try:
                return q.get_nowait()
            except queue.Empty:
                return None
        item = await _qget(q, timeout)
        if item is _Q_EMPTY:
            continue
        return item


async def _live_stream(vid, text, instruction, fmt, settings, request=None, hid=None):
    """Flush each slice as produced, and keep the whole reading under the id in the headers."""
    spec = _as_spec(fmt)
    # Concatenated WAV headers are not a file; the client wraps one RIFF.
    wire = OutSpec("pcm", spec.sr, None, spec.token) if spec.kind == "wav" else spec
    q = queue.Queue()
    err = []
    stop = threading.Event()
    ctx = tasks.stream_ctx(stop)
    sr_box = {"sr": spec.sr or _state.get("sample_rate") or 24000}
    mime = _FORMATS[wire.kind][2]
    meta = _store().get(vid)
    if hid and _was_refused(hid):
        # Stopped while this request was still on its way; do not speak it.
        return Response(status_code=499)
    try:
        record = _history().begin(
            voice_id=(meta or {}).get("voice_id") or vid or "",
            voice_name=(meta or {}).get("name") or "",
            voice_category=(meta or {}).get("category") or "",
            text=text or "",
            settings=settings or {},
            content_type=mime,
            output_format=spec.token,
            hid=hid,
        )
    except FileExistsError:
        raise HTTPException(status_code=409,
                            detail="history_item_id %r is already taken" % hid)
    hid = record["history_item_id"]
    live = _live_begin(hid, stop)

    def produce():
        n = 0
        mux = None
        completed = False
        try:
            backend = _backend()
            for wave, sr in _iter_speak_voice(vid, text, instruction=instruction,
                                              ctx=ctx, settings=settings):
                live.slice_done()
                if stop.is_set():
                    log.info("stream stopped after %s slices", n)
                    return
                sr_box["sr"] = wire.sr or sr
                if mux is None:
                    mux = _LiveMux(wire, sr)
                    mux.start()
                if n:
                    gap = _stream_gap(backend, text, sr)
                    if gap is not None:
                        mux.write(gap, sr)
                mux.write(_stream_next(backend, wave, sr, n), sr)
                for chunk in mux.drain():
                    _history().append(hid, chunk)
                    if live.readers > 0:
                        q.put(chunk)
                n += 1
            if mux is not None:
                for chunk in mux.close():
                    _history().append(hid, chunk)
                    if live.readers > 0:
                        q.put(chunk)
            if n == 0 and not stop.is_set():
                raise RuntimeError("engine produced no audio")
            completed = not stop.is_set()
        except tasks.Cancelled:
            log.info("stream canceled after %s slices", n)
        except Exception as e:
            err.append(e)
            log.exception("stream failed")
        finally:
            if mux is not None and getattr(mux, "proc", None) is not None:
                try:
                    mux.proc.kill()
                except Exception:
                    pass
            # Whatever was spoken stays readable; silence leaves no item.
            if _history().size(hid) > 0:
                state = "created" if completed else ("failed" if err else "canceled")
                duration = mux.output_duration_seconds if completed and mux is not None else None
                _history().finish(hid, state, duration)
            else:
                _history().delete(hid)
            _live_end(hid)
            q.put(_STREAM_DONE)

    th = threading.Thread(target=produce, daemon=True, name="el-stream")
    th.start()

    def finish():
        stop.set()
        th.join(timeout=5)

    first = await _poll_q(q, stop, request)
    if first is None or first is _STREAM_DONE:
        finish()
        if stop.is_set() and not err:
            return Response(status_code=499)
        _raise_speak(err[0] if err else RuntimeError("engine produced no audio"))

    async def rest():
        try:
            yield first
            while True:
                item = await _poll_q(q, stop, request, live=live)
                if item is None or item is _STREAM_DONE:
                    break
                yield item
        finally:
            # Ending this response does not end the reading; the watchdog decides.
            live.readers -= 1
            live.touch()

    return StreamingResponse(rest(), media_type=mime,
                             headers=_headers(wire.kind, sr_box["sr"], hid))


def _headers(fmt, sr, hid=None):
    out = {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "tts",
           "X-Audio-Format": fmt, "X-Audio-Sample-Rate": str(sr)}
    if hid:
        # EL names the generation in the headers, and exposes them to a browser.
        out["history-item-id"] = hid
        out["request-id"] = hid
        out["access-control-expose-headers"] = "request-id, history-item-id"
    return out


def _remember_tts(meta, text, settings, spec, body, content_type):
    if not body:
        return None
    spec = _as_spec(spec)
    try:
        return _history().add(
            voice_id=(meta or {}).get("voice_id") or "",
            voice_name=(meta or {}).get("name") or "",
            voice_category=(meta or {}).get("category") or "",
            text=text or "",
            settings=settings or {},
            content_type=content_type,
            output_format=spec.token,
            audio=body,
        )
    except Exception:
        log.exception("history write failed")
    return None


def _audio_response(audio, sr, fmt, stream=False, *, meta=None, text="", settings=None):
    spec = _as_spec(fmt)
    body, fmt = _encode_edge(audio, sr, spec)
    mime = _FORMATS[fmt][2]
    doc = _remember_tts(meta, text, settings, spec, body, mime)
    headers = _headers(fmt, spec.sr or sr, (doc or {}).get("history_item_id"))
    if stream:
        def chunks():
            step = 4096
            for i in range(0, len(body), step):
                yield body[i:i + step]
        return StreamingResponse(chunks(), media_type=mime, headers=headers)
    return tasks.Binary(body, mime, suffix="." + fmt, headers=headers)


def build_app(supports, module=None):
    module = module or _MODULE["name"] or "firered"
    _MODULE["name"] = module
    app = FastAPI(title="audio-tts (%s, ElevenLabs shape)" % module)
    mount_metrics(app)
    register(app, model_name=MODEL_NAME, module=module, served=supports, repo=repo_id(),
             is_ready=lambda: _state["ready"], error=lambda: _state["error"],
             task_api=True, task_legacy=False,
             sample_rate=_state.get("sample_rate") or 24000)
    _args.warn_unclaimed(log)

    # Operator probe for a wedged generation; not part of the model's contract.
    @app.get("/debug/stacks", include_in_schema=False)
    def _debug_stacks():
        return Response(content=thread_stacks(), media_type="text/plain")

    has_tts = "tts" in supports
    has_clone = "tts_clone" in supports

    def _text_of(payload):
        return str(payload.get("text") or payload.get("input") or "").strip()

    def _voice_of(payload):
        return str(payload.get("voice_id") or payload.get("voice") or "").strip()

    async def _synthesize_payload(payload, async_flag, *, stream=False, default_fmt="wav"):
        _require_ready()
        if stream and tasks.truthy(async_flag):
            raise HTTPException(status_code=400,
                                detail="stream does not support async=1; "
                                       "use POST /v1/text-to-speech/{voice_id}?async=1")
        text = _text_of(payload)
        if not text:
            raise HTTPException(status_code=400, detail="text (the words to speak) is required")
        fmt = _output_format(payload.get("output_format") or payload.get("response_format"),
                             default_fmt)
        voice = _voice_of(payload)
        instructions = str(payload.get("instructions") or payload.get("voice_description") or "").strip()
        ref = payload.get("ref_audio")
        ref_text = str(payload.get("ref_text") or payload.get("description") or "").strip()
        settings = _merged_settings(_store().get(voice) if voice else None, payload)

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="synthesize")
            if ref:
                data, _suffix = _ref_bytes(ref)
                if not ref_text and not has_clone:
                    raise HTTPException(status_code=400, detail="this instance does not clone")
                if not ref_text:
                    raise HTTPException(status_code=400,
                                        detail="ref_text (the exact transcript of the reference) is required")
                _check_ref_audio(data)
                audio, sr = _speak_ref(text, data, ref_text, instruction=instructions, ctx=ctx,
                                       settings=settings)
            elif voice:
                audio, sr = _speak_voice(voice, text, instruction=instructions, ctx=ctx,
                                         settings=settings)
            elif instructions:
                _check_design_text(instructions)
                audio, sr = _speak_design(instructions, text, ctx=ctx, settings=settings)
            else:
                voices = _store().list()
                if not voices:
                    raise HTTPException(status_code=400,
                                        detail="voice_id is required (this instance has no voices yet)")
                audio, sr = _speak_voice(voices[0]["voice_id"], text, ctx=ctx,
                                         settings=_merged_settings(voices[0], payload))
            ctx.meter(output_seconds=seconds(audio, sr))
            ctx.progress(ratio=1.0, stage="done")
            voice_meta = _store().get(voice) if voice else None
            return _audio_response(audio, sr, fmt, stream=stream,
                                   meta=voice_meta, text=text, settings=settings)

        return await tasks.dispatch(async_flag, "tts", MODEL_NAME, _work,
                                    fail="speech synthesis failed")

    if has_tts:
        @app.get("/v1/voices")
        def list_voices():
            _require_ready()
            return {"voices": [_public_voice(m) for m in _store().list()]}

        @app.get("/v1/voices/settings/default")
        def default_voice_settings():
            return dict(_DEFAULT_SETTINGS)

        @app.get("/v1/history")
        def list_history(page_size: int = 100,
                         start_after_history_item_id: str = None,
                         voice_id: str = None):
            _require_ready()
            return _history().list(page_size=page_size,
                                   start_after=start_after_history_item_id,
                                   voice_id=voice_id)

        @app.post("/v1/history/download")
        async def download_history(request: Request):
            _require_ready()
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            ids = payload.get("history_item_ids") or []
            if isinstance(ids, str):
                ids = [ids]
            ids = [str(x).strip() for x in ids if str(x).strip()]
            if not ids:
                raise HTTPException(status_code=400, detail="history_item_ids is required")
            if len(ids) == 1:
                return _history_audio_response(ids[0])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for hid in ids:
                    doc = _history().get(hid)
                    path = _history().audio_path(hid)
                    if doc is None or path is None:
                        raise HTTPException(status_code=404, detail="unknown history_item_id %r" % hid)
                    ext = "mp3" if "mpeg" in str(doc.get("content_type") or "") else "wav"
                    zf.write(path, "%s.%s" % (hid, ext))
            return Response(content=buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": "attachment; filename=history.zip"})

        @app.get("/v1/history/{history_item_id}")
        def get_history_item(history_item_id: str):
            _require_ready()
            doc = _history().get(history_item_id)
            if doc is None:
                raise HTTPException(status_code=404,
                                    detail="unknown history_item_id %r" % history_item_id)
            return doc

        @app.get("/v1/history/{history_item_id}/audio")
        async def get_history_audio(history_item_id: str, request: Request):
            _require_ready()
            doc = _history().get(history_item_id)
            if doc is None or _history().audio_path(history_item_id) is None:
                raise HTTPException(status_code=404,
                                    detail="unknown history_item_id %r" % history_item_id)
            rng = _parse_range(request.headers.get("range"))
            if rng is not None:
                return await _history_audio_range(history_item_id, doc, rng, request)
            if doc.get("state") != "processing":
                return _history_audio_response(history_item_id)
            # Still being spoken: give it as it comes, resumable with Range.
            return StreamingResponse(
                _follow_audio(history_item_id, 0, request,
                              time.time() + _RESUME_WAIT_SECONDS),
                media_type=doc.get("content_type") or "application/octet-stream",
                headers={"Accept-Ranges": "bytes"})

        @app.delete("/v1/history/{history_item_id}")
        def delete_history_item(history_item_id: str):
            _require_ready()
            # Throwing away a reading means to stop speaking it, not to wait.
            live = _live_get(history_item_id)
            if live is not None:
                live.stop.set()
            if not _history().delete(history_item_id):
                # Named but not started yet: 404 would let it start with nobody left to stop it.
                if live is None and _ID_RE.match(history_item_id or ""):
                    _remember_refusal(history_item_id)
                    return {"status": "ok"}
                raise HTTPException(status_code=404,
                                    detail="unknown history_item_id %r" % history_item_id)
            return {"status": "ok"}

        @app.get("/v1/voices/{voice_id}")
        def get_voice(voice_id: str):
            _require_ready()
            meta = _store().get(voice_id)
            if meta is None:
                raise HTTPException(status_code=404, detail="unknown voice_id %r" % voice_id)
            return _public_voice(meta)

        @app.get("/v1/voices/{voice_id}/settings")
        def get_voice_settings(voice_id: str):
            _require_ready()
            return _store().settings(voice_id)

        @app.post("/v1/voices/{voice_id}/settings/edit")
        async def edit_voice_settings(voice_id: str, request: Request):
            _require_ready()
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            return _store().edit_settings(voice_id, payload)

        @app.delete("/v1/voices/{voice_id}")
        def delete_voice(voice_id: str):
            _require_ready()
            return _store().delete(voice_id)

        @app.post("/v1/voices/{voice_id}/edit")
        async def edit_voice(voice_id: str,
                             name: str = Form(...),
                             description: str = Form(default=None),
                             labels: str = Form(default=""),
                             ref_text: str = Form(default=""),
                             files: list[UploadFile] | None = File(default=None),
                             file: UploadFile | None = File(default=None)):
            _require_ready()
            parsed_labels = None
            if labels and labels.strip():
                try:
                    doc = json.loads(labels)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail="labels must be a JSON object") from e
                if not isinstance(doc, dict):
                    raise HTTPException(status_code=400, detail="labels must be a JSON object")
                parsed_labels = {str(k): str(v) for k, v in doc.items()}
            uploads = [u for u in (files or []) if u is not None]
            if file is not None:
                uploads.append(file)
            wav_bytes = None
            if uploads:
                raw = await uploads[0].read()
                if not raw:
                    raise HTTPException(status_code=400, detail="reference audio is empty")
                _check_ref_audio(raw)
                wav_bytes = _as_prompt_wav(raw)
            transcript = (ref_text or description or "").strip() or None
            _store().edit(voice_id, name=name.strip(),
                          description=None if description is None else description,
                          labels=parsed_labels, wav_bytes=wav_bytes, transcript=transcript)
            return {"status": "ok"}

        @app.post("/v1/text-to-voice/design")
        async def design_preview(request: Request):
            _require_ready()
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            instruction = str(payload.get("voice_description") or payload.get("instruction")
                              or "").strip()
            if not instruction:
                raise HTTPException(status_code=400, detail="voice_description is required")
            _check_design_text(instruction)
            text = str(payload.get("text") or "").strip() or _preview_text(instruction)
            async_ = request.query_params.get("async")
            takes = _preview_takes(payload)

            def _work(ctx):
                ctx.progress(ratio=0.0, stage="design")
                previews = []
                for take in range(takes):
                    # Take 0 is the one this endpoint has always returned.
                    with _gen_lock:
                        audio, sr, extra = _backend().design(instruction, text, ctx=ctx,
                                                             seed_jitter=take)
                    gid, wav, sr = _store().put_preview(instruction, audio, sr, text, extra=extra)
                    ctx.meter(output_seconds=seconds(audio, sr))
                    ctx.progress(ratio=(take + 1) / float(takes), stage="design")
                    previews.append({
                        "generated_voice_id": gid,
                        "audio_base_64": base64.b64encode(wav).decode("ascii"),
                        "media_type": "audio/wav",
                        "duration_secs": round(len(_to_mono(audio)) / float(sr), 3),
                        "language": None,
                        "plan": (extra or {}).get("plan"),
                    })
                ctx.progress(ratio=1.0, stage="done")
                return {"previews": previews}

            return await tasks.dispatch(async_, "tts", MODEL_NAME, _work, fail="voice design failed")

        @app.post("/v1/text-to-voice")
        async def create_voice(request: Request):
            _require_ready()
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            gid = str(payload.get("generated_voice_id") or "").strip()
            name = str(payload.get("voice_name") or payload.get("name") or "").strip()
            if not gid:
                raise HTTPException(status_code=400, detail="generated_voice_id is required")
            if not name:
                raise HTTPException(status_code=400, detail="voice_name is required")
            desc = str(payload.get("voice_description") or "").strip()
            async_ = request.query_params.get("async")

            def _work(ctx):
                ctx.progress(ratio=0.0, stage="save")
                job_tick(ctx, 0, 1, stage="save")
                meta = _store().promote_preview(gid, name, desc)
                job_tick(ctx, 1, 1, stage="save")
                ctx.progress(ratio=1.0, stage="done")
                return _public_voice(meta)

            return await tasks.dispatch(async_, "tts", MODEL_NAME, _work,
                                        fail="persist designed voice failed")

        @app.post("/v1/text-to-speech/{voice_id}/stream")
        async def el_stream(voice_id: str, request: Request):
            _require_ready()
            if tasks.truthy(request.query_params.get("async")):
                raise HTTPException(status_code=400,
                                    detail="stream does not support async=1; "
                                           "use POST /v1/text-to-speech/{voice_id}?async=1")
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            text = _text_of(payload)
            if not text:
                raise HTTPException(status_code=400, detail="text (the words to speak) is required")
            meta = _store().get(voice_id)
            if meta is None:
                raise HTTPException(status_code=404, detail="unknown voice_id %r" % voice_id)
            fmt = _output_format(
                request.query_params.get("output_format")
                or payload.get("output_format")
                or payload.get("response_format"),
                "mp3_44100_128")
            instructions = str(payload.get("instructions")
                               or payload.get("voice_description") or "").strip()
            # Naming the reading up front lets a client stop it before this response returns the name.
            named = _named_id(request.query_params.get("history_item_id")
                              or payload.get("history_item_id"), "hi")
            return await _live_stream(voice_id, text, instructions, fmt,
                                      _merged_settings(meta, payload), request, hid=named)

        @app.post("/v1/text-to-speech/{voice_id}")
        async def el_speak(voice_id: str, request: Request):
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            payload["voice_id"] = voice_id
            fmt = _output_format(
                request.query_params.get("output_format")
                or payload.get("output_format")
                or payload.get("response_format"),
                "mp3_44100_128")
            payload["output_format"] = fmt.token
            return await _synthesize_payload(payload, request.query_params.get("async"),
                                             default_fmt="mp3_44100_128")

    if has_clone:
        @app.post("/v1/voices/add")
        async def add_voice(name: str = Form(...),
                            description: str = Form(default=""),
                            labels: str = Form(default=""),
                            ref_text: str = Form(default=""),
                            files: list[UploadFile] | None = File(default=None),
                            file: UploadFile | None = File(default=None),
                            async_: str = Query(default=None, alias="async")):
            _require_ready()
            uploads = [u for u in (files or []) if u is not None]
            if file is not None:
                uploads.append(file)
            if not uploads:
                raise HTTPException(status_code=400, detail="files[] (reference audio) is required")
            data = await uploads[0].read()
            if not data:
                raise HTTPException(status_code=400, detail="reference audio is empty")
            _check_ref_audio(data)
            transcript = (ref_text or description or "").strip()
            if labels.strip().startswith("{"):
                try:
                    doc = json.loads(labels)
                    transcript = transcript or str(doc.get("transcript") or doc.get("ref_text") or "")
                except json.JSONDecodeError:
                    pass
            if not transcript:
                raise HTTPException(status_code=400,
                                    detail="description or ref_text must be the exact transcript of the reference")
            wav = _as_prompt_wav(data)

            def _work(ctx):
                ctx.progress(ratio=0.0, stage="clone")
                job_tick(ctx, 0, 1, stage="clone")
                meta = _store().add_clone(name, wav, transcript, description=description)
                job_tick(ctx, 1, 1, stage="clone")
                ctx.progress(ratio=1.0, stage="done")
                return {"voice_id": meta["voice_id"]}

            return await tasks.dispatch(async_, "tts_clone", MODEL_NAME, _work,
                                        fail="voice clone failed")

    return app


def _ref_bytes(ref):
    raw = str(ref or "")
    if raw.startswith("data:"):
        try:
            header, b64 = raw.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="ref_audio data URL is malformed")
        try:
            data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="ref_audio data URL is not valid base64")
        mime = header[5:].split(";", 1)[0] or "audio/wav"
        suffix = ".wav"
        if "mpeg" in mime or "mp3" in mime:
            suffix = ".mp3"
        elif "flac" in mime:
            suffix = ".flac"
        return data, suffix
    raise HTTPException(status_code=400, detail="ref_audio must be a data: URL")


def install(backend, store=None, sample_rate=24000):
    """Tests and a completed load both land here so the HTTP surface never imports the model."""
    _state["backend"] = backend
    if store is None:
        cards = seed_builtins(_MODULE["name"], VOICE_DIR, backend=backend)
        store = VoiceStore(VOICE_DIR, cards or list(backend.presets() or []))
    _state["store"] = store
    _state["history"] = HistoryStore(_history_dir(store))
    _state["sample_rate"] = int(getattr(backend, "sample_rate", None) or sample_rate)
    _state["ready"] = True
    _state["error"] = None
    return _state["store"]


def run(supports, *, module, watchdog, load, sample_rate=24000):
    _MODULE["name"] = module

    def _boot():
        try:
            backend = load()
            install(backend, sample_rate=getattr(backend, "sample_rate", sample_rate) or sample_rate)
            log.info("%s ready, %d premade voices, voice_dir=%s",
                     watchdog, len(backend.presets() or []), VOICE_DIR)
        except Exception as e:
            _state["error"] = hfgate.explain(repo_id(), e)
            log.exception("load failed: %s", _state["error"])
            raise

    _runtime.serve(supports, _boot, lambda caps: build_app(caps, module=module), watchdog,
                   timeout_s=BOOT_TIMEOUT_S + WARMUP_TIMEOUT_S)
