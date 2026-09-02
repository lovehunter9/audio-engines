# ElevenLabs-shaped TTS: list / design / clone / speak on voice_id; premade slots freeze a design sample on first speak.
import base64
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .. import hfgate
from .. import tasks
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
SEED = int(_args.number("--seed", 42))
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
}
# Public FRP / LLM gateway resets around 10–25 MB (530). Speech mp3 stays under this.
_EDGE_SAFE_BYTES = 6 * 1024 * 1024

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


def _encode_mp3_fit(audio, sr):
    """Speech mp3 sized to fit the public hop. 64 kbps is the ceiling; longer clips drop further."""
    dur = max(float(seconds(audio, sr) or 0.0), 0.25)
    kbps = min(64, max(24, int(_EDGE_SAFE_BYTES * 8 / dur / 1000)))
    pcm = _to_mono(audio).astype("<f4").tobytes()
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-nostdin",
             "-f", "f32le", "-ar", str(int(sr)), "-ac", "1", "-i", "pipe:0",
             "-b:a", "%dk" % kbps, "-f", "mp3", "pipe:1"],
            input=pcm, capture_output=True, check=False)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        log.warning("ffmpeg mp3 fit rc=%s: %s", r.returncode, (r.stderr or b"")[-200:])
    except OSError as e:
        log.warning("ffmpeg mp3 fit missing: %s", e)
    return _encode(audio, sr, "mp3")


def _encode_edge(audio, sr, fmt):
    """Encode `fmt`, but never hand the public hop a body over _EDGE_SAFE_BYTES."""
    body = _encode(audio, sr, fmt)
    if len(body) <= _EDGE_SAFE_BYTES:
        return body, fmt
    compact = _encode_mp3_fit(audio, sr)
    if compact and len(compact) < len(body):
        log.info("edge-safe %s %dB -> mp3 %dB", fmt, len(body), len(compact))
        return compact, "mp3"
    return body, fmt


def _output_format(raw, default="mp3"):
    """ElevenLabs `mp3_44100_128` / OpenAI `wav` -> one of _FORMATS."""
    token = str(raw or default).strip().lower()
    if not token:
        token = default
    head = token.split("_", 1)[0]
    if head not in _FORMATS:
        raise HTTPException(status_code=400,
                            detail="output_format / response_format must start with one of %s"
                                   % ", ".join(sorted(_FORMATS)))
    return head


def _explain(e):
    name = type(e).__name__
    text = str(e).strip()
    if name == "NoBackendError" or "Format not recognised" in text:
        return ("could not decode the reference audio: no decoder for that container. "
                "Convert it to wav/mp3/flac and retry")
    return text or name


def _new_id(prefix):
    return "%s%s" % (prefix, uuid.uuid4().hex[:20])


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


def _store():
    store = _state.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail=_state.get("error") or "engine not ready")
    return store


def _backend():
    b = _state.get("backend")
    if b is None:
        raise HTTPException(status_code=503, detail=_state.get("error") or "engine not ready")
    return b


def _require_ready():
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail=_state.get("error") or "engine not ready")


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


def _speak_voice(vid, text, instruction=None, ctx=None):
    backend = _backend()
    meta, wav_path, transcript = _ensure_frozen(vid, ctx=ctx)
    # FireRed design-identity stays on generate_voice_design; Breeze factory blurbs are not Voice Direction.
    if getattr(backend, "prefer_design_speak", False) and _design_identity(meta):
        speak_as = (instruction or _design_instruction(meta) or "").strip()
        if speak_as:
            with _gen_lock:
                audio, sr, extra = backend.design(speak_as, text, ctx=ctx)
            if extra and extra.get("plan") and not (meta or {}).get("plan"):
                _store().remember_plan(vid, extra["plan"])
            return audio, sr
    speak_as = (instruction or "").strip()
    if not speak_as and getattr(backend, "card_instruction_is_direction", True):
        speak_as = ((meta or {}).get("instruction") or "").strip()
    prompt, sr = _load_prompt_wav(wav_path)
    with _gen_lock:
        return backend.clone(text, prompt, sr, transcript or text,
                             instruction=speak_as, ctx=ctx)


def _speak_ref(text, wav_bytes, transcript, instruction=None, ctx=None):
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
                             instruction=(instruction or "").strip(), ctx=ctx)


def _speak_design(instruction, text, ctx=None):
    backend = _backend()
    with _gen_lock:
        audio, sr, _extra = backend.design(instruction, text, ctx=ctx)
    return audio, sr


def _headers(fmt, sr):
    return {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "tts",
            "X-Audio-Format": fmt, "X-Audio-Sample-Rate": str(sr)}


def _audio_response(audio, sr, fmt, stream=False):
    body, fmt = _encode_edge(audio, sr, fmt)
    mime = _FORMATS[fmt][2]
    headers = _headers(fmt, sr)
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
                audio, sr = _speak_ref(text, data, ref_text, instruction=instructions, ctx=ctx)
            elif voice:
                audio, sr = _speak_voice(voice, text, instruction=instructions, ctx=ctx)
            elif instructions:
                _check_design_text(instructions)
                audio, sr = _speak_design(instructions, text, ctx=ctx)
            else:
                voices = _store().list()
                if not voices:
                    raise HTTPException(status_code=400,
                                        detail="voice_id is required (this instance has no voices yet)")
                audio, sr = _speak_voice(voices[0]["voice_id"], text, ctx=ctx)
            ctx.meter(output_seconds=seconds(audio, sr))
            ctx.progress(ratio=1.0, stage="done")
            return _audio_response(audio, sr, fmt, stream=stream)

        return await tasks.dispatch(async_flag, "tts", MODEL_NAME, _work,
                                    fail="speech synthesis failed")

    if has_tts:
        @app.get("/v1/voices")
        def list_voices():
            _require_ready()
            return {"voices": [_public_voice(m) for m in _store().list()]}

        @app.get("/v1/voices/{voice_id}")
        def get_voice(voice_id: str):
            _require_ready()
            meta = _store().get(voice_id)
            if meta is None:
                raise HTTPException(status_code=404, detail="unknown voice_id %r" % voice_id)
            return _public_voice(meta)

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

            def _work(ctx):
                ctx.progress(ratio=0.0, stage="design")
                with _gen_lock:
                    audio, sr, extra = _backend().design(instruction, text, ctx=ctx)
                gid, wav, sr = _store().put_preview(instruction, audio, sr, text, extra=extra)
                ctx.meter(output_seconds=seconds(audio, sr))
                ctx.progress(ratio=1.0, stage="done")
                return {
                    "previews": [{
                        "generated_voice_id": gid,
                        "audio_base_64": base64.b64encode(wav).decode("ascii"),
                        "media_type": "audio/wav",
                        "duration_secs": round(len(_to_mono(audio)) / float(sr), 3),
                        "language": None,
                        "plan": (extra or {}).get("plan"),
                    }]
                }

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
            try:
                payload = dict(await request.json())
            except Exception:
                raise HTTPException(status_code=400, detail="body must be JSON")
            payload["voice_id"] = voice_id
            return await _synthesize_payload(payload, request.query_params.get("async"),
                                             stream=True, default_fmt="mp3")

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
                "mp3")
            payload["output_format"] = fmt
            return await _synthesize_payload(payload, request.query_params.get("async"),
                                             default_fmt="mp3")

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
