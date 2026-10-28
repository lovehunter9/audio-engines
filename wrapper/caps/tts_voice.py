# One voice library for every HTTP surface. The card's task is decided here; a backend applies a knob only when that task accepts it.
import json
import shutil
from pathlib import Path


class VoiceError(Exception):
    def __init__(self, status, detail):
        self.status_code = int(status)
        self.detail = detail
        super().__init__(detail)


class Turn:
    """One utterance, one task. `knobs` holds EL settings and Ming controls this build has not wired; empty means the caller asked for none."""

    def __init__(self, task, text, voice_id="", ip="", prompt_wav="", prompt_text="",
                 style="", language_id="", ref_audio="", knobs=None, origin=""):
        self.task = task
        self.text = text
        self.voice_id = voice_id or ""
        self.ip = ip or ""
        self.prompt_wav = prompt_wav or ""
        self.prompt_text = prompt_text or ""
        self.style = style or ""
        self.language_id = language_id or ""
        self.ref_audio = ref_audio or ""
        self.knobs = dict(knobs or {})
        self.origin = origin or ""


def pack_cards(pack):
    """The four shared voices. `origin` is stamped on the planted copy, so the files in the repo stay untouched."""
    pack = Path(pack) if pack else None
    if pack is None or not pack.is_dir():
        return []
    cards = []
    for child in sorted(pack.iterdir()):
        meta_path = child / "meta.json"
        if not child.is_dir() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or meta.get("category") != "premade":
            continue
        card = dict(meta)
        card["voice_id"] = child.name
        card["origin"] = "pack"
        labels = dict(card.get("labels") or {})
        labels["origin"] = "pack"
        card["labels"] = labels
        cards.append(card)
    return cards


def _read_meta(path):
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


def _premade(meta):
    if not meta:
        return False
    if meta.get("category") == "premade":
        return True
    return meta.get("origin") in ("model", "pack")


def _voice_id(name):
    vid = str(name or "").strip()
    if not vid or vid in (".", "..") or any(ch in vid for ch in "/\\\x00"):
        raise VoiceError(400, "name is required and must be a single path segment")
    return vid


class VoiceLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def reconcile(self, model_cards, pack):
        """Plant premade cards. User clone / design directories stay."""
        want = {}
        for card in list(model_cards or []) + pack_cards(pack):
            vid = str(card.get("voice_id") or "").strip()
            if vid:
                want[vid] = card
        for child in list(self.root.iterdir()):
            if not child.is_dir():
                continue
            meta = _read_meta(child / "meta.json")
            if _premade(meta) and child.name not in want:
                shutil.rmtree(child)
        pack = Path(pack) if pack else None
        for vid, card in want.items():
            target = self.root / vid
            if target.exists() and not _premade(_read_meta(target / "meta.json")):
                continue
            self._plant(target, card, pack)
        return [self.get(vid) for vid in want if self.get(vid)]

    def _plant(self, target, card, pack):
        src = (pack / target.name) if pack is not None else None
        if src is not None and src.is_dir() and (src / "meta.json").is_file():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        else:
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
        meta = dict(card)
        meta["voice_id"] = target.name
        meta.setdefault("category", "premade")
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def get(self, voice_id):
        meta = _read_meta(self.root / voice_id / "meta.json")
        if not meta:
            return None
        meta.setdefault("voice_id", voice_id)
        return meta

    def list(self):
        rows = []
        for child in sorted(self.root.iterdir()):
            # `pending` is the EL preview scratch dir. It is not a voice.
            if not child.is_dir() or child.name == "pending":
                continue
            meta = self.get(child.name)
            if meta:
                rows.append(meta)
        return rows

    def public(self, meta):
        return {
            "id": meta["voice_id"],
            "voice": meta["voice_id"],
            "name": meta.get("name") or meta["voice_id"],
            "origin": meta.get("origin") or "",
            "source": meta.get("source") or "",
            "category": meta.get("category") or "",
            "description": meta.get("description") or "",
            "labels": meta.get("labels") or {},
        }

    def delete(self, voice_id):
        meta = self.get(voice_id)
        if meta is None:
            raise VoiceError(404, "unknown voice %r" % voice_id)
        if _premade(meta):
            raise VoiceError(400, "premade voice %r cannot be deleted" % voice_id)
        shutil.rmtree(self.root / voice_id)
        return {"status": "ok"}

    def edit(self, voice_id, **fields):
        """Present so an ElevenLabs edit route can call it. Premade stays fixed."""
        meta = self.get(voice_id)
        if meta is None:
            raise VoiceError(404, "unknown voice %r" % voice_id)
        if _premade(meta):
            raise VoiceError(400, "premade voice %r cannot be edited" % voice_id)
        for key, value in fields.items():
            if value is not None:
                meta[key] = value
        (self.root / voice_id / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def _claim(self, name):
        vid = _voice_id(name)
        if (self.root / vid).exists():
            raise VoiceError(400, "voice %r already exists" % vid)
        return vid

    def add_clone(self, voice_id, name, wav_bytes, transcript, description=""):
        """A user clone: the reference wav and its transcript. OpenAI `POST /v1/audio/voices` calls this."""
        transcript = str(transcript or "").strip()
        if not transcript:
            raise VoiceError(400, "ref_text (the exact transcript of the reference) is required")
        if not wav_bytes:
            raise VoiceError(400, "reference audio is empty")
        vid = self._claim(voice_id)
        target = self.root / vid
        target.mkdir(parents=True)
        (target / "prompt.wav").write_bytes(wav_bytes)
        (target / "prompt.txt").write_text(transcript)
        meta = {
            "voice_id": vid,
            "name": name or vid,
            "category": "cloned",
            "source": "clone",
            "origin": "user",
            "description": description or "",
            "labels": {"origin": "user"},
        }
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def add_design(self, voice_id, name, instruction, description=""):
        """A user design: the instruction is the voice. The next speak runs design again from that text."""
        instruction = str(instruction or "").strip()
        if not instruction:
            raise VoiceError(400, "instructions (the voice description) is required")
        vid = self._claim(voice_id)
        target = self.root / vid
        target.mkdir(parents=True)
        meta = {
            "voice_id": vid,
            "name": name or vid,
            "category": "generated",
            "source": "design",
            "origin": "user",
            "instruction": instruction,
            "description": description or instruction,
            "labels": {"origin": "user"},
        }
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    def turn_from_card(self, card, text, language_id=""):
        vid = card["voice_id"]
        source = (card.get("source") or "").strip().lower()
        origin = card.get("origin") or ""
        if source == "ip":
            return Turn(task="ip", text=text, voice_id=vid, ip=card.get("ip") or vid,
                        language_id=language_id, origin=origin)
        if source == "clone":
            wav = self.root / vid / "prompt.wav"
            transcript = ""
            txt = self.root / vid / "prompt.txt"
            if txt.is_file():
                transcript = txt.read_text().strip()
            if not wav.is_file() or not transcript:
                raise VoiceError(400, "clone voice %r is missing prompt.wav / prompt.txt" % vid)
            return Turn(task="clone", text=text, voice_id=vid, prompt_wav=str(wav),
                        prompt_text=transcript, language_id=language_id, origin=origin)
        if source == "design":
            style = (card.get("instruction") or "").strip()
            if not style:
                raise VoiceError(400, "design voice %r has no instruction" % vid)
            return Turn(task="design", text=text, voice_id=vid, style=style,
                        language_id=language_id, origin=origin)
        raise VoiceError(400, "voice %r has no speak task" % vid)


_KNOB_KEYS = ("stability", "similarity_boost", "style", "emotion", "dialect",
              "pitch", "volume")


def resolve_openai(payload, library):
    """OpenAI speech body -> one Turn. Exactly one of voice, ref_audio, instructions."""
    text = str(payload.get("input") or "").strip()
    if not text:
        raise VoiceError(400, "input (the text to speak) is required")
    voice = str(payload.get("voice") or "").strip()
    instructions = str(payload.get("instructions") or "").strip()
    ref = payload.get("ref_audio")
    ref_text = str(payload.get("ref_text") or "").strip()
    language = str(payload.get("language") or "").strip()
    knobs = {}
    if payload.get("speed") not in (None, ""):
        try:
            speed = float(payload.get("speed"))
        except (TypeError, ValueError):
            raise VoiceError(400, "speed must be a number")
        if abs(speed - 1.0) > 1e-6:
            knobs["speed"] = speed
    for key in _KNOB_KEYS:
        if payload.get(key) not in (None, ""):
            knobs[key] = payload.get(key)
    if knobs:
        raise VoiceError(400, "these controls are not wired yet: %s" % ", ".join(sorted(knobs)))
    chosen = []
    if voice:
        chosen.append("voice")
    if ref:
        chosen.append("ref_audio")
    if instructions:
        chosen.append("instructions")
    if len(chosen) != 1:
        raise VoiceError(
            400, "send exactly one of voice, ref_audio, or instructions (got %s)"
            % (", ".join(chosen) or "none"))
    if voice:
        card = library.get(voice)
        if card is None:
            raise VoiceError(400, "unknown voice %r; see GET /v1/audio/voices" % voice)
        return library.turn_from_card(card, text, language)
    if ref:
        if not isinstance(ref, str) or not ref.startswith("data:"):
            raise VoiceError(400, "ref_audio must be a data: URL (base64 reference audio)")
        if not ref_text:
            raise VoiceError(400, "ref_text (the exact transcript of the reference) is required")
        return Turn(task="clone", text=text, prompt_text=ref_text, language_id=language,
                    ref_audio=ref, origin="request")
    return Turn(task="design", text=text, style=instructions, language_id=language,
                origin="request")
