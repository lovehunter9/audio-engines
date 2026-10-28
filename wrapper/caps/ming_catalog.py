# Ming's built-in IP list. The shared library plants these cards; other TTS models keep their own catalog.
import json
from pathlib import Path


def catalog_path():
    return Path(__file__).resolve().parents[1] / "data" / "ming_ips.json"


def load_ip_catalog(path=None):
    path = Path(path) if path else catalog_path()
    rows = json.loads(path.read_text())
    cards = []
    for row in rows:
        vid = str(row.get("voice_id") or "").strip()
        if not vid:
            continue
        labels = {
            "origin": "model",
            "language": row.get("language") or "",
            "gender": row.get("gender") or "",
            "group": row.get("group") or "",
        }
        if row.get("recommended"):
            labels["recommended"] = "true"
        cards.append({
            "voice_id": vid,
            "name": row.get("name") or vid,
            "category": "premade",
            "source": "ip",
            "origin": "model",
            "ip": vid,
            "description": row.get("description") or "",
            "labels": labels,
        })
    return cards
