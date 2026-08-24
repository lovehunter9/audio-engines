"""Voxtral-4B-TTS preset names. The ggml binding's speakers() is empty for this backend."""

VOXTRAL_PRESETS = (
    "casual_male", "casual_female", "cheerful_female", "neutral_male", "neutral_female",
    "fr_male", "fr_female", "es_male", "es_female", "de_male", "de_female",
    "it_male", "it_female", "pt_male", "pt_female", "nl_male", "nl_female",
    "ar_male", "hi_male", "hi_female",
)


def listed_voices(from_session, backend):
    """Prefer the binding's list; an empty voxtral-tts list is a gap, not zero voices."""
    names = [str(s) for s in (from_session or []) if str(s).strip()]
    if names:
        return names
    kind = (backend or "").replace("_", "-").lower()
    if kind == "voxtral-tts":
        return list(VOXTRAL_PRESETS)
    return []
