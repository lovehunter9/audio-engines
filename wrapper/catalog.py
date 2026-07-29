# One table for the audio bases: what an image implements, and what each capability mounts.
from . import tasks

# base -> ordered [(caps served together, module)]; first match wins, one instance = one model.
BASES = {
    "qwen": [
        (("stt", "stt_stream"), "stt_stream"),
        (("align",), "align"),
    ],
    "fasterwhisper": [
        (("stt",), "whisper"),
    ],
    "pyannote": [
        (("vad",), "vad"),
        (("diar",), "diar"),
        (("speaker_embed",), "embed"),
        (("enhance",), "enhance"),
    ],
    "nemo": [
        (("diar_stream",), "diar_stream"),
    ],
}

# module -> the model family it runs; parameter size is read off the model id instead.
FAMILIES = {
    "stt_stream": "qwen3-asr",
    "align": "qwen3-forced-aligner",
    "whisper": "faster-whisper",
    "vad": "silero-vad",
    "diar": "pyannote",
    "embed": "pyannote",
    "enhance": "speechbrain",
    "diar_stream": "nemo-sortformer",
}

# (module, capability) -> [(method, path, description, takes async=1)] that capability mounts.
_MOUNTS = {
    ("stt_stream", "stt"): [
        ("POST", "/v1/audio/transcriptions",
         "Offline transcription (single / batch segments)", True),
    ],
    ("stt_stream", "stt_stream"): [
        ("WS", "/v1/audio/stream", "Streaming ASR (WebSocket)", False),
    ],
    ("whisper", "stt"): [
        ("POST", "/v1/audio/transcriptions",
         "Offline transcription (single / batch segments)", True),
        ("POST", "/v1/audio/translations",
         "Speech -> English (Whisper translate task)", True),
    ],
    ("align", "align"): [
        ("POST", "/v1/audio/align", "Forced alignment (single / batch segments)", True),
    ],
    ("vad", "vad"): [
        ("POST", "/v1/audio/vad", "Voice activity detection (speech segments)", True),
    ],
    ("diar", "diar"): [
        ("POST", "/v1/audio/diarization", "Speaker diarization (who spoke when)", True),
    ],
    ("embed", "speaker_embed"): [
        ("POST", "/v1/audio/embeddings", "Speaker embedding (one vector per clip)", True),
    ],
    ("enhance", "enhance"): [
        ("POST", "/v1/audio/enhance",
         "Speech enhancement / denoise (16k mono, format=wav|flac|ogg, default wav)", True),
    ],
    ("diar_stream", "diar_stream"): [
        ("WS", "/v1/audio/diarize/stream", "Streaming speaker diarization (WebSocket)", False),
    ],
}


def implements(base):
    """Every capability the base image can serve, in routing order."""
    return [cap for caps, _mod in BASES.get(base, ()) for cap in caps]


def module_of(base, cap):
    """The module that serves cap on this base, or None."""
    for caps, mod in BASES.get(base, ()):
        if cap in caps:
            return mod
    return None


def _row(cap, mount, available, reason=None):
    method, path, desc, is_async = mount
    if is_async:
        desc = "%s; %s" % (desc, tasks.ASYNC_HINT)
    row = {"capability": cap, "method": method, "path": path, "description": desc,
           "available": available}
    if reason:
        row["reason"] = reason
    return row


def endpoints(module, served):
    """The capability endpoints this process mounts."""
    return [_row(cap, m, True) for cap in served for m in _MOUNTS.get((module, cap), ())]


def spec_endpoints(base, served, declared):
    """Every capability endpoint of the base: the ones mounted here, and why the rest are not."""
    rows = []
    for cap in implements(base):
        mounts = _MOUNTS.get((module_of(base, cap), cap), ())
        if cap in served:
            rows.extend(_row(cap, m, True) for m in mounts)
            continue
        why = ("declared, but needs its own instance of this base" if cap in declared
               else "not declared in MODEL_SUPPORTS")
        rows.extend(_row(cap, m, False, why) for m in mounts)
    return rows
