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
    # Qwen3-TTS in-process: CustomVoice weights serve tts, Base weights serve tts_clone.
    "qwen3tts": [
        (("tts", "tts_clone"), "tts"),
    ],
    # Dasheng-AudioGen: sound effects from English-only captions, so translation sits upstream.
    "dasheng": [
        (("sound_fx",), "sound_fx"),
    ],
    # SoulX-Podcast: one script, later turns conditioned on earlier; zero-shot cloning only.
    "soulx": [
        (("tts_dialogue",), "tts_dialogue"),
    ],
    # Voxtral-4B-TTS on ggml. tts only: the published weights ship no encoder to clone a voice with.
    "crispasr": [
        (("tts",), "crispasr_tts"),
    ],
    # audio.cpp as a child engine. One image covers many families (VoxCPM2, MOSS-TTS-Nano, ...),
    # each serving tts and tts_clone from one set of weights, so both caps share one module and
    # what a given model can actually do is read off the engine at boot. Streaming lives inside
    # tts rather than a capability of its own, and a family without a streaming decode withholds
    # that one route (see catalog.spec_endpoints) instead of dropping the capability.
    "audiocpp": [
        (("tts", "tts_clone"), "audiocpp_tts"),
    ],
    # audio_llm and audio_s2s stay reserved names: no base implements them yet.
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
    "tts": "qwen3-tts",
    "sound_fx": "dasheng-audiogen",
    "tts_dialogue": "soulx-podcast",
    "crispasr_tts": "voxtral-tts",
    # Only a fallback: this module drives whichever audio.cpp family the weights turn out to be,
    # and reports that one (register(family=...)) once they are loaded.
    "audiocpp_tts": "audiocpp",
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
    ("tts", "tts"): [
        ("POST", "/v1/audio/speech",
         "Text to speech, OpenAI shape (JSON in, audio out; stream=1 streams instead)", True),
        ("POST", "/v1/audio/speech/batch",
         "Batch TTS (JSON items[] 1–32, base64 audio out)", True),
        ("WS", "/v1/audio/speech/stream",
         "Streaming text-in TTS (WebSocket; sentence-scoped audio out)", False),
        ("GET", "/v1/audio/voices", "List preset voices", False),
    ],
    ("tts", "tts_clone"): [
        ("POST", "/v1/audio/speech",
         "TTS with a ref_audio data: URL (Base weights; OpenAI JSON shape)", True),
        ("POST", "/v1/audio/speech/batch",
         "Batch TTS with uploaded voice / ref_audio", True),
        ("WS", "/v1/audio/speech/stream",
         "Streaming text-in TTS (WebSocket; sentence-scoped audio out)", False),
        ("POST", "/v1/audio/speech/clone",
         "Voice cloning from reference audio (multipart: file + input + ref_text)", True),
    ],
    # Shared speech routes live only under tts. Qwen3-TTS lists them again under tts_clone because
    # a given instance serves one cap or the other; these two models serve both at once, and the
    # same path under two caps is what the dashboard then prints twice. Cloning via `ref_audio` on
    # /v1/audio/speech still works — that is a field of the same endpoint, not a second one.
    # No /v1/audio/voices: these families have no built-in speakers.
    ("audiocpp_tts", "tts"): [
        ("POST", "/v1/audio/speech",
         "Text to speech, OpenAI shape (JSON in, audio out; stream=1 streams instead; "
         "ref_audio as a data: URL clones)", True),
        ("POST", "/v1/audio/speech/batch",
         "Batch TTS (JSON items[] 1–32, base64 audio out; ref_audio clones per item)", True),
        ("WS", "/v1/audio/speech/stream",
         "Streaming text-in TTS (WebSocket; sentence-scoped audio out)", False),
    ],
    ("audiocpp_tts", "tts_clone"): [
        ("POST", "/v1/audio/speech/clone",
         "Voice cloning from reference audio (multipart: file + input + ref_text)", True),
    ],
    ("sound_fx", "sound_fx"): [
        ("POST", "/v1/audio/speech",
         "Sound effect / ambience from an English description (OpenAI shape; optional "
         "sfx/env/music/speech/asr aspects; length is inferred from the text, not set)", True),
        ("POST", "/v1/audio/speech/batch",
         "Native batch (JSON items[] 1–8, one denoising pass, base64 audio out)", True),
    ],
    # No WebSocket: the binding synthesizes whole utterances, so stream=1 is sentence-scoped.
    ("crispasr_tts", "tts"): [
        ("POST", "/v1/audio/speech",
         "Text to speech, OpenAI shape (JSON in, audio out; stream=1 streams sentence by sentence)",
         True),
        ("POST", "/v1/audio/speech/batch",
         "Batch TTS (JSON items[] 1–32, base64 audio out)", True),
        ("GET", "/v1/audio/voices", "List preset voices", False),
    ],
    # One endpoint: a script is already the unit of work, and the model has no streaming decode.
    ("tts_dialogue", "tts_dialogue"): [
        ("POST", "/v1/audio/speech",
         "Multi-speaker dialogue from a script (speakers[] of {ref_audio, ref_text} + turns[] of "
         "{speaker, text}; whole conversation in one pass, per_turn=true splits the response)",
         True),
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
           "available": available, "async_supported": is_async}
    if reason:
        row["reason"] = reason
    return row


def endpoints(module, served):
    """The capability endpoints this process mounts."""
    return [_row(cap, m, True)
            for cap in served for m in _MOUNTS.get((module, cap), ())]


def spec_endpoints(base, served, declared, withheld=()):
    """Every capability endpoint of the base: the ones mounted here, and why the rest are not.

    withheld is [(method, path, reason)] a cap chose not to mount even though it serves that
    capability — a route the model behind it cannot honour (no streaming decode, no cloning). Two
    models on one base can differ that way, so the table below is what the base can mount and this
    is what this instance did.
    """
    holds = {(str(m).upper(), p): why for m, p, why in withheld or ()}
    rows = []
    for cap in implements(base):
        module = module_of(base, cap)
        mounts = _MOUNTS.get((module, cap), ())
        if cap in served:
            for m in mounts:
                why = holds.get((str(m[0]).upper(), m[1]))
                rows.append(_row(cap, m, why is None, why))
            continue
        why = ("declared, but needs its own instance of this base" if cap in declared
               else "not declared in MODEL_SUPPORTS")
        rows.extend(_row(cap, m, False, why) for m in mounts)
    return rows
