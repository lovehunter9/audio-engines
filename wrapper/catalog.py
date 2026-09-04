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
    # Qwen3-TTS in-process: which routes a checkpoint can honour is read off the weights
    # (e.g. CustomVoice -> tts, Base -> tts_clone, VoiceDesign -> tts, etc.).
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
    # FireRedTTS3-Instruct: one weight does preset, clone, and design; tts_design advertises, HTTP mounts under tts.
    "firered": [
        (("tts", "tts_clone", "tts_design"), "firered"),
    ],
    # Breeze TTS 2: the same three slots on one 3B checkpoint (zh/en).
    "breeze": [
        (("tts", "tts_clone", "tts_design"), "breeze"),
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
    "firered": "fireredtts3-instruct",
    "breeze": "breeze-tts-2",
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
        ("POST", "/v1/audio/diarization",
         "Speaker diarization (who spoke when; exclusive=1 for non-overlapping turns)", True),
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
    # ElevenLabs voice_id + /v1/tasks only. No OpenAI /v1/audio/* aliases.
    ("firered", "tts"): [
        ("GET", "/v1/voices", "List voices (ElevenLabs shape: premade + cloned + designed)", False),
        ("GET", "/v1/voices/settings/default", "Default voice_settings (ElevenLabs)", False),
        ("GET", "/v1/voices/{voice_id}", "Get one voice by id", False),
        ("GET", "/v1/voices/{voice_id}/settings", "Get stored voice_settings for a voice", False),
        ("POST", "/v1/voices/{voice_id}/settings/edit",
         "Edit voice_settings (mapped onto CFG / seed / instruction / speed); premade allowed", False),
        ("DELETE", "/v1/voices/{voice_id}",
         "Delete a cloned or designed voice; premade voices return 400", False),
        ("POST", "/v1/voices/{voice_id}/edit",
         "Edit a cloned or designed voice (multipart: name required); premade returns 400", False),
        ("POST", "/v1/text-to-speech/{voice_id}",
         "Speak with a stored voice_id (ElevenLabs JSON: text, not input)", True),
        ("POST", "/v1/text-to-speech/{voice_id}/stream",
         "Speak and flush each slice (output_format: mp3_44100_128 / wav_24000 / …)", False),
    ],
    ("firered", "tts_design"): [
        ("POST", "/v1/text-to-voice/design",
         "Voice design preview (JSON voice_description; returns generated_voice_id + audio)", True),
        ("POST", "/v1/text-to-voice",
         "Persist a design preview as a voice_id (JSON generated_voice_id + voice_name)", True),
    ],
    ("firered", "tts_clone"): [
        ("POST", "/v1/voices/add",
         "Clone a voice (multipart name + files[]; description/ref_text is the transcript)", True),
    ],
    ("breeze", "tts"): [
        ("GET", "/v1/voices", "List voices (ElevenLabs shape: premade + cloned + designed)", False),
        ("GET", "/v1/voices/settings/default", "Default voice_settings (ElevenLabs)", False),
        ("GET", "/v1/voices/{voice_id}", "Get one voice by id", False),
        ("GET", "/v1/voices/{voice_id}/settings", "Get stored voice_settings for a voice", False),
        ("POST", "/v1/voices/{voice_id}/settings/edit",
         "Edit voice_settings (mapped onto CFG / seed / instruction / speed); premade allowed", False),
        ("DELETE", "/v1/voices/{voice_id}",
         "Delete a cloned or designed voice; premade voices return 400", False),
        ("POST", "/v1/voices/{voice_id}/edit",
         "Edit a cloned or designed voice (multipart: name required); premade returns 400", False),
        ("POST", "/v1/text-to-speech/{voice_id}",
         "Speak with a stored voice_id (ElevenLabs JSON: text, not input)", True),
        ("POST", "/v1/text-to-speech/{voice_id}/stream",
         "Speak and flush each slice (output_format: mp3_44100_128 / wav_24000 / …)", False),
    ],
    ("breeze", "tts_design"): [
        ("POST", "/v1/text-to-voice/design",
         "Voice design preview (JSON voice_description; returns generated_voice_id + audio)", True),
        ("POST", "/v1/text-to-voice",
         "Persist a design preview as a voice_id (JSON generated_voice_id + voice_name)", True),
    ],
    ("breeze", "tts_clone"): [
        ("POST", "/v1/voices/add",
         "Clone a voice (multipart name + files[]; description/ref_text is the transcript)", True),
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


def spec_endpoints(base, served, declared):
    """Every capability endpoint of the base: the ones mounted here, and why the rest are not."""
    rows = []
    for cap in implements(base):
        module = module_of(base, cap)
        mounts = _MOUNTS.get((module, cap), ())
        if cap in served:
            rows.extend(_row(cap, m, True) for m in mounts)
            continue
        why = ("declared, but needs its own instance of this base" if cap in declared
               else "not declared in MODEL_SUPPORTS")
        rows.extend(_row(cap, m, False, why) for m in mounts)
    return rows
