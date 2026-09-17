# One table for the audio bases: what an image implements, and what each capability mounts.
from . import tasks

# base -> ordered [(caps served together, module)]; first match wins, one instance = one model.
BASES = {
    "qwen": [
        (("stt", "stt_stream"), "stt_stream"),
        (("align",), "align"),
    ],
    # Intel iGPU / Arc: OpenVINO GenAI ASRPipeline. Same image for both Olares modes;
    # stt_stream is decoder-token streaming after the utterance, not vLLM incremental audio.
    "ov": [
        (("stt", "stt_stream"), "stt_stream"),
        (("align",), "align"),
    ],
    # Intel Whisper: OpenVINO WhisperPipeline only. Not the Qwen ov image.
    "whisperov": [
        (("stt",), "whisper_ov"),
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
    # Intel Enhance: SpeechBrain on XPU. Same enhance module, no CUDA fallback.
    "enhancexpu": [
        (("enhance",), "enhance"),
    ],
    "nemo": [
        (("diar_stream",), "diar_stream"),
    ],
    # speakrs: the pyannote community-1 pipeline rewritten in Rust, inference on ONNX Runtime.
    # A separate image because its dependencies share nothing with the torch stack, per the one
    # image per engine family rule. pyannote keeps its own diar: which of the two an installation
    # runs is decided by the image its chart deploys, not here.
    "speakrs": [
        (("diar",), "diar_speakrs"),
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
    "whisper_ov": "whisper",
    "vad": "silero-vad",
    "diar": "pyannote",
    "diar_speakrs": "speakrs",
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
    ("whisper_ov", "stt"): [
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
    ("diar_speakrs", "diar"): [
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
         "Speak and flush each slice (output_format: pcm_24000 / mp3_44100_128 / wav_24000 / …)", False),
        ("GET", "/v1/history", "List generated speech history", False),
        ("POST", "/v1/history/download", "Download multiple history items", False),
        ("GET", "/v1/history/{history_item_id}", "Read speech history metadata", False),
        ("GET", "/v1/history/{history_item_id}/audio", "Read generated speech audio", False),
        ("DELETE", "/v1/history/{history_item_id}", "Delete generated speech history", False),
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
         "Speak and flush each slice (output_format: pcm_24000 / mp3_44100_128 / wav_24000 / …)", False),
        ("GET", "/v1/history", "List generated speech history", False),
        ("POST", "/v1/history/download", "Download multiple history items", False),
        ("GET", "/v1/history/{history_item_id}", "Read speech history metadata", False),
        ("GET", "/v1/history/{history_item_id}/audio", "Read generated speech audio", False),
        ("DELETE", "/v1/history/{history_item_id}", "Delete generated speech history", False),
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


_OPERATION_IDS = {
    ("GET", "/v1/models"): "model.list",
    ("POST", "/v1/audio/transcriptions"): "audio.transcribe",
    ("POST", "/v1/audio/translations"): "audio.translate",
    ("WS", "/v1/audio/stream"): "audio.transcribe.stream",
    ("POST", "/v1/audio/align"): "audio.align",
    ("POST", "/v1/audio/vad"): "audio.vad",
    ("POST", "/v1/audio/diarization"): "audio.diarize",
    ("WS", "/v1/audio/diarize/stream"): "audio.diarize.stream",
    ("POST", "/v1/audio/embeddings"): "audio.speaker_embed",
    ("POST", "/v1/audio/enhance"): "audio.enhance",
    ("POST", "/v1/audio/speech"): "speech.synthesize",
    ("POST", "/v1/audio/speech/batch"): "speech.synthesize.batch",
    ("WS", "/v1/audio/speech/stream"): "speech.synthesize.stream",
    ("GET", "/v1/audio/voices"): "voice.list",
    ("POST", "/v1/audio/speech/clone"): "speech.synthesize.reference",
    ("GET", "/v1/voices"): "voice.list",
    ("GET", "/v1/voices/settings/default"): "voice.settings.default",
    ("GET", "/v1/voices/{voice_id}"): "voice.read",
    ("GET", "/v1/voices/{voice_id}/settings"): "voice.settings.read",
    ("POST", "/v1/voices/{voice_id}/settings/edit"): "voice.settings.update",
    ("DELETE", "/v1/voices/{voice_id}"): "voice.delete",
    ("POST", "/v1/voices/{voice_id}/edit"): "voice.update",
    ("POST", "/v1/text-to-speech/{voice_id}"): "speech.synthesize",
    ("POST", "/v1/text-to-speech/{voice_id}/stream"): "speech.synthesize.stream",
    ("POST", "/v1/text-to-voice/design"): "voice.design.preview",
    ("POST", "/v1/text-to-voice"): "voice.design.save",
    ("POST", "/v1/voices/add"): "voice.clone",
    ("GET", "/v1/history"): "history.list",
    ("POST", "/v1/history/download"): "history.download",
    ("GET", "/v1/history/{history_item_id}"): "history.read",
    ("GET", "/v1/history/{history_item_id}/audio"): "history.audio",
    ("DELETE", "/v1/history/{history_item_id}"): "history.delete",
    ("GET", "/v1/tasks"): "task.list",
    ("GET", "/v1/tasks/{id}"): "task.read",
    ("GET", "/v1/tasks/{id}/result"): "task.result",
    ("DELETE", "/v1/tasks/{id}"): "task.cancel",
    ("GET", "/v1/audio/tasks"): "task.list",
    ("GET", "/v1/audio/tasks/{id}"): "task.read",
    ("GET", "/v1/audio/tasks/{id}/result"): "task.result",
    ("DELETE", "/v1/audio/tasks/{id}"): "task.cancel",
}


def _protocol(path):
    if path.startswith(("/v1/voices", "/v1/text-to-speech", "/v1/text-to-voice")):
        return "elevenlabs.voice.v1"
    if path in ("/v1/audio/transcriptions", "/v1/audio/translations", "/v1/audio/speech"):
        return "openai.audio.v1"
    return "olares.audio.v1"


def _operation_id(module, cap, method, path):
    if cap == "tts_dialogue" and path == "/v1/audio/speech":
        return "speech.dialogue"
    if cap == "sound_fx":
        return "sound.generate.batch" if path.endswith("/batch") else "sound.generate"
    if cap == "tts_clone" and path in ("/v1/audio/speech", "/v1/audio/speech/batch", "/v1/audio/speech/stream"):
        return {"/v1/audio/speech": "speech.synthesize.reference",
                "/v1/audio/speech/batch": "speech.synthesize.reference.batch",
                "/v1/audio/speech/stream": "speech.synthesize.reference.stream"}[path]
    return _OPERATION_IDS.get((method, path), "")


def _modalities(operation_id):
    if operation_id.startswith("speech.synthesize.reference"):
        return ["text", "audio"], ["audio"]
    if operation_id.startswith(("speech.synthesize", "speech.dialogue", "sound.generate")):
        return ["text"], ["audio"]
    if operation_id == "voice.clone":
        return ["audio", "text"], ["voice"]
    if operation_id == "voice.design.preview":
        return ["text"], ["voice", "audio"]
    if operation_id.startswith("audio."):
        return ["audio"], ["audio"] if operation_id == "audio.enhance" else ["text"]
    return [], []


def _parameters(module, operation_id):
    params = []
    if operation_id.startswith(("speech.synthesize", "speech.dialogue", "sound.generate")):
        if module in ("firered", "breeze"):
            params.append({"name": "output_format", "type": "string", "default": "mp3_44100_128",
                           "enum": ["pcm_24000", "wav_24000", "mp3_44100_128"]})
        else:
            params.append({"name": "response_format", "type": "string", "default": "wav",
                           "enum": ["wav", "pcm", "mp3", "flac", "ogg"]})
    if module in ("firered", "breeze") and operation_id.startswith(("speech.", "voice.settings")):
        params.extend([
            {"name": "speed", "type": "number", "default": 1.0, "minimum": 0.25, "maximum": 4.0},
            {"name": "stability", "type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
            {"name": "similarity_boost", "type": "number", "default": 0.75, "minimum": 0.0, "maximum": 1.0},
            {"name": "style", "type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
            {"name": "use_speaker_boost", "type": "boolean", "default": True},
        ])
    return params


def _constraints(module, operation_id):
    formats, rates, limits = [], [], {}
    if operation_id.startswith(("speech.synthesize", "speech.dialogue", "sound.generate")):
        formats = (["pcm_24000", "wav_24000", "mp3_44100_128"]
                   if module in ("firered", "breeze") else ["wav", "pcm", "mp3", "flac", "ogg"])
    if module in ("firered", "breeze"):
        rates = [24000]
    if operation_id == "speech.synthesize.batch":
        limits["max_batch_items"] = 8 if module == "sound_fx" else 32
    if operation_id == "voice.clone":
        limits["reference_audio_seconds"] = {"minimum": 5, "maximum": 30}
    return formats, rates, limits


def describe_endpoint(module, cap, method, path, is_async):
    operation_id = _operation_id(module, cap, method, path)
    inputs, outputs = _modalities(operation_id)
    formats, rates, limits = _constraints(module, operation_id)
    scope = "model"
    if operation_id.startswith("voice."):
        scope = "voice"
    elif operation_id.startswith("history."):
        scope = "history"
    elif operation_id.startswith("task."):
        scope = "task"
    return {
        "operation_id": operation_id,
        "protocol": _protocol(path),
        "transport": "websocket" if method == "WS" else "http",
        "sync_supported": method != "WS",
        "streaming": method == "WS" or path.endswith("/stream"),
        "async_supported": is_async,
        "required_supports": [cap] if cap else [],
        "input_modalities": inputs,
        "output_modalities": outputs,
        "output_formats": formats,
        "sample_rates": rates,
        "parameters": _parameters(module, operation_id),
        "limits": limits,
        "resource_scope": scope,
    }


def _row(module, cap, mount, available, reason=None):
    method, path, desc, is_async = mount
    if is_async:
        desc = "%s; %s" % (desc, tasks.ASYNC_HINT)
    row = {"capability": cap, "method": method, "path": path, "description": desc,
           "available": available}
    row.update(describe_endpoint(module, cap, method, path, is_async))
    if reason:
        row["reason"] = reason
    return row


def endpoints(module, served):
    """The capability endpoints this process mounts."""
    return [_row(module, cap, m, True)
            for cap in served for m in _MOUNTS.get((module, cap), ())]


def spec_endpoints(base, served, declared):
    """Every capability endpoint of the base: the ones mounted here, and why the rest are not."""
    rows = []
    for cap in implements(base):
        module = module_of(base, cap)
        mounts = _MOUNTS.get((module, cap), ())
        if cap in served:
            rows.extend(_row(module, cap, m, True) for m in mounts)
            continue
        why = ("declared, but needs its own instance of this base" if cap in declared
               else "not declared in MODEL_SUPPORTS")
        rows.extend(_row(module, cap, m, False, why) for m in mounts)
    return rows
