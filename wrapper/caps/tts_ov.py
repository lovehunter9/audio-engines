# Qwen3-TTS on Intel GPU. The HTTP server stays wrapper.caps.tts; only the load is swapped.
#
# PyPI openvino-genai 2026.4.1 predates Text2SpeechPipeline (merged 2026-10-14, release 2026.5.dev).
# optimum-intel at the GPU fix drives the same qwen_tts calls through OpenVINO graphs, so the
# OpenAI routes, voice checks, and error text stay the ones tts.py already serves.
#
# The stock checkpoint has no codec-step stream. A stream request still speaks one sentence at a
# time and returns that waveform as one pcm chunk, which is the socket the CUDA server speaks.
import os

import numpy as np

from . import tts

log = tts.log

# The codec graph's time axis is dynamic, and a new length pays the full compile again.
# One sentence is one chunk (qwen_tts windows at 300 frames plus 25 of left context),
# so four buckets cover it. The decoder is causal: right-padding does not move the
# samples the window keeps, and those samples are cropped back off the pad.
_CODEC_BUCKETS = (64, 128, 256, 384)


def _codec_bucket(frames):
    frames = int(frames)
    for bucket in _CODEC_BUCKETS:
        if frames <= bucket:
            return bucket
    return frames


def _pad_time(codes, pad):
    extra = codes.shape[:-1] + (pad,)
    if hasattr(codes, "new_zeros"):
        import torch
        return torch.cat((codes, codes.new_zeros(extra)), dim=-1)
    return np.concatenate((codes, np.zeros(extra, dtype=codes.dtype)), axis=-1)


def _crop_time(wav, frames, bucket):
    if frames == bucket:
        return wav
    keep = int(round(wav.shape[-1] * frames / bucket))
    return wav[..., :keep]


def _bucketing_forward(orig):
    """Pad the code time axis to a bucket, then drop the padded tail of the wave."""
    seen = set()

    def forward(codes):
        frames = int(codes.shape[-1])
        bucket = _codec_bucket(frames)
        if bucket not in seen:
            seen.add(bucket)
            log.info("codec frames %s -> bucket %s", frames, bucket)
        if bucket != frames:
            codes = _pad_time(codes, bucket - frames)
        wav = orig(codes)
        return _crop_time(wav, frames, bucket)

    forward._codec_buckets = True
    return forward


def _install_codec_buckets(ov):
    model = getattr(ov, "model", None)
    tokenizer = getattr(model, "speech_tokenizer", None) if model is not None else None
    inner = getattr(tokenizer, "model", None) if tokenizer is not None else None
    decoder = getattr(inner, "decoder", None) if inner is not None else None
    if decoder is None or not callable(getattr(decoder, "forward", None)):
        log.warning("codec buckets skipped; the loaded model has no decoder")
        return
    if getattr(decoder.forward, "_codec_buckets", False):
        return
    decoder.forward = _bucketing_forward(decoder.forward)
    log.info("codec decode buckets %s", ",".join(str(b) for b in _CODEC_BUCKETS))


def _device():
    """GPU when this pod is an Intel mode. --device in ENGINE_ARGS replaces that."""
    asked = str(tts._args.text("--device", "") or "").strip()
    if asked:
        return asked
    mode = (os.environ.get("OLARES_GPU_MODE") or "").strip().lower()
    if mode.startswith("intel"):
        return "GPU"
    raise RuntimeError(
        "qwen3ttsov runs on Intel GPU; OLARES_GPU_MODE=%r has no --device" % mode)


def _hub_cache():
    return (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.path.expanduser("~/.cache/huggingface/hub"))


class _OpenVINOQwen3TTS:
    """The method names _synthesize_blocking and _stream_blocking already call."""

    def __init__(self, ov, sample_rate):
        self._ov = ov
        self.sample_rate = int(sample_rate)

    def warmup(self, prefill_len=100):
        return None

    @staticmethod
    def _audio(wav):
        if isinstance(wav, (list, tuple)):
            wav = wav[0] if len(wav) == 1 else np.concatenate(
                [np.asarray(a).reshape(-1) for a in wav])
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        return np.asarray(wav, dtype="float32")

    def _speak(self, **kwargs):
        kwargs.pop("chunk_size", None)
        wav, sr = self._ov.generate(return_sample_rate=True, **kwargs)
        self.sample_rate = int(sr)
        return self._audio(wav), int(sr)

    def generate_custom_voice(self, text, speaker, language="Auto", instruct=None, **kw):
        return self._speak(text=text, speaker=speaker, language=language or "Auto",
                           instruct=instruct, **kw)

    def generate_voice_design(self, text, language="Auto", instruct=None, **kw):
        return self._speak(text=text, language=language or "Auto", instruct=instruct, **kw)

    def generate_voice_clone(self, text, language="Auto", ref_audio=None, ref_text="",
                             xvec_only=False, instruct=None, **kw):
        # faster-qwen3-tts spells this xvec_only; qwen_tts spells it x_vector_only_mode.
        return self._speak(text=text, language=language or "Auto", ref_audio=ref_audio,
                           ref_text=ref_text or None, x_vector_only_mode=bool(xvec_only),
                           instruct=instruct, **kw)

    def generate_custom_voice_streaming(self, **kw):
        audio, sr = self.generate_custom_voice(**kw)
        yield audio, sr, None

    def generate_voice_design_streaming(self, **kw):
        audio, sr = self.generate_voice_design(**kw)
        yield audio, sr, None

    def generate_voice_clone_streaming(self, **kw):
        audio, sr = self.generate_voice_clone(**kw)
        yield audio, sr, None


def _load():
    from optimum.intel import OVModelForTextToSpeechSeq2Seq

    path = tts._model_path()
    device = _device()
    cache = os.path.join(_hub_cache(), "ov-cache")
    os.makedirs(cache, exist_ok=True)
    log.info("loading %s from %s on OpenVINO %s", tts.MODEL_NAME, path, device)
    ov = OVModelForTextToSpeechSeq2Seq.from_pretrained(
        path, export=True, device=device, local_files_only=True,
        cache_dir=_hub_cache(), token=tts.HF_TOKEN,
        ov_config={"CACHE_DIR": cache})
    kind = str(getattr(ov, "_tts_model_type", "") or "").strip().lower()
    speakers = []
    if kind == "custom_voice":
        speakers = list(ov.get_supported_speakers() or [])
    elif not kind:
        kind = "base"
    rate = int(getattr(ov, "sampling_rate", tts.OUT_SR) or tts.OUT_SR)
    _install_codec_buckets(ov)
    return (_OpenVINOQwen3TTS(ov, rate), speakers,
            kind == "custom_voice", kind == "voice_design")


def build_app(supports):
    return tts.build_app(supports)


def run(supports):
    tts._load = _load
    tts.run(supports)
