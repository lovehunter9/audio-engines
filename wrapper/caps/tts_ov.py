# Qwen3-TTS on Intel. HTTP stays wrapper.caps.tts; discrete weights stay fp32 and the iGPU uses int8.
import os

import numpy as np

from . import tts

log = tts.log

# Pad the codec time axis to 64/128/256/384 and crop the wave, so a repeat reuses the compiled graph.
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


def compress_frame_int8(xml):
    """8-bit the fused predictor frame. load_in_8bit covers the talker and embeddings; the waveform decoder stays fp32."""
    import gc

    import openvino as ov
    from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
    from optimum.intel.openvino.quantization import _weight_only_quantization

    core = ov.Core()
    core.set_property({"ENABLE_MMAP": False})
    model = core.read_model(xml)
    compressed = _weight_only_quantization(model, OVWeightQuantizationConfig(bits=8))
    staged = xml[:-4] + ".compressed.xml"
    ov.save_model(compressed, staged, compress_to_fp16=False)
    del compressed, model
    gc.collect()
    os.replace(staged, xml)
    os.replace(staged[:-4] + ".bin", xml[:-4] + ".bin")
    log.info("code predictor frame int8 %s", xml)


def _igpu_int8():
    """8-bit weights on the integrated GPU only. Discrete stays full precision."""
    return (os.environ.get("OLARES_GPU_MODE") or "").strip().lower() == "intel"


def _exporter_auto_int8(model_name):
    """Optimum int8-compresses a submodel at 1e9 parameters. 1.7B's talker crosses it; 0.6B does not."""
    import re
    match = re.search(r"(\d+(?:\.\d+)?)b", str(model_name).lower())
    if not match:
        return False
    return float(match.group(1)) >= 1.0


def _load_in_8bit(int8, model_name):
    """Pass the flag only when export will not already compress. A second compress_irs is refused."""
    if not int8:
        return False
    return not _exporter_auto_int8(model_name)


def _load():
    from optimum.intel import OVModelForTextToSpeechSeq2Seq

    path = tts._model_path()
    device = _device()
    cache = os.path.join(_hub_cache(), "ov-cache")
    os.makedirs(cache, exist_ok=True)
    int8 = _igpu_int8()
    load_8bit = _load_in_8bit(int8, tts.MODEL_NAME)
    log.info(
        "loading %s from %s on OpenVINO %s %s",
        tts.MODEL_NAME, path, device, "int8" if int8 else "fp32")
    ov = OVModelForTextToSpeechSeq2Seq.from_pretrained(
        path, export=True, device=device, local_files_only=True,
        cache_dir=_hub_cache(), token=tts.HF_TOKEN,
        load_in_8bit=load_8bit,
        ov_config={"CACHE_DIR": cache})
    kind = str(getattr(ov, "_tts_model_type", "") or "").strip().lower()
    speakers = []
    if kind == "custom_voice":
        speakers = list(ov.get_supported_speakers() or [])
    elif not kind:
        kind = "base"
    rate = int(getattr(ov, "sampling_rate", tts.OUT_SR) or tts.OUT_SR)
    _install_codec_buckets(ov)
    from . import tts_ov_frame
    tts_ov_frame.install(ov, path, device, cache, compress_frame_int8 if int8 else None)
    return (_OpenVINOQwen3TTS(ov, rate), speakers,
            kind == "custom_voice", kind == "voice_design")


def load_checkpoint(path, model_name):
    """Compile one checkpoint. The three-in-one image calls this per repo; _load stays the single-checkpoint path."""
    from optimum.intel import OVModelForTextToSpeechSeq2Seq

    device = _device()
    slug = "".join(ch if ch.isalnum() else "-" for ch in str(model_name))[:80].strip("-") or "model"
    cache = os.path.join(_hub_cache(), "ov-cache", slug)
    os.makedirs(cache, exist_ok=True)
    int8 = _igpu_int8()
    load_8bit = _load_in_8bit(int8, model_name)
    log.info(
        "loading %s from %s on OpenVINO %s %s",
        model_name, path, device, "int8" if int8 else "fp32")
    ov = OVModelForTextToSpeechSeq2Seq.from_pretrained(
        path, export=True, device=device, local_files_only=True,
        cache_dir=_hub_cache(), token=tts.HF_TOKEN,
        load_in_8bit=load_8bit,
        ov_config={"CACHE_DIR": cache})
    kind = str(getattr(ov, "_tts_model_type", "") or "").strip().lower()
    speakers = []
    if kind == "custom_voice":
        speakers = list(ov.get_supported_speakers() or [])
    elif not kind:
        kind = "base"
    rate = int(getattr(ov, "sampling_rate", tts.OUT_SR) or tts.OUT_SR)
    _install_codec_buckets(ov)
    from . import tts_ov_frame
    tts_ov_frame.install(ov, path, device, cache, compress_frame_int8 if int8 else None)
    return (_OpenVINOQwen3TTS(ov, rate), speakers,
            kind == "custom_voice", kind == "voice_design")


def build_app(supports):
    return tts.build_app(supports)


def run(supports):
    tts._load = _load
    tts.FRAME_BUDGET = False
    tts.run(supports)
