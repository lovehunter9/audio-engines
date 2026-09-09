# Shared audio conversion and decoding; optional numeric dependencies are imported on demand.
import io
import logging
import os
import tempfile

log = logging.getLogger("audio-io")


# What lets a restart recognise and sweep whatever a crash stranded in the temp directory.
SPILL_PREFIX = "upload-"

# One read of an upload being streamed to disk. Big enough that the syscalls do not dominate,
# small enough that the resident cost of a request does not depend on the size of its upload.
SPILL_CHUNK = 1 << 20


def spill(data, filename=None, default_suffix=".wav"):
    # Upload bytes -> a temp path the models can open. Blocking, so handlers thread it.
    suffix = os.path.splitext(filename or "")[1] or default_suffix
    with tempfile.NamedTemporaryFile(prefix=SPILL_PREFIX, suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name


async def spill_upload(file, max_bytes=None, default_suffix=".wav"):
    """Stream an upload to a temp path, without ever holding the whole of it.

    Returns (path, size, over_limit). `over_limit` says the upload was longer than max_bytes;
    the partial file is already gone in that case and path is None, so the caller only has to
    decide what to answer.

    The alternative, `await file.read()`, resolves to the entire body in memory before a single
    bound can be applied to it -- the check would run after the damage. Starlette has already
    spooled a large upload to its own temp file by then, so this copies from disk to disk.
    """
    suffix = os.path.splitext(getattr(file, "filename", "") or "")[1] or default_suffix
    f = tempfile.NamedTemporaryFile(prefix=SPILL_PREFIX, suffix=suffix, delete=False)
    size = 0
    try:
        while True:
            chunk = await file.read(SPILL_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                f.close()
                unlink(f.name)
                return None, size, True
            f.write(chunk)
    finally:
        if not f.closed:
            f.close()
    return f.name, size, False


def unlink(path):
    # spill's counterpart: the upload is gone the moment the job that owned it ends.
    try:
        os.unlink(path)
    except Exception:
        pass


def seconds(samples, sample_rate):
    """Duration of an already-decoded buffer, mono (time,) or (channels, time)."""
    import numpy as np

    a = np.asarray(samples)
    if not a.ndim or not sample_rate:
        return None
    return round(int(a.shape[-1]) / float(sample_rate), 3)


def wav_seconds(data):
    """Duration of RIFF WAV bytes, from the header alone, or None if they are not one.

    For a caller that produced the WAV itself and runs in an image without
    soundfile; probe_seconds is the general form and prefers soundfile.
    """
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            if wf.getframerate():
                return round(float(wf.getnframes()) / float(wf.getframerate()), 3)
    except Exception:
        pass
    return None


def probe_seconds(src):
    """Duration in seconds without decoding the clip, or None when nothing here can read it.

    For caps that hand the file straight to a model and so never hold samples of their own.
    None is a real answer: billing reports "not measured" rather than a guess.
    """
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    why = None
    try:
        import soundfile as sf

        info = sf.info(buf)
        if info.samplerate:
            return round(float(info.frames) / float(info.samplerate), 3)
    except Exception as e:
        why = e
    if isinstance(buf, io.BytesIO):
        buf.seek(0)
    try:
        import wave

        with wave.open(buf, "rb") as wf:
            if wf.getframerate():
                return round(float(wf.getnframes()) / float(wf.getframerate()), 3)
    except Exception as e:
        why = why or e
    # The caller goes on to process this audio and now cannot report how much
    # of it there was, so the call bills as unmeasured. Silent is the one thing
    # that must not happen: underbilling has no other symptom.
    log.warning("could not probe audio duration, the call will bill as unmeasured: %s", why)
    return None


def pcm16_to_float32(data):
    import numpy as np

    if not data:
        return np.zeros((0,), dtype="float32")
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0


def resample_linear(samples, source_rate, target_rate=16000):
    import numpy as np

    if source_rate == target_rate or samples.shape[0] == 0:
        return samples.astype("float32", copy=False)
    duration = samples.shape[0] / float(source_rate)
    target_length = round(duration * target_rate)
    if target_length <= 0:
        return np.zeros((0,), dtype="float32")
    source_points = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_points = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_points, source_points, samples).astype("float32")


def decode(src):
    # src: a path or raw bytes -> ((channels, time) float32 torch tensor, sample_rate).
    import torch

    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    try:
        import soundfile as sf

        data, sr = sf.read(buf, dtype="float32", always_2d=True)  # (time, channels)
        return torch.from_numpy(data.T.copy()), int(sr)
    except Exception:
        pass
    if isinstance(buf, io.BytesIO):
        buf.seek(0)
    try:
        import torchaudio

        wav, sr = torchaudio.load(buf)  # (channels, time)
        return wav.float(), int(sr)
    except Exception:
        pass
    import wave

    import numpy as np

    if isinstance(buf, io.BytesIO):
        buf.seek(0)
    with wave.open(buf, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if sw == 2:
        a /= 32768.0
    elif sw == 4:
        a /= 2147483648.0
    else:
        a = (a - 128.0) / 128.0
    a = a.reshape(-1, ch).T
    return torch.from_numpy(a.copy()), int(sr)


def mono(wav, sr, target_sr):
    # (channels, time) at sr -> (1, time) mono at target_sr.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sr):
        import torchaudio

        wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
    return wav


def decode_mono(src, target_sr):
    wav, sr = decode(src)
    return mono(wav, sr, target_sr)
