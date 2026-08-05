# Shared audio conversion and decoding; optional numeric dependencies are imported on demand.
import io
import os
import tempfile


def spill(data, filename=None, default_suffix=".wav"):
    # Upload bytes -> a temp path the models can open. Blocking, so handlers thread it.
    suffix = os.path.splitext(filename or "")[1] or default_suffix
    # The prefix is what lets a restart recognise and sweep whatever a crash stranded here.
    with tempfile.NamedTemporaryFile(prefix="upload-", suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name


def unlink(path):
    # spill's counterpart: the upload is gone the moment the job that owned it ends.
    try:
        os.unlink(path)
    except Exception:
        pass


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
