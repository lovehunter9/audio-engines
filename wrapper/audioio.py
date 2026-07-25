# Audio decoding shared by the torch-based caps (vad / diar / embed / enhance).
# Everything goes through soundfile first and falls back to torchaudio, then to
# the stdlib wave reader: the pyannote image ships a broken torchcodec
# AudioDecoder that torchaudio.load reaches for on some inputs, and feeding
# pyannote an in-memory waveform sidesteps it entirely.
import io


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
