"""Author audio feature equations exported to OpenVINO GPU."""
import hashlib
from pathlib import Path

import numpy as np

from .s3_ov_graphs import export_module
from .ov_gpu_graph import infer_gpu
from .s3_ov_source import frontend_definitions


class WhisperFeatures:
    def __init__(self, root, target):
        import torch
        from torch import nn

        native, source_hash = frontend_definitions(root)
        class Features(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("window", torch.hann_window(native.win_length))
                self.register_buffer("filters", native.mel_filters("cpu", native.n_mels, native.filters_path))

            def forward(self, wave, lengths):
                spec = torch.stft(wave, native.n_fft, native.hop_length,
                                  window=self.window, return_complex=True)
                power = spec[..., :-1].abs().square()
                mel = (self.filters @ power).clamp(min=1e-10).log10()
                floor = mel.new_full((wave.shape[0], 1, 1), 1.3904691073 - 8.)
                return (torch.maximum(mel, floor)+4.)/4., lengths//native.hop_length

        self.compiled = export_module(Features().eval(),
            (torch.zeros(1, 16000), torch.tensor([16000], dtype=torch.int64)),
            [[1, -1], [1]], target,
            source={"original_frontend": source_hash,
                    "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def __call__(self, wave):
        samples = np.ascontiguousarray(wave, dtype=np.float32).reshape(1, -1)
        result = infer_gpu(self.compiled, [samples, np.array([samples.shape[1]], dtype=np.int64)], release=True)
        return np.asarray(result[0]).copy(), np.asarray(result[1]).copy()


class KaldiFeatures:
    def __init__(self, target, *, center=True):
        import torch
        from torch import nn
        from torchaudio.compliance import kaldi

        class Features(nn.Module):
            def __init__(self):
                super().__init__()
                window = kaldi._feature_window_function("povey", 400, .42, "cpu", torch.float32)
                banks, _ = kaldi.get_mel_banks(80, 512, 16000, 20., 0., 100., -500., 1.)
                self.register_buffer("window", window)
                self.register_buffer("banks", torch.nn.functional.pad(banks, (0, 1)))
                phase = -2*torch.pi*torch.arange(512, dtype=torch.float64)[:, None]*torch.arange(257, dtype=torch.float64)[None, :]/512
                self.register_buffer("cosine", phase.cos().float())
                self.register_buffer("sine", phase.sin().float())

            def forward(self, wave):
                count = 1+(wave.shape[1]-400)//160
                indices = torch.arange(count, device=wave.device)[:, None]*160 + torch.arange(400, device=wave.device)[None, :]
                frames = wave[0, indices]
                frames = frames-frames.mean(dim=1, keepdim=True)
                previous = torch.cat((frames[:, :1], frames[:, :-1]), dim=1)
                frames = (frames-.97*previous)*self.window
                padded = torch.nn.functional.pad(frames, (0, 112))
                spectrum = (padded @ self.cosine).square()+(padded @ self.sine).square()
                feature = (spectrum @ self.banks.T).clamp(min=torch.finfo(torch.float32).eps).log()
                return (feature-feature.mean(dim=0, keepdim=True) if center else feature).unsqueeze(0)

        self.compiled = export_module(Features().eval(), (torch.zeros(1, 16000),), [[1, -1]], target,
            source={"method": "original_kaldi_fbank_80_dither0_16k_explicit_frame_gather_gpu_dft", "center": center,
                    "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def __call__(self, wave):
        return np.asarray(infer_gpu(self.compiled, [np.ascontiguousarray(wave, dtype=np.float32).reshape(1, -1)])[0]).copy()
