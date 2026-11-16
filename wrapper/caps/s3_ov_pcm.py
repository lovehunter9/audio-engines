"""Author S3 PCM control with FM, BigVGAN and CAMPPlus on OpenVINO GPU."""
from collections import OrderedDict
import hashlib
import inspect
import json
import textwrap
import threading
import time
from pathlib import Path

import numpy as np

from .s3_ov_graphs import compile_gpu
from .s3_ov_source import REVISION, definitions


def _author_pcm(source_root, chunk):
    import torch
    import torchaudio

    source_root = Path(source_root)
    if (source_root / "COMMIT").read_text().strip() != REVISION:
        raise RuntimeError("S3 PCM requires the pinned original source")
    namespace = {"torch": torch, "torchaudio": torchaudio, "np": np,
                 "time": time, "threading": threading, "hashlib": hashlib,
                 "json": json, "Path": Path, "_S3_ROOT": source_root}
    author = definitions(source_root / "inference/runtime/pcm_runtime.py",
                         ["PCMRuntime", "Playback"], namespace)
    # This is the author's s3_grid_diagnostic override, applied without importing
    # the CUDA Thinker/Talker scheduler and its transport implementation.
    for name in ("on_codec", "decode"):
        old = inspect.unwrap(getattr(author.PCMRuntime, name))
        source = textwrap.dedent(inspect.getsource(old)).replace("@torch.inference_mode()\n", "")
        source = source.replace("n%15", "n%fm_chunk").replace("c+15", "c+fm_chunk")
        source = source.replace("(n-c)%15", "(n-c)%fm_chunk")
        source = source.replace("block=15*ratio", "block=fm_chunk*ratio")
        source = source.replace("//15)", "//fm_chunk)")
        source = source.replace("chunk_size=15,", "chunk_size=fm_chunk,")
        scope = {**old.__globals__, "fm_chunk": chunk}
        result = {}
        exec(compile(source, str(source_root / "inference/runtime/s3_grid_diagnostic.py"), "exec"), scope, result)
        setattr(author.PCMRuntime, name, torch.inference_mode()(result[name]))
    return author


class SourceSpeaker:
    """Uncropped author kaldi features; original CAMPPlus graph on GPU only."""

    def __init__(self, source_root):
        from .s3_ov_features import KaldiFeatures

        from .s3_ov_pcm_graph import _asset

        self.features = KaldiFeatures(Path(source_root)/"cache/openvino/speaker-features")
        path, self.evidence = _asset(Path(source_root), "speaker/campplus.onnx")
        self.path = path
        self.compiled_shapes = OrderedDict()
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def extract(self, wave):
        import torch

        wave = torch.as_tensor(np.asarray(wave, dtype=np.float32).copy()).reshape(1, -1)
        if wave.shape[1] < 400:
            raise ValueError("insufficient source prefix for speaker embedding")
        feat = self.features(wave.numpy())
        shape = tuple(feat.shape)
        compiled = self.compiled_shapes.get(shape)
        if compiled is None:
            compiled = compile_gpu(self.path, input_shapes={0: list(shape)})
            if len(compiled.inputs) != 1 or len(compiled.outputs) != 1:
                raise RuntimeError("Unexpected original CAMPPlus interface")
            self.compiled_shapes[shape] = compiled
            if len(self.compiled_shapes) > 8:
                self.compiled_shapes.popitem(last=False)
        self.compiled_shapes.move_to_end(shape)
        output = compiled([feat])[0]
        x = np.asarray(output, dtype=np.float32).reshape(1, -1).copy()
        if not np.isfinite(x).all() or np.linalg.norm(x) <= 0:
            raise RuntimeError("Invalid original CAMPPlus output")
        return torch.from_numpy(x), dict(
            samples=int(wave.shape[1]),
            source_prefix_sha256=hashlib.sha256(wave.numpy().tobytes()).hexdigest(),
            embedding_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            norm=float(np.linalg.norm(x)), execution_device="GPU",
            model_sha256=self.model_sha256)


class PCMRuntime:
    """Compatible with the final s3_syncgraphs_entry acoustic emission contract."""

    def __init__(self, root, device, config, *, source_root="/opt/s3", graph_root=None):
        import torch
        from .s3_ov_pcm_graph import FlowMatching, Vocoder

        if str(device).upper() != "GPU":
            raise ValueError("The S3 OpenVINO PCM neural device must be GPU")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.fm_chunk = int(config.get("fm_chunk", 8))
        if self.fm_chunk not in (1, 4, 7, 8, 15):
            raise ValueError("Unsupported author FM chunk arm")
        self.author = _author_pcm(source_root, self.fm_chunk)
        # Only integer control/slicing/output tensors use host Torch. Every
        # neural module below compiles with literal GPU and verifies devices.
        self.device = torch.device("cpu")
        graph_root = Path(graph_root or Path(source_root) / "cache/openvino/pcm")
        self.speaker = SourceSpeaker(source_root)
        self.fm = FlowMatching(source_root, graph_root / "fm", self.config)
        self.voc = Vocoder(source_root, graph_root / "vocoder", self.config)
        self.sr = self.voc.sample_rate
        self.play = self.author.Playback()
        self.active = False

    def begin(self, index):
        self.author.PCMRuntime.begin(self, index)
        self.speaker_refreshes = []
        self._speaker_refresh_lock = threading.RLock()
        self.audit = []

    def prefix(self, wave, available):
        return self.author.PCMRuntime.prefix(self, wave, available)

    def fix_speaker(self):
        import torch

        with self._speaker_refresh_lock:
            available = int(self.available)
            started = time.monotonic()
            emb, info = self.speaker.extract(self.wave[:available])
            if info["samples"] != available:
                raise RuntimeError("Speaker extraction did not use the complete consumed prefix")
            info.update(available_source_ms=available / 16.,
                        refresh_index=len(self.speaker_refreshes),
                        wall_s=time.monotonic() - self.play.start,
                        extraction_s=time.monotonic() - started)
            self.emb = emb.to(dtype=torch.float16)
            self.speaker_info = info
            self.speaker_refreshes.append(dict(info))

    def on_codec(self, token):
        return self.author.PCMRuntime.on_codec(self, token)

    def decode(self, final, prefix_n=None):
        with self._speaker_refresh_lock:
            started = time.monotonic()
            result = self.author.PCMRuntime.decode(self, final, prefix_n=prefix_n)
            self.audit.append(dict(kind="pcm", source_ms=self.source_ms,
                                   wall_s=time.monotonic() - self.play.start,
                                   seconds=time.monotonic() - started,
                                   finalize=bool(final)))
            return result

    def gate(self):
        return self.author.PCMRuntime.gate(self)

    def finish(self, key, truncated=False, allow_empty=True):
        # Final curve explicitly preserves genuine empty outputs. The inherited
        # finish retains unflushed-envelope failure and writes original PCM16.
        result = self.author.PCMRuntime.finish(self, key, truncated=truncated, allow_empty=True)
        result["speaker_refreshes"] = self.speaker_refreshes
        result["speaker_protocol"] = "all_consumed_prefix_each_nonempty_write_no_crop"
        if len(self.speaker_refreshes) != len(self.text_events):
            raise RuntimeError("Each nonempty WRITE must refresh the source speaker")
        for refresh, text in zip(self.speaker_refreshes, self.text_events):
            if abs(refresh["available_source_ms"] - text["source_ms"]) >= .001:
                raise RuntimeError("Speaker refresh and WRITE consumed-prefix mismatch")
        result["latency_audit"] = self.audit
        result["diagnostic_contract"] = dict(deliver_read_hidden=True, fm_chunk=self.fm_chunk,
                                             model_unchanged=True)
        result["neural_runtime"] = "OpenVINO GPU"
        result["acoustic_models"] = {"flow_matching": self.fm.evidence,
                                     "vocoder": self.voc.evidence,
                                     "speaker": self.speaker.evidence}
        (self.root / key / "trace.json").write_text(json.dumps(result, indent=2))
        return result
