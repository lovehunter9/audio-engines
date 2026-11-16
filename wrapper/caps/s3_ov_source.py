"""Load unchanged author neural definitions without importing the CUDA scheduler."""
import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path


REVISION = "1ca52ea064ec2081f12675d853525d394571bf73"


def definitions(path, names, namespace):
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    symbols = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols[target.id] = node
    selected = set()

    def include(name):
        if name in selected or name in namespace:
            return
        if name not in symbols:
            return
        selected.add(name)
        for child in ast.walk(symbols[name]):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                include(child.id)

    for name in names:
        if name not in symbols:
            raise RuntimeError("Required original definition is absent: " + name)
        include(name)
    nodes = [node for node in tree.body if any(symbols[name] is node for name in selected)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module_ast = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module_name = "s3_ov_author_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module = types.ModuleType(module_name)
    module.__dict__.update(namespace)
    module.__file__ = str(path)
    module.__spec__ = importlib.util.spec_from_file_location(module_name, path)
    sys.modules[module_name] = module
    exec(compile(module_ast, str(path), "exec"), module.__dict__)
    module._source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return module


def neural_definitions(root):
    import logging
    import math
    import os
    from types import MethodType
    from typing import Optional, Tuple
    import numpy as np
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from transformers.activations import ACT2FN

    root = Path(root)
    if (root / "COMMIT").read_text().strip() != REVISION:
        raise RuntimeError("The OpenVINO adapter requires the pinned original S3 source")
    models = root / "vendor/vllm/model_executor/models"
    base = {"torch": torch, "nn": nn, "F": F, "math": math, "os": os, "np": np,
            "Tensor": Tensor, "Optional": Optional, "Tuple": Tuple, "ACT2FN": ACT2FN,
            "logger": logging.getLogger("s3-openvino"), "flash_attn_varlen_func": None}
    encoder = definitions(models / "omnidrca2_encoder.py", ["SenseVoiceEncoder", "apply_rotary_pos_emb"], base)
    adaptor = definitions(models / "omnidrca2_next_thinker.py",
                          ["Transformer", "encoder_conf", "audio_adaptor_conf", "load_pretrained_module"],
                          {**base, "flash_attn": None})
    streaming = definitions(root / "inference/streaming/true_streaming_encoder.py",
                            ["enable_true_streaming_encoder", "enable_true_streaming_adaptor"],
                            {**base, "MethodType": MethodType,
                             "apply_rotary_pos_emb": encoder.apply_rotary_pos_emb})
    return encoder, adaptor, streaming


def load_audio_encoder(root, model_dir):
    import json
    from torch import nn

    original, adaptor, streaming = neural_definitions(root)
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    encoder = original.SenseVoiceEncoder(**adaptor.encoder_conf).float().eval()
    projector = adaptor.Transformer(**{**adaptor.audio_adaptor_conf,
        "encoder_dim": adaptor.encoder_conf["linear_units"],
        "llm_dim": config["text_config"]["hidden_size"]}).float().eval()
    counts = {}
    counts["encoder"] = adaptor.load_pretrained_module(
        str(model_dir / "audio_encoder.pt"), encoder, remove_prefix="audio_encoder.")
    counts["adaptor"] = adaptor.load_pretrained_module(
        str(model_dir / "audio_adaptor.pt"), projector, remove_prefix="audio_adaptor.")
    streaming.enable_true_streaming_encoder(encoder, chunk_ms=640)
    streaming.enable_true_streaming_adaptor(projector, encoder.streaming_chunk_frames)

    class AudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.adaptor = encoder, projector

        def forward(self, features, lengths):
            encoded, encoded_lengths = self.encoder(features, lengths)
            return self.adaptor(encoded, encoded_lengths)

    manifest = model_dir / "release.json"
    if not manifest.is_file():
        manifest = Path(root) / ".model-installed"
    if not manifest.is_file():
        raise RuntimeError("The verified S3 release manifest is missing")
    return AudioEncoder().eval(), {"weight_loading": {key: list(value) for key, value in counts.items()},
        "release_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "original_source_sha256": {"encoder": original._source_sha256,
                                   "adaptor": adaptor._source_sha256,
                                   "streaming": streaming._source_sha256}}


def frontend_definitions(root):
    import math
    import os
    from typing import Tuple
    import numpy as np
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence

    root = Path(root)
    base = {"torch": torch, "nn": nn, "np": np, "os": os, "math": math,
            "Tuple": Tuple, "pad_sequence": pad_sequence}
    frontend = definitions(root / "vendor/vllm/model_executor/models/whisper.py",
                           ["WhisperFrontend"], base)
    calibrated = definitions(root / "inference/streaming/streaming_core.py",
                             ["FixedMaxWhisperFrontend"],
                             {**base, "VllmWhisperFrontend": frontend.WhisperFrontend})
    return calibrated.FixedMaxWhisperFrontend(
        fs=16000, n_mels=128, do_pad_trim=False,
        filters_path=str(root / "models/frontend/mel_filters.npz")), calibrated._source_sha256
