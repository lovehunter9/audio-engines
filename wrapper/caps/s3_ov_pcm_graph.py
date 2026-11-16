"""Export unchanged author acoustic networks and explicit streaming state to GPU."""
import gc
import hashlib
import json
import logging
import math
import os
import random
import sys
import threading
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .s3_ov_graphs import compile_gpu, export_module
from .s3_ov_source import REVISION, definitions


class _Registry:
    def __init__(self):
        self.model_classes = {}

    def register(self, category, name):
        if category != "model_classes":
            raise ValueError(category)

        def register(cls):
            self.model_classes[name] = cls
            return cls
        return register


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _asset(root, relative):
    path = root / "models" / relative
    manifest = root / "models/streaming/release.json"
    release = json.loads(manifest.read_text())
    record = next((item for item in release["files"] if item["path"] == "assets/" + relative), None)
    if record is None or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise RuntimeError("Original S3 acoustic release checksum mismatch: " + relative)
    return path, {"revision": REVISION, "release_manifest_sha256": _sha256(manifest),
                  "weight": record}


def _base():
    import torch
    from torch import nn
    from torch.nn import functional as F

    return {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
            "os": os, "random": random, "time": time, "logging": logging,
            "Dict": Dict, "Optional": Optional, "Tuple": Tuple, "tables": _Registry()}


def _dynamic_pad_mask(lengths):
    # The author's helper converts lengths to Python lists. This tensor form is
    # the same comparison, so a traced graph retains dynamic prefix lengths.
    import torch

    positions = torch.arange(lengths.max(), device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(-1)


def _dynamic_chunk_mask(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk,
                        decoding_chunk_size, static_chunk_size, num_decoding_left_chunks,
                        enable_full_context=True):
    import torch

    if use_dynamic_chunk or use_dynamic_left_chunk or decoding_chunk_size != 0:
        raise RuntimeError("Only the author's fixed streaming FM inference arm is exported")
    if static_chunk_size <= 0:
        return masks
    positions = torch.arange(xs.shape[1], device=xs.device)
    block = torch.div(positions, static_chunk_size, rounding_mode="trunc")
    allowed = positions.unsqueeze(0) < ((block + 1) * static_chunk_size).unsqueeze(1)
    if num_decoding_left_chunks >= 0:
        start = ((block - num_decoding_left_chunks) * static_chunk_size).clamp(min=0)
        allowed = allowed & (positions.unsqueeze(0) >= start.unsqueeze(1))
    return masks & allowed.unsqueeze(0)


def _load_state(module, path, *, fm=False):
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    elif isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if fm and any(key.startswith("fm_model.") for key in state):
        state = {key[len("fm_model."):]: value for key, value in state.items()
                 if key.startswith("fm_model.")}
    target = set(module.state_dict())
    old, new = "dit_model.rotary_embed.inv_freq", "dit_model.rotary_embed.freqs"
    if old in state and new in target and old not in target:
        state[new] = state.pop(old)
    ignored = len(state) - len(target.intersection(state))
    missing, unexpected = module.load_state_dict({key: value for key, value in state.items()
                                                 if key in target}, strict=False)
    parameters = set(dict(module.named_parameters()))
    missing_parameters = [key for key in missing if key in parameters]
    if missing_parameters or unexpected:
        raise RuntimeError("Original acoustic weight mismatch: " + repr((missing_parameters, unexpected)))
    return dict(missing_buffers=list(missing), ignored_checkpoint_entries=ignored,
                trainable_parameters=len(parameters), missing_parameters=[])


def _fm_author(root):
    import torch
    from einops import repeat
    from librosa.filters import mel
    from torch.utils.checkpoint import checkpoint
    from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

    base = _base()
    location = root / "vendor/FunASR/funasr/models/cosyvoice"
    rotary = definitions(root / "vendor/FunASR/x_transformers/x_transformers.py",
                         ["apply_rotary_pos_emb"], {**base, "apply_rotary_emb": apply_rotary_emb})
    modules = definitions(location / "modules/dit_flow_matching/dit_modules.py",
                          ["TimestepEmbedding", "CausalConvPositionEmbedding", "DiTBlock",
                           "AdaLayerNormZero_Final"],
                          {**base, "apply_rotary_pos_emb": rotary.apply_rotary_pos_emb})
    dit = definitions(location / "modules/dit_flow_matching/dit_model.py", ["DiT"],
                      {**base, "repeat": repeat, "checkpoint": checkpoint,
                       "RotaryEmbedding": RotaryEmbedding,
                       "TimestepEmbedding": modules.TimestepEmbedding,
                       "CausalConvPositionEmbedding": modules.CausalConvPositionEmbedding,
                       "DiTBlock": modules.DiTBlock,
                       "AdaLayerNormZero_Final": modules.AdaLayerNormZero_Final,
                       "add_optional_chunk_mask": _dynamic_chunk_mask})
    masks = definitions(root / "vendor/FunASR/funasr/models/transformer/utils/mask.py",
                        ["causal_block_mask"], base)
    dit.causal_block_mask = masks.causal_block_mask
    fm = definitions(location / "flow_matching_model.py", ["CosyVoiceFlowMatching"],
                     {**base, "librosa_mel_fn": mel,
                      "make_pad_mask": _dynamic_pad_mask,
                      "add_optional_chunk_mask": _dynamic_chunk_mask,
                      "dtype_map": {"fp32": torch.float32, "fp16": torch.float16,
                                    "bf16": torch.bfloat16}})
    # Only this isolated package exists; no funasr/vLLM/CUDA scheduler imports.
    fm.__package__ = fm.__name__
    fm.__path__ = []
    for suffix in (".modules", ".modules.dit_flow_matching"):
        package = types.ModuleType(fm.__name__ + suffix)
        package.__path__ = []
        sys.modules[package.__name__] = package
    sys.modules[fm.__name__ + ".modules.dit_flow_matching.dit_model"] = dit
    return fm, {"fm": fm._source_sha256, "dit": dit._source_sha256,
                "dit_modules": modules._source_sha256, "mask": masks._source_sha256,
                "rotary_shim": rotary._source_sha256}


def _vocoder_author(root):
    import torch
    import typing as tp
    from torch.nn.utils import weight_norm, remove_weight_norm

    base = {**_base(), "tp": tp, "weight_norm": weight_norm,
            "remove_weight_norm": remove_weight_norm}
    location = root / "vendor/FunASR/funasr/models/cosyvoice"
    helpers = definitions(location / "hifigan_module/__init__.py", ["init_weights", "get_padding"], base)
    activation = definitions(location / "hifigan_module/activations.py", ["Snake", "SnakeBeta"],
                             {**base, "Parameter": torch.nn.Parameter,
                              "sin": torch.sin, "pow": torch.pow})
    filters = definitions(location / "alias_free_activation/torch/filter.py",
                          ["LowPassFilter1d", "kaiser_sinc_filter1d"],
                          {**base, "sinc": torch.sinc})
    resample = definitions(location / "alias_free_activation/torch/resample.py",
                           ["UpSample1d", "DownSample1d"],
                           {**base, "LowPassFilter1d": filters.LowPassFilter1d,
                            "kaiser_sinc_filter1d": filters.kaiser_sinc_filter1d})
    antialias = definitions(location / "alias_free_activation/torch/act.py", ["Activation1d"],
                            {**base, "UpSample1d": resample.UpSample1d,
                             "DownSample1d": resample.DownSample1d})
    causal = definitions(location / "causal_hifigan.py",
                         ["LookRightConv1d", "LookLeftConv1d", "LookLeftConv1dWithDilation",
                          "LookLeftConvTranspose1d"], base)
    original = definitions(location / "causal_bigvgan.py", ["CausalBigVGan"],
                           {**base, "activations": activation,
                            "TorchActivation1d": antialias.Activation1d,
                            "init_weights": helpers.init_weights,
                            "get_padding": helpers.get_padding,
                            "LookRightConv1d": causal.LookRightConv1d,
                            "LookLeftConv1d": causal.LookLeftConv1d,
                            "LookLeftConv1dWithDilation": causal.LookLeftConv1dWithDilation,
                            "LookLeftConvTranspose1d": causal.LookLeftConvTranspose1d})
    return original, {"vocoder": original._source_sha256, "causal": causal._source_sha256,
                      "activation": activation._source_sha256, "filter": filters._source_sha256,
                      "resample": resample._source_sha256, "antialias": antialias._source_sha256}


class FlowMatching:
    def __init__(self, source_root, graph_root, config):
        import torch
        import yaml
        from torch import nn

        root = Path(source_root)
        if (root / "COMMIT").read_text().strip() != REVISION:
            raise RuntimeError("Original S3 revision mismatch")
        original, hashes = _fm_author(root)
        path = Path(config.get("fm_config") or root / "configs/fm_config.yaml")
        definition = yaml.safe_load(path.read_text().replace("${S3_ROOT}", str(root)))
        if "model" not in definition:
            definition = definition["model_conf"]["fm_conf"]
        if definition["model"] != "CosyVoiceFlowMatching":
            raise RuntimeError("The original streaming FM model is required")
        conf = dict(definition["model_conf"])
        conf["mel_feat_conf"] = {**conf["mel_feat_conf"], "device": "cpu"}
        fm = original.CosyVoiceFlowMatching(**conf).eval()
        weight, self.evidence = _asset(root, "flow_matching/model.pt")
        self.evidence.update(adapter_sha256=_sha256(__file__), source_sha256=hashes, config_sha256=_sha256(path),
                             weight_loading=_load_state(fm, weight, fm=True))
        fm = fm.half().eval()
        if fm.unified or fm.upsample_method != "repeat" or fm.attn_type != "v1":
            raise RuntimeError("Unexpected original streaming FM inference configuration")
        self.context_size = int(fm.context_size)
        self.feat_token_ratio = int(fm.feat_token_ratio)
        self.num_mels = int(fm.num_mels)
        self.chunk = int(config.get("fm_chunk", 8))
        self.graphs = {}
        self.lock = threading.RLock()
        # Match original get_rand_noise: CPU float32 RNG, seed 0, then fp16.
        # A dedicated generator retains that stream without changing Thinker RNG.
        self.noise = torch.randn((1, 50 * 300, self.num_mels),
                                 generator=torch.Generator(device="cpu").manual_seed(0)).half().numpy()

        class FullSolver(nn.Module):
            def __init__(self, module, final, chunk):
                super().__init__()
                self.fm = module
                self.final = final
                self.chunk = chunk

            def forward(self, codec, xvec, noise):
                self.fm.rand_noise = noise
                lengths = torch.ones((1,), dtype=torch.int64, device=codec.device) * codec.shape[1]
                return self.fm._inference(codec=codec, codec_lengths=lengths,
                    chunk_size=self.chunk, finalize=self.final, xvec=xvec,
                    xvec_lengths=torch.ones((1,), dtype=torch.int64, device=codec.device),
                    fm_dtype="fp16", infer_causal_mask_type=0)

        example_n = self.context_size + self.chunk
        example = (torch.zeros((1, example_n), dtype=torch.int64),
                   torch.ones((1, conf.get("xvec_size", 192)), dtype=torch.float16),
                   torch.from_numpy(self.noise.copy()))
        for final in (False, True):
            source = {**self.evidence, "finalize": final, "fm_chunk": self.chunk,
                      "solver": "original_solve_euler_10_steps_cfg_0.7_cosine",
                      "shape_adaptation": "tensor_pad_mask_vectorized_same_chunk_mask"}
            self.graphs[final] = export_module(FullSolver(fm, final, self.chunk).eval(), example,
                [[1, -1], [1, conf.get("xvec_size", 192)], [1, 15000, self.num_mels]],
                Path(graph_root) / ("final" if final else "stream"), source=source, precision="f16")
        self.evidence.update(execution_device="GPU", solver_steps=10, fm_chunk=self.chunk)
        del fm
        gc.collect()

    def _inference(self, *, codec, codec_lengths, chunk_size, finalize, xvec,
                   xvec_lengths, fm_dtype, infer_causal_mask_type):
        import torch

        n = int(codec.shape[1])
        if (chunk_size != self.chunk or fm_dtype != "fp16" or infer_causal_mask_type != 0
                or codec.shape[0] != 1 or int(codec_lengths[0]) != n
                or int(xvec_lengths[0]) != 1 or n * self.feat_token_ratio > self.noise.shape[1]
                or bool((codec < 0).any()) or bool((codec >= 59049).any())):
            raise ValueError("S3 FM input differs from the original discrete streaming contract")
        if not finalize and n <= self.context_size:
            raise ValueError("Nonfinal FM prefix must contain its genuine lookahead context")
        with self.lock:
            out = self.graphs[bool(finalize)]([codec.numpy(), xvec.numpy(), self.noise])[0]
            expected = (n if finalize else n - self.context_size) * self.feat_token_ratio
            if out.shape != (1, expected, self.num_mels) or not np.isfinite(out).all():
                raise RuntimeError("Invalid OpenVINO original FM output")
            return torch.from_numpy(np.asarray(out).copy())


def _flatten(tree):
    if isinstance(tree, dict):
        tensors, children = [], []
        for key, value in tree.items():
            flat, spec = _flatten(value)
            tensors.extend(flat)
            children.append((key, spec))
        return tensors, ("dict", children)
    if isinstance(tree, (list, tuple)):
        tensors, children = [], []
        for value in tree:
            flat, spec = _flatten(value)
            tensors.extend(flat)
            children.append(spec)
        return tensors, ("tuple" if isinstance(tree, tuple) else "list", children)
    return [tree], ("tensor", None)


def _unflatten(values, spec):
    kind, children = spec
    if kind == "tensor":
        return next(values)
    if kind == "dict":
        return {key: _unflatten(values, child) for key, child in children}
    result = [_unflatten(values, child) for child in children]
    return tuple(result) if kind == "tuple" else result


class _VocoderGPU:
    def __init__(self, generator, graph_root, evidence):
        import torch
        from torch import nn

        self.h = generator.h
        self.lock = threading.RLock()
        self.graphs = {}
        self.specializations = OrderedDict()
        example_feat = torch.zeros((1, self.h.num_mels, 32), dtype=torch.float32)
        with torch.inference_mode():
            _, initial = generator.inference_cache(example_feat, finalize=False, cache={})
        example_cache, spec = _flatten(initial)
        self.cache_shapes = [tuple(value.shape) for value in example_cache]
        self.cache_dtypes = [str(value.dtype) for value in example_cache]

        class CachedGenerator(nn.Module):
            def __init__(self, final, has_cache):
                super().__init__()
                self.generator = generator
                self.final, self.has_cache = final, has_cache

            def forward(self, feat, *cache):
                state = _unflatten(iter(cache), spec) if self.has_cache else {}
                wave, state = self.generator.inference_cache(feat, finalize=self.final, cache=state)
                leaves, _ = _flatten(state)
                return (wave, *leaves)

        for final in (False, True):
            for cached in (False, True):
                inputs = (example_feat, *example_cache) if cached else (example_feat,)
                shapes = [[1, self.h.num_mels, -1]] + (list(self.cache_shapes) if cached else [])
                source = {**evidence, "finalize": final, "has_cache": cached,
                          "cache_shapes": [list(shape) for shape in self.cache_shapes],
                          "cache_dtypes": self.cache_dtypes,
                          "method": "original_generator.inference_cache"}
                key = (final, cached)
                self.graphs[key] = export_module(CachedGenerator(final, cached).eval(), inputs,
                    shapes, Path(graph_root) / ("final" if final else "stream") / ("cached" if cached else "first"),
                    source=source, precision="f32", prepare_only=True)
        self.cache_count = len(example_cache)

    def inference_cache(self, feat, finalize=True, cache=None):
        import openvino as ov
        import torch

        cache = cache or {}
        cached = bool(cache)
        key = (bool(finalize), cached)
        if feat.shape[0] != 1 or feat.shape[1] != self.h.num_mels:
            raise ValueError("Invalid original vocoder feature shape")
        if not finalize and feat.shape[2] <= self.h.conv_pre_look_right:
            raise ValueError("Nonfinal vocoder needs genuine right context")
        with self.lock:
            specialization = (*key, int(feat.shape[2]))
            if specialization not in self.specializations:
                compiled = compile_gpu(self.graphs[key], input_shapes={0: list(feat.shape)})
                if len(compiled.outputs) != self.cache_count + 1:
                    raise RuntimeError("OpenVINO export omitted original vocoder cache outputs")
                self.specializations[specialization] = (compiled, [compiled.create_infer_request() for _ in range(2)], 0)
            compiled, requests, turn = self.specializations[specialization]
            request = requests[turn % 2]
            self.specializations[specialization] = (compiled, requests, turn + 1)
            self.specializations.move_to_end(specialization)
            if len(self.specializations) > 8:
                self.specializations.popitem(last=False)
            request.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(feat.numpy())))
            if cached:
                if len(cache["tensors"]) != self.cache_count:
                    raise RuntimeError("Incomplete original vocoder streaming cache")
                for index, tensor in enumerate(cache["tensors"], 1):
                    request.set_input_tensor(index, tensor)
            request.infer()
            wave = np.asarray(request.get_output_tensor(0).data).copy()
            frames = feat.shape[2] - (0 if finalize else self.h.conv_pre_look_right)
            expected = frames * int(np.prod(self.h.upsample_rates))
            if wave.shape != (1, expected) or not np.isfinite(wave).all():
                raise RuntimeError("Invalid original vocoder output shape or samples")
            tensors = tuple(request.get_output_tensor(i + 1) for i in range(self.cache_count))
            if any(tuple(value.shape) != shape for value, shape in zip(tensors, self.cache_shapes)):
                raise RuntimeError("Original vocoder streaming cache shape mismatch")
            state = {"tensors": tensors, "owner": request}
            return torch.from_numpy(wave), state


class Vocoder:
    def __init__(self, source_root, graph_root, config):
        import yaml

        root = Path(source_root)
        original, hashes = _vocoder_author(root)
        weight, self.evidence = _asset(root, "vocoder/model.pt")
        if config.get("vocoder_checkpoint") and Path(config["vocoder_checkpoint"]).resolve() != weight.resolve():
            raise ValueError("Only the verified original vocoder checkpoint is accepted")
        verified_config, _ = _asset(root, "vocoder/config.yaml")
        path = Path(config.get("vocoder_config") or verified_config)
        if path.resolve() != verified_config.resolve():
            raise ValueError("Only the verified original vocoder configuration is accepted")
        definition = yaml.safe_load(path.read_text())
        if definition["model"] != "CausalBigVGan":
            raise RuntimeError("Original streaming CausalBigVGan configuration required")
        model = original.CausalBigVGan(**definition["model_conf"]).float().eval()
        self.evidence.update(adapter_sha256=_sha256(__file__), source_sha256=hashes, config_sha256=_sha256(path),
                             weight_loading=_load_state(model, weight))
        self.sample_rate = int(model.sample_rate)
        model.generator.remove_weight_norm()
        self.generator = _VocoderGPU(model.generator, graph_root, self.evidence)
        self.evidence.update(execution_device="GPU", explicit_cache_tensors=self.generator.cache_count,
                             cache_shapes=[list(shape) for shape in self.generator.cache_shapes])
        del model
        gc.collect()
