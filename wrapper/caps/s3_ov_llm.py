"""Original S3 Thinker graphs, with neural execution restricted to OpenVINO GPU.

Torch is used only to load/export the original weights. Tokenization, placeholder
expansion, greedy choice, and streaming timeline bookkeeping stay on the host.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .s3_ov_graphs import compile_gpu

ADAPTER_VERSION = 1
REVISION = "1ca52ea064ec2081f12675d853525d394571bf73"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(model_dir, files):
    model_dir = Path(model_dir)
    manifest = model_dir / "release.json"
    if not manifest.is_file():
        raise RuntimeError("The verified original S3 release.json is required")
    release = json.loads(manifest.read_text())
    if release.get("format_version") != 1 or release.get("mode") != "streaming":
        raise RuntimeError("S3 Thinker/Talker require the original streaming release")
    records = {record["path"]: record for record in release["files"]}
    hashes = {}
    for name in files:
        path = model_dir / name
        if name not in records or not path.is_file():
            raise RuntimeError("Required original S3 weight/config is absent: " + name)
        record = records[name]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError("Original S3 release checksum mismatch: " + name)
        hashes[name] = record["sha256"]
    return {"adapter_version": ADAPTER_VERSION, "original_revision": REVISION,
            "release_manifest_sha256": _sha256(manifest), "files": hashes}


def _check_cache(graph, source):
    graph = Path(graph)
    if not graph.is_file():
        return False
    metadata = graph.with_suffix(".source.json")
    if not metadata.is_file() or json.loads(metadata.read_text()) != source:
        raise RuntimeError("S3 graph cache has a different or missing source identity: " + str(graph))
    if not graph.with_suffix(".bin").is_file():
        raise RuntimeError("S3 OpenVINO graph has no weights: " + str(graph))
    return True


def _save_graph(model, graph, source):
    import openvino as ov

    graph = Path(graph)
    graph.parent.mkdir(parents=True, exist_ok=True)
    model.validate_nodes_and_infer_types()
    ov.save_model(model, str(graph), compress_to_fp16=False)
    graph.with_suffix(".source.json").write_text(json.dumps(source, indent=2))


def _numpy(value):
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise ValueError("S3 OpenVINO inputs must be CPU tensors or NumPy arrays")
        # BF16 tensors cannot expose NumPy storage; float() preserves their exact values.
        value = value.detach().float().numpy()
    result = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite S3 neural input")
    return result


def _weight(path, *, keys=None):
    import torch

    value = torch.load(str(path), map_location="cpu", weights_only=True)
    if keys is not None:
        if not isinstance(value, dict) or set(value) != set(keys):
            raise RuntimeError("Unexpected original weight keys: " + str(path))
        return {key: _numpy(value[key]) for key in keys}
    if isinstance(value, dict):
        if set(value) != {"weight"}:
            raise RuntimeError("Expected exactly one bias-free weight: " + str(path))
        value = value["weight"]
    result = _numpy(value)
    if result.ndim != 2:
        raise RuntimeError("Expected a matrix in " + str(path))
    return result



def _export_causal(model_dir, target, role, files=None):
    import subprocess
    import sys

    command = [sys.executable, "/opt/s3_adapter/export_causal.py", "--role", role,
               "--model", str(model_dir), "--output", str(target)]
    if files is not None:
        command.extend(["--files", json.dumps(files)])
    subprocess.run(command, check=True)
    graph = Path(target) / "openvino_model.xml"
    if not graph.is_file() or not graph.with_suffix(".bin").is_file():
        raise RuntimeError("Original S3 graph conversion did not produce weights")
    return graph


def _export_thinker(model_dir, target, source):
    graph = Path(target) / "openvino_model.xml"
    if _check_cache(graph, source):
        return graph
    graph = _export_causal(model_dir, target, "thinker")
    graph.with_suffix(".source.json").write_text(json.dumps(source, indent=2))
    return graph


def _embedding(model, width):
    candidates = [node for node in model.get_ordered_ops()
                  if node.get_type_name() == "Gather" and ".embed_tokens/" in node.get_friendly_name()
                  and node.output(0).get_partial_shape().rank.get_length() == 3]
    if len(candidates) != 1:
        raise RuntimeError("Cannot identify the original token embedding Gather")
    output = candidates[0].output(0)
    if output.get_partial_shape()[-1].get_length() != width:
        raise RuntimeError("Original S3 embedding width disagrees with its config")
    return output


def _head(model, width):
    candidates = [node for node in model.get_ordered_ops()
                  if node.get_type_name() == "MatMul" and ".lm_head/" in node.get_friendly_name()]
    if len(candidates) != 1:
        raise RuntimeError("Cannot identify the original S3 language-model head")
    hidden = candidates[0].input_value(0)
    if hidden.get_partial_shape().rank.get_length() != 3 or hidden.get_partial_shape()[-1].get_length() != width:
        raise RuntimeError("The S3 head does not consume complete final normalized hidden states")
    return candidates[0], hidden


def _last(hidden):
    from openvino import opset13 as ops

    return ops.gather(hidden, np.array([-1], dtype=np.int64), np.int64(1))


def _override_embeddings(model, width):
    import openvino as ov
    from openvino import opset13 as ops

    embedding = _embedding(model, width)
    consumers = list(embedding.get_target_inputs())
    override = ops.parameter([-1, -1, width], ov.Type.f32, name="embedding_override")
    mask = ops.parameter([-1, -1, 1], ov.Type.boolean, name="embedding_override_mask")
    original = ops.convert(embedding, ov.Type.f32)
    selected = ops.select(mask, override, original)
    selected = ops.convert(selected, embedding.get_element_type())
    # Capture consumers before constructing Select, so Select never rewires itself.
    for consumer in consumers:
        consumer.replace_source_output(selected.output(0))
    model.add_parameters([override, mask])
    return selected.output(0)


def _fusion(model_dir, config):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "thinker_layer_fusion.json").read_text())
    text = config["text_config"]
    expected_layers = list(range(4, int(text["num_hidden_layers"]) + 1, 4))
    if (metadata.get("version") != 1
        or metadata.get("hidden_convention") != "intermediate_post_block_final_post_norm"
        or metadata.get("accumulation_dtype") != "float32"
        or metadata.get("talker_profile", "segmentpad") != "continuous"
        or metadata.get("layers") != expected_layers
        or metadata.get("hidden_size") != int(text["hidden_size"])
        or config.get("thinker_layer_fusion") != metadata
        or metadata.get("weights_sha256") != _sha256(model_dir / "thinker_layer_fusion.pt")):
        raise RuntimeError("Original S3 streaming layer-fusion metadata is inconsistent")
    weights = _weight(model_dir / "thinker_layer_fusion.pt", keys=["weights"])["weights"]
    if weights.shape != (len(expected_layers), int(text["hidden_size"])):
        raise RuntimeError("Original S3 layer-fusion weight shape is inconsistent")
    return metadata, weights


def _fused_hidden(model, base_ir, last_hidden, metadata, weights):
    from openvino import opset13 as ops

    # Optimum records post-block outputs and the post-norm final output explicitly.
    record = ET.parse(base_ir).getroot().find("./rt_info/hidden_states_decoder_layers")
    if record is None:
        raise RuntimeError("The official Optimum IR is missing decoder-layer producer metadata")
    mapping = json.loads(record.attrib["value"])["layers"]
    nodes = {node.get_friendly_name(): node for node in model.get_ordered_ops()}
    combined = None
    layers = metadata["layers"]
    for index, number in enumerate(layers):
        producer = mapping.get(str(number - 1))
        if not producer or producer["producer"] not in nodes:
            raise RuntimeError("Original S3 fusion layer is absent from IR: " + str(number))
        hidden = nodes[producer["producer"]].output(int(producer["output_index"]))
        if number == layers[-1] and hidden != last_hidden:
            raise RuntimeError("Original S3 fusion final layer is not the final normalized hidden")
        if hidden.get_partial_shape() != last_hidden.get_partial_shape():
            raise RuntimeError("Original S3 fusion layer shape mismatch")
        term = ops.multiply(ops.convert(hidden, "f32"), ops.constant(weights[index]))
        combined = term if combined is None else ops.add(combined, term)
    return combined


def _gate_logits(hidden, model_dir, width):
    import torch
    from openvino import opset13 as ops

    saved = torch.load(str(Path(model_dir) / "streaming_gate.pt"), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise TypeError("Original streaming gate checkpoint is malformed")
    keys = {"net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias", "net.4.weight", "net.4.bias"}
    state = saved["state_dict"]
    if set(state) != keys:
        raise RuntimeError("Original streaming gate has unexpected weight keys")
    state = {key: _numpy(value) for key, value in state.items()}
    shapes = {"net.0.weight": (width,), "net.0.bias": (width,), "net.1.weight": (256, width),
              "net.1.bias": (256,), "net.4.weight": (2, 256), "net.4.bias": (2,)}
    if any(state[key].shape != shape for key, shape in shapes.items()):
        raise RuntimeError("Original streaming gate weight shape mismatch")
    norm = ops.mvn(ops.convert(_last(hidden), "f32"), np.array([-1], dtype=np.int64),
                   normalize_variance=True, eps=1e-5, eps_mode="INSIDE_SQRT")
    norm = ops.add(ops.multiply(norm, state["net.0.weight"]), state["net.0.bias"])
    first = ops.add(ops.matmul(norm, state["net.1.weight"], False, True), state["net.1.bias"])
    silu = ops.multiply(first, ops.sigmoid(first))
    return ops.add(ops.matmul(silu, state["net.4.weight"], False, True), state["net.4.bias"])


def _add_output(model, output, name):
    from openvino import opset13 as ops

    if hasattr(output, "output"):
        output = output.output(0)
    # Identity keeps original shared-tensor names (especially logits) intact.
    named = ops.convert(output, output.get_element_type())
    named.output(0).get_tensor().set_names({name})
    model.add_outputs([named.output(0)])


def _adapt_thinker(base_ir, target, model_dir, config, source, residual):
    import openvino as ov
    from openvino import opset13 as ops

    target = Path(target)
    if _check_cache(target, source):
        return target
    model = ov.Core().read_model(str(base_ir))
    width = int(config["text_config"]["hidden_size"])
    head, hidden = _head(model, width)
    embedded = _override_embeddings(model, width)
    metadata, weights = _fusion(model_dir, config)
    mixed = _fused_hidden(model, base_ir, hidden, metadata, weights)
    projection_weight = _weight(Path(model_dir) / "pre_matching.pt")
    if projection_weight.shape[1] != width:
        raise RuntimeError("Original S3 pre_matching input width mismatch")
    projection_input = ops.add(mixed, embedded) if residual else mixed
    projected = ops.matmul(projection_input, projection_weight, False, True)
    gate = _gate_logits(hidden, model_dir, width)
    head.set_argument(0, _last(hidden).output(0))
    _add_output(model, hidden, "final_hidden")
    _add_output(model, projected, "projected_hidden")
    _add_output(model, gate, "gate_logits")
    _save_graph(model, target, source)
    del model
    gc.collect()
    return target


class StatefulGPU:
    """One serial, batch-one OpenVINO request with persistent KV/SSM state."""

    def __init__(self, path, width):
        self.compiled = compile_gpu(path, precision="f32")
        self.request = self.compiled.create_infer_request()
        self.width = int(width)
        self.inputs = {port.get_any_name() for port in self.compiled.inputs}
        expected = {"input_ids", "attention_mask", "position_ids", "beam_idx",
                    "embedding_override", "embedding_override_mask"}
        if self.inputs - expected or "embedding_override" not in self.inputs:
            raise RuntimeError("Unexpected stateful S3 OpenVINO input signature: " + repr(self.inputs))
        if not self.request.query_state():
            raise RuntimeError("S3 autoregressive execution requires stateful OpenVINO KV/SSM caches")
        self.outputs = {}
        known_outputs = {"logits", "final_hidden", "projected_hidden", "gate_logits"}
        for port in self.compiled.outputs:
            names = port.get_names() & known_outputs
            if len(names) != 1:
                raise RuntimeError("Unexpected S3 OpenVINO output aliases: " + repr(port.get_names()))
            name = names.pop()
            if name in self.outputs:
                raise RuntimeError("Duplicate S3 OpenVINO output: " + name)
            self.outputs[name] = port
        self.reset()

    def reset(self):
        for state in self.request.query_state():
            state.reset()
        self.position = 0

    def infer(self, token_ids, *, embeddings=None, mask=None):
        ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
        length = ids.shape[1]
        if length == 0:
            raise ValueError("Cannot infer an empty S3 token sequence")
        if embeddings is None:
            embeddings = np.zeros((1, length, self.width), dtype=np.float32)
            mask = np.zeros((1, length, 1), dtype=np.bool_)
        else:
            embeddings = _numpy(embeddings).reshape(1, length, self.width)
            mask = (np.ones((1, length, 1), dtype=np.bool_) if mask is None
                    else np.asarray(mask, dtype=np.bool_).reshape(1, length, 1))
        values = {"input_ids": ids,
                  "attention_mask": np.ones((1, self.position + length), dtype=np.int64),
                  "position_ids": np.arange(self.position, self.position + length, dtype=np.int64)[None],
                  "beam_idx": np.array([0], dtype=np.int32), "embedding_override": embeddings,
                  "embedding_override_mask": mask}
        try:
            self.request.infer({key: values[key] for key in self.inputs}, share_inputs=False)
            # InferRequest output buffers are reused by the next continuation.
            result = {name: np.array(self.request.get_tensor(port).data, copy=True)
                      for name, port in self.outputs.items()}
            self.position += length
            return result
        except Exception:
            self.reset()
            raise


class Thinker:
    def __init__(self, model_dir, cache_dir, *, base_ir=None, max_new_tokens=256,
                 max_codec_tokens=2048, blank_count=5):
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.width = int(self.config["text_config"]["hidden_size"])
        self.max_new_tokens = int(max_new_tokens)
        self.max_codec_tokens, self.blank_count = int(max_codec_tokens), int(blank_count)
        if self.max_new_tokens <= 0 or self.max_codec_tokens <= 0 or self.blank_count != 5:
            raise ValueError("Invalid original S3 generation/alignment policy")
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        files = ["config.json", "model.safetensors.index.json", *sorted(set(index["weight_map"].values())),
                 "pre_matching.pt", "streaming_gate.pt", "thinker_layer_fusion.json", "thinker_layer_fusion.pt",
                 "tokenizer.json", "tokenizer_config.json"]
        source = _source(self.model_dir, files)
        residual = os.environ.get("OMNIDRCA2_TALKER_ADD_INPUT_EMBEDDING", "0").lower() in ("1", "true", "yes", "on")
        cache_dir = Path(cache_dir)
        export_source = {**source, "role": "original_dense_qwen3_5_causal_lm"}
        if base_ir is None:
            base_ir = _export_thinker(self.model_dir, cache_dir / "thinker-base", export_source)
            base_identity = {"export_source": export_source}
        else:
            base_ir = Path(base_ir)
            if base_ir.is_dir():
                base_ir = base_ir / "openvino_model.xml"
            # Explicit reuse records the exact supplied IR; provenance is the caller's responsibility.
            base_identity = {"caller_supplied_ir_xml_sha256": _sha256(base_ir),
                             "caller_supplied_ir_bin_sha256": _sha256(base_ir.with_suffix(".bin"))}
        self.evidence = {**source, "role": "streaming_thinker_with_original_fusion_gate_projection",
                         "add_input_embedding": residual, "base_ir": base_identity,
                         "neural_device": "OpenVINO GPU", "sampling": "greedy"}
        graph = _adapt_thinker(base_ir, cache_dir / "thinker" / "model.xml", self.model_dir,
                               self.config, self.evidence, residual)
        self.runtime = StatefulGPU(graph, self.width)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True,
                                                       trust_remote_code=False)
        self.audio_pad_id = int(self.tokenizer.convert_tokens_to_ids("<|audio_pad|>"))
        self.im_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        if self.audio_pad_id != int(self.config["audio_token_index"]):
            raise RuntimeError("Original S3 audio placeholder token disagrees with config")
        if self.tokenizer.encode("<|im_end|>", add_special_tokens=False) != [self.im_end_id]:
            raise RuntimeError("Original S3 chat stop token is not atomic")
        eos = self.config["text_config"].get("eos_token_id")
        self.stop_ids = {self.im_end_id, *(eos if isinstance(eos, list) else [eos])} - {None}
        self._lock = threading.RLock()

    def _prompt(self, prompt, segments, extra_ids=()):
        if isinstance(prompt, str):
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        elif isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            tokens = list(prompt["prompt_token_ids"])
        else:
            tokens = list(prompt)
        segments = list(segments)
        if tokens.count(self.audio_pad_id) != len(segments):
            raise ValueError("Each original audio_pad placeholder needs exactly one audio segment")
        expanded, spans = [], []
        segment_index = 0
        for token in tokens:
            if token != self.audio_pad_id:
                expanded.append(int(token))
                continue
            segment = _numpy(segments[segment_index])
            if segment.ndim != 2 or segment.shape[1] != self.width or not len(segment):
                raise ValueError("S3 audio embeddings must be nonempty [frames, Thinker width]")
            spans.append((len(expanded), segment))
            expanded.extend([self.audio_pad_id] * len(segment))
            segment_index += 1
        prefix_len = len(expanded)
        expanded.extend(int(token) for token in extra_ids)
        if len(expanded) > int(self.config["text_config"]["max_position_embeddings"]):
            raise ValueError("S3 expanded prompt exceeds the original context limit")
        override = np.zeros((1, len(expanded), self.width), dtype=np.float32)
        mask = np.zeros((1, len(expanded), 1), dtype=np.bool_)
        for start, segment in spans:
            override[0, start:start + len(segment)] = segment
            mask[0, start:start + len(segment)] = True
        return expanded, override, mask, prefix_len

    def _prefill(self, prompt, segments, extra_ids=()):
        ids, override, mask, prefix_len = self._prompt(prompt, segments, extra_ids)
        self.runtime.reset()
        output = self.runtime.infer(ids, embeddings=override, mask=mask)
        return output, prefix_len

    def generate(self, prompt, segments):
        with self._lock:
            output, _ = self._prefill(prompt, segments)
            generated = []
            for index in range(self.max_new_tokens):
                token = int(np.argmax(output["logits"].reshape(-1, output["logits"].shape[-1])[-1]))
                if token in self.stop_ids:
                    break
                generated.append(token)
                if index + 1 < self.max_new_tokens:
                    output = self.runtime.infer([token])
            return self.tokenizer.decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).replace("<|im_end|>", "").strip()

    def gate(self, prompt, segments, threshold=0.5):
        threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
        with self._lock:
            output, _ = self._prefill(prompt, segments)
            logits = output["gate_logits"].reshape(-1, 2)[-1].copy()
            logits[1] -= math.log(threshold / (1.0 - threshold))
            return int(np.argmax(logits)) == 1

    def alignment(self, prompt, segments, forced_text, final, finalize_only=False):
        with self._lock:
            real_ids = [] if finalize_only else self.tokenizer.encode(str(forced_text).strip(), add_special_tokens=False)
            if not finalize_only:
                real_ids += self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            pad_id = int(self.tokenizer.pad_token_id or 0)
            pad_count = self.max_codec_tokens + self.blank_count + 2 if final else 0
            output, prefix_len = self._prefill(prompt, segments, real_ids + [pad_id] * pad_count)
            hidden = output["projected_hidden"]
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError("S3 aligned projected hidden output shape is invalid")
            if hidden.shape[1] != prefix_len + len(real_ids) + pad_count:
                raise RuntimeError("S3 aligned hidden timeline length changed")
            return hidden[0].copy(), prefix_len, prefix_len + len(real_ids), self.blank_count
