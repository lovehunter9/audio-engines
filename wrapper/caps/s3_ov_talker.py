"""Original S3 CosyVoice Qwen2 Talker, with stateful OpenVINO GPU inference."""
from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

import numpy as np

from .s3_ov_graphs import compile_gpu
from .s3_ov_llm import (
    StatefulGPU,
    _add_output,
    _check_cache,
    _export_causal,
    _head,
    _last,
    _numpy,
    _override_embeddings,
    _save_graph,
    _source,
    _weight,
)


def _export_talker(model_dir, target, files, source):
    graph = Path(target) / "openvino_model.xml"
    if _check_cache(graph, source):
        return graph
    graph = _export_causal(model_dir, target, "talker", files)
    graph.with_suffix(".source.json").write_text(json.dumps(source, indent=2))
    return graph


def _adapt_talker(base_ir, target, width, source):
    import openvino as ov

    if _check_cache(target, source):
        return Path(target)
    model = ov.Core().read_model(str(base_ir))
    head, hidden = _head(model, width)
    _override_embeddings(model, width)
    head.set_argument(0, _last(hidden).output(0))
    _add_output(model, _last(hidden), "final_hidden")
    _save_graph(model, target, source)
    del model
    gc.collect()
    return Path(target)


def _codec_graphs(model_dir, cache_dir, files, width, thinker_width, source):
    import openvino as ov
    from openvino import opset13 as ops

    embedding = _weight(Path(model_dir) / files["embed"])
    projection = _weight(Path(model_dir) / files["projection"])
    if embedding.shape[1] != width or projection.shape != (width, thinker_width):
        raise RuntimeError("Original S3 codec embedding/input projection shape mismatch")
    paths = {name: Path(cache_dir) / name / "model.xml" for name in ("codec-embed", "input-proj", "codec-add")}
    for name, path in paths.items():
        identity = {**source, "role": name}
        if _check_cache(path, identity):
            continue
        if name == "input-proj":
            hidden = ops.parameter([1, -1, thinker_width], ov.Type.f32, name="hidden")
            result = ops.matmul(hidden, projection, False, True)
            model = ov.Model([result], [hidden], name="s3_original_talker_input_projection")
        else:
            ids = ops.parameter([1, -1], ov.Type.i64, name="codec_ids")
            # Native embed_input_ids clamps negative alignment mask/BOS ids too.
            clamped = ops.minimum(ops.maximum(ids, np.int64(0)), np.int64(len(embedding) - 1))
            result = ops.gather(embedding, clamped, np.int64(0))
            parameters = [ids]
            if name == "codec-add":
                hidden = ops.parameter([1, -1, width], ov.Type.f32, name="hidden")
                result = ops.add(hidden, result)
                parameters.append(hidden)
            model = ov.Model([result], parameters, name="s3_original_" + name.replace("-", "_"))
        _save_graph(model, path, identity)
    return {name: compile_gpu(path, precision="f32") for name, path in paths.items()}, embedding.shape[0]


class Talker:
    """The caller owns original sampling and streaming alignment decisions.

    ``forward`` accepts already aligned [L,H] embeddings. With ``codec_ids``, it
    executes the original external-hidden += codec-embedding operation on GPU.
    ``last_hidden`` exposes the copied post-norm codec-LLM state needed by FM.
    """

    def __init__(self, model_dir, cache_dir, *, prepare_only=False):
        self.model_dir = Path(model_dir)
        config = json.loads((self.model_dir / "config.json").read_text())
        codex = config.get("cosyvoice_codex", {})
        if not codex.get("enabled"):
            raise RuntimeError("S3 OpenVINO requires the original CosyVoice codex Talker")
        config_path = Path(codex["llm_init_param_path"])
        if config_path.is_absolute() or ".." in config_path.parts:
            raise RuntimeError("S3 Talker config must be part of the verified model release")
        files = {"llm": codex.get("llm_file", "talker.pt"),
                 "head": codex.get("codec_head_file", "talker_codec_head.pt"),
                 "embed": codex.get("codec_embed_file", "talker_codec_embed.pt"),
                 "projection": codex.get("input_proj_file", "talker_input_proj.pt"),
                 "config": str(config_path / "config.json")}
        for filename in files.values():
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise RuntimeError("S3 Talker weights must belong to the verified model release")
        source = _source(self.model_dir, ["config.json", *files.values()])
        self.width = int(codex["llm_hidden_size"])
        self.thinker_width = int(config["text_config"]["hidden_size"])
        self.codec_unit = int(codex["codec_unit"])
        self.evidence = {**source, "role": "original_cosyvoice_codex_qwen2_talker", "neural_device": "OpenVINO GPU"}
        # The original Qwen2 config keeps its text vocab; codec IO has independent learned weights.
        talker_config = json.loads((self.model_dir / files["config"]).read_text())
        if int(talker_config["hidden_size"]) != self.width:
            raise RuntimeError("Original S3 Talker config hidden size disagrees with codex config")
        base = _export_talker(self.model_dir, Path(cache_dir) / "talker-base", files, self.evidence)
        graph = _adapt_talker(base, Path(cache_dir) / "talker" / "model.xml", self.width, self.evidence)
        if prepare_only:
            return
        self.runtime = StatefulGPU(graph, self.width)
        self.components, self.codec_vocab_size = _codec_graphs(
            self.model_dir, cache_dir, files, self.width, self.thinker_width, self.evidence)
        head = _weight(self.model_dir / files["head"])
        if len(head) != self.codec_vocab_size or len(head) != int(codex["codec_vocab_size"]):
            raise RuntimeError("Original S3 codec head/embedding/config vocabulary sizes disagree")
        self._lock = threading.RLock()
        self.last_hidden = None

    def reset(self):
        with self._lock:
            self.runtime.reset()
            self.last_hidden = None

    @staticmethod
    def _infer_component(compiled, values):
        request = compiled.create_infer_request()
        request.infer(values, share_inputs=False)
        return np.array(request.get_output_tensor(0).data, copy=True)[0]

    def embed(self, codec_ids):
        ids = np.asarray(codec_ids, dtype=np.int64).reshape(1, -1)
        if ids.shape[1] == 0:
            return np.empty((0, self.width), dtype=np.float32)
        return self._infer_component(self.components["codec-embed"], {"codec_ids": ids})

    def project(self, hidden):
        value = _numpy(hidden)
        if value.ndim == 1:
            value = value[None]
        if value.ndim != 2:
            raise ValueError("S3 Talker hidden must have shape [positions, width]")
        if value.shape[1] == self.width:
            return value.copy()
        if value.shape[1] != self.thinker_width:
            raise ValueError("S3 Talker input has neither original Thinker nor Talker width")
        return self._infer_component(self.components["input-proj"], {"hidden": value[None]})

    def compose(self, hidden, codec_ids):
        value = self.project(hidden)
        ids = np.asarray(codec_ids, dtype=np.int64).reshape(1, -1)
        if ids.shape[1] != len(value):
            raise ValueError("S3 Talker codec ids and external hidden must share the original timeline")
        if (ids < 0).any() or (ids >= self.codec_vocab_size).any():
            raise ValueError("Original S3 aligned codec id is outside its embedding table; use None for the -60 gap")
        return self._infer_component(self.components["codec-add"], {"hidden": value[None], "codec_ids": ids})

    def forward(self, inputs_embeds, codec_ids=None):
        with self._lock:
            value = (self.project(inputs_embeds) if codec_ids is None
                     else self.compose(inputs_embeds, codec_ids))
            if not len(value):
                raise ValueError("Cannot infer an empty S3 Talker sequence")
            output = self.runtime.infer(np.zeros(len(value), dtype=np.int64), embeddings=value)
            self.last_hidden = output["final_hidden"].reshape(-1, self.width)[-1:].copy()
            return output["logits"].reshape(-1, self.codec_vocab_size)[-1].copy()

    step = forward
