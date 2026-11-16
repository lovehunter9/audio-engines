"""Strict OpenVINO GPU compilation for original S3 neural modules."""
import gc
import json
import logging
import time
from pathlib import Path


from .ov_gpu_graph import compile_gpu


def export_module(module, example, shapes, target, *, source, precision="f32", prepare_only=False):
    import openvino as ov
    import torch
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    graph = target / "model.xml"
    marker = target / "source.json"
    if not graph.exists() or not marker.exists() or json.loads(marker.read_text()) != source:
        started = time.monotonic()
        logging.getLogger("s3-openvino").info("Tracing original graph %s", target)
        with torch.inference_mode():
            traced = torch.jit.trace(module, example, check_trace=False, strict=False)
            logging.getLogger("s3-openvino").info("Converting original graph %s after %.2fs", target, time.monotonic()-started)
            converted = ov.convert_model(traced, input=[ov.PartialShape(shape) for shape in shapes])
        del traced
        ov.save_model(converted, str(graph), compress_to_fp16=False)
        (target / "source.json").write_text(json.dumps(source, indent=2))
        del converted
        gc.collect()
    return graph if prepare_only else compile_gpu(graph, precision=precision)


def export_audio_encoder(source_root, model_dir, target, *, prepare_only=False):
    import torch
    from .s3_ov_source import load_audio_encoder
    module, evidence = load_audio_encoder(source_root, model_dir)
    example = (torch.zeros(1, 128, 96), torch.tensor([96], dtype=torch.int64))
    compiled = export_module(module, example, [[1, 128, -1], [1]], target, source=evidence, prepare_only=prepare_only)
    del module
    gc.collect()
    return compiled, evidence
