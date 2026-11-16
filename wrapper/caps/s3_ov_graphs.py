"""Strict OpenVINO GPU compilation for original S3 neural modules."""
import gc
import json
import logging
import time
from pathlib import Path


def compile_gpu(path, *, precision="f32"):
    import openvino as ov
    core = ov.Core()
    model = core.read_model(str(path))
    compiled = core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": precision})
    devices = [str(device) for device in compiled.get_property("EXECUTION_DEVICES")]
    if not devices or any(not device.startswith("GPU") for device in devices):
        raise RuntimeError("S3 neural execution must use OpenVINO GPU: " + repr(devices))
    return compiled


def export_module(module, example, shapes, target, *, source, precision="f32"):
    import openvino as ov
    import torch
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    graph = target / "model.xml"
    if not graph.exists():
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
    elif json.loads((target / "source.json").read_text()) != source:
        raise RuntimeError("S3 OpenVINO graph source identity differs from the current original module")
    return compile_gpu(graph, precision=precision)


def export_audio_encoder(source_root, model_dir, target):
    import torch
    from .s3_ov_source import load_audio_encoder
    module, evidence = load_audio_encoder(source_root, model_dir)
    example = (torch.zeros(1, 128, 96), torch.tensor([96], dtype=torch.int64))
    compiled = export_module(module, example, [[1, 128, -1], [1]], target, source=evidence)
    del module
    gc.collect()
    return compiled, evidence
