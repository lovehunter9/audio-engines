"""Install the author's model namespace in every native vLLM process."""
import logging
import os
from pathlib import Path


def register():
    raw = os.environ.get("S3_NATIVE_OVERRIDE_DIR")
    if not raw:
        return
    directory = Path(raw)
    if not (directory / "omnidrca2_next_thinker.py").is_file():
        raise RuntimeError("The original S3 streaming model definitions are absent")
    import vllm.model_executor.models as models
    if str(directory) not in models.__path__:
        models.__path__.insert(0, str(directory))
    models.ModelRegistry.register_model(
        "Qwen3_5ForCausalLM", "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM")
    import torch
    from vllm import _custom_ops as ops
    if not hasattr(torch.ops._C, "flashinfer_gemma_rms_norm"):
        def gemma_norm(out, value, weight, epsilon=1e-6, enable_pdl=False):
            if not value.is_cuda:
                raise RuntimeError("Native S3 normalization requires CUDA")
            from flashinfer.norm import gemma_rmsnorm
            gemma_rmsnorm(value, weight, eps=epsilon, out=out, enable_pdl=enable_pdl)
        ops.flashinfer_gemma_rms_norm = gemma_norm
    if not hasattr(torch.ops._C, "flashinfer_gemma_fused_add_rms_norm"):
        def gemma_add_norm(value, residual, weight, epsilon=1e-6, enable_pdl=False):
            if not value.is_cuda or not residual.is_cuda:
                raise RuntimeError("Native S3 normalization requires CUDA")
            from flashinfer.norm import gemma_fused_add_rmsnorm
            gemma_fused_add_rmsnorm(value, residual, weight, eps=epsilon, enable_pdl=enable_pdl)
        ops.flashinfer_gemma_fused_add_rms_norm = gemma_add_norm
    logging.getLogger("s3-native-runtime").info("Installed original streaming model namespace: %s", directory)
