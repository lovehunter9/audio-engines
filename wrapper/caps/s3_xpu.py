"""S3 XPU recurrence with the existing BF16 query/key normalization."""

import sys


def install(model):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, l2norm

    sys.path.insert(0, "/opt/s3-xpu-kernels")
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    finally:
        sys.path.pop(0)

    def delta(query, key, value, **kwargs):
        if kwargs.pop("use_qk_l2norm_in_kernel", False):
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        return fused_recurrent_gated_delta_rule(
            query, key, value, **kwargs, use_qk_l2norm_in_kernel=False
        )

    count = 0
    for layer in model.modules():
        if isinstance(layer, Qwen3_5GatedDeltaNet):
            layer.recurrent_gated_delta_rule = delta
            layer.chunk_gated_delta_rule = delta
            count += 1
    if not count:
        raise RuntimeError("S3 XPU GatedDeltaNet layers are missing")
