"""Repair optional imports without changing the selected original inference math."""
import hashlib
import json
from pathlib import Path

root = Path("/opt/s3")
path = root / "vendor/vllm/_custom_ops.py"
original = path.read_text()
anchor = '    @register_fake("_C::cutlass_encode_and_reorder_int4b_grouped")\n    def cutlass_encode_and_reorder_int4b_grouped_fake(b: torch.Tensor) -> torch.Tensor:\n        return torch.empty_like(b, memory_format=torch.contiguous_format)'
if original.count(anchor) != 1:
    raise RuntimeError("The pinned optional INT4 registration changed")
updated = original.replace(anchor, 'if hasattr(torch.ops._C, "cutlass_encode_and_reorder_int4b_grouped"):\n\n' + anchor)
path.write_text(updated)
records = [{
    "path": str(path.relative_to(root)),
    "reason": "Check the optional grouped INT4 operator itself before fake registration; the published dense S3 model never calls it",
    "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
    "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
}]
(root / "PATCHES.json").write_text(json.dumps(records, indent=2))
