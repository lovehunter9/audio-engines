"""Export the pinned upstream PyTorch frontend without its serving framework."""

import ast
from pathlib import Path

root = Path("/opt/s3")
out = root / "native"
out.mkdir(exist_ok=True)
header = """# SPDX-License-Identifier: Apache-2.0
# Adapted from the pinned QwenAudio/S3-S2ST native inference sources.
from __future__ import annotations
import math
import os
import logging
from typing import Optional, Tuple, Any, List
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)
flash_attn = None
"""
source = root / "vendor/vllm/model_executor/models/omnidrca2_encoder.py"
tree = ast.parse(source.read_text())
nodes = [
    n
    for n in tree.body
    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign)) and 581 <= n.lineno <= 1462
]
assert {"SenseVoiceEncoder", "apply_rotary_pos_emb"} <= {getattr(n, "name", "") for n in nodes}
(out / "s3_encoder.py").write_text(header + "\n\n".join(ast.unparse(n) for n in nodes) + "\n")
source = root / "inference/streaming/vllm_overrides/omnidrca2_next_thinker.py"
tree = ast.parse(source.read_text())
names = {
    "LayerNorm",
    "EncoderLayer",
    "sequence_mask",
    "PositionwiseFeedForward",
    "MultiHeadedAttention",
    "Transformer",
}
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"encoder_conf", "audio_adaptor_conf"}
        for t in node.targets
    ):
        nodes.append(node)
assert len(nodes) == len(names) + 2
(out / "s3_adaptor.py").write_text(header + "\n\n".join(ast.unparse(n) for n in nodes) + "\n")
for name in ["true_streaming_encoder.py", "block640_lookahead160_encoder.py"]:
    text = (
        (root / "inference/streaming" / name)
        .read_text()
        .replace(
            "from vllm.model_executor.models.omnidrca2_encoder import apply_rotary_pos_emb",
            "from s3_encoder import apply_rotary_pos_emb",
        )
    )
    (out / name).write_text(text)
source = root / "inference/runtime/pcm_runtime.py"
tree = ast.parse(source.read_text())
nodes = [
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"Playback", "PCMRuntime"}
]


class FM8(ast.NodeTransformer):
    def visit_Constant(self, node):
        return (
            ast.copy_location(ast.Constant(8), node)
            if type(node.value) is int and node.value == 15
            else node
        )


for cls in nodes:
    if cls.name == "PCMRuntime":
        cls.body = [
            n
            for n in cls.body
            if not isinstance(n, ast.FunctionDef) or n.name not in {"__init__", "finish", "gate"}
        ]
        for n in cls.body:
            if isinstance(n, ast.FunctionDef) and n.name in {"decode", "on_codec"}:
                FM8().visit(n)
extra = "import time, threading, hashlib, json\nfrom pathlib import Path\nimport torchaudio\n"
(out / "s3_pcm.py").write_text(header + extra + "\n\n".join(ast.unparse(n) for n in nodes) + "\n")

# Keep optional dataset I/O outside the native waveform inference imports.
import copy

package = root / "vendor/FunASR/funasr"
(package / "__init__.py").write_text(
    'from pathlib import Path\n__version__ = (Path(__file__).parent/"version.txt").read_text().strip()\n'
)
source = package / "models/cosyvoice/flow_matching_model.py"
original = source.read_text()
tree = ast.parse(original)
optional = [
    n
    for n in tree.body
    if isinstance(n, ast.ImportFrom)
    and n.module in {"funasr.utils.load_utils", "funasr.utils.oss_io"}
]
protected = {
    id(n): ast.dump(n)
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "_inference"
}
for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
        continue
    refs = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    imports = [
        copy.deepcopy(imp) for imp in optional if refs & {a.asname or a.name for a in imp.names}
    ]
    offset = (
        1
        if node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
        else 0
    )
    node.body[offset:offset] = imports
for node in ast.walk(tree):
    if id(node) in protected:
        assert ast.dump(node) == protected[id(node)], (
            "Native FM inference must not acquire generic audio I/O"
        )
tree.body = [n for n in tree.body if n not in optional]
prefix = "\n".join(original.splitlines()[: tree.body[0].lineno - 1])
source.write_text(prefix + "\n" + ast.unparse(tree) + "\n")
