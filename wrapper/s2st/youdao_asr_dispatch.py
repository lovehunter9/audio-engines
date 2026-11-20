"""Audited HF dispatch for the pinned original R2T2 streaming algorithms."""

import ast
import hashlib
import inspect
import linecache
import textwrap


METHODS = (
    "init_streaming_state",
    "streaming_transcribe",
    "finish_streaming_transcribe",
    "streaming_transcribe_no_reset",
    "finish_streaming_transcribe_no_reset",
)


def rewrite(source, name):
    """Fail closed if the pinned method no longer has the known call shape."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != name:
        raise ValueError("Unexpected R2T2 method")
    changes = []
    guards = imports = calls = 0
    for node in ast.walk(function):
        if isinstance(node, ast.If):
            check = ast.unparse(node.test)
            if check == "self.backend != 'vllm'":
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Raise) or node.orelse:
                    raise ValueError("Unknown backend guard")
                changes.append((node.lineno, node.end_lineno, ""))
                guards += 1
            elif (
                check == "max_new_tokens is not None"
                and len(node.body) == 1
                and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module == "vllm"
            ):
                if [a.name for a in node.body[0].names] != ["SamplingParams"] or node.orelse:
                    raise ValueError("Unknown sampling import")
                changes.append((node.lineno, node.end_lineno, ""))
                imports += 1
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if ast.unparse(node.targets[0]) != "gen_text":
                continue
            if ast.unparse(node.value) != "outputs[0].outputs[0].text":
                continue
            # Locate precisely the adjacent input / sampling / extraction block.
            parent = next(
                (
                    p
                    for p in ast.walk(function)
                    if hasattr(p, "body") and isinstance(p.body, list) and node in p.body
                ),
                None,
            )
            statements = parent.body
            index = statements.index(node)
            before, sample = statements[index - 2 : index]
            if not (
                isinstance(before, ast.Assign)
                and ast.unparse(before.targets[0]) == "inp"
                and isinstance(sample, ast.If)
                and ast.unparse(sample.test) == "max_new_tokens is not None"
            ):
                raise ValueError("Unknown generation block")
            actual = ast.unparse(before.value)
            if actual != "{'prompt': prompt, 'multi_modal_data': {'audio': [state.audio_accum]}}":
                raise ValueError("Unknown source audio binding")
            generate_calls = [
                n
                for n in ast.walk(sample)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.model.generate"
            ]
            if len(generate_calls) != 2:
                raise ValueError("Unknown sampling dispatch")
            replacement = (
                " " * before.col_offset
                + "gen_text = self._streaming_generate(prompt, state.audio_accum, max_new_tokens)\n"
            )
            changes.append((before.lineno, node.end_lineno, replacement))
            calls += 1
    expected = (1, 0, 0) if name == "init_streaming_state" else (1, 1, 1)
    if (guards, imports, calls) != expected:
        raise ValueError(f"R2T2 dispatch shape changed: {name}: {(guards, imports, calls)}")
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(changes, reverse=True):
        lines[start - 1 : end] = [replacement] if replacement else []
    patched = "".join(lines)
    compile(patched, "<audited-hf-dispatch>", "exec")
    return patched, dict(
        method=name,
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        removed_backend_guards=guards,
        removed_sampling_imports=imports,
        replaced_model_calls=calls,
    )


def generate(self, prompt, audio, max_new_tokens=None):
    import torch

    budget = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("Invalid original adaptive ASR token budget")
    inputs = self.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
    inputs = inputs.to(self.model.device).to(self.model.dtype)
    with torch.inference_mode():
        # The pinned Qwen3-ASR entry already sets return_dict_in_generate on
        # its thinker call; forwarding it again creates a duplicate keyword.
        result = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False)
    text = self.processor.batch_decode(
        result.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(text) != 1:
        raise RuntimeError("Unexpected streaming ASR batch size")
    return text[0]


def install(model_class):
    if getattr(model_class, "_audited_hf_dispatch", False):
        raise RuntimeError("HF dispatch was already installed")
    audit = []
    namespace = dict(vars(__import__(model_class.__module__, fromlist=["_"])))
    for name in METHODS:
        original = getattr(model_class, name)
        patched, entry = rewrite(inspect.getsource(original), name)
        compiled = dict(namespace)
        filename = inspect.getfile(original) + ":hf-dispatch:" + name
        linecache.cache[filename] = (
            len(patched),
            None,
            patched.splitlines(keepends=True),
            filename,
        )
        exec(compile(patched, filename, "exec"), compiled)
        setattr(model_class, name, compiled[name])
        audit.append(entry)
    model_class._streaming_generate = generate
    model_class._audited_hf_dispatch = True
    model_class._hf_dispatch_audit = audit
    return audit


def load_prepared(model_path, max_new_tokens=4):
    import torch
    from r2t2 import R2T2ASRModel

    if not getattr(R2T2ASRModel, "_audited_hf_dispatch", False):
        raise RuntimeError("HF dispatch was not explicitly installed")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("Invalid original default ASR budget")
    model = R2T2ASRModel.from_pretrained(
        str(model_path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
        local_files_only=True,
    )
    model.model.eval()
    devices = sorted({str(parameter.device) for parameter in model.model.parameters()})
    if not devices or any(not value.startswith("cuda") for value in devices):
        raise RuntimeError("R2T2 neural parameters must all be CUDA")
    buffers = {
        name: str(value.device) for name, value in model.model.named_buffers() if value.numel()
    }
    if any(not device.startswith("cuda") for device in buffers.values()):
        raise RuntimeError("R2T2 nonempty model buffers are not wholly CUDA")
    model.decode_backend = "transformers"
    model.device_audit = dict(
        backend="official HF forward, audited streaming dispatch",
        parameter_devices=devices,
        nonempty_buffer_devices=buffers,
        method_patches=R2T2ASRModel._hf_dispatch_audit,
    )
    return model
