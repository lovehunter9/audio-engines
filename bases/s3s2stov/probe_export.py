"""Export verified native S3 weights in an isolated CPU conversion process."""
import argparse
import json
from pathlib import Path

def _strict_loading(info):
    bad = {key: info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if bad:
        raise RuntimeError("Incomplete original S3 language-model loading: " + repr(bad))

def _export_causal(model, target):
    from optimum.exporters.openvino import export_from_model
    from optimum.intel.openvino import OVConfig

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    model.eval()
    # Qwen3.5 export mutates the reference model. Never reuse it for native checks.
    export_from_model(model, output=target, task="text-generation-with-past", stateful=True,
                      ov_config=OVConfig(dtype="fp16"), patch_16bit_model=True)
    graph = target / "openvino_model.xml"
    if not graph.is_file() or not graph.with_suffix(".bin").is_file():
        raise RuntimeError("Official Optimum export did not produce the stateful S3 graph")
    return graph


def thinker(model_dir, target):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**json.loads((Path(model_dir) / "config.json").read_text())["text_config"])
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        str(model_dir), config=config, key_mapping={r"^model\.language_model\.": "model."},
        dtype=torch.float16, attn_implementation="eager", local_files_only=True,
        output_loading_info=True,
    )
    _strict_loading(info)
    return _export_causal(model, target)


def talker(model_dir, target, files):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config.from_pretrained(str(Path(model_dir) / files["config"]), local_files_only=True)
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default", "rope_theta": getattr(config, "rope_theta", 1000000.0)}
    config._attn_implementation = "eager"
    state = torch.load(str(Path(model_dir) / files["llm"]), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Original S3 talker.pt must be a complete Qwen2 state dict")
    if all(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    head = torch.load(str(Path(model_dir) / files["head"]), map_location="cpu", weights_only=True)
    if not isinstance(head, dict) or set(head) != {"weight"}:
        raise RuntimeError("Original S3 codec head must contain exactly weight")
    head_weight = head["weight"]
    if head_weight.ndim != 2 or head_weight.shape[1] != int(config.hidden_size):
        raise RuntimeError("Original S3 codec head and Qwen2 hidden widths disagree")
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config)
        model.lm_head = torch.nn.Linear(int(config.hidden_size), int(head_weight.shape[0]), bias=False)
    # Strict=True checks every Qwen2 parameter, including its unused text embedding.
    model.model.load_state_dict(state, strict=True, assign=True)
    model.lm_head.load_state_dict(head, strict=True, assign=True)
    # RoPE frequencies are nonpersistent buffers, absent from every state dict.
    # Recreate them with the original Transformers constructor after meta loading.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device="cpu")
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError("An original S3 Talker parameter/buffer was not materialized")
    model.config.tie_word_embeddings = False
    model.config.vocab_size = int(head_weight.shape[0])
    model.vocab_size = int(head_weight.shape[0])
    model.float().eval()
    return _export_causal(model, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("thinker", "talker"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", default="{}")
    args = parser.parse_args()
    if args.role == "thinker":
        thinker(args.model, args.output)
    else:
        talker(args.model, args.output, json.loads(args.files))
