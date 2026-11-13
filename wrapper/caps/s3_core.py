"""Single-process S3-S2ST with the published block-causal frontend and weights."""

import json
import os
from pathlib import Path
from types import SimpleNamespace


class Engine:
    def __init__(self, repo, *, dtype="bf16", threshold=None):
        import torch

        torch.set_num_threads(1)
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from s3_encoder import SenseVoiceEncoder
        from s3_adaptor import Transformer, encoder_conf, audio_adaptor_conf
        from true_streaming_encoder import enable_true_streaming_adaptor
        from block640_lookahead160_encoder import enable_block640_lookahead160_encoder
        from .s3_speech import SpeechDecoder

        self.torch = torch
        self.device = "xpu:0" if os.environ.get("AUDIO_BASE") == "s3s2stxpu" else "cuda:0"
        backend = torch.xpu if self.device.startswith("xpu") else torch.cuda
        if not backend.is_available():
            raise RuntimeError("S3-S2ST requires its selected GPU backend")
        if dtype != "bf16":
            raise ValueError("S3-S2ST uses the published bf16 language-model weights")
        self.dtype = torch.bfloat16
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError("--decision-threshold must be between 0 and 1")
        self.threshold = threshold
        self.root = (
            Path(
                snapshot_download(
                    repo,
                    local_files_only=True,
                    allow_patterns=["streaming/**"],
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                    token=os.environ.get("HF_TOKEN") or None,
                )
            )
            / "streaming"
        )
        self.config = json.loads((self.root / "config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.root, local_files_only=True)
        self.stop_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.audio_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        cfg = Qwen3_5TextConfig(**self.config["text_config"])
        cfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            self.model = Qwen3_5ForCausalLM(cfg)
        rotary = self.model.model.rotary_emb
        self.model.model.rotary_emb = type(rotary)(cfg, device="cpu")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        loaded = set()
        for shard in sorted(set(index["weight_map"].values())):
            state = {}
            with safe_open(self.root / shard, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if not key.startswith("model.language_model."):
                        raise ValueError("Unexpected S3 language weight: " + key)
                    name = "model." + key[len("model.language_model.") :]
                    state[name] = source.get_tensor(key)
            self.model.load_state_dict(state, strict=False, assign=True)
            loaded.update(state)
        expected = set(self.model.state_dict()) - {"lm_head.weight"}
        if loaded != expected:
            raise RuntimeError("S3 language checkpoint does not match its architecture")
        self.model.tie_weights()
        self.model.to(device=self.device).eval()
        self.fusion_layers = self.config["thinker_layer_fusion"]["layers"]
        self.fusion = torch.load(
            self.root / "thinker_layer_fusion.pt", map_location="cpu", weights_only=True
        )["weights"].to(self.device)
        self.capture = {}
        for number in self.fusion_layers[:-1]:
            self.model.model.layers[number - 1].register_forward_hook(self._hook(number))
        self.model.model.norm.register_forward_hook(self._hook(self.fusion_layers[-1]))
        self.projection = self._linear("pre_matching.pt")
        gate = torch.nn.Sequential(
            torch.nn.LayerNorm(cfg.hidden_size),
            torch.nn.Linear(cfg.hidden_size, 256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 2),
        )
        state = torch.load(self.root / "streaming_gate.pt", map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        gate.load_state_dict({k.removeprefix("net."): v for k, v in state.items()}, strict=True)
        self.gate = gate.to(self.device, dtype=self.dtype).eval()
        self.encoder = SenseVoiceEncoder(**encoder_conf).float().eval()
        adaptor_cfg = {**audio_adaptor_conf, "llm_dim": cfg.hidden_size}
        self.adaptor = Transformer(**adaptor_cfg).float().eval()
        self._load(self.encoder, "audio_encoder.pt", "audio_encoder.")
        self._load(self.adaptor, "audio_adaptor.pt", "audio_adaptor.")
        self.encoder.to(self.device)
        self.adaptor.to(self.device)
        enable_block640_lookahead160_encoder(self.encoder, chunk_ms=640, lookahead_ms=160)
        enable_true_streaming_adaptor(self.adaptor, self.encoder.streaming_chunk_frames)
        import numpy as np

        self.filters = torch.from_numpy(
            np.load(self.root / "assets/frontend/mel_filters.npz")["mel_128"]
        ).to(self.device)
        self.window = torch.hann_window(400, device=self.device)
        self.speech = SpeechDecoder(self.root, self.device)
        self.languages = self.speech_languages = {"cmn", "eng"}
        from .s3_cache import install

        install(self.model)

    def _hook(self, number):
        def capture(_module, _inputs, output):
            if self.capture is not None:
                self.capture[number] = output.detach()

        return capture

    def _load(self, module, filename, prefix=""):
        state = self.torch.load(self.root / filename, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix(prefix): v for k, v in state.items()}
        module.load_state_dict(state, strict=True)

    def _linear(self, filename):
        torch = self.torch
        state = torch.load(self.root / filename, map_location="cpu", weights_only=True)
        weight = state["weight"]
        result = torch.nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in state)
        result.load_state_dict(state, strict=True)
        return result.to(self.device, dtype=self.dtype).eval()

    def session(self, target, *, text_only=False):
        return SimpleNamespace(
            samples=[],
            gate_cache=None,
            gate_prefix=None,
            gate_hidden=None,
            gate_result=None,
            next_closed=10240,
            encoded=0,
            pending=None,
            committed="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            segments=[],
            text_only=text_only,
            finished=False,
            speech=None if text_only else self.speech.session(),
            target=target,
        )

    def _features(self, samples):
        torch = self.torch
        wave = torch.as_tensor(samples, dtype=torch.float32, device=self.device).reshape(1, -1)
        if wave.shape[-1] < 400:
            wave = torch.nn.functional.pad(wave, (0, 400 - wave.shape[-1]))
        stft = torch.stft(wave, 400, 160, window=self.window, return_complex=True)
        mel = (self.filters @ stft[..., :-1].abs().square()).clamp_min(1e-10).log10()
        mel = torch.maximum(mel, mel.new_tensor(1.3904691073 - 8))
        features = (mel + 4) / 4
        lengths = torch.tensor([features.shape[-1]], device=self.device)
        encoded, lengths = self.encoder(features, lengths)
        embeds, lengths = self.adaptor(encoded, lengths)
        return embeds[0, : int(lengths[0])].to(self.dtype)

    def _prompt(self, state, *, open_write):
        direction = (
            "Translate into Chinese:" if state.target == "cmn" else "Translate into English:"
        )
        prefix = direction if not state.segments else ""
        text = state.committed + "<|im_start|>user\n" + prefix + "<|audio_start|><|audio_pad|>"
        if open_write:
            text += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        parts = []
        cursor = 0
        segments = state.segments + [state.pending]
        torch = self.torch
        embedded = self.model.get_input_embeddings()(torch.tensor(ids, device=self.device))
        left = 0
        for i, token in enumerate(ids):
            if token == self.audio_id:
                if i > left:
                    parts.append(embedded[left:i])
                parts.append(segments[cursor])
                cursor += 1
                left = i + 1
        if left < len(ids):
            parts.append(embedded[left:])
        if cursor != len(segments):
            raise RuntimeError("S3 audio placeholder mismatch")
        return torch.cat(parts, dim=0).unsqueeze(0), text

    def _forward(self, embeds, cache=None, *, project=True):
        self.capture = {} if project else None
        out = self.model(
            inputs_embeds=embeds,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        if not project:
            return out, None, None
        hidden = self.capture[self.fusion_layers[0]].float() * self.fusion[0].float()
        for i, layer in enumerate(self.fusion_layers[1:], 1):
            hidden = hidden + self.capture[layer].float() * self.fusion[i].float()
        projected = self.projection(hidden.to(self.dtype))[0]
        return out, projected, self.capture[self.fusion_layers[-1]][:, -1]

    def _gate_forward(self, state, embeds):
        torch = self.torch
        previous = state.gate_prefix
        reuse = previous is not None and embeds.shape[1] >= previous.shape[1]
        if reuse:
            reuse = torch.equal(embeds[:, : previous.shape[1]], previous)
        if reuse and embeds.shape[1] == previous.shape[1]:
            return state.gate_result
        suffix = embeds[:, previous.shape[1] :] if reuse else embeds
        output, new_hidden, last = self._forward(suffix, state.gate_cache if reuse else None)
        hidden = torch.cat([state.gate_hidden, new_hidden]) if reuse else new_hidden
        state.gate_cache = output.past_key_values
        state.gate_prefix, state.gate_hidden = embeds, hidden
        state.gate_result = output, hidden, last
        return state.gate_result

    def _threshold(self, state):
        if self.threshold is not None:
            return self.threshold
        if state.speech is None:
            return 0.5
        return state.speech.pcm.play.threshold(0.1, 0.99, 1.0, 0.5)[0]

    def _generate(self, embeds):
        torch = self.torch
        out, _, _ = self._forward(embeds, project=False)
        ids = []
        for _ in range(256):
            token = int(out.logits[0, -1].argmax())
            if token == self.stop_id:
                break
            ids.append(token)
            step = self.model.get_input_embeddings()(torch.tensor([[token]], device=self.device))
            out, _, _ = self._forward(step, out.past_key_values, project=False)
        else:
            raise RuntimeError("S3 translation exceeded its token limit")
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def push(self, state, samples, target, finished):
        import numpy as np

        torch = self.torch
        if state.finished:
            return [], True
        state.samples.extend(samples)
        events = []
        with torch.inference_mode():
            while state.next_closed + 2600 <= len(state.samples) or finished:
                final = bool(finished and state.next_closed + 2600 >= len(state.samples))
                available = len(state.samples) if final else state.next_closed + 2600
                embeds = self._features(np.asarray(state.samples[:available], dtype=np.float32))
                count = len(embeds) if final else min(len(embeds), state.next_closed // 10240 * 8)
                new = embeds[state.encoded : count]
                state.encoded = count
                if len(new):
                    state.pending = (
                        new if state.pending is None else torch.cat([state.pending, new])
                    )
                if state.pending is None or not len(state.pending):
                    if final:
                        if state.speech is not None:
                            events.extend(("samples", x) for x in self.speech.finish(state.speech))
                        state.finished = True
                    else:
                        state.next_closed += 10240
                    if final:
                        break
                    continue
                prompt, _ = self._prompt(state, open_write=False)
                _, hidden, last = self._gate_forward(state, prompt)
                if state.speech is not None:
                    events.extend(("samples", x) for x in self.speech.update(state.speech, hidden))
                write = final or float(
                    self.gate(last).float().softmax(-1)[0, 1]
                ) >= self._threshold(state)
                if write:
                    prompt, prompt_text = self._prompt(state, open_write=True)
                    text = self._generate(prompt)
                    final_embeds = prompt
                    if text:
                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        events.append(("text", text))
                        suffix = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                        ending = self.model.get_input_embeddings()(
                            torch.tensor([ids + suffix], device=self.device)
                        )
                        final_embeds = torch.cat([prompt, ending], dim=1)
                        _, hidden, _ = self._forward(final_embeds)
                        if state.speech is not None:
                            events.extend(
                                ("samples", x)
                                for x in self.speech.update(
                                    state.speech,
                                    hidden,
                                    text_start=prompt.shape[1],
                                    source=state.samples[:available],
                                    final=final,
                                )
                            )
                        state.committed = prompt_text + text + "<|im_end|>\n"
                        state.segments.append(state.pending)
                        state.pending = None
                    elif final and not state.segments:
                        raise RuntimeError("S3 final translation returned empty text")
                if final:
                    if state.speech is not None:
                        pad = self.tokenizer.pad_token_id or 0
                        out, projection, _ = self._forward(final_embeds)
                        tail = []
                        while state.speech is not None and not state.speech.complete:
                            tail.append(pad)
                            if len(tail) > 2048:
                                raise RuntimeError("S3 final audio did not reach EOS")
                            step = self.model.get_input_embeddings()(
                                torch.tensor([[pad]], device=self.device)
                            )
                            out, projected, _ = self._forward(step, out.past_key_values)
                            projection = torch.cat([projection, projected])
                            events.extend(
                                ("samples", x)
                                for x in self.speech.update(state.speech, projection, final=True)
                            )
                        events.extend(("samples", x) for x in self.speech.finish(state.speech))
                    state.finished = True
                    break
                state.next_closed += 10240
        return events, state.finished
