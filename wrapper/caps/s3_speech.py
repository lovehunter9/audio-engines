"""Published S3 codec alignment and FM8 synthesis, with one persistent Talker."""

import os
from pathlib import Path
from types import SimpleNamespace


class SpeechDecoder:
    def __init__(self, root, device):
        import torch
        from transformers import Qwen2Config, Qwen2Model
        from streaming_codex_to_wav import _load_model_from_config
        from s3_pcm import PCMRuntime, Playback

        self.torch = torch
        self.root = Path(root)
        self.device = device
        self.PCMRuntime, self.Playback = PCMRuntime, Playback
        cfg = Qwen2Config.from_pretrained(root / "talker_config", local_files_only=True)
        cfg._attn_implementation = "sdpa"
        self.model = Qwen2Model(cfg).eval()
        state = torch.load(root / "talker.pt", map_location="cpu", weights_only=True)
        state = {k.removeprefix("model."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=device, dtype=torch.bfloat16)
        state = torch.load(root / "talker_codec_embed.pt", map_location="cpu", weights_only=True)
        self.embedding = torch.nn.Embedding(*state["weight"].shape)
        self.embedding.load_state_dict(state, strict=True)
        self.embedding.to(device=device, dtype=torch.bfloat16).eval()
        state = torch.load(root / "talker_codec_head.pt", map_location="cpu", weights_only=True)
        self.head = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0], bias=False)
        self.head.load_state_dict(state, strict=True)
        self.head.to(device=device, dtype=torch.bfloat16).eval()
        source = Path(os.environ.get("S3_ROOT", "/opt/s3"))
        self.fm = (
            _load_model_from_config(
                source / "configs/fm_config.yaml",
                root / "assets/flow_matching/model.pt",
                torch.device("cpu"),
            )
            .to(device=device, dtype=torch.float16)
            .eval()
        )
        self.voc = (
            _load_model_from_config(
                root / "assets/vocoder/config.yaml",
                root / "assets/vocoder/model.pt",
                torch.device("cpu"),
            )
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.speaker = ort.InferenceSession(
            str(root / "assets/speaker/campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def session(self):
        torch = self.torch
        pcm = self.PCMRuntime.__new__(self.PCMRuntime)
        pcm.device = torch.device(self.device)
        pcm.fm, pcm.voc, pcm.sr = self.fm, self.voc, 24000
        pcm.play = self.Playback()
        pcm.begin(0)
        return SimpleNamespace(
            pcm=pcm,
            wave_cursor=0,
            cache=None,
            position=-1,
            specs=[],
            next_spec=0,
            previous_boundary=-1,
            active=None,
            pending=None,
            complete=False,
            final=False,
            hidden=None,
            generator=torch.Generator(device=self.device).manual_seed(0),
        )

    def _speaker(self, samples):
        import numpy as np
        from torchaudio.compliance import kaldi

        torch = self.torch
        wave = torch.as_tensor(np.asarray(samples, dtype=np.float32).copy()).reshape(1, -1)
        if wave.shape[-1] < 400:
            raise ValueError("source prefix is too short for speaker conditioning")
        features = kaldi.fbank(wave, num_mel_bins=80, dither=0, sample_frequency=16000)
        features = features - features.mean(dim=0, keepdim=True)
        output = self.speaker.run(
            None, {self.speaker.get_inputs()[0].name: features.unsqueeze(0).numpy()}
        )[0]
        return torch.as_tensor(output.reshape(1, -1), device=self.device, dtype=torch.float16)

    def _sample(self, state, logits):
        torch = self.torch
        logits = logits.float().clone()
        eos = logits[59050].clone()
        logits[59049:] = -float("inf")
        # EOS participates in sampling; controller-only boundary tokens do not.
        logits[59050] = eos
        values, ids = logits.topk(5)
        probs = (values / 0.8).softmax(-1)
        remove = probs.cumsum(-1) > 0.8
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        probs = probs.masked_fill(remove, 0)
        index = torch.multinomial(probs, 1, generator=state.generator)
        return int(ids[index].item())

    def _step(self, state, end, codec=None):
        torch = self.torch
        start = state.position + 1
        if end < start:
            raise RuntimeError("Talker timeline moved backwards")
        embeds = state.hidden[start : end + 1].unsqueeze(0).clone()
        if codec is not None:
            embeds[:, -1] += self.embedding(torch.tensor([codec], device=self.device))
        output = self.model(
            inputs_embeds=embeds, past_key_values=state.cache, use_cache=True, return_dict=True
        )
        state.cache = output.past_key_values
        state.position = end
        return self.head(output.last_hidden_state[0, -1])

    def _waves(self, state):
        result = state.pcm.waves[state.wave_cursor :]
        state.wave_cursor = len(state.pcm.waves)
        return result

    def update(self, state, hidden, *, text_start=None, source=None, final=False):
        state.hidden = hidden
        if text_start is not None:
            state.specs.append({"start": text_start, "final": final})
            state.pcm.emb = self._speaker(source)
        if final:
            state.final = True
            if text_start is None and state.next_spec == len(state.specs):
                if state.active is not None:
                    state.active["final"] = True
                elif state.specs and state.previous_boundary >= 0:
                    state.complete = True
        if state.complete:
            return self._waves(state)
        for _ in range(8192):
            available = len(hidden)
            if state.pending is not None:
                next_pos = state.position + 1
                if next_pos >= available:
                    break
                token = state.pending
                state.pending = None
                logits = self._step(state, next_pos, token)
                if state.active is None:
                    continue
            elif state.active is not None:
                raise RuntimeError("active Talker has no pending codec feedback")
            elif state.next_spec < len(state.specs):
                spec = state.specs[state.next_spec]
                bos = max(spec["start"] + 5, state.previous_boundary + 1)
                if bos >= available:
                    break
                logits = self._step(state, bos, 59049)
                state.active = spec
                state.next_spec += 1
            else:
                if available > state.position + 1:
                    self._step(state, available - 1)
                break
            token = self._sample(state, logits)
            state.pcm.on_codec(token)
            if token == 59050:
                state.previous_boundary = state.position + 1
                if state.active["final"]:
                    state.complete = True
                    state.active = None
                    break
                state.active = None
            state.pending = token
        else:
            raise RuntimeError("S3 codec exceeded its bounded generation limit")
        return self._waves(state)

    def finish(self, state):
        if state.specs and not state.complete:
            raise RuntimeError("S3 audio did not finish at its final EOS")
        return self._waves(state)
