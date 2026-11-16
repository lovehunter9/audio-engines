"""OpenVINO GPU graphs around the unchanged Meta streaming policies."""
import gc
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .ov_gpu_graph import compile_gpu

log = logging.getLogger("seamless-openvino")


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class Graph:
    def __init__(self, module, examples, shapes, directory, *, static_inputs=False):
        import torch
        import openvino as ov

        directory.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        graph, marker = directory/"model.xml", directory/"source.json"
        if not graph.exists() or not marker.exists() or json.loads(marker.read_text())["adapter"] != identity:
            log.info("Tracing official Seamless graph %s", directory.name)
            with torch.inference_mode():
                traced = torch.jit.trace(module.eval(), examples, check_trace=False, strict=False)
                model = ov.convert_model(traced, input=[ov.PartialShape(shape) for shape in shapes])
            ov.save_model(model, str(graph), compress_to_fp16=False)
            marker.write_text(json.dumps({"adapter": identity}))
            del traced, model
            gc.collect()
        self.graph, self.static_inputs = graph, static_inputs
        self.entries = OrderedDict()
        if not static_inputs:
            self._entry(None)

    def _entry(self, shapes):
        if shapes not in self.entries:
            compiled = compile_gpu(self.graph, input_shapes=None if shapes is None else dict(enumerate(shapes)))
            self.entries[shapes] = (compiled, [compiled.create_infer_request() for _ in range(2)], 0)
        compiled, requests, turn = self.entries[shapes]
        self.entries[shapes] = (compiled, requests, turn + 1)
        self.entries.move_to_end(shapes)
        if len(self.entries) > 8:
            self.entries.popitem(last=False)
        return compiled, requests[turn % 2]

    def infer(self, inputs, *, retain=0):
        import openvino as ov
        shapes = tuple(tuple(value.shape) for value in inputs) if self.static_inputs else None
        compiled, request = self._entry(shapes)
        if len(inputs) != len(compiled.inputs):
            raise RuntimeError("Official Seamless graph input signature changed")
        for index, value in enumerate(inputs):
            tensor = value if isinstance(value, ov.Tensor) else ov.Tensor(np.ascontiguousarray(_numpy(value)))
            request.set_input_tensor(index, tensor)
        request.infer()
        count = len(compiled.outputs)
        copied = [np.asarray(request.get_output_tensor(i).data).copy() for i in range(count-retain)]
        return copied+[request.get_output_tensor(i) for i in range(count-retain, count)]


def _encoder(model, root):
    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = model.speech_encoder_frontend
            self.encoder = model.speech_encoder

        def forward(self, feature):
            encoded, mask = self.frontend(feature, None)
            return self.encoder(encoded, mask)[0]

    graph = Graph(Encoder(), (torch.zeros(1, 400, 80),), [[1, -1, 80]], root/"encoder")
    def encode_speech(seqs, padding_mask):
        if padding_mask is not None and not bool(padding_mask.seq_lens.eq(seqs.shape[1]).all()):
            raise ValueError("Seamless streaming uses one unpadded source per session")
        return torch.from_numpy(graph.infer([seqs])[0]), None
    return SimpleNamespace(encode_speech=encode_speech)


def _decoder(agent, root):
    import torch
    from torch import nn
    from fairseq2.nn.incremental_state import IncrementalStateBag
    from fairseq2.nn.transformer.multihead_attention import AttentionState
    from fairseq2.nn.transformer.attention_mask import CustomAttentionMask

    model = agent.model
    layers = list(model.text_decoder.layers)
    attentions = [layer.self_attn for layer in layers]
    heads, width = attentions[0].num_key_value_heads, model.text_decoder.model_dim//attentions[0].num_heads
    count = len(layers)

    class State(AttentionState):
        def __init__(self, k, v):
            self.k, self.v = k, v

        def append(self, k, v):
            self.k, self.v = torch.cat((self.k, k), 2), torch.cat((self.v, v), 2)

        def get(self):
            return self.k, self.v

        def reorder(self, order):
            self.k, self.v = self.k.index_select(0, order), self.v.index_select(0, order)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, tokens, encoded, *cache):
            bag = IncrementalStateBag(4096)
            bag.step_nr = cache[0].shape[2]
            for i, attention in enumerate(attentions):
                bag.set_state(attention, State(cache[2*i], cache[2*i+1]))
            def mask_factory(seqs, keys, **kwargs):
                query = torch.arange(seqs.shape[1], device=seqs.device)[:, None]+cache[0].shape[2]
                key = torch.arange(cache[0].shape[2]+seqs.shape[1], device=seqs.device)[None, :]
                return CustomAttentionMask(torch.where(key <= query, torch.zeros((), dtype=seqs.dtype),
                    torch.full((), float("-inf"), dtype=seqs.dtype)))
            self.model.text_decoder.self_attn_mask_factory = mask_factory
            feature, _, choose = self.model.decode(tokens, None, encoded, None, state_bag=bag)
            logits = self.model.project(feature)
            probability = choose[:, -1, -1].amin()
            flat = []
            for attention in attentions:
                state = bag.get_state(attention, AttentionState)
                flat.extend(state.get())
            return (feature, logits[0, -1].argmax(), probability, *flat)

    cache = tuple(torch.zeros(1, heads, 2, width) for _ in range(2*count))
    examples = (torch.ones(1, 2, dtype=torch.int64), torch.zeros(1, 20, model.text_decoder.model_dim), *cache)
    shapes = [[1, -1], [1, -1, model.text_decoder.model_dim]]+[[1, heads, -1, width] for _ in cache]
    graph = Graph(Decoder(), examples, shapes, root/"monotonic-decoder")
    if agent.block_ngrams or agent.decision_method != "min" or agent.p_choose_start_layer != 0:
        raise ValueError("The OpenVINO port requires the published Seamless decoding defaults")

    def run_decoder(self, states, predicted):
        bag = self.state_bag
        if not predicted:
            self.enforce_tgt_lang_in_prefix(states)
            tokens = self.prefix_indices+states.target_indices
        else:
            tokens = predicted[-1:]
        old = getattr(bag, "seamless_gpu_cache", None)
        if old is None:
            if bag.step_nr != 0:
                raise RuntimeError("Official decoder position has no matching cache")
            old = [np.zeros((1, heads, 0, width), np.float32) for _ in range(2*count)]
        if old[0].shape[2] != bag.step_nr:
            raise RuntimeError("Official decoder position and OpenVINO cache diverged")
        result = graph.infer([np.array([tokens], np.int64), states.source, *old], retain=2*count)
        feature, index, probability = result[:3]
        bag.seamless_gpu_cache = result[3:]
        return int(index), float(probability), torch.from_numpy(feature)

    agent.run_decoder = MethodType(run_decoder, agent)
    agent.model = None


def _hard_upsampling(module):
    import torch
    def forward(self, seqs, durations):
        lengths = durations.sum(dim=1)
        ends = durations[0].cumsum(0)
        positions = torch.arange(lengths[0], device=durations.device)
        indices = (positions[:, None] >= ends[None, :]).sum(-1)
        return seqs[:, indices], lengths
    module.forward = MethodType(forward, module)


def _unit_decoder(agent, root):
    import torch
    from torch import nn
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.data.segments import TextSegment
    from seamless_communication.models.unity.length_regulator import HardUpsampling
    from seamless_communication.models.unity.nar_decoder_frontend import NARDecoderFrontend

    model, frontend = agent.model, agent.model.decoder_frontend
    tokenizer = SimpleNamespace(text_tokenizer=frontend.text_tokenizer, char_tokenizer=frontend.char_tokenizer,
        tag_manager=frontend.tag_manager, unk_idx=frontend.unk_idx, pad_idx=frontend.pad_idx)
    for name in ("indices_to_subwords", "text_to_char_seqs", "count_character_length_in_subword", "get_char_seqs"):
        setattr(tokenizer, name, MethodType(getattr(NARDecoderFrontend, name), tokenizer))
    for module in model.modules():
        if isinstance(module, HardUpsampling):
            _hard_upsampling(module)

    class Units(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.prepared = None
            frontend.text_to_char_seqs = lambda unused: self.prepared

        def forward(self, feature, chars, lengths, character_counts):
            self.prepared = (chars, lengths, character_counts)
            output, _, durations = self.model(feature, None, text_seqs=chars,
                duration_factor=agent.d_factor)
            return output.logits[0].argmax(dim=-1), durations

    dimension = model.model_dim
    examples = (torch.zeros(1, 5, dimension), torch.ones(1, 5, dtype=torch.int64),
                torch.tensor([5], dtype=torch.int64), torch.ones(1, 5, dtype=torch.int64))
    graph = Graph(Units(), examples, [[1, -1, dimension], [1, -1], [1], [1, -1]], root/"text-to-units")

    def policy(self, states):
        if states.target_finished:
            return WriteAction("", finished=True)
        if len(states.source_token_list) < 2:
            return WriteAction("", finished=True) if states.source_finished else ReadAction()
        chars, lengths, counts = tokenizer.text_to_char_seqs(states.source_indices)
        units, duration_array = graph.infer([states.source, chars, lengths, counts])
        durations = duration_array[0]
        if states.source_finished and states.duration_start_index > 0:
            if int(durations[states.duration_start_index:].sum()) == 0:
                return WriteAction("", finished=True)
            states.duration_start_index = max(states.duration_start_index-1, 0)
        current = int(durations[states.duration_start_index:].sum())
        if current < self.min_unit_chunk_size:
            if not states.source_finished:
                return ReadAction()
            if current == 0:
                return WriteAction("", finished=True)
        offset = int(durations[:states.duration_start_index].sum())
        decoded = self.token_decoder(torch.from_numpy(units[offset:])[None])
        states.duration_start_index = len(durations)-1
        return WriteAction(TextSegment(content=decoded, finished=states.source_finished,
            tgt_lang=states.tgt_lang), finished=states.source_finished)

    agent.policy = MethodType(policy, agent)
    agent.model = None


def _vocoder(agent, root):
    import torch
    from torch import nn

    model = agent.vocoder
    mapping = model.lang_spkr_idx_map
    class Vocoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.generator = model.code_generator

        def forward(self, units, language, speaker):
            return self.generator({"code": units, "lang": language, "spkr": speaker}, False)

    graph = Graph(Vocoder(), (torch.ones(1, 50, dtype=torch.int64), torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, dtype=torch.int64)), [[1, -1], [1, 1], [1, 1]], root/"vocoder", static_inputs=True)
    def vocoder(units, language, speaker, dur_prediction=False):
        if dur_prediction:
            raise ValueError("Official Seamless streaming disables vocoder duration prediction")
        if language not in mapping["multilingual"]:
            raise ValueError("Unsupported official vocoder language")
        language_id = mapping["multilingual"][language]
        speaker_id = mapping["multispkr"][language][0] if speaker == -1 else speaker
        output = graph.infer([np.asarray(units, dtype=np.int64).reshape(1, -1),
            np.array([[language_id]], np.int64), np.array([[speaker_id]], np.int64)])[0]
        return torch.from_numpy(output)
    agent.vocoder = vocoder


def install(agent, snapshot):
    import torch
    from seamless_communication.streaming.agents.online_feature_extractor import OnlineFeatureExtractorAgent
    from seamless_communication.streaming.agents.offline_w2v_bert_encoder import OfflineWav2VecBertEncoderAgent
    from seamless_communication.streaming.agents.online_text_decoder import UnitYMMATextDecoderAgent
    from seamless_communication.streaming.agents.online_unit_decoder import NARUnitYUnitDecoderAgent
    from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
    from .s3_ov_features import KaldiFeatures

    torch.set_num_threads(4)
    root = Path('/tmp/seamless-openvino')/Path(snapshot).name
    for module in agent.module_list:
        if isinstance(module, OnlineFeatureExtractorAgent):
            features = KaldiFeatures(root/"features", center=False)
            def convert(data, features=features):
                return {"fbank": torch.from_numpy(features(data["waveform"].numpy())[0])}
            module.convert_to_fbank = convert
        elif isinstance(module, OfflineWav2VecBertEncoderAgent):
            module.model = _encoder(module.model, root)
        elif isinstance(module, UnitYMMATextDecoderAgent):
            _decoder(module, root)
        elif isinstance(module, NARUnitYUnitDecoderAgent):
            _unit_decoder(module, root)
        elif isinstance(module, VocoderAgent):
            _vocoder(module, root)
        else:
            raise RuntimeError("Unexpected official Seamless streaming stage")
        gc.collect()
