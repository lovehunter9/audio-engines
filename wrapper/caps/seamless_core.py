"""Official SeamlessStreaming policies with local, shared-cache asset paths."""
import argparse
import copy
import os
from pathlib import Path


class Engine:
    def __init__(self, repo, *, dtype="fp16", threshold=0.5):
        import torch
        from fairseq2.assets import InProcAssetMetadataProvider, asset_store
        from huggingface_hub import snapshot_download
        from seamless_communication.streaming.agents.detokenizer import UnitYDetokenizerAgent
        from seamless_communication.streaming.agents.offline_w2v_bert_encoder import OfflineWav2VecBertEncoderAgent
        from seamless_communication.streaming.agents.online_feature_extractor import OnlineFeatureExtractorAgent
        from seamless_communication.streaming.agents.online_text_decoder import UnitYMMATextDecoderAgent
        from seamless_communication.streaming.agents.online_unit_decoder import NARUnitYUnitDecoderAgent
        from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
        from seamless_communication.streaming.agents.unity_pipeline import UnitYAgentTreePipeline, UnitYPipelineMixin

        if not torch.cuda.is_available():
            raise RuntimeError("SeamlessStreaming CUDA engine requires CUDA")
        if dtype not in ("fp16", "fp32"):
            raise ValueError("--dtype must be fp16 or fp32")
        snap = Path(snapshot_download(repo, local_files_only=True,
                                     cache_dir=os.environ.get("HF_HUB_CACHE"),
                                     token=os.environ.get("HF_TOKEN") or None))
        def uri(filename):
            path = snap / filename
            if not path.is_file():
                raise RuntimeError("model snapshot is missing %s" % filename)
            return path.as_uri()
        provider = InProcAssetMetadataProvider([
            {"name": "seamless_streaming_unity@user",
             "checkpoint": uri("seamless_streaming_unity.pt"),
             "tokenizer": uri("tokenizer.model"),
             "char_tokenizer": uri("spm_char_lang38_tc.model")},
            {"name": "seamless_streaming_monotonic_decoder@user",
             "checkpoint": uri("seamless_streaming_monotonic_decoder.pt")},
            {"name": "vocoder_v2@user", "checkpoint": uri("vocoder_v2.pt")},
        ])
        asset_store.user_metadata_providers.append(provider)
        self.languages = set(asset_store.retrieve_card("seamless_streaming_unity").field("langs").as_(list))
        self.speech_languages = set(asset_store.retrieve_card("vocoder_v2").field("model_config").as_(dict)["lang_spkr_idx_map"]["multilingual"])

        class JointPipeline(UnitYAgentTreePipeline):
            pipeline = {
                OnlineFeatureExtractorAgent: [OfflineWav2VecBertEncoderAgent],
                OfflineWav2VecBertEncoderAgent: [UnitYMMATextDecoderAgent],
                UnitYMMATextDecoderAgent: [UnitYDetokenizerAgent, NARUnitYUnitDecoderAgent],
                UnitYDetokenizerAgent: [],
                NARUnitYUnitDecoderAgent: [VocoderAgent],
                VocoderAgent: [],
            }

        parser = argparse.ArgumentParser(conflict_handler="resolve")
        JointPipeline.add_args(parser)
        args = parser.parse_args([])
        args.device = "cuda:0"
        args.dtype = dtype
        args.fp16 = False
        args.task = "s2st"
        args.tgt_lang = "eng"
        args.min_starting_wait_w2vbert = 192
        args.decision_threshold = threshold
        args.no_early_stop = True
        args.max_len_a = 0
        args.max_len_b = 200
        args.min_unit_chunk_size = 50
        args.denormalize = True
        args.sample_rate = 16000
        models = UnitYPipelineMixin.load_model(args)
        speech_modules = {cls: cls.from_args(args, **models) for cls in JointPipeline.pipeline}
        graph = {speech_modules[cls]: [speech_modules[child] for child in children]
                 for cls, children in JointPipeline.pipeline.items()}
        self.speech_agent = UnitYAgentTreePipeline.__new__(JointPipeline)
        from simuleval.agents import TreeAgentPipeline
        TreeAgentPipeline.__init__(self.speech_agent, graph, args)
        text_graph = {key: value for key, value in graph.items()
                      if key not in (speech_modules[NARUnitYUnitDecoderAgent], speech_modules[VocoderAgent])}
        text_graph[speech_modules[UnitYMMATextDecoderAgent]] = [speech_modules[UnitYDetokenizerAgent]]
        self.text_agent = UnitYAgentTreePipeline.__new__(JointPipeline)
        TreeAgentPipeline.__init__(self.text_agent, text_graph, copy.copy(args))

    def session(self, target, *, text_only=False):
        if not text_only and target not in self.speech_languages:
            raise ValueError("target language has text output only; request response_format=text")
        agent = self.text_agent if text_only else self.speech_agent
        return agent, agent.build_states()

    def push(self, session, samples, target, finished):
        import torch
        from simuleval.data.segments import SpeechSegment

        agent, states = session
        with torch.inference_mode():
            output = agent.pushpop(SpeechSegment(content=samples, sample_rate=16000,
                                                tgt_lang=target, finished=finished), states)
        pieces = []
        outputs = output if isinstance(output, list) else [output]
        for segment in outputs:
            if segment.is_empty:
                continue
            if segment.data_type == "text":
                if segment.content:
                    pieces.append(("text", str(segment.content)))
            elif segment.data_type == "speech" and len(segment.content):
                pieces.append(("samples", segment.content))
        return pieces, all(segment.finished for segment in outputs)
