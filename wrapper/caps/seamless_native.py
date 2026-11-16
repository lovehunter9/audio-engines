"""Transport and local assets around Meta's unchanged streaming S2ST agent."""
import argparse
import io
import threading
import time
from pathlib import Path

import numpy as np


def _segments(samples, finished):
    while len(samples) >= 5520 or (finished and samples):
        count = len(samples) if finished and len(samples) < 5520 else 5120
        if finished and count < 400:
            raise ValueError("Seamless requires at least 25 milliseconds of source audio")
        chunk = samples[:count]
        del samples[:count]
        yield chunk, finished and not samples


class Engine:
    sample_rate = 16000

    def __init__(self, snapshot, backend="cuda"):
        import torch
        from fairseq2.assets import InProcAssetMetadataProvider, asset_store
        from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent
        from seamless_communication.streaming.agents.online_text_decoder import UnitYMMATextDecoderAgent

        if backend not in ("cuda", "openvino"):
            raise ValueError("Unsupported Seamless inference backend")
        if backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The native Seamless CUDA runtime requires CUDA")
        self.lock = threading.Lock()
        snapshot = Path(snapshot)
        names = ("seamless_streaming_unity.pt", "seamless_streaming_monotonic_decoder.pt",
                 "tokenizer.model", "spm_char_lang38_tc.model", "vocoder_v2.pt")
        for name in names:
            if not (snapshot / name).is_file():
                raise RuntimeError("Missing official Seamless asset: " + name)
        def uri(name):
            return (snapshot / name).as_uri()
        asset_store.user_metadata_providers.append(InProcAssetMetadataProvider([
            {"name": "seamless_streaming_unity@user", "checkpoint": uri(names[0]),
             "tokenizer": uri(names[2]), "char_tokenizer": uri(names[3])},
            {"name": "seamlessM4T_v2_large@user", "tokenizer": uri(names[2]),
             "char_tokenizer": uri(names[3])},
            {"name": "seamless_streaming_monotonic_decoder@user", "checkpoint": uri(names[1])},
            {"name": "vocoder_v2@user", "checkpoint": uri(names[4])}]))
        parser = argparse.ArgumentParser(conflict_handler="resolve")
        SeamlessStreamingS2STAgent.add_args(parser)
        args = parser.parse_args(["--min-unit-chunk-size", "50"])
        args.device = "cpu" if backend == "openvino" else "cuda:0"
        args.dtype, args.fp16 = ("fp32" if backend == "openvino" else "fp16"), False
        args.task, args.tgt_lang = "s2st", "eng"
        args.min_starting_wait_w2vbert, args.decision_threshold = 192, .5
        args.no_early_stop, args.max_len_a, args.max_len_b = True, 0, 100
        args.denormalize, args.sample_rate = False, 16000
        self.agent = SeamlessStreamingS2STAgent.from_args(args)
        if backend == "openvino":
            from .seamless_ov import install
            install(self.agent, snapshot)
        self.target_languages = set(asset_store.retrieve_card("seamless_streaming_unity").field("unit_langs").as_(list))
        self.pending = []
        decoder = next(module for module in self.agent.module_list if isinstance(module, UnitYMMATextDecoderAgent))
        original = decoder.postprocess
        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            self.pending.extend(result.content.tokens)
            return result
        decoder.postprocess = observe

    def new_session(self, target):
        if target not in self.target_languages:
            raise ValueError("Unsupported Seamless speech target: " + target)
        self.agent.reset()
        self.pending.clear()
        return {"states": self.agent.build_states(), "target": target, "samples": [], "finished": False}

    def push(self, session, samples, finished):
        import torch
        from simuleval.data.segments import SpeechSegment

        if session["finished"]:
            raise ValueError("Session already finished")
        session["samples"].extend(np.asarray(samples, dtype=np.float32).tolist())
        events = []
        with torch.inference_mode():
            for chunk, last in _segments(session["samples"], finished):
                result = self.agent.pushpop(SpeechSegment(content=chunk, sample_rate=16000,
                    tgt_lang=session["target"], finished=last), session["states"])
                if self.pending:
                    text = "".join(self.pending).replace("▁", " ").replace("</s>", "").replace("<s>", "")
                    self.pending.clear()
                    if text:
                        events.append(("text", text))
                if not result.is_empty and len(result.content):
                    audio = np.asarray(result.content, dtype=np.float32)
                    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                    events.append(("pcm", pcm))
                if last:
                    session["finished"] = bool(result.finished)
                    if not session["finished"]:
                        raise RuntimeError("The original Seamless agent did not finish after final input")
                    break
        return events

    def translate(self, data, source, target):
        from ..audioio import decode_mono
        import soundfile as sf

        samples = decode_mono(io.BytesIO(data), 16000)[0].numpy()
        started = time.monotonic()
        with self.lock:
            session = self.new_session(target)
            events = self.push(session, samples, True)
        pcm = b"".join(payload for kind, payload in events if kind == "pcm")
        if not pcm:
            raise RuntimeError("The original Seamless agent produced no speech")
        audio = np.frombuffer(pcm, dtype="<i2")
        output = io.BytesIO()
        sf.write(output, audio, 16000, format="WAV", subtype="PCM_16")
        return {"text": "".join(payload for kind, payload in events if kind == "text").strip(),
                "elapsed_s": time.monotonic()-started}, output.getvalue()
