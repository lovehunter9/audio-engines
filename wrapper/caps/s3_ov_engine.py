"""OpenVINO adapters for the original S3 streaming controller and codec timeline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from .s3_native import OfficialEngine
from .s3_ov_source import definitions


def _record_functions(root):
    import torch
    import soundfile as sf

    def read_audio(path, use_ffmpeg=False):
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000 or wave.ndim != 1:
            raise ValueError("Official make_input must produce mono 16 kHz source audio")
        return wave

    namespace = {"np": np, "torch": torch, "argparse": argparse, "Path": Path,
                 "json": json, "hashlib": hashlib, "time": time, "re": re,
                 "load_audio_text_image_video": read_audio, "OmniDRCA2Engine": object}
    core = definitions(root / "inference/streaming/streaming_core.py",
                       ["evaluate_one_streaming_thinker_deferred_talker"], namespace)
    core.STREAMING_LOOKAHEAD_MS = 162.5
    stream = definitions(root / "inference/streaming/run_block640_lookahead160.py",
                         ["_stream_prefixes"], namespace)
    finalizer = definitions(root / "inference/streaming/infer_deferred_s2st.py",
                            ["StreamingS2STBackend"], {**namespace, "s2tt": core})
    return core, stream._stream_prefixes, finalizer.StreamingS2STBackend.finish_sample


class AudioEncoder:
    def __init__(self, root, compiled, pcm, stream):
        self.compiled, self.pcm, self._stream = compiled, pcm, stream
        from .s3_ov_features import WhisperFeatures
        self.frontend = WhisperFeatures(root, Path(root)/"cache/openvino/frontend")
        self.true_streaming = True
        self.adaptor = SimpleNamespace(streaming_chunk_size=8)
        self.wave = None

    def __call__(self, wave):
        import torch

        if self.pcm.config.get("paced", True):
            wait = self.pcm.play.start + len(wave) / 16000. - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self.pcm.prefix(self.wave, len(wave))
        features, lengths = self.frontend(wave)
        result = self.compiled([features, lengths])
        embeds = np.asarray(result[0]).copy()
        length = int(np.asarray(result[1]).reshape(-1)[0])
        return torch.from_numpy(embeds[0, :length]).to(torch.bfloat16).contiguous()

    def stream_prefixes(self, wave, chunk_samples, lookahead_samples):
        self.wave = wave
        return self._stream(self, wave, chunk_samples, lookahead_samples)


class CodecTimeline:
    def __init__(self, talker, pcm, maximum=2048, seed=0):
        self.talker, self.pcm = talker, pcm
        self.maximum, self.seed = maximum, seed
        self.reset()

    def reset(self):
        self.talker.reset()
        self.rng = np.random.default_rng(self.seed)
        self.hidden = None
        self.available = self.real_end = 0
        self.specs = []
        self.next_spec = 0
        self.previous_boundary = -1
        self.position = -1
        self.started = self.done = False
        self.active = None
        self.tokens = []
        self.pending_token = None
        self.expect = False
        self.steps = 0

    def _next_bos(self):
        if self.next_spec == len(self.specs):
            return None
        spec = self.specs[self.next_spec]
        return max(spec["start"] + 5, self.previous_boundary + 1)

    def _sample(self, logits):
        logits = np.asarray(logits, dtype=np.float32).copy()
        logits[[59049, 59051]] -= 10000.
        logits /= np.float32(.8)
        top = np.partition(logits, -5)[-5]
        logits[logits < top] = -np.inf
        order = np.argsort(logits)[::-1]
        selected = logits[order]
        probabilities = np.exp(selected - selected[0])
        probabilities /= probabilities.sum()
        remove = np.cumsum(probabilities) - probabilities > np.float32(.8)
        selected[remove] = -np.inf
        filtered = np.full_like(logits, -np.inf)
        filtered[order] = selected
        probabilities = np.exp(filtered - np.max(filtered))
        probabilities /= probabilities.sum()
        # The author samples the entire codec vocabulary, including ignored gaps.
        noise = self.rng.exponential(size=len(probabilities)).astype(np.float32)
        return int(np.argmax(probabilities / noise))

    def write(self, hidden, prefix_len, real_end, *, final, finalize_only=False):
        if self.done or real_end < self.real_end:
            raise RuntimeError("The original S3 codec timeline regressed or was already finalized")
        self.hidden = np.asarray(hidden, dtype=np.float32)
        self.real_end = real_end
        self.available = len(self.hidden) if final else real_end
        if finalize_only:
            if not self.specs:
                raise RuntimeError("Cannot finalize a codec timeline with no WRITE")
            self.specs[-1]["final"] = True
            if self.active is not None:
                self.active["final"] = True
            elif self.started and self.next_spec == len(self.specs) and self.tokens[-1] == 59050:
                self.done = True
                return
        else:
            self.specs.append({"start": prefix_len, "final": bool(final)})
        if not self.started:
            bos = self._next_bos()
            if bos is None or bos >= self.available:
                return
            if bos:
                self.talker.forward(self.hidden[:bos])
            logits = self.talker.forward(self.hidden[bos:bos+1], codec_ids=np.array([59049]))
            self.position, self.expect = bos, True
            self.active = self.specs[0]
            self.next_spec = 1
            self.tokens.append(59049)
            self.started = True
        else:
            logits = self._advance()
            if logits is None:
                return
        while True:
            if self.steps >= self.maximum:
                raise RuntimeError("Original Talker token limit reached before final EOS")
            token = self._sample(logits)
            self.steps += 1
            if self.expect:
                if token in (59049, 59051) or not 0 <= token < self.talker.codec_unit:
                    raise RuntimeError("Unexpected token in the original codec envelope: " + str(token))
                self.tokens.append(token)
                self.pcm.on_codec(token)
                if token == 59050:
                    self.previous_boundary = self.position + 1
                    final_envelope = self.active["final"]
                    self.active = None
                    if final_envelope:
                        self.done = True
                        return
                    self.expect = False
                self.pending_token = token
            else:
                self.pending_token = None
            logits = self._advance()
            if logits is None:
                return

    def _advance(self):
        position = self.position + 1
        if position >= self.available:
            return None
        codec = self.pending_token
        expect = codec is not None and codec != 59050
        if codec is None:
            bos = self._next_bos()
            if bos is not None and position > bos:
                raise RuntimeError("Missed an injected S3 BOS position")
            if position == bos:
                codec, expect = 59049, True
                self.active = self.specs[self.next_spec]
                self.next_spec += 1
                self.tokens.append(59049)
        logits = self.talker.forward(self.hidden[position:position+1],
                                     codec_ids=None if codec is None else np.array([codec]))
        self.position, self.expect, self.pending_token = position, expect, None
        return logits


class Backend:
    def __init__(self, thinker, talker, pcm, finalizer):
        self.thinker, self.pcm, self._finalizer = thinker, pcm, finalizer
        self.timeline = CodecTimeline(talker, pcm)
        self.codec_bos_id = self.codec_final_bos_id = 59049
        self.codec_eos_id, self.codec_pause_id = 59050, 59051
        self.codec_boundary_policy = "unified_eos"
        self.s2st_engine = SimpleNamespace(_get_talker_hf_config=lambda:SimpleNamespace(codec_unit=talker.codec_unit))

    def start_sample(self, index):
        self.timeline.reset()
        self._segments = []
        self._global_codec_full = None
        self.pcm.begin(index)

    def gate_decision(self, prompt, segments):
        answer = self.thinker.gate(prompt, segments, self.pcm.gate())
        self.pcm.gates[-1]["write"] = answer
        return answer

    def generate_text(self, prompt, segments):
        text = self.thinker.generate(prompt, segments)
        if text:
            self.pcm.text_events.append(dict(source_ms=self.pcm.source_ms,
                wall_s=time.monotonic()-self.pcm.play.start,text=text))
            self.pcm.fix_speaker()
        return text

    def set_write_context(self, *, final_write):
        self.final_write = final_write

    def generate_forced_speech(self, prompt, segments, text, *, record_segment=True):
        hidden, prefix, end, blank = self.thinker.alignment(
            prompt, segments, text, self.final_write, finalize_only=not record_segment and not text)
        if blank != 5:
            raise RuntimeError("Original S3 blank count must remain five")
        self.timeline.write(hidden, prefix, end, final=self.final_write,
                            finalize_only=not record_segment and not text)
        if self.final_write:
            if not self.timeline.done or self.timeline.tokens[-1] != self.codec_eos_id:
                raise RuntimeError("OpenVINO Talker did not reach the original final EOS")
            self._global_codec_full = np.asarray(self.timeline.tokens, dtype=np.int64)
        if record_segment:
            self._segments.append(dict(text=text,decoded_text=text,global_final_write=self.final_write))
        return text

    def finalize_without_new_write(self, prompt, segments):
        self.final_write = True
        self.generate_forced_speech(prompt, segments, "", record_segment=False)
        self._segments[-1]["global_final_write"] = True

    def finish_sample(self):
        return self._finalizer(self)


class OpenVINOEngine(OfficialEngine):
    def _start(self):
        import torch
        torch.set_num_threads(4)
        from .s3_ov_graphs import export_audio_encoder
        from .s3_ov_llm import Thinker
        from .s3_ov_talker import Talker
        from .s3_ov_pcm import PCMRuntime

        cache = self.root / "cache/openvino"
        self.core, stream, finalizer = _record_functions(self.root)
        policies = {"adaptive":(.1,.99,.5),"fixed-0.1":(.1,.1,1.),"fixed-0.5":(.5,.5,1.)}
        base, cap, power = policies[self.preset]
        config = dict(deliver_read_hidden=True,fm_chunk=8,diagnostic=True,base=base,cap=cap,
                      saturation=1.,power=power,paced=True,
                      fm_config=str(self.root/'configs/fm_config.yaml'),
                      vocoder_checkpoint=str(self.root/'models/vocoder/model.pt'),
                      vocoder_config=str(self.root/'models/vocoder/config.yaml'))
        pcm = PCMRuntime(self.root/'pcm', "GPU", config, source_root=self.root, graph_root=cache/'pcm')
        thinker = Thinker(self.root/'models/streaming',cache/'thinker')
        talker = Talker(self.root/'models/streaming',cache/'talker')
        compiled, evidence = export_audio_encoder(self.root,self.root/'models/streaming',cache/'encoder')
        self.encoder = AudioEncoder(self.root,compiled,pcm,stream)
        self.backend = Backend(thinker,talker,pcm,finalizer)
        self.args = SimpleNamespace(encoder_chunk_ms=640.,audio_sample_rate=16000)
        self.index = 0
        self.encoder_evidence = evidence

    def close(self):
        pass

    def translate(self, audio, source, target):
        if (source,target) not in (("Chinese","English"),("English","Chinese")):
            raise ValueError("S3 supports Chinese/English bidirectional translation")
        import uuid
        with self.lock:
            job = self.root/'requests'/uuid.uuid4().hex
            job.mkdir(parents=True)
            incoming = job/(job.name+'.wav')
            incoming.write_bytes(audio)
            inputs = job/'input.jsonl'
            self._make_input(incoming,inputs,source)
            row = json.loads(inputs.read_text())
            started = time.monotonic()
            self.backend.start_sample(self.index)
            result = self.core.evaluate_one_streaming_thinker_deferred_talker(
                self.index,row,self.encoder,self.backend,self.args)
            codec, segments = self.backend.finish_sample()
            result.update(codec_len=len(codec),codec_segments=segments,
                          live_pcm=self.backend.pcm.finish(result['key']),
                          elapsed_s=time.monotonic()-started,
                          neural_device="OpenVINO GPU")
            self.index += 1
            wav = Path(result['live_pcm']['wav'])
            (job/'result.json').write_text(json.dumps(result,ensure_ascii=False))
            return result,wav.read_bytes()
