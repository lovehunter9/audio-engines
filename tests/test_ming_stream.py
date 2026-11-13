import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from wrapper.caps import ming_ov_stream, ming_stream


class TailTests(unittest.TestCase):
    def test_irregular_chunks_match_whole_utterance_tail_trim(self):
        sr = 1000
        wave = np.r_[np.zeros(137), np.full(1555, .1), np.zeros(1200)].astype("float32")
        tail = ming_stream.SilenceTail(sr)
        parts = [tail.push(wave[i:i + 113]) for i in range(0, len(wave), 113)]
        parts.append(tail.push([], final=True))
        np.testing.assert_array_equal(np.concatenate(parts), wave[:2000])

    def test_silent_and_invalid_outputs_fail(self):
        for wave in (np.zeros(1000), np.full(100, np.nan)):
            with self.subTest(wave=wave[:1]), self.assertRaises(RuntimeError):
                ming_stream.SilenceTail(1000).push(wave, final=True)


class BackendStreamTests(unittest.TestCase):
    def test_native_clone_and_design_use_incremental_path_with_original_identity(self):
        from pathlib import Path
        from wrapper.caps import ming_backend

        model = SimpleNamespace(sample_rate=24000, _prompt_cache={})
        backend = ming_backend.MingBackend(model)
        requests = []

        def generate(model, kwargs, ctx=None):
            requests.append(dict(kwargs))
            if kwargs.get("prompt_wav_path"):
                self.assertTrue(Path(kwargs["prompt_wav_path"]).is_file())
                model._prompt_cache[kwargs["prompt_wav_path"]] = "prepared"
            yield np.full(2400, .1, dtype="float32")
            yield np.full(2400, .2, dtype="float32")

        with patch("wrapper.caps.ming_stream.iter_synthesize", generate):
            native = list(backend.iter_speak_native({"ip": "US_James"}, "Hello.", instruction="ignored"))
            cloned = list(backend.iter_clone("Hello.", np.full(2400, .1), 24000, "reference", instruction="calm"))
            designed = list(backend.iter_design("warm", "Hello."))
        for chunks in (native, cloned, designed):
            self.assertEqual(len(chunks), 2)
            self.assertEqual(sum(len(wave) for wave, sr in chunks), 4800)
            self.assertTrue(all(sr == 24000 for wave, sr in chunks))
        self.assertEqual([r["task"] for r in requests], ["IP", "TTS", "VOICE_DESIGN"])
        self.assertEqual(requests[0]["controls"], {"ip": "US_James"})
        self.assertNotIn("instruction", requests[0])
        self.assertEqual(requests[1]["instruction"], "calm")
        self.assertEqual(requests[1]["prompt_text"], "reference")
        self.assertEqual(requests[2]["controls"], {"style": "warm"})
        self.assertFalse(Path(requests[1]["prompt_wav_path"]).exists())
        self.assertEqual(model._prompt_cache, {})


class DecoderTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from audio_tokenizer.vae_modules import Decoder
        except ImportError:
            self.skipTest("Ming tokenizer and torch are required")
        torch.set_num_threads(1)
        self.torch = torch
        torch.manual_seed(42)
        args = dict(vocab_size=16, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=2048, attention_dropout=0)
        self.decoder = Decoder(args, output_dim=4, latent_dim=8, patch_size=4).eval()
        self.decoder.decoder.config._attn_implementation = "eager"

    def runner(self):
        decoder = self.decoder.decoder

        class Runner:
            def reset(self):
                self.cache = None

            def step(self, hidden):
                output = decoder(inputs_embeds=hidden, past_key_values=self.cache, use_cache=True)
                self.cache = output.past_key_values
                return output.last_hidden_state

        return Runner()

    @staticmethod
    def outputs(module, *inputs):
        result = module(*inputs)
        return list(result) if isinstance(result, tuple) else [result]

    def test_intel_chunk_boundaries_match_full_decoder(self):
        torch, decoder = self.torch, self.decoder
        heads = {}
        for mode in ("first", "middle", "last", "whole"):
            for n in range(1, ming_stream.STREAM_FRAMES + 1):
                heads[mode, n] = ming_ov_stream.Head(decoder.head, mode).module
        upsample = ming_ov_stream.Upsample(decoder).module
        stream = ming_ov_stream.Decoder(upsample, self.runner(), heads, 4)
        with torch.inference_mode(), patch.object(ming_ov_stream, "_outputs", self.outputs):
            for length in (1, 4, 8, 9, 16, 17, 23, 32):
                with self.subTest(length=length):
                    latent = torch.randn(1, length, 8)
                    full = decoder.low_level_reconstruct(latent, last_chunk=True,
                                                        stream_state=(None, None, None))[0]
                    cache, state, waves = None, (None, None, None), []
                    for at in range(0, length, ming_stream.STREAM_FRAMES):
                        batch = latent[:, at:at + ming_stream.STREAM_FRAMES]
                        wave, state, cache = stream(batch, cache, True, state,
                                                    at + ming_stream.STREAM_FRAMES >= length)
                        waves.append(wave)
                    actual = torch.cat(waves, dim=-1)
                    torch.testing.assert_close(actual, full, rtol=2e-5, atol=2e-5)

    def test_overlap_head_export_uses_a_valid_window_state(self):
        torch, decoder = self.torch, self.decoder
        overlap = decoder.head.istft.buffer_len
        example = (torch.zeros(1, 32, 32), torch.zeros(1, overlap), torch.ones(1, overlap))
        for mode in ("middle", "last"):
            head = ming_ov_stream.Head(decoder.head, mode).module.eval()
            with torch.inference_mode():
                traced = torch.jit.trace(head, example)
                expected = head(*example)
                actual = traced(*example)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference)

    def test_cuda_yields_audio_before_latent_generation_finishes(self):
        torch, decoder = self.torch, self.decoder
        latent = torch.randn(1, 40, 8)
        consumed = []

        def sample(**kwargs):
            for i in range(40):
                consumed.append(i)
                yield latent[:, i:i + 1], i == 39

        audio = SimpleNamespace(decode=lambda *a, **kw: decoder.low_level_reconstruct(*a, **kw))
        model = SimpleNamespace(sample=sample, audio=audio)
        ming = SimpleNamespace(model=model)
        generator = ming_stream._decode(ming, {}, None)
        first = next(generator)
        self.assertGreater(len(first), 0)
        self.assertLess(len(consumed), 40)
        waves = [first, *generator]
        with torch.inference_mode():
            full = decoder.low_level_reconstruct(latent, last_chunk=True,
                                                stream_state=(None, None, None))[0]
        np.testing.assert_allclose(np.concatenate(waves), full.numpy().reshape(-1),
                                   rtol=2e-5, atol=2e-5)

    def test_cancel_stops_sampling_between_latent_steps(self):
        torch, decoder = self.torch, self.decoder
        consumed = []

        def sample(**kwargs):
            for i in range(40):
                consumed.append(i)
                yield torch.ones(1, 1, 8), False

        class Context:
            canceled = False

            def checkpoint(self):
                if self.canceled:
                    raise RuntimeError("cancelled")

        context = Context()
        audio = SimpleNamespace(decode=lambda *a, **kw: decoder.low_level_reconstruct(*a, **kw))
        generator = ming_stream._decode(SimpleNamespace(model=SimpleNamespace(sample=sample, audio=audio)), {}, context)
        next(generator)
        context.canceled = True
        count = len(consumed)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            next(generator)
        self.assertLessEqual(len(consumed), count + 1)


if __name__ == "__main__":
    unittest.main()
