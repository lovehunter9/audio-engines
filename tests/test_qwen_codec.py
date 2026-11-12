import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from wrapper.caps import qwen_codec


class CachedCodecTest(unittest.TestCase):
    def test_accumulated_reference_and_sliding_windows_do_not_feed_tokens_twice(self):
        torch = types.SimpleNamespace(cat=lambda parts, dim: np.concatenate(parts, axis=dim),
                                      equal=np.array_equal)
        with patch.dict(sys.modules, torch=torch):
            state = qwen_codec._CachedCodec(types.SimpleNamespace(total_upsample=2))
        advanced = []

        def advance(codes, context):
            advanced.extend(codes.flatten().tolist())
            return np.repeat(codes, 2, axis=-1)

        state._advance = advance
        codes = np.arange(40).reshape(1, 1, -1)
        np.testing.assert_array_equal(state.decode(codes[..., :18]), np.repeat(codes[..., :18], 2, -1))
        np.testing.assert_array_equal(state.decode(codes[..., :26]), np.repeat(codes[..., :26], 2, -1))
        np.testing.assert_array_equal(state.decode(codes[..., 1:34]), np.repeat(codes[..., 1:34], 2, -1))
        np.testing.assert_array_equal(state.decode(codes[..., 9:40]), np.repeat(codes[..., 9:40], 2, -1))
        self.assertEqual(advanced, list(range(40)))

    def test_identical_silence_codes_still_advance_each_stream_chunk(self):
        torch = types.SimpleNamespace(cat=lambda parts, dim: np.concatenate(parts, axis=dim),
                                      equal=np.array_equal)
        with patch.dict(sys.modules, torch=torch):
            state = qwen_codec._CachedCodec(types.SimpleNamespace(total_upsample=2), stream_chunk_size=8)
        advanced = []

        def advance(codes, context):
            start = sum(advanced)
            advanced.append(codes.shape[-1])
            return np.repeat(np.arange(start, start + codes.shape[-1]).reshape(1, 1, -1), 2, -1)

        state._advance = advance
        for count in [8, 16, 24, 32, 33, 33, 26]:
            result = state.decode(np.zeros((1, 1, count), dtype='int64'))
            self.assertEqual(result.shape[-1], count * 2)
        self.assertEqual(advanced, [8, 8, 8, 8, 8, 8, 1])
        self.assertEqual(state.codes.shape[-1], 49)

    def test_decoder_override_is_scoped_to_one_request_and_restored_on_failure(self):
        class Decoder:
            pre_transformer = object()

            def chunked_decode(self, codes):
                return codes

        decoder = Decoder()
        model = types.SimpleNamespace(speech_tokenizer=types.SimpleNamespace(model=types.SimpleNamespace(decoder=decoder)))
        state = types.SimpleNamespace(decode=lambda codes: codes + 1)
        with patch.object(qwen_codec, "_CachedCodec", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "canceled"):
                with qwen_codec.causal_decode(model):
                    self.assertEqual(decoder.chunked_decode(2), 3)
                    raise RuntimeError("canceled")
        self.assertNotIn("chunked_decode", vars(decoder))
        self.assertEqual(decoder.chunked_decode(2), 2)

    def test_non_codec_backends_keep_the_original_path(self):
        with patch.object(qwen_codec, "_CachedCodec") as factory:
            with qwen_codec.causal_decode(types.SimpleNamespace()):
                pass
        factory.assert_not_called()
