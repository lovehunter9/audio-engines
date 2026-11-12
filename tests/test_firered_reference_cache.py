import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from wrapper.caps import firered


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype="float32")

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.values

    def clone(self):
        return Tensor(self.values.copy())


class ReferenceCacheTest(unittest.TestCase):
    def model(self):
        state = {"active": False, "entries": 0}

        class Autocast:
            def __enter__(self):
                state.update(active=True, entries=state["entries"] + 1)

            def __exit__(self, *args):
                state["active"] = False

        def raw_encode(audio):
            self.assertTrue(state["active"])
            return Tensor([[audio.values.sum()]])

        original_encode = mock.Mock(side_effect=raw_encode)
        acoustic_encoder = SimpleNamespace(forward=original_encode)

        def tokenize(audio, sr):
            self.assertFalse(state["active"])
            return acoustic_encoder.forward(audio)

        encoder = mock.Mock(side_effect=tokenize)
        model = SimpleNamespace(_tokenize_audio=encoder,
                                redae=SimpleNamespace(encoder=acoustic_encoder))
        times = {}
        autocast = mock.Mock(side_effect=lambda *a, **kw: Autocast())
        with mock.patch.dict("sys.modules", {
            "torch": SimpleNamespace(autocast=autocast, bfloat16="bfloat16"),
        }):
            firered._install_firered_redae_cache(model, times)
        model.autocast_calls = autocast
        model.autocast_state = state
        return model, encoder, times

    def test_encoding_receives_original_reference_without_extra_silence(self):
        for sr in (16000, 24000, 44100, 48000):
            with self.subTest(sr=sr):
                model, encoder, _ = self.model()
                source = Tensor(np.linspace(-.5, .5, sr)[None])
                before = source.values.copy()
                model._tokenize_audio(source, sr)
                audio, rate = encoder.call_args.args
                self.assertIs(audio, source)
                self.assertEqual(rate, sr)
                np.testing.assert_array_equal(audio.values, before)

    def test_cached_return_values_cannot_mutate_the_saved_encoding(self):
        model, encoder, times = self.model()
        source = Tensor([[.2, .3, .4]])
        first = model._tokenize_audio(source, 24000)
        expected = first.values.copy()
        second = model._tokenize_audio(source.clone(), 24000)
        second.values[:] = -1
        third = model._tokenize_audio(source, 24000)
        np.testing.assert_array_equal(third.values, expected)
        self.assertEqual(encoder.call_count, 1)
        self.assertEqual(times["redae_hits"], 2)
        self.assertEqual(model.autocast_state["entries"], 1)
        model.autocast_calls.assert_called_once_with("cpu", dtype="bfloat16")

    def test_reference_content_and_sample_rate_invalidate_cache(self):
        model, encoder, _ = self.model()
        source = Tensor([[.2, .3, .4]])
        model._tokenize_audio(source, 24000)
        source.values[0, 1] += .1
        model._tokenize_audio(source, 24000)
        model._tokenize_audio(source, 48000)
        self.assertEqual(encoder.call_count, 3)
