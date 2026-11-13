import contextlib
import sys
import types
import unittest
from unittest.mock import patch

from wrapper.caps.seamless_core import Engine


class CoreDrainTests(unittest.TestCase):
    def run_engine(self, outputs, finished, samples=None):
        calls = []
        source = object()
        state = types.SimpleNamespace(source=[[0.25]], source_finished=False)

        def pushpop(segment, states):
            calls.append(segment)
            return next(outputs)

        def push_impl(decoder, segment, states, upstream):
            calls.append(segment)

        agent = types.SimpleNamespace(source_module=source, pushpop=pushpop,
                                      push_impl=push_impl, pop=lambda states: next(outputs))
        torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        segments = types.SimpleNamespace(
            SpeechSegment=lambda **kw: types.SimpleNamespace(**kw),
            EmptySegment=lambda **kw: types.SimpleNamespace(content=[], **kw),
        )
        with patch.dict(sys.modules, {'torch': torch, 'simuleval.data.segments': segments}):
            engine = Engine.__new__(Engine)
            engine.decoder = object()
            result = engine._decode((agent, {source: state}), [0.5] if samples is None else samples, 'eng', finished)
        return result, calls, state

    @staticmethod
    def segment(kind, content, finished=False):
        return types.SimpleNamespace(data_type=kind, content=content, finished=finished,
                                     is_empty=not bool(content))

    def test_eof_drains_both_text_and_speech_without_replaying_samples(self):
        outputs = iter([
            [self.segment('text', 'first'), self.segment('speech', [0.1])],
            [self.segment('text', 'last', True), self.segment('speech', [0.2], True)],
        ])
        (pieces, done), calls, state = self.run_engine(outputs, True)
        self.assertEqual(pieces, [('text', 'first'), ('samples', [0.1]),
                                  ('text', 'last'), ('samples', [0.2])])
        self.assertTrue(done)
        self.assertEqual([s.content for s in calls], [[0.5], []])
        self.assertTrue(all(s.finished for s in calls))
        self.assertEqual(state.source, [[0.25]])
        self.assertTrue(state.source_finished)

    def test_empty_final_packet_closes_decoder_without_reusing_waveform(self):
        (pieces, done), calls, state = self.run_engine(
            iter([[self.segment('text', 'last', True), self.segment('speech', [0.2], True)]]),
            True, samples=[])
        self.assertTrue(done)
        self.assertEqual(pieces, [('text', 'last'), ('samples', [0.2])])
        self.assertEqual([x.content for x in calls], [[]])
        self.assertEqual(state.source, [[0.25]])

    def test_live_input_returns_without_forcing_eof(self):
        (pieces, done), calls, state = self.run_engine(
            iter([[self.segment('text', 'partial')]]), False)
        self.assertEqual(pieces, [('text', 'partial')])
        self.assertFalse(done)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state.source, [[0.25]])

    def test_stalled_eof_raises_instead_of_silently_truncating(self):
        outputs = iter([[self.segment('text', '')]] * 2)
        with self.assertRaisesRegex(RuntimeError, 'stalled'):
            self.run_engine(outputs, True)


if __name__ == '__main__':
    unittest.main()
