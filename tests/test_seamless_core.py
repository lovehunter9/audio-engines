import contextlib
import sys
import types
import unittest
from unittest.mock import patch

from wrapper.caps.seamless_core import Engine


class CoreDrainTests(unittest.TestCase):
    def run_engine(self, outputs, finished):
        calls = []
        source = object()
        state = types.SimpleNamespace(source=[[0.25]])

        def pushpop(segment, states):
            calls.append(segment)
            return next(outputs)

        agent = types.SimpleNamespace(source_module=source, pushpop=pushpop)
        torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        segments = types.SimpleNamespace(SpeechSegment=lambda **kw: types.SimpleNamespace(**kw))
        with patch.dict(sys.modules, {'torch': torch, 'simuleval.data.segments': segments}):
            result = Engine.__new__(Engine)._decode((agent, {source: state}), [0.5], 'eng', finished)
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
        self.assertEqual(state.source, [])

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
