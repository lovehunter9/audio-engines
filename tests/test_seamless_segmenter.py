import unittest

import numpy as np

from wrapper.caps.seamless_segmenter import UtteranceBuffer


class SegmenterTests(unittest.TestCase):
    def test_packet_boundaries_do_not_change_audio_or_utterances(self):
        audio = np.concatenate([np.zeros(24000), np.full(24000, 0.1), np.zeros(24000),
                                np.full(16000, 0.2), np.zeros(16000)]).astype(np.float32)
        outputs = []
        for size in [len(audio), 5120, 1137]:
            state = UtteranceBuffer(lambda frame: float(np.max(frame) > 0.05))
            events = []
            for start in range(0, len(audio), size):
                events.extend(state.feed(audio[start:start + size], False))
            events.extend(state.feed([], True))
            outputs.append(events)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(sum(final for _, final in outputs[0]), 2)
        self.assertTrue(any(not final for _, final in outputs[0]))

    def test_short_input_and_final_partial_frame_are_preserved(self):
        state = UtteranceBuffer(lambda frame: float(np.max(frame) > 0.05))
        events = state.feed(np.full(333, 0.1), True)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0][1])
        self.assertEqual(len(events[0][0]), 333)

    def test_exact_chunk_boundary_still_flushes_decoder_eof(self):
        state = UtteranceBuffer(lambda _: 1.0)
        events = state.feed(np.full(5120, 0.1), True)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1], ([], True))

    def test_silence_produces_no_translation_input(self):
        state = UtteranceBuffer(lambda _: 0.0)
        self.assertEqual(state.feed(np.zeros(48000), True), [])

    def test_subthreshold_pause_keeps_one_utterance(self):
        state = UtteranceBuffer(lambda frame: float(np.max(frame) > 0.05))
        data = np.concatenate([np.full(8000, 0.1), np.zeros(16000), np.full(8000, 0.1)])
        events = state.feed(data, True)
        self.assertEqual(sum(last for _, last in events), 1)
        self.assertEqual(sum(len(samples) for samples, _ in events), len(data))


if __name__ == '__main__':
    unittest.main()
