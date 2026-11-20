import unittest

from wrapper.caps import diar_asr


class ParseMossTest(unittest.TestCase):
    def test_reads_every_segment_and_drops_an_unfinished_tail(self):
        raw = ("[0.48][S01]Welcome everyone[1.66][12.26][S02]我们开始吧[13.81]"
               "[14.36][S01]Great, include the[")
        got = diar_asr.parse_moss(raw)
        self.assertEqual(got, [
            {"speaker": "S01", "start": 0.48, "end": 1.66, "text": "Welcome everyone"},
            {"speaker": "S02", "start": 12.26, "end": 13.81, "text": "我们开始吧"},
        ])

    def test_skips_empty_segments(self):
        self.assertEqual(diar_asr.parse_moss("[1][S01] [2]"), [])


class ParseVibeVoiceTest(unittest.TestCase):
    def test_reads_the_json_list_and_numbers_speakers_like_moss(self):
        raw = ('<|im_start|>assistant\n[{"Start":0,"End":15.43,"Speaker":0,"Content":"Hello"},'
               '{"Start":15.43,"End":21.05,"Speaker":1,"Content":"Thanks"}]<|im_end|>')
        got = diar_asr.parse_vibevoice(raw)
        self.assertEqual([(s["speaker"], s["start"], s["text"]) for s in got],
                         [("S01", 0.0, "Hello"), ("S02", 15.43, "Thanks")])

    def test_keeps_complete_objects_when_the_budget_cut_the_list(self):
        raw = ('[{"Start":0,"End":1,"Speaker":0,"Content":"one"},'
               '{"Start":1,"End":2,"Speaker":1,"Content":"two"},{"Start":2,"End"')
        got = diar_asr.parse_vibevoice(raw)
        self.assertEqual([s["text"] for s in got], ["one", "two"])


class JoinedTextTest(unittest.TestCase):
    def test_spaces_only_between_latin_segments(self):
        segs = [{"start": 2, "text": "world"}, {"start": 0, "text": "hello"},
                {"start": 3, "text": "你好"}, {"start": 4, "text": "再见"}]
        self.assertEqual(diar_asr.joined_text(segs), "hello world你好再见")


class KindTest(unittest.TestCase):
    def test_reads_the_family_off_model_type(self):
        self.assertEqual(diar_asr.kind_of({"model_type": "moss_transcribe_diarize"}), diar_asr.KIND_MOSS)
        self.assertEqual(diar_asr.kind_of({"model_type": "vibevoice_asr"}), diar_asr.KIND_VIBEVOICE)
        self.assertIsNone(diar_asr.kind_of({"model_type": "qwen3_asr"}))


if __name__ == "__main__":
    unittest.main()
