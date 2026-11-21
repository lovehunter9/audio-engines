import sys
import types
import unittest
from unittest import mock

from wrapper import gpu
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


class TurnsTest(unittest.TestCase):
    SEGS = [{"speaker": "S01", "start": 0.0, "end": 10.0, "text": "a"},
            {"speaker": "S02", "start": 2.0, "end": 4.0, "text": "b"},
            {"speaker": "S03", "start": 3.0, "end": 5.0, "text": "c"},
            {"speaker": "S01", "start": 12.0, "end": 13.0, "text": "d"},
            {"speaker": "S01", "start": 13.0, "end": 14.0, "text": "e"},
            {"speaker": "S02", "start": 20.0, "end": 20.0, "text": "empty"}]

    def turn(self, out):
        return [(t["speaker"], t["start"], t["end"]) for t in out]

    def test_overlapping_turns_keep_every_segment(self):
        self.assertEqual(self.turn(diar_asr.turns(self.SEGS, False)), [
            ("S01", 0.0, 10.0), ("S02", 2.0, 4.0), ("S03", 3.0, 5.0),
            ("S01", 12.0, 13.0), ("S01", 13.0, 14.0)])

    def test_exclusive_turns_give_the_floor_to_the_latest_starter(self):
        self.assertEqual(self.turn(diar_asr.turns(self.SEGS, True)), [
            ("S01", 0.0, 2.0), ("S02", 2.0, 3.0), ("S03", 3.0, 5.0),
            ("S01", 5.0, 10.0), ("S01", 12.0, 14.0)])

    def test_exclusive_turns_never_overlap(self):
        out = diar_asr.turns(self.SEGS, True)
        for prev, cur in zip(out, out[1:]):
            self.assertLessEqual(prev["end"], cur["start"])


class DiarizeTest(unittest.TestCase):
    def test_answers_in_the_diar_shape_from_one_transcription(self):
        segs = [{"speaker": "S01", "start": 0.0, "end": 3.0, "text": "a"},
                {"speaker": "S02", "start": 2.0, "end": 4.0, "text": "b"}]
        orig = diar_asr._transcribe
        diar_asr._transcribe = lambda data, fn, prompt, ctx: ("raw", segs, 42, 4.0, False)
        try:
            body = diar_asr._diarize(b"", "a.wav", True)
        finally:
            diar_asr._transcribe = orig
        self.assertEqual(body["speakers"], ["S01", "S02"])
        self.assertEqual(body["num_segments"], 2)
        self.assertTrue(body["exclusive"])
        self.assertEqual(body["segments"][0]["end"], 2.0)
        self.assertEqual(len(body["nonexclusive_segments"]), 2)
        self.assertFalse(body["truncated"])


class KindTest(unittest.TestCase):
    def test_reads_the_family_off_model_type(self):
        self.assertEqual(diar_asr.kind_of({"model_type": "moss_transcribe_diarize"}), diar_asr.KIND_MOSS)
        self.assertEqual(diar_asr.kind_of({"model_type": "vibevoice_asr"}), diar_asr.KIND_VIBEVOICE)
        self.assertIsNone(diar_asr.kind_of({"model_type": "qwen3_asr"}))


def _fake_torch(cuda, xpu, xpu_mem=(6 << 30, 32 << 30)):
    t = types.SimpleNamespace()
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: int(cuda))
    if xpu is not None:
        t.xpu = types.SimpleNamespace(is_available=lambda: xpu, mem_get_info=lambda: xpu_mem,
                                      utilization=lambda: 40)
    return t


class DeviceTest(unittest.TestCase):
    def test_prefers_cuda_then_xpu_then_cpu(self):
        self.assertEqual(diar_asr.pick_device(_fake_torch(True, True)), "cuda")
        self.assertEqual(diar_asr.pick_device(_fake_torch(False, True)), "xpu")
        self.assertEqual(diar_asr.pick_device(_fake_torch(False, False)), "cpu")

    def test_a_torch_without_the_xpu_module_is_cpu(self):
        self.assertEqual(diar_asr.pick_device(_fake_torch(False, None)), "cpu")

    def test_metrics_report_the_xpu_card(self):
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(False, True)}):
            text = gpu.gpu_metrics_text()
        self.assertIn("gpu_present 1", text)
        self.assertIn("gpu_mem_total_bytes %d" % (32 << 30), text)
        self.assertIn("gpu_mem_used_bytes %d" % (26 << 30), text)
        self.assertIn("gpu_util_ratio 0.4000", text)


if __name__ == "__main__":
    unittest.main()
