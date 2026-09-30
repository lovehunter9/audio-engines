"""Nemotron's --min-duration-off / --min-duration-on, and the frame length its streaming path reads."""
import io
import os
import sys
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper.caps import diar_stream as stream  # noqa: E402


def seg(s, e, spk):
    return {"start": s, "end": e, "speaker": spk}


class _PerSpeaker:
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.was = stream.PER_SPEAKER
        stream.PER_SPEAKER = self.on

    def __exit__(self, *exc):
        stream.PER_SPEAKER = self.was


class FramesTest(unittest.TestCase):
    def test_high_resolution_rows_are_10ms(self):
        # Nemotron 3: high_resolution true, output_subsampling_factor 1. Reading it as 80ms stretched time 8x.
        m = type("M", (), {"high_resolution": True, "output_subsampling_factor": 1})()
        self.assertAlmostEqual(stream._frame_sec(m, 8), 0.01)

    def test_encoder_rate_models_keep_80ms(self):
        self.assertAlmostEqual(stream._frame_sec(object(), 8), 0.08)
        m = type("M", (), {"high_resolution": False, "output_subsampling_factor": 1})()
        self.assertAlmostEqual(stream._frame_sec(m, 8), 0.08)


class TidyTest(unittest.TestCase):
    def test_pause_is_filled_per_speaker_across_another_voice(self):
        segs = [seg(0.0, 1.0, "spk_0"), seg(1.2, 1.5, "spk_1"), seg(2.0, 3.0, "spk_0")]
        with _PerSpeaker(True):
            out = stream._tidy(segs, 1.25, 0.0)
        self.assertIn(seg(0.0, 3.0, "spk_0"), out)
        self.assertIn(seg(1.2, 1.5, "spk_1"), out)

    def test_sortformer_keeps_its_neighbour_only_merge(self):
        segs = [seg(0.0, 1.0, "spk_0"), seg(1.2, 1.5, "spk_1"), seg(2.0, 3.0, "spk_0")]
        with _PerSpeaker(False):
            out = stream._tidy(segs, 1.25, 0.0)
        self.assertEqual(len(out), 3)

    def test_longer_pause_is_kept(self):
        with _PerSpeaker(True):
            out = stream._tidy([seg(0.0, 1.0, "spk_0"), seg(2.5, 3.0, "spk_0")], 1.25, 0.0)
        self.assertEqual(len(out), 2)

    def test_zero_off_fills_nothing(self):
        with _PerSpeaker(True):
            out = stream._tidy([seg(0.0, 1.0, "spk_0"), seg(1.01, 2.0, "spk_0")], 0.0, 0.0)
        self.assertEqual(len(out), 2)

    def test_on_drops_short_turns_after_filling(self):
        segs = [seg(0.0, 0.1, "spk_0"), seg(0.3, 0.4, "spk_0"), seg(5.0, 5.1, "spk_1")]
        with _PerSpeaker(True):
            out = stream._tidy(segs, 0.5, 0.3)
        self.assertEqual(out, [seg(0.0, 0.4, "spk_0")])

    def test_output_is_time_ordered(self):
        segs = [seg(3.0, 4.0, "spk_1"), seg(0.0, 1.0, "spk_0"), seg(1.5, 2.0, "spk_0")]
        with _PerSpeaker(True):
            out = stream._tidy(segs, 1.0, 0.0)
        self.assertEqual([s["start"] for s in out], sorted(s["start"] for s in out))


class SecondsTest(unittest.TestCase):
    def test_absent_is_the_default(self):
        self.assertEqual(stream._seconds(None, 1.25, "x"), 1.25)
        self.assertEqual(stream._seconds("  ", 1.25, "x"), 1.25)

    def test_numbers(self):
        self.assertEqual(stream._seconds("0.5", 1.25, "x"), 0.5)
        self.assertEqual(stream._seconds(0, 1.25, "x"), 0.0)

    def test_bad_values_raise(self):
        for bad in ("abc", "-1", "nan", "inf"):
            with self.assertRaises(ValueError, msg=bad):
                stream._seconds(bad, 1.25, "x")

    def test_start_message_overrides_one_connection(self):
        knobs = {"off": 1.25, "on": 0.0}
        stream._start_knobs({"type": "start", "min_duration_off": 0.5}, knobs)
        self.assertEqual(knobs, {"off": 0.5, "on": 0.0})
        with self.assertRaises(ValueError):
            stream._start_knobs({"type": "start", "min_duration_on": "x"}, knobs)


def _wav(seconds=1.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


class _FakeModel:
    def diarize(self, audio, batch_size, sample_rate):
        return [["0.0 1.0 speaker_0", "1.2 1.5 speaker_1", "2.0 3.0 speaker_0", "4.0 4.1 speaker_1"]]


class OfflineEndpointTest(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as e:  # pragma: no cover
            self.skipTest("fastapi test client unavailable: %s" % e)
        from wrapper.caps import diar_nemotron as nm
        self.nm = nm
        self.saved = dict(nm._state)
        nm._state.update(model=_FakeModel(), device="cpu", ready=True, error=None)
        nm._bind_stream()
        import numpy as np
        self.mono = nm._mono16k
        nm._mono16k = lambda path: np.zeros(16000, dtype="float32")
        self.client = TestClient(nm.build_app(["diar"]))

    def tearDown(self):
        self.nm._mono16k = self.mono
        self.nm._state.clear()
        self.nm._state.update(self.saved)
        stream.PER_SPEAKER = False

    def post(self, **form):
        return self.client.post("/v1/audio/diarization", files={"file": ("a.wav", _wav(), "audio/wav")},
                                data=form)

    def test_engine_defaults_apply_and_are_echoed(self):
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["min_duration_off"], d["min_duration_on"]), (1.25, 0.0))
        self.assertIn(seg(0.0, 3.0, "spk_0"), d["segments"])

    def test_request_overrides_for_one_job(self):
        d = self.post(min_duration_off="0", min_duration_on="0.2").json()
        self.assertEqual((d["min_duration_off"], d["min_duration_on"]), (0.0, 0.2))
        self.assertEqual(d["num_segments"], 3)
        self.assertEqual(self.post().json()["min_duration_off"], 1.25)

    def test_bad_override_is_a_400(self):
        self.assertEqual(self.post(min_duration_off="-1").status_code, 400)


if __name__ == "__main__":
    unittest.main()
