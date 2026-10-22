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
        self.assertIs(d["exclusive"], False)

    def test_exclusive_without_frame_scores_is_refused(self):
        self.nm._state["streaming_ok"] = False
        self.assertEqual(self.post(exclusive="1", exclusive_method="score_aware_v1").status_code, 400)

    def test_an_unknown_exclusive_method_is_refused(self):
        self.assertEqual(self.post(exclusive="1", exclusive_method="legacy_binary_v1").status_code, 400)
        self.assertEqual(self.post(exclusive="0", exclusive_method="score_aware_v1").status_code, 400)

    def test_exclusive_request_returns_aligned_score_evidence(self):
        self.nm._state["streaming_ok"] = True
        saved = self.nm._diarize_chunked

        def fake(ctx, path, seconds, keep_scores=False):
            self.assertTrue(keep_scores)
            return ([seg(0.0, 1.0, "spk_0"), seg(0.5, 1.5, "spk_1")], 1.0,
                    [row(0.9, 0.2), row(0.2, 0.95)], 1.0)

        self.nm._diarize_chunked = fake
        try:
            r = self.post(exclusive="1", exclusive_method="score_aware_v1")
        finally:
            self.nm._diarize_chunked = saved
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertIs(d["exclusive"], True)
        self.assertEqual(d["exclusive_method"], "score_aware_v1")
        self.assertEqual(d["exclusive_evidence"]["method"], "score_aware_v1")
        self.assertEqual(d["exclusive_evidence"]["frame_step_ms"], 1000.0)
        ev = d["exclusive_evidence"]["segments"]
        self.assertEqual(len(ev), len(d["segments"]))
        for i, turn in enumerate(d["segments"]):
            self.assertEqual((ev[i]["ordinal"], ev[i]["start"], ev[i]["end"], ev[i]["speaker"]),
                             (i, turn["start"], turn["end"], turn["speaker"]))
        self.assertEqual([s["speaker"] for s in d["segments"]], ["spk_0", "spk_1"])
        self.assertLessEqual(d["segments"][0]["end"], d["segments"][1]["start"])
        self.assertEqual(len(d["nonexclusive_segments"]), 2)

    def test_request_overrides_for_one_job(self):
        d = self.post(min_duration_off="0", min_duration_on="0.2").json()
        self.assertEqual((d["min_duration_off"], d["min_duration_on"]), (0.0, 0.2))
        self.assertEqual(d["num_segments"], 3)
        self.assertEqual(self.post().json()["min_duration_off"], 1.25)

    def test_bad_override_is_a_400(self):
        self.assertEqual(self.post(min_duration_off="-1").status_code, 400)


class StreamerStateTest(unittest.TestCase):
    def test_initial_state_matches_the_model_update_mode(self):
        import types
        from unittest import mock
        import numpy as np

        # The real runner needs only zeros during initialization. Keep this
        # regression GPU-free so every image's PR checks exercise both modes.
        torch = types.SimpleNamespace(zeros=lambda shape, device=None: np.zeros(shape))
        for mode in (False, True, None):
            with self.subTest(mode=mode):
                def init_state(batch_size, async_streaming, device):
                    return types.SimpleNamespace(
                        cache_frames=264 if async_streaming else 0,
                        fifo_frames=40 if async_streaming else 0)

                sm = types.SimpleNamespace(
                    chunk_len=340, chunk_right_context=40,
                    init_streaming_state=mock.Mock(side_effect=init_state))
                model = types.SimpleNamespace(sortformer_modules=sm)
                if mode is not None:
                    model.async_streaming = mode
                state = dict(model=model, device="cpu", n_spk=8, subsampling=8)
                with mock.patch.dict(sys.modules, {"torch": torch}), \
                        mock.patch.object(stream, "_state", state):
                    runner = stream._Streamer(bounded=True, keep_scores=True)
                    expected = mode if mode is not None else False
                    sm.init_streaming_state.assert_called_once_with(
                        batch_size=1, async_streaming=expected, device="cpu")
                    self.assertEqual(runner.state.cache_frames, 264 if expected else 0)
                    self.assertEqual(runner.state.fifo_frames, 40 if expected else 0)
                    # A fresh recording must reset all output clocks and scores.
                    runner.runs.add(np.ones((5, 8), dtype="float32"))
                    runner.score_rows.append(np.ones((5, 8), dtype="float32"))
                    runner.committed = 2
                    runner.reset()
                    self.assertEqual(runner.committed, 0)
                    self.assertEqual(runner.segments(), [])
                    self.assertEqual(runner.scores().shape, (0, 8))
                    self.assertEqual(sm.init_streaming_state.call_count, 2)


class RunsTest(unittest.TestCase):
    """Turns built chunk by chunk must equal the ones read off the whole array at once."""

    @staticmethod
    def whole(rows, frame_sec):
        r = stream._Runs(rows.shape[1], frame_sec)
        r.add(rows)
        return sorted((s["start"], s["end"], s["speaker"]) for s in r.segments())

    def test_a_single_block_reads_each_run(self):
        import numpy as np

        rows = np.zeros((10, 2), dtype="float32")
        rows[2:5, 0] = 0.9
        rows[7:10, 1] = 0.6
        self.assertEqual(self.whole(rows, 0.01),
                         [(0.02, 0.05, "spk_0"), (0.07, 0.1, "spk_1")])

    def test_any_split_gives_the_same_turns(self):
        import numpy as np

        rng = np.random.default_rng(7)
        for trial in range(40):
            T, S = int(rng.integers(1, 400)), int(rng.integers(1, 5))
            # Sticky activity so runs cross block edges, which is the case being guarded.
            rows = (rng.random((T, S)) < 0.5).astype("float32")
            for k in range(S):
                for t in range(1, T):
                    if rng.random() < 0.8:
                        rows[t, k] = rows[t - 1, k]
            want = self.whole(rows, 0.01)
            r = stream._Runs(S, 0.01)
            cuts = sorted(set(int(c) for c in rng.integers(0, T + 1, size=int(rng.integers(0, 8)))))
            prev = 0
            for c in cuts + [T]:
                r.add(rows[prev:c])
                prev = c
            got = sorted((s["start"], s["end"], s["speaker"]) for s in r.segments())
            self.assertEqual(got, want, "trial %d cuts %s" % (trial, cuts))


@unittest.skipUnless(__import__("shutil").which("ffmpeg"), "ffmpeg not installed")
class PcmBlocksTest(unittest.TestCase):
    def test_blocks_cover_the_whole_clip_resampled_to_16k(self):
        import tempfile
        from wrapper.caps import diar_nemotron as nm

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(8000)
                w.writeframes(b"\x10\x00\x20\x00" * 8000 * 5)
            f.write(buf.getvalue())
        try:
            blocks = list(nm._pcm_blocks(f.name, seconds=2.0))
            self.assertEqual([len(b) for b in blocks[:2]], [32000, 32000])
            self.assertAlmostEqual(sum(len(b) for b in blocks) / 16000.0, 5.0, places=1)
        finally:
            os.unlink(f.name)

    def test_an_undecodable_upload_raises(self):
        import tempfile
        from wrapper.caps import diar_nemotron as nm

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"not audio at all" * 100)
        try:
            with self.assertRaises(RuntimeError):
                list(nm._pcm_blocks(f.name))
        finally:
            os.unlink(f.name)


def row(*scores):
    return list(scores)


class ExclusiveTimelineTest(unittest.TestCase):
    def test_a_clear_lead_is_unique_support(self):
        segs, evidence = stream.exclusive_timeline([row(0.9, 0.1)], 1.0, 0.0, 0.0)
        self.assertEqual(segs, [seg(0.0, 1.0, "spk_0")])
        self.assertEqual(evidence[0]["decision"], "unique_support")
        self.assertEqual(evidence[0]["runner_up_speaker"], "spk_1")
        self.assertAlmostEqual(evidence[0]["support_sum"], 0.9)
        self.assertAlmostEqual(evidence[0]["runner_up_support_sum"], 0.1)
        self.assertEqual(evidence[0]["ordinal"], 0)
        self.assertEqual(evidence[0]["start"], segs[0]["start"])
        self.assertEqual(evidence[0]["end"], segs[0]["end"])
        self.assertEqual(evidence[0]["speaker"], segs[0]["speaker"])

    def test_overlap_keeps_one_speaker(self):
        # Both above 0.5 in the same frame. The louder one owns that instant.
        segs, evidence = stream.exclusive_timeline(
            [row(0.2, 0.95), row(0.8, 0.7)], 1.0, 0.0, 0.0)
        self.assertEqual([s["speaker"] for s in segs], ["spk_1", "spk_0"])
        self.assertEqual(len(evidence), len(segs))
        self.assertTrue(segs[0]["end"] <= segs[1]["start"])

    def test_a_close_call_keeps_the_speaker_already_talking(self):
        # Frame 1's louder channel is spk_0, but the margin is under 0.05 and spk_1 is still active.
        segs, evidence = stream.exclusive_timeline(
            [row(0.1, 0.9), row(0.70, 0.67), row(0.70, 0.67)], 1.0, 0.0, 0.0)
        self.assertEqual(segs, [seg(0.0, 3.0, "spk_1")])
        self.assertEqual(evidence[0]["decision"], "tie_continuity")
        self.assertEqual(evidence[0]["tie_continuity_frames"], 2)
        self.assertEqual(evidence[0]["unique_support_frames"], 1)

    def test_an_unanchored_tie_falls_back_to_the_louder_lower_index(self):
        segs, evidence = stream.exclusive_timeline([row(0.70, 0.67)], 1.0, 0.0, 0.0)
        self.assertEqual(segs, [seg(0.0, 1.0, "spk_0")])
        self.assertEqual(evidence[0]["decision"], "deterministic_fallback")

    def test_equal_scores_break_toward_the_lower_index(self):
        segs, _ = stream.exclusive_timeline([row(0.8, 0.8)], 1.0, 0.0, 0.0)
        self.assertEqual(segs[0]["speaker"], "spk_0")

    def test_a_short_same_speaker_gap_is_bridged_without_overlap(self):
        segs, evidence = stream.exclusive_timeline(
            [row(0.9, 0.0), row(0.0, 0.0), row(0.9, 0.0)], 1.0, 1.25, 0.0)
        self.assertEqual(segs, [seg(0.0, 3.0, "spk_0")])
        self.assertEqual(evidence[0]["unique_support_frames"], 2)
        self.assertEqual(evidence[0]["start"], segs[0]["start"])
        self.assertEqual(evidence[0]["end"], segs[0]["end"])

    def test_someone_else_in_the_gap_blocks_the_bridge(self):
        segs, _ = stream.exclusive_timeline(
            [row(0.9, 0.0), row(0.0, 0.9), row(0.9, 0.0)], 1.0, 5.0, 0.0)
        self.assertEqual([s["speaker"] for s in segs], ["spk_0", "spk_1", "spk_0"])

    def test_min_duration_on_drops_the_evidence_with_the_turn(self):
        segs, evidence = stream.exclusive_timeline(
            [row(0.9, 0.0), row(0.0, 0.9), row(0.0, 0.9)], 1.0, 0.0, 1.5)
        self.assertEqual(segs, [seg(1.0, 3.0, "spk_1")])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["ordinal"], 0)
        self.assertEqual(evidence[0]["speaker"], "spk_1")

    def test_silence_is_empty(self):
        self.assertEqual(stream.exclusive_timeline([row(0.1, 0.2), row(0.0, 0.4)], 0.01, 1.25, 0.0),
                         ([], []))

    def test_below_threshold_is_not_speech(self):
        segs, _ = stream.exclusive_timeline([row(0.49, 0.0)], 1.0, 0.0, 0.0)
        self.assertEqual(segs, [])


if __name__ == "__main__":
    unittest.main()
