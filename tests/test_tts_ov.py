# The Intel loader must not replace the CUDA one until its own image calls run().
import asyncio
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from wrapper.caps import tts
from wrapper.caps import tts_ov
from wrapper.caps import tts_ov_stream

UP = 4  # samples per codec frame in these fakes
GROUPS = 3


class _FakeDecoder:
    """Frame t of a window decodes to UP samples of its first code."""

    total_upsample = UP

    def __init__(self):
        self.windows = []

    def __call__(self, codes):
        codes = codes.numpy() if hasattr(codes, "numpy") else codes
        self.windows.append(codes[0, 0].tolist())
        return np.repeat(codes[0, 0].astype("float32"), UP)[None, None]


class _FakeModel:
    """qwen_tts' talker loop: one forward per frame, codec ids of the previous frame on each."""

    def __init__(self, frames, gate=None, fail_at=None, delay=0.0):
        self.frames = frames
        self.gate = gate
        self.fail_at = fail_at
        self.delay = delay
        self.forwards = 0
        self.decoder = _FakeDecoder()
        self.speech_tokenizer = SimpleNamespace(model=SimpleNamespace(decoder=self.decoder))
        self.talker = SimpleNamespace(forward=self._talker_forward)

    def _talker_forward(self, step):
        self.forwards += 1
        codes = None if step == 0 else np.full((1, GROUPS), step, dtype=np.int64)
        return SimpleNamespace(hidden_states=((), codes))

    def generate(self, voice_clone_prompt=None, **_kw):
        for step in range(self.frames + 1):
            if self.gate is not None and step == self.gate[0] + 1:
                self.gate[1].wait(5)
            if step == self.fail_at:
                raise RuntimeError("talker died")
            time.sleep(self.delay)
            self.talker.forward(step)
        return [np.arange(1, self.frames + 1)], None


class _FakeOV:
    def __init__(self, frames=2, ref=None, **kw):
        self.calls = []
        self.ref = ref
        self.model = _FakeModel(frames, **kw)

    def generate(self, return_sample_rate=False, **kwargs):
        self.calls.append(kwargs)
        prompt = None if self.ref is None else {"ref_code": [self.ref]}
        codes, _ = self.model.generate(voice_clone_prompt=prompt)
        wav = self.model.decoder(codes[0].reshape(1, 1, -1))
        return wav.reshape(-1), 24000


def _frames(chunks):
    return [int(x) for chunk in chunks for x in chunk[0][::UP]]


def _adapter(ov):
    """The adapter with the codec fed numpy; the image hands the decoder a torch tensor."""
    model = tts_ov._OpenVINOQwen3TTS(ov, 24000)
    model._decode = lambda codes: ov.model.decoder(codes).reshape(-1)
    return model


class OpenVINOAdapterTest(unittest.TestCase):
    def test_import_leaves_the_cuda_loader_in_place(self):
        self.assertEqual(tts._load.__module__, "wrapper.caps.tts")
        self.assertIn("faster_qwen3_tts", tts._load.__code__.co_names)

    def test_custom_voice_and_clone_use_the_qwen_tts_spellings(self):
        ov = _FakeOV()
        model = _adapter(ov)
        audio, sr = model.generate_custom_voice(
            text="你好", speaker="aiden", language="Chinese", instruct=None, chunk_size=8)
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape, (2 * UP,))
        self.assertNotIn("chunk_size", ov.calls[0])
        self.assertEqual(ov.calls[0]["speaker"], "aiden")

        model.generate_voice_clone(
            text="hi", language="English", ref_audio="/tmp/ref.wav", ref_text="",
            xvec_only=True, chunk_size=4)
        clone = ov.calls[1]
        self.assertTrue(clone["x_vector_only_mode"])
        self.assertIsNone(clone["ref_text"])
        self.assertNotIn("xvec_only", clone)
        self.assertNotIn("chunk_size", clone)

    def test_streaming_decodes_each_chunk_as_its_frames_land(self):
        ov = _FakeOV(frames=20)
        model = _adapter(ov)
        chunks = list(model.generate_custom_voice_streaming(
            text="Hi.", speaker="aiden", language="English", instruct=None, chunk_size=8))
        self.assertEqual([len(c[0]) for c in chunks], [8 * UP, 8 * UP, 4 * UP])
        self.assertEqual(_frames(chunks), list(range(1, 21)))
        self.assertTrue(all(sr == 24000 and timing is None for _a, sr, timing in chunks))
        self.assertEqual(ov.model.decoder.windows, [
            list(range(1, 9)), list(range(1, 17)), list(range(1, 21))])

    def test_the_first_chunk_plays_before_the_sentence_is_done(self):
        release = threading.Event()
        ov = _FakeOV(frames=16, gate=(8, release))
        model = _adapter(ov)
        it = model.generate_custom_voice_streaming(
            text="Hi.", speaker="aiden", language="English", instruct=None, chunk_size=8)
        first = next(it)
        self.assertEqual(_frames([first]), list(range(1, 9)))
        self.assertLess(ov.model.forwards, 17)
        release.set()
        self.assertEqual(_frames(list(it)), list(range(9, 17)))

    def test_a_clone_reference_is_context_and_never_played(self):
        ov = _FakeOV(frames=4, ref=np.full((5, GROUPS), 99, dtype=np.int64))
        model = _adapter(ov)
        chunks = list(model.generate_voice_clone_streaming(
            text="hi", language="English", ref_audio="/tmp/ref.wav", ref_text="hello", chunk_size=8))
        self.assertEqual(_frames(chunks), [1, 2, 3, 4])
        self.assertEqual(ov.model.decoder.windows, [[99] * 5 + [1, 2, 3, 4]])

    def test_a_failed_generation_reaches_the_reader(self):
        model = _adapter(_FakeOV(frames=20, fail_at=12))
        it = model.generate_custom_voice_streaming(
            text="Hi.", speaker="aiden", language="English", instruct=None, chunk_size=8)
        self.assertEqual(_frames([next(it)]), list(range(1, 9)))
        with self.assertRaisesRegex(RuntimeError, "talker died"):
            next(it)

    def test_a_reader_that_leaves_stops_the_talker_and_frees_the_engine(self):
        release = threading.Event()
        ov = _FakeOV(frames=400, gate=(8, release), delay=0.002)
        model = _adapter(ov)
        state = {"model": model, "voice_design": False}

        async def run():
            agen = tts._aiter(lambda: tts._stream_blocking({"input": "Hi.", "voice": "aiden"}, None, 8))
            first, _sr = await agen.__anext__()
            release.set()
            await agen.aclose()
            for _ in range(200):
                if tts._gen_lock.acquire(blocking=False):
                    tts._gen_lock.release()
                    return first
                await asyncio.sleep(0.01)
            self.fail("the engine lock was never released")

        with mock.patch.dict(tts._state, state):
            first = asyncio.run(run())
        self.assertEqual(_frames([(first,)]), list(range(1, 9)))
        self.assertLess(ov.model.forwards, 400)
        self.assertIsNone(model._tap.sink)
        self.assertFalse(any(t.name == "tts-ov-stream" for t in threading.enumerate()))

    def test_plain_synthesis_still_decodes_the_whole_sentence(self):
        ov = _FakeOV(frames=20)
        model = _adapter(ov)
        list(model.generate_custom_voice_streaming(
            text="Hi.", speaker="aiden", language="English", instruct=None, chunk_size=8))
        audio, _sr = model.generate_custom_voice(
            text="Hi.", speaker="aiden", language="English", instruct=None)
        self.assertEqual(audio.shape, (20 * UP,))
        self.assertEqual(ov.model.decoder.windows[-1], list(range(1, 21)))

    def test_intel_mode_selects_gpu_unless_device_is_set(self):
        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel", "ENGINE_ARGS": ""},
                             clear=False):
            fresh = tts.EngineArgs("")
            with mock.patch.object(tts, "_args", fresh):
                self.assertEqual(tts_ov._device(), "GPU")
                asked = tts.EngineArgs("--device CPU")
                with mock.patch.object(tts, "_args", asked):
                    self.assertEqual(tts_ov._device(), "CPU")

    def test_a_non_intel_pod_without_a_device_fails_closed(self):
        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "nvidia"}, clear=False):
            with mock.patch.object(tts, "_args", tts.EngineArgs("")):
                with self.assertRaises(RuntimeError):
                    tts_ov._device()


class StreamWindowTest(unittest.TestCase):
    def test_a_window_fills_its_bucket_with_left_context(self):
        history = [np.full(GROUPS, i) for i in range(100)]
        codes, lead = tts_ov_stream.window(history, 64, tts_ov._codec_bucket)
        self.assertEqual(codes.shape, (1, GROUPS, 64))
        self.assertEqual(lead, 28)
        self.assertEqual(int(codes[0, 0, lead]), 64)
        codes, lead = tts_ov_stream.window(history[:72], 64, tts_ov._codec_bucket)
        self.assertEqual((codes.shape[-1], lead), (64, 56))

    def test_a_wide_chunk_keeps_the_context_floor(self):
        history = [np.full(GROUPS, i) for i in range(200)]
        codes, lead = tts_ov_stream.window(history, 150, tts_ov._codec_bucket)
        self.assertGreaterEqual(lead, tts_ov_stream.CONTEXT_FLOOR)
        self.assertEqual(codes.shape[-1], 128)


if __name__ == "__main__":
    unittest.main()
