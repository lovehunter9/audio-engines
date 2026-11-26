"""Unit tests for wrapper/caps/stt_stream.py that need neither a model nor a card."""
import importlib
import os
import sys
import types
import unittest
from unittest import mock


def _stt(engine_args=""):
    env = {"AUDIO_BASE": "qwen", "MODEL_SUPPORTS": "stt,stt_stream", "ENGINE_ARGS": engine_args}
    with mock.patch.dict(os.environ, env, clear=False):
        return importlib.reload(importlib.import_module("wrapper.caps.stt_stream"))


class PatchMaxInputTest(unittest.TestCase):
    """The clip cap has to reach the package, and say so only where it actually did.

    🔴 Assigning to a module that no longer carries the constant creates it and changes
    nothing, so the split length stays at whatever the package now uses while the log still
    reports the patch. The failure is silent in the direction that matters: a cap nobody is
    enforcing reads exactly like one that is.
    """

    def _run(self, has_attr):
        m = _stt()
        utils = types.ModuleType("qwen_asr.inference.utils")
        mod = types.ModuleType("qwen_asr.inference.qwen3_asr")
        if has_attr:
            utils.MAX_ASR_INPUT_SECONDS = 540
            mod.MAX_ASR_INPUT_SECONDS = 540
        pkg = types.ModuleType("qwen_asr")
        inf = types.ModuleType("qwen_asr.inference")
        said = []
        with mock.patch.dict(sys.modules, {"qwen_asr": pkg, "qwen_asr.inference": inf,
                                           "qwen_asr.inference.utils": utils,
                                           "qwen_asr.inference.qwen3_asr": mod}):
            with mock.patch.object(m, "_p", said.append):
                m._patch_max_input(630)
        return utils, mod, said

    def test_the_attribute_is_patched_when_the_package_has_it(self):
        utils, mod, said = self._run(True)
        self.assertEqual((utils.MAX_ASR_INPUT_SECONDS, mod.MAX_ASR_INPUT_SECONDS), (630, 630))
        self.assertTrue(any(x.startswith("patched") for x in said), said)

    def test_a_renamed_attribute_is_said_and_not_created(self):
        utils, mod, said = self._run(False)
        self.assertFalse(hasattr(utils, "MAX_ASR_INPUT_SECONDS"))
        self.assertFalse(hasattr(mod, "MAX_ASR_INPUT_SECONDS"))
        self.assertTrue(any("no longer has MAX_ASR_INPUT_SECONDS" in x for x in said), said)

    def test_one_module_moving_it_does_not_stop_the_other_being_patched(self):
        m = _stt()
        utils = types.ModuleType("qwen_asr.inference.utils")
        utils.MAX_ASR_INPUT_SECONDS = 540
        mod = types.ModuleType("qwen_asr.inference.qwen3_asr")  # this one lost it
        said = []
        with mock.patch.dict(sys.modules, {"qwen_asr": types.ModuleType("qwen_asr"),
                                           "qwen_asr.inference": types.ModuleType("qwen_asr.inference"),
                                           "qwen_asr.inference.utils": utils,
                                           "qwen_asr.inference.qwen3_asr": mod}):
            with mock.patch.object(m, "_p", said.append):
                m._patch_max_input(630)
        self.assertEqual(utils.MAX_ASR_INPUT_SECONDS, 630)
        self.assertFalse(hasattr(mod, "MAX_ASR_INPUT_SECONDS"))
        self.assertTrue(any("qwen3_asr" in x and "no longer has" in x for x in said), said)
        self.assertTrue(any(x.startswith("patched") and "utils" in x for x in said), said)


class StreamStartLanguageTest(unittest.TestCase):
    """The language in start has to reach the state the audio is decoded with.

    🔴 The state is built when the socket opens, before start is read. Keeping that state means
    every utterance shorter than a roll auto-detects, so English came back as Spanish or Chinese
    while the caller had asked for English.
    """

    def _stream(self, *frames):
        import json
        from fastapi.testclient import TestClient
        m = _stt()
        built = []

        def init(_asr, language=None, **_):
            built.append(language)
            return types.SimpleNamespace(language=language or "", text="")

        with mock.patch.dict(os.environ, {"AUDIO_BASE": "qwen"}), \
                mock.patch.object(m, "_stream_init", init), \
                mock.patch.object(m, "_stream_step", lambda _asr, _pcm, st: None), \
                mock.patch.object(m, "_stream_finish", lambda _asr, st: None), \
                mock.patch.dict(m._state, {"ready": True, "asr": object(), "error": None}):
            with TestClient(m.build_app(["stt", "stt_stream"])) as c:
                with c.websocket_connect("/v1/audio/stream") as ws:
                    self.assertEqual(json.loads(ws.receive_text())["type"], "ready")
                    for frame in frames:
                        (ws.send_text if isinstance(frame, str) else ws.send_bytes)(frame)
                    ws.send_text(json.dumps({"type": "stop"}))
                    while json.loads(ws.receive_text())["type"] not in ("closed", "error"):
                        pass
        return built

    def test_start_language_reaches_the_first_state(self):
        import json
        built = self._stream(json.dumps({"type": "start", "language": "en"}), b"\x00\x00" * 8000)
        self.assertEqual(built[-1], "en", built)

    def test_start_after_audio_keeps_the_state_being_decoded(self):
        import json
        built = self._stream(b"\x00\x00" * 8000, json.dumps({"type": "start", "language": "en"}))
        self.assertEqual(built, [None], built)


class OpenVinoLiveStreamTest(unittest.TestCase):
    """OpenVINO streams the way qwen-asr's own streaming does: every chunk, the audio so far is
    decoded again under the text already decoded, so partials arrive while the caller still talks.

    🔴 It used to buffer until stop and decode once, so a caller holding a turn open for a pause
    had no words to judge it by; a voice session could only guess from silence.
    """

    class _Tokenizer:
        def encode(self, text):
            return list(text)

        def decode(self, ids):
            return "".join(ids)

    class _Engine:
        def __init__(self, words):
            self.processor = types.SimpleNamespace(tokenizer=OpenVinoLiveStreamTest._Tokenizer())
            self.words = words
            self.calls = []

        def continue_text(self, clip, text_prompt, max_new_tokens):
            self.calls.append((len(clip), text_prompt))
            heard = self.words[: len(self.calls)]
            prefix = text_prompt.split("<asr_text>", 1)[1]
            return heard[len(prefix):] if heard.startswith(prefix) else heard

    def _qwen_asr(self):
        utils = types.ModuleType("qwen_asr.inference.utils")
        utils.SAMPLE_RATE = 16000
        utils.parse_asr_output = lambda raw, user_language=None: (user_language or "", raw)
        return {"qwen_asr": types.ModuleType("qwen_asr"),
                "qwen_asr.inference": types.ModuleType("qwen_asr.inference"),
                "qwen_asr.inference.utils": utils}

    def _session(self, engine, start, seconds, frame_sec=0.25):
        import json
        from fastapi.testclient import TestClient
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "ov"}):
            m = _stt()
        frame = b"\x00\x00" * int(16000 * frame_sec)
        live = []
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "ov"}), \
                mock.patch.dict(sys.modules, self._qwen_asr()), \
                mock.patch.dict(m._state, {"ready": True, "ov_batch": engine, "asr": None, "error": None}):
            with TestClient(m.build_app(["stt", "stt_stream"])) as c:
                with c.websocket_connect("/v1/audio/stream") as ws:
                    self.assertEqual(json.loads(ws.receive_text())["type"], "ready")
                    ws.send_text(json.dumps({"type": "start", **start}))
                    for _ in range(int(seconds / frame_sec)):
                        ws.send_bytes(frame)
                    # Every whole step of audio is answered before stop is even sent.
                    for _ in range(int(seconds / 0.5)):
                        live.append(json.loads(ws.receive_text()))
                    ws.send_text(json.dumps({"type": "stop"}))
                    rest = []
                    while not rest or rest[-1]["type"] not in ("closed", "error"):
                        rest.append(json.loads(ws.receive_text()))
        return live, rest

    def test_partials_arrive_every_chunk_while_audio_is_still_coming(self):
        engine = self._Engine("帮我把灯变成绿色")
        live, rest = self._session(engine, {"language": "zh", "chunk_ms": 500, "step_ms": 500}, 2.0)
        self.assertEqual([x["type"] for x in live], ["partial"] * 4)
        self.assertEqual([x["text"] for x in live], ["帮", "帮我", "帮我把", "帮我把灯"])
        # The whole take is decoded each time, not just the newest chunk.
        self.assertEqual([n for n, _ in engine.calls[:4]], [8000, 16000, 24000, 32000])
        self.assertEqual(rest[-2]["type"], "final")

    def test_the_text_so_far_rides_in_the_prompt_after_the_unfixed_chunks(self):
        engine = self._Engine("帮我把灯变成绿色")
        self._session(engine, {"language": "zh", "chunk_ms": 500, "step_ms": 500}, 2.0)
        prompts = [p for _, p in engine.calls]
        self.assertTrue(prompts[0].endswith("language Chinese<asr_text>"), prompts[0])
        self.assertTrue(prompts[1].endswith("<asr_text>"), prompts[1])
        # From the third chunk on, the decoded text minus its last five tokens is the prefix.
        self.assertTrue(prompts[3].endswith("<asr_text>"), prompts[3])
        engine = self._Engine("帮我把灯变成绿色了吗你好")
        self._session(engine, {"language": "zh", "chunk_ms": 500, "step_ms": 500}, 4.0)
        self.assertTrue(engine.calls[7][1].endswith("<asr_text>帮我"), engine.calls[7][1])

    def test_start_context_reaches_the_system_prompt(self):
        engine = self._Engine("点点头")
        self._session(engine, {"language": "zh", "chunk_ms": 500, "context": "点点头 摇摇头"}, 1.0)
        self.assertIn("<|im_start|>system\n点点头 摇摇头<|im_end|>", engine.calls[0][1])

    def _flush_session(self, base):
        import json
        from fastapi.testclient import TestClient
        engine = self._Engine("帮我把灯变成绿色")
        with mock.patch.dict(os.environ, {"AUDIO_BASE": base}):
            m = _stt()
        if base == "ov":
            loaded = {"ov_batch": engine, "asr": None}
            decode = mock.patch.object(m, "_tf_generate", None)
        else:
            # The CUDA image decodes with qwen-asr's own load through `_tf_generate`.
            engine._build_text_prompt = lambda context="", force_language=None: (
                "language %s<asr_text>" % force_language)
            loaded = {"asr": engine}
            decode = mock.patch.object(
                m, "_tf_generate", lambda asr, prompt, wav, n: asr.continue_text(wav, prompt, n))
        frame = b"\x00\x00" * 4000
        with mock.patch.dict(os.environ, {"AUDIO_BASE": base}), decode, \
                mock.patch.dict(sys.modules, self._qwen_asr()), \
                mock.patch.dict(m._state, {"ready": True, "error": None, **loaded}):
            with TestClient(m.build_app(["stt", "stt_stream"])) as c:
                with c.websocket_connect("/v1/audio/stream") as ws:
                    self.assertEqual(json.loads(ws.receive_text())["type"], "ready")
                    ws.send_text(json.dumps({"type": "start", "language": "zh", "chunk_ms": 500, "step_ms": 500}))
                    for _ in range(3):
                        ws.send_bytes(frame)
                    self.assertEqual(json.loads(ws.receive_text())["type"], "partial")
                    ws.send_text(json.dumps({"type": "flush"}))
                    flushed = json.loads(ws.receive_text())
                    # The quarter second short of a step is in it too: the decode covers the pause.
                    self.assertEqual(flushed["type"], "flushed")
                    self.assertEqual(flushed["text"], "帮我")
                    self.assertEqual(engine.calls[-1][0], 12000)
                    ws.send_text(json.dumps({"type": "flush"}))
                    again = json.loads(ws.receive_text())
                    self.assertEqual((again["type"], again["text"]), ("flushed", "帮我"))
                    self.assertEqual(len(engine.calls), 2)
                    for _ in range(2):
                        ws.send_bytes(frame)
                    later = json.loads(ws.receive_text())
                    self.assertEqual(later["type"], "partial")
                    self.assertEqual(engine.calls[-1][0], 20000)
                    ws.send_text(json.dumps({"type": "stop"}))
                    rest = []
                    while not rest or rest[-1]["type"] not in ("closed", "error"):
                        rest.append(json.loads(ws.receive_text()))
        self.assertEqual([x["type"] for x in rest], ["final", "closed"])

    def test_flush_decodes_everything_received_and_keeps_the_stream_open_on_openvino(self):
        self._flush_session("ov")

    def test_flush_decodes_everything_received_and_keeps_the_stream_open_on_cuda(self):
        self._flush_session("qwen")

    def test_without_chunk_ms_a_decode_waits_for_the_default_chunk(self):
        engine = self._Engine("帮我把灯变成绿色")
        live, _ = self._session(engine, {"language": "zh"}, 2.0)
        self.assertEqual([x["text"] for x in live], ["", "", "", "帮"])
        self.assertEqual(len(engine.calls) >= 1 and engine.calls[0][0], 32000)


if __name__ == "__main__":
    unittest.main()
