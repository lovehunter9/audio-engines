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

        def init(_asr, language=None):
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


if __name__ == "__main__":
    unittest.main()
