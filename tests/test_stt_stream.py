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


if __name__ == "__main__":
    unittest.main()
