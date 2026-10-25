# The rewritten suppress mask must be the shipped one, built in two writes instead of 1023.
import importlib.util
import os
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

from wrapper.caps import tts_mask

# The shape faster-qwen3-tts 0.3.2 ships in streaming.py and generate.py.
SHIPPED = textwrap.dedent("""\
    def build(vocab_size, eos_id, np):
        suppress_mask = np.zeros(vocab_size, dtype=bool)
        suppress_start = max(0, vocab_size - 1024)
        for i in range(suppress_start, vocab_size):
            if i != eos_id:
                suppress_mask[i] = True
        return suppress_mask
    """)


def _run(src, *args):
    ns = {}
    exec(src, ns)
    return ns["build"](*args)


class RewriteTest(unittest.TestCase):
    def test_same_mask_for_every_eos_position(self):
        new, n = tts_mask.rewrite(SHIPPED)
        self.assertEqual(n, 1)
        self.assertNotIn("for i in range", new)
        # 3072 is the 12Hz talker vocab; eos inside, at either edge and outside the suppressed band.
        for vocab, eos in ((3072, 2150), (3072, 2048), (3072, 3071), (3072, 100), (500, 499)):
            np.testing.assert_array_equal(_run(new, vocab, eos, np), _run(SHIPPED, vocab, eos, np),
                                          err_msg="vocab=%d eos=%d" % (vocab, eos))

    def test_other_source_is_left_alone(self):
        src = "def f():\n    return 1\n"
        self.assertEqual(tts_mask.rewrite(src), (src, 0))


class PatchFunctionTest(unittest.TestCase):
    def test_function_is_redefined_in_its_module(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fake_streaming.py")
            with open(path, "w") as f:
                # The module's own numpy import, as the library's functions use its torch.
                f.write("import numpy as np\n\n" + SHIPPED.replace("np):", "_np=None):"))
            spec = importlib.util.spec_from_file_location("fake_streaming", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            before, shipped = mod.build(3072, 2150), mod.build.__code__
            self.assertEqual(tts_mask._patch_function(mod, "build"), 1)
            self.assertEqual(tts_mask._patch_function(mod, "missing"), 0)
            self.assertIsNot(mod.build.__code__, shipped)
            # The loop's per-index store is gone from the bytecode the module now runs.
            self.assertNotIn("i", mod.build.__code__.co_varnames)
            np.testing.assert_array_equal(mod.build(3072, 2150), before)

    def test_apply_runs_once(self):
        with mock.patch.object(tts_mask, "_applied", []), \
                mock.patch.object(tts_mask, "_patch_function", return_value=1) as patch:
            with mock.patch.dict("sys.modules", {"faster_qwen3_tts": mock.MagicMock()}):
                self.assertEqual(tts_mask.apply(), 3)
                self.assertEqual(tts_mask.apply(), 3)
            self.assertEqual(patch.call_count, 3)


if __name__ == "__main__":
    unittest.main()
