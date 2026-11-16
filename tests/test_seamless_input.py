import ast
from pathlib import Path
import unittest

source = Path(__file__).parents[1] / "wrapper/caps/seamless_native.py"
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_segments")
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
segments = namespace["_segments"]


class SeamlessInputTests(unittest.TestCase):
    def test_final_nonempty_frame_preserves_every_sample(self):
        for length in [400, 401, 5119, 5120, 5121, 5280, 5519, 5520, 10240, 168960, 585920]:
            for packet in [160, 5120, 6337]:
                with self.subTest(length=length, packet=packet):
                    pending, sent = [], []
                    for start in range(0, length, packet):
                        pending.extend(range(start, min(start + packet, length)))
                        sent.extend(segments(pending, False))
                    sent.extend(segments(pending, True))
                    self.assertFalse(pending)
                    self.assertEqual([sample for chunk, _ in sent for sample in chunk], list(range(length)))
                    self.assertEqual([last for _, last in sent], [False] * (len(sent) - 1) + [True])
                    self.assertGreaterEqual(len(sent[-1][0]), 400)

    def test_short_input_does_not_replay_previous_audio(self):
        with self.assertRaisesRegex(ValueError, "25 milliseconds"):
            list(segments([0] * 399, True))
        self.assertEqual(list(segments([], True)), [])
