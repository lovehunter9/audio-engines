import tempfile
import unittest
from pathlib import Path

from wrapper.caps.s3_native import _stage_source


class SourceLayoutTests(unittest.TestCase):
    def test_nested_neural_sources_survive_and_incomplete_cache_is_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary)/"source", Path(temporary)/"target"
            neural = source/"vendor/FunASR/funasr/models/cosyvoice/modules/network.py"
            neural.parent.mkdir(parents=True)
            neural.write_text("original neural definition")
            (source/"COMMIT").write_text("revision\n")
            (source/"models").mkdir()
            (source/"models/weights.bin").write_bytes(b"not baked into image")
            target.mkdir()
            (target/".source-revision").write_text("revision\n")
            _stage_source(source, target)
            self.assertEqual((target/neural.relative_to(source)).read_text(), neural.read_text())
            self.assertFalse((target/"models/weights.bin").exists())
            self.assertEqual((target/".source-layout").read_text().strip(), "revision:2")

    def test_different_original_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary)/"source", Path(temporary)/"target"
            source.mkdir(); target.mkdir()
            (source/"COMMIT").write_text("new")
            (target/".source-revision").write_text("old")
            with self.assertRaisesRegex(RuntimeError, "different official S3"):
                _stage_source(source, target)
