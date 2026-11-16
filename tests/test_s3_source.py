import tempfile
import unittest
from pathlib import Path

from wrapper.caps.s3_native import _stage_source
from wrapper.caps.s3_native_compat import prepare_checkpoint_loader


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
            self.assertEqual((target/".source-layout").read_text().strip(), "revision:3:native")

    def test_compatibility_patch_change_refreshes_cached_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary)/"source", Path(temporary)/"target"
            source.mkdir()
            (source/"COMMIT").write_text("revision\n")
            (source/"module.py").write_text("original")
            _stage_source(source, target)
            (source/"PATCHES.json").write_text('[{"patch": "capability-guard"}]')
            (source/"module.py").write_text("guarded optional registration")
            _stage_source(source, target)
            self.assertEqual((target/"module.py").read_text(), "guarded optional registration")

    def test_different_original_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary)/"source", Path(temporary)/"target"
            source.mkdir()
            target.mkdir()
            (source/"COMMIT").write_text("new")
            (target/".source-revision").write_text("old")
            with self.assertRaisesRegex(RuntimeError, "different official S3"):
                _stage_source(source, target)


class CheckpointLoadingTests(unittest.TestCase):
    def test_deserialization_stays_on_host_and_warm_start_is_idempotent(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "inference/streaming/streaming_codex_to_wav.py"
            path.parent.mkdir(parents=True)
            path.write_text('def load(ckpt_path, device):\n    state_dict = torch.load(str(ckpt_path), map_location=device)\n    return state_dict\n')
            prepare_checkpoint_loader(root)
            prepared = path.read_bytes()
            prepare_checkpoint_loader(root)
            self.assertEqual(path.read_bytes(), prepared)
            calls = []
            scope = {"torch": SimpleNamespace(load=lambda *args, **kwargs: calls.append(kwargs) or {"weight": 42})}
            exec(compile(path.read_text(), str(path), "exec"), scope)
            self.assertEqual(scope["load"]("weights.pt", "cuda:0"), {"weight": 42})
            self.assertEqual(calls, [{"map_location": "cpu", "mmap": True}])
