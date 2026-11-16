import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeDependencyIdentityTests(unittest.TestCase):
    def test_changes_to_copied_source_patches_and_plugins_change_the_deps_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"scripts").mkdir()
            original = Path(__file__).resolve().parents[1]/"scripts/deps-image.sh"
            script = root/"scripts/deps-image.sh"
            shutil.copyfile(original, script)
            base = root/"bases/example"
            (base/"native_plugin").mkdir(parents=True)
            (base/"deps.Dockerfile").write_text("COPY bases/example/native_plugin /opt/plugin\n")
            (base/"apply_compat.py").write_text("original patch")
            (base/"native_plugin/module.py").write_text("original plugin")
            (base/"native_plugin/pyproject.toml").write_text("original registration")
            def tag():
                return subprocess.check_output(["bash", str(script), "example", "namespace/runtime"], text=True).strip()
            baseline = tag()
            for filename in ("apply_compat.py", "native_plugin/module.py", "native_plugin/pyproject.toml"):
                path = base/filename
                old = path.read_text()
                path.write_text("changed dependency")
                self.assertNotEqual(tag(), baseline, filename)
                path.write_text(old)
                self.assertEqual(tag(), baseline)
            (base/"README.md").write_text("unrelated documentation")
            self.assertEqual(tag(), baseline)


if __name__ == "__main__":
    unittest.main()
