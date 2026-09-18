"""How long is this clip -- answered four different ways, and they do not agree."""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wrapper.audioio import probe_seconds          # noqa: E402
from wrapper.limits import duration                # noqa: E402

CAPS = pathlib.Path(__file__).resolve().parent.parent / "wrapper" / "caps"
FFMPEG = shutil.which("ffmpeg")


def _make(dirpath, name, *args):
    """One second of tone in whatever container `args` names."""
    out = dirpath / name
    assert FFMPEG                                   # the class is skipped without it
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1",
                    *args, str(out)], check=True)
    return out


@unittest.skipUnless(FFMPEG, "ffmpeg is not installed; the sample clips cannot be generated")
class TheReadersContract(unittest.TestCase):
    """What each reader answers, per container."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(cls._tmp.name)
        cls.wav = _make(d, "tiny.wav")
        cls.mp3 = _make(d, "tiny.mp3", "-c:a", "libmp3lame")
        cls.m4a = _make(d, "tiny.m4a", "-c:a", "aac")
        cls.flac = _make(d, "whole.flac", "-c:a", "flac")
        cls.cut = d / "half.flac"
        whole = cls.flac.read_bytes()
        cls.cut.write_bytes(whole[: len(whole) // 2])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


    DECODABLE = ("wav", "mp3", "m4a")

    def test_the_wide_reader_measures_everything_the_engines_decode(self):
        """🔴 The one that matters. A container the engine reads and nothing can measure is"""
        for ext in self.DECODABLE:
            path = getattr(self, ext)
            wide = duration(str(path))
            self.assertIsNotNone(
                wide, "%s decodes in the engine and no reader here can measure it, so every "
                      "duration limit is unenforceable on it" % path.name)
            self.assertTrue(0.5 <= float(wide) <= 2.0,
                            "%s measured %.3fs for a one-second clip -- that is not padding, "
                            "it is a different unit" % (path.name, float(wide)))

    def test_the_fallback_only_adds_answers(self):
        """Never disagrees. A fallback that answered differently would move every limit"""
        for ext in self.DECODABLE:
            path = getattr(self, ext)
            narrow = probe_seconds(str(path))
            if narrow is None:
                continue
            wide = duration(str(path))
            self.assertIsNotNone(wide, path.name)
            self.assertLess(
                abs(float(narrow) - float(wide)), 0.1 * max(float(narrow), float(wide)) + 0.1,
                "%s: the two readers disagree by more than padding (%.3f vs %.3f), so every "
                "limit downstream moves depending on which one answered"
                % (path.name, float(narrow), float(wide)))

    def test_the_fallback_is_load_bearing_on_this_machine(self):
        """🔴 At least one, not a named one. Which container the narrow reader misses is a"""
# At least one, not a named one: which container the narrow reader misses depends on
# the libsndfile this image carries. What must hold is that the two readers do not
# cover the same set -- the day they do, somebody deletes the fallback and the next
# thinner libsndfile silently loses every duration limit.
        missed = [ext for ext in self.DECODABLE
                  if probe_seconds(str(getattr(self, ext))) is None]
        self.assertTrue(
            missed,
            "soundfile/wave now measure every container the engines decode, so the ffprobe "
            "fallback reads as dead code here. It is not: the gap is between two libraries "
            "that move independently, and a thinner libsndfile in the next base image opens "
            "it again with nothing to show for it. Before removing anything, check what this "
            "image actually carries")

    def test_a_body_that_will_not_decode_is_not_measured_into_acceptance(self):
        """⚠️ Whether a truncated body measures at all is the container's business and"""
        whole = duration(str(self.flac))
        got = duration(str(self.cut))
        if got is not None and whole is not None:
            self.assertLessEqual(
                float(got), float(whole) * 1.1,
                "the half-written clip measured %.3fs against %.3fs for the whole one -- a "
                "length check handed that number accepts a clip the decoder then refuses"
                % (float(got), float(whole)))


class WhoMeasuresWhat(unittest.TestCase):
    """🔴 The half that keeps paying. The align capability read the narrow one for a day and"""

    DECLARED = {
        "align": (("audioio.probe_seconds",), "passes through: ffprobe may be absent, and a clip "
                                     "three readers cannot measure is one the engine fails "
                                     "on by itself"),
        "diar": (("Bounds",), "shared upload guard"),
        "diar_speakrs": (("Bounds",), "shared upload guard"),
        "embed": (("Bounds",), "shared upload guard"),
        "enhance": (("Bounds",), "shared upload guard"),
        "vad": (("Bounds",), "shared upload guard"),
        "tts_el": (("audioio.wav_seconds", "audioio.probe_seconds"),
                   "refuses with 400. The WAV header first, the general probe after it -- "
                   "and neither reaches ffprobe, so an m4a reference the engine could have "
                   "used is turned away"),
        "whisper": (("audioio.wav_seconds",),
                    "falls back to the requested slice length"),
    }

    FROM_SAMPLES = re.compile(r"\bseconds\(")

    READERS = tuple(
        (name, re.compile(r"from\s+\.\.%s\s+import\s*\(?[\s\S]{0,400}?\b%s\b"
                          % (module, attr)))
        for name, module, attr in (
            ("limits.duration", "limits", "duration"),
            ("Bounds", "limits", "Bounds"),
            ("audioio.probe_seconds", "audioio", "probe_seconds"),
            ("audioio.wav_seconds", "audioio", "wav_seconds")))

    @staticmethod
    def _code(path):
        """The file with comment lines dropped -- a reader NAMED in a comment is not a call,"""
        return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                         if not line.lstrip().startswith("#"))

    def test_every_capability_that_measures_a_clip_is_declared(self):
        found = {}
        for path in sorted(CAPS.glob("*.py")):
            if path.name == "__init__.py":
                continue
            code = self._code(path)
            used = {name for name, pat in self.READERS if pat.search(code)}
            if used:
                found[path.stem] = used

        undeclared = sorted(set(found) - set(self.DECLARED))
        self.assertEqual(undeclared, [], "\n".join([
            "these capabilities measure a clip and are not in DECLARED: %s" % undeclared,
            "Say which reader each uses and what it does when the read comes back empty.",
            "The choice that matters: `audioio.probe_seconds` reads through soundfile and",
            "the stdlib wave module and cannot measure m4a or aac, while the engines decode",
            "them through ffmpeg -- so a limit keyed to it does not fire on a clip that runs.",
            "`limits.duration` is the same readers plus an ffprobe fallback."]))

        gone = sorted(set(self.DECLARED) - set(found))
        self.assertEqual(gone, [], "declared here but no longer measuring anything: %s. "
                                   "Drop the entry in the same commit." % gone)

    def test_the_declared_reader_is_the_one_in_the_file(self):
        for cap, (readers, _) in sorted(self.DECLARED.items()):
            path = CAPS / (cap + ".py")
            if not path.exists():
                continue
            code = self._code(path)
            imported = {n for n, pat in self.READERS if pat.search(code)}
            self.assertEqual(
                imported, set(readers),
                "%s imports %s and the table says %s -- list every reader it measures with, "
                "in the order it tries them" % (cap, sorted(imported), sorted(readers)))

    def test_the_pattern_sees_every_shape_an_import_is_written_in(self):
        """🔴 A positive control, because the tree cannot provide one. None of these three"""
        shapes = {
            "    from ..limits import duration": "limits.duration",
            "from ..limits import (\n    Bounds,\n)": "Bounds",
            "from ..audioio import probe_seconds as _p": "audioio.probe_seconds",
        }
        for source, want in shapes.items():
            seen = [n for n, pat in self.READERS if pat.search(source)]
            self.assertIn(want, seen,
                          "the registry cannot see %r, so a capability importing a reader "
                          "that way is invisible to this whole table" % source)

    def test_the_scan_is_not_blind(self):
        """Without this the table passes over an empty scan -- a rename of any reader, and"""
        hits = sum(1 for path in CAPS.glob("*.py")
                   for _, pat in self.READERS if pat.search(self._code(path)))
        expected = sum(len(readers) for readers, _ in self.DECLARED.values())
        self.assertGreaterEqual(
            hits, expected,
            "the reader scan sees %d imports against the %d this table names. Either a "
            "capability stopped measuring -- then edit DECLARED in the same commit -- or a "
            "pattern in READERS stopped matching a shape it used to see" % (hits, expected))


if __name__ == "__main__":
    unittest.main()
