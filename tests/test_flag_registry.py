"""One ENGINE_ARGS string reaches every capability in the image, so two of them"""
import pathlib
import re
import unittest

WRAPPER = pathlib.Path(__file__).resolve().parent.parent / "wrapper"
# Any receiver ending in `args`, and single quotes too: three flags are read through a
# parameter, among them the collision above -- the check could not see its own example.
_CLAIM = re.compile(
    r"""[A-Za-z_]*args\.(switch|number|count|text)\(\s*['"](--[a-z0-9][a-z0-9-]*)['"]""")
_VALUED = {"number", "count", "text"}


def claims():
    """{flag: {(reader, method)}} read off the source rather than by importing."""
    found = {}
    for path in sorted(WRAPPER.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        reader = path.stem if path.parent.name == "caps" else "shared:" + path.stem
        for method, flag in _CLAIM.findall(path.read_text(encoding="utf-8")):
            found.setdefault(flag, set()).add((reader, method))
    return found


#: A ratchet, not a floor: "more than twenty" passes just as well after the scan goes
#: half-blind, which is how both widenings were found by reading, not by a red test.
FLAGS_SEEN = 59


class FlagRegistryTest(unittest.TestCase):
    def test_the_scan_finds_the_flags_that_are_there(self):
        found = claims()
        self.assertGreaterEqual(
            len(found), FLAGS_SEEN,
            "the flag scan sees %d flags, fewer than the %d recorded here. Either flags were "
            "removed -- then lower FLAGS_SEEN in the same commit -- or _CLAIM stopped "
            "matching a shape it used to see, which is the failure this file is about"
            % (len(found), FLAGS_SEEN))
        self.assertIn("--enforce-eager", found)   # on main since before this check
        self.assertIn("--repetition-detection", found)
        self.assertIn("--max-upload-mb", found)

    def test_the_pattern_sees_every_shape_a_flag_is_read_in(self):
        """A positive control: the tree has no `engine_args.` receiver and no single-quoted"""
        shapes = {
            '_args.number("--a", 1)': "--a",                 # the module-level reader
            'args.number("--b", 1)': "--b",                  # a parameter called args
            'engine_args.number("--c", 1)': "--c",           # any prefix, no word boundary
            "_args.switch('--d')": "--d",                    # single quotes
        }
        for source, flag in shapes.items():
            found = [f for _, f in _CLAIM.findall(source)]
            self.assertEqual(found, [flag],
                             "the flag scan cannot see %r, so every capability reading its "
                             "flags that way is invisible to this whole file" % source)

    def test_no_flag_is_a_switch_in_one_capability_and_a_value_in_another(self):
        """Two DIFFERENT readers. One capability reading a flag both ways is deliberate and"""
        bad = []
        for flag, uses in sorted(claims().items()):
            switched = {r for r, m in uses if m == "switch"}
            valued = {r for r, m in uses if m in _VALUED}
# A reader that ONLY switches. Subtracting reader sets instead let a capability reading
# both ways empty the difference, and a second one switching the same name went green.
            switch_only = switched - valued
            if switch_only and valued:
                bad.append("%s: %s" % (flag, ", ".join(sorted("%s reads it with %s" % u
                                                             for u in uses))))
        self.assertEqual(bad, [], "\n".join([
            "a flag is a bare switch in one capability and takes a value in another.",
            "On an image serving both, the value form is claimed by the switch and read as",
            "OFF, and the unclaimed-flag warning does not fire because it was claimed.",
            "Give one of them its own name (--<capability>-<flag>).", *bad]))

    def test_value_taking_methods_that_disagree_are_reported(self):
        soft, poly = [], []
        for flag, uses in sorted(claims().items()):
            readers = {r for r, _ in uses}
            methods = {m for _, m in uses}
            if "switch" not in methods and len(methods) > 1:
                soft.append("%s: %s" % (flag, ", ".join(sorted("%s:%s" % u for u in uses))))
            elif "switch" in methods and methods & _VALUED and len(readers) == 1:
                poly.append("%s: %s reads it both ways" % (flag, readers.pop()))
        if soft:
            print("\nflags read by different value methods (not an error):\n  "
                  + "\n  ".join(soft))
        if poly:
            print("\nflags one capability deliberately reads both ways (not an error):\n  "
                  + "\n  ".join(poly))


if __name__ == "__main__":
    unittest.main()
