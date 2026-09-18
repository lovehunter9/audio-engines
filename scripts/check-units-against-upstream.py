#!/usr/bin/env python3
"""Diff the hand-copied `_units_of` against the library function it was copied from."""
import argparse, ast, random, sys, unicodedata
from pathlib import Path
from typing import List

UPSTREAM_METHODS = ("is_kept_char", "clean_token", "is_cjk_char",
                    "split_segment_with_chinese", "tokenize_space_lang")

OURS_FUNCTIONS = ("_is_kept", "_is_cjk", "_units_of")

ALPHABET = list("abcXYZ '.,!?0129") + list(
    "你好世界"      # CJK
    "あカん"            # kana
    "가한글"            # hangul
    "éüñ"            # accented Latin
    "аб"                  # Cyrillic
    "、。，…"      # CJK punctuation
    "\U00020000"                    # astral-plane ideograph (is CJK)
    "\U0001f600")                   # emoji (is not)

SMOKE_CASES = [
    "hi there 你好",
    "你abc好",
    "啊……对，对，对。",
    "a.b.c",
    "……!!",
    "don't stop",
    "",
    " ",
    "あいうえお",
]


# Lifted out of the library's source rather than imported: the class needs a loaded
# checkpoint to construct, and these functions do not depend on one.
def _lift(path, names, is_method):
    """exec the named top-level functions (or one class's methods) out of a source file."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    wanted, found = list(names), {}
    bodies = []
    if is_method:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name in wanted:
                        bodies.append(sub)
    else:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                bodies.append(node)
    for b in bodies:
        found[b.name] = b
    missing = [n for n in wanted if n not in found]
    if missing:
        sys.exit("FAIL: %s does not define %s -- it was renamed, moved or removed, "
                 "which is itself the drift this checks for" % (path, ", ".join(missing)))
    ns = {"unicodedata": unicodedata, "List": List}
    exec(compile(ast.Module(body=[found[n] for n in wanted], type_ignores=[]),
                 str(path), "exec"), ns)
    return ns


def _show(s):
    return "%r  (%s)" % (s, " ".join("U+%04X" % ord(c) for c in s))


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="qwen3_forced_aligner.py; default: ask the "
                                     "installed qwen_asr where it is")
    ap.add_argument("--ours", default=str(here / "wrapper" / "caps" / "align.py"))
    ap.add_argument("--strings", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    src = a.source
    if not src:
        import importlib.util
        spec = importlib.util.find_spec("qwen_asr.inference.qwen3_forced_aligner")
        if spec is None or not spec.origin:
            sys.exit("FAIL: qwen_asr is not installed and --source was not given. This "
                     "check only means anything where the library is; run it in the "
                     "base-image build, not in CI.")
        src = spec.origin

    up = _lift(src, UPSTREAM_METHODS, is_method=True)
    ours = _lift(a.ours, OURS_FUNCTIONS, is_method=False)

    class _Self:
        """Enough of the processor for five methods that only call each other."""

    me = _Self()
    for _name in UPSTREAM_METHODS:
        setattr(me, _name, (lambda f: lambda *args: f(me, *args))(up[_name]))
    tokenize = me.tokenize_space_lang

    print("upstream : %s" % src)
    print("ours     : %s" % a.ours)
    print("seed     : %d, %d random strings" % (a.seed, a.strings))

    bad = 0

    def compare(label, text):
        nonlocal bad
        mine, theirs = ours["_units_of"](text), tokenize(text)
        if mine != theirs:
            bad += 1
            if bad <= 20:
                print("\nMISMATCH (%s)\n  input    %s\n  ours     %r\n  upstream %r"
                      % (label, _show(text), mine, theirs))

    for s in SMOKE_CASES:
        compare("smoke case", s)

    rnd = random.Random(a.seed)
    for _ in range(a.strings):
        n = rnd.randint(0, 24)
        compare("random draw", "".join(rnd.choice(ALPHABET) for _ in range(n)))

    kept = cjk = 0
    for cp in range(0x3000):
        ch = chr(cp)
        if bool(ours["_is_kept"](ch)) != bool(me.is_kept_char(ch)):
            kept += 1
            if kept <= 10:
                print("\nMISMATCH (is_kept) %s  ours %r  upstream %r"
                      % (_show(ch), ours["_is_kept"](ch), me.is_kept_char(ch)))
    for cp in range(0x30000):
        ch = chr(cp)
        if bool(ours["_is_cjk"](ch)) != bool(me.is_cjk_char(ch)):
            cjk += 1
            if cjk <= 10:
                print("\nMISMATCH (is_cjk) %s  ours %r  upstream %r"
                      % (_show(ch), ours["_is_cjk"](ch), me.is_cjk_char(ch)))

    kana = "あいうえお"
    intent = 0
    for lang in ("japanese", "Japanese", "KOREAN", "korean"):
        got = ours["_units_of"](kana, lang)
        if got != list(kana):
            intent += 1
            print("\nMISMATCH (deliberate divergence, %s): expected one unit a character, "
                  "got %r" % (lang, got))

    total = bad + kept + cjk + intent
    print("\n%d strings compared, %d disagreed" % (len(SMOKE_CASES) + a.strings, bad))
    print("is_kept over U+0000..U+2FFF: %d disagreed" % kept)
    print("is_cjk  over U+0000..U+2FFFF: %d disagreed" % cjk)
    print("deliberate japanese/korean divergence: %d broken" % intent)
    if total:
        sys.exit("\nFAIL: %d disagreements. The copy in %s no longer matches %s."
                 % (total, a.ours, src))
    print("\nOK: the copy matches the library.")


if __name__ == "__main__":
    main()
