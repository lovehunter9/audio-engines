"""Retain a trailing source number until its actual scale/unit arrives."""

import re

_DIGITS = r"[0-9０-９零〇一二两三四五六七八九十百千万亿兆点]"
_NUMBER = r"[+-]?" + _DIGITS + r"+(?:[.,:/．^xX×eE-]" + _DIGITS + r"*)*"
_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|point)"
_TERM = "(?:" + _NUMBER + "|" + _WORD + ")"
_TAIL = re.compile(r"(?<![A-Za-z0-9])(?:第)?" + _TERM + r"(?:[ -]+" + _TERM + r")*$|第$", re.I)


def boundary(text, end):
    match = _TAIL.search(text[:end].rstrip())
    return match.start() if match else end


def decimal_separator(text, index):
    if text[index] not in ".,":
        return False
    digits = "0123456789０１２３４５６７８９"
    return (
        index > 0
        and text[index - 1] in digits
        and (index + 1 == len(text) or text[index + 1] in digits)
    )
