# Speak digit runs in Chinese: 22号楼 is 二十二号楼, 202户 is 二零二户. The FST normalizer needs pynini, which this image does not ship.
import re

_DIGITS = "零一二三四五六七八九"
# Longer suffixes first. 号楼 must win over 号, or "22号楼" becomes "二十二号" plus a stray 楼.
_SUFFIX = re.compile(r"(?<![\d.])(\d+)(号楼|栋|幢|单元|楼|层|户|室)")
_BARE = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
_DIGIT_SUFFIX = ("户", "室")


def _cardinal(n):
    n = int(n)
    if n < 0:
        return "负" + _cardinal(-n)
    if n < 10:
        return _DIGITS[n]
    if n < 20:
        return "十" if n == 10 else "十" + _DIGITS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        tail = "" if ones == 0 else _DIGITS[ones]
        return _DIGITS[tens] + "十" + tail
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = _DIGITS[hundreds] + "百"
        if rest == 0:
            return head
        if rest < 10:
            return head + "零" + _DIGITS[rest]
        if rest < 20:
            return head + "一" + _cardinal(rest)
        return head + _cardinal(rest)
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        head = _DIGITS[thousands] + "千"
        if rest == 0:
            return head
        if rest < 100:
            return head + "零" + _cardinal(rest)
        return head + _cardinal(rest)
    return "".join(_DIGITS[int(ch)] for ch in str(n))


def _digits(num):
    return "".join(_DIGITS[int(ch)] for ch in num)


def spoken_zh_numbers(text):
    """Rewrite digit runs in a Chinese sentence. English text is left alone."""
    raw = text or ""
    if not any("\u4e00" <= ch <= "\u9fff" for ch in raw):
        return raw

    def suffix(match):
        num, kind = match.group(1), match.group(2)
        # A door number is read one digit at a time. 202户 is 二零二户, not 二百零二户.
        words = _digits(num) if kind in _DIGIT_SUFFIX else _cardinal(int(num))
        return words + kind

    spoken = _SUFFIX.sub(suffix, raw)
    return _BARE.sub(lambda m: _cardinal(int(m.group(1))), spoken)
