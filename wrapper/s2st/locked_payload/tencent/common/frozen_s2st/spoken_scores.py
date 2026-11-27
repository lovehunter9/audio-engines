import re


PAIR = re.compile(
    r"(?<![\w.+\-])(\d{1,2})\s*[-–]\s*(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b(?![\w\-]|\.\d)",
    re.I,
)
SPORT = re.compile(
    r"\b(score|goals?|leading|lead|trailing|ahead|playing at home|won|wins?|defeated|beat)\b", re.I
)
MATH = re.compile(r"(?:[=+*/]|\b(?:equals?|minus|subtract|difference|formula)\b)", re.I)


def spoken_scores(text, context=""):
    def render(match):
        before = text[max(0, match.start() - 80) : match.start()]
        after = text[match.end() : match.end() + 80]
        nearby = before + " " + after
        if MATH.search(nearby) or not SPORT.search(context[-160:] + " " + nearby):
            return match.group(0)
        return match.group(1) + " to " + match.group(2)

    return PAIR.sub(render, text)
