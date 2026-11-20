"""Lossless English delivery boundary; no target text or TTS parameter changes."""

import re


class SpeechWordTail:
    OPEN = frozenset(
        "a an the to of in on at for from by with without into onto upon about under over between through during before after as than and or but if because although while whose which that is are was were be been being has have had do does did can could will would shall should may might must".split()
    )

    def __init__(self, direction):
        self.direction = direction
        self.pending = ""
        self.received = ""
        self.delivered = ""

    def feed(self, text, final=False):
        self.received += text
        self.pending += text
        cut = len(self.pending)
        if self.direction == "zh2en" and not final:
            words = list(re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?", self.pending))
            # Hold only a trailing function-word chain, never the whole sentence.
            if words and not self.pending[words[-1].end() :].strip().strip(".,;:!?").strip():
                index = len(words) - 1
                while index >= 0 and words[index].group().lower() in self.OPEN:
                    cut = words[index].start()
                    index -= 1
        ready, self.pending = self.pending[:cut], self.pending[cut:]
        self.delivered += ready
        if not self.received.startswith(self.delivered):
            raise RuntimeError("Youdao speech boundary revised delivered text")
        if final and (self.pending or self.received != self.delivered):
            raise RuntimeError("Youdao speech boundary did not flush all text")
        return ready
