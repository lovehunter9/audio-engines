"""Packet-independent speech boundaries for the native streaming decoder."""

import numpy as np


class UtteranceBuffer:
    def __init__(self, probability, *, chunk_size=5120, silence_ms=1200):
        self.probability = probability
        self.chunk_size = chunk_size
        self.silence_limit = int(silence_ms * 16)
        self.pending = np.empty(0, dtype=np.float32)
        self.preroll = []
        self.audio = []
        self.silence = []
        self.active = False
        self.gain = None

    def _take(self, count):
        if not count:
            return []
        result, self.audio = self.audio[:count], self.audio[count:]
        if self.gain is None:
            rms = float(np.sqrt(np.mean(np.square(result))))
            self.gain = min(100.0, max(0.25, 0.1 / max(0.0001, rms)))
        return (np.asarray(result, dtype=np.float32) * self.gain).tolist()

    def _close(self):
        self.audio.extend(self.silence[:1600])
        result = [(self._take(len(self.audio)), True)]
        self.preroll = self.silence[-1600:]
        self.audio, self.silence = [], []
        self.active, self.gain = False, None
        return result

    def feed(self, samples, finished):
        self.pending = np.concatenate((self.pending, np.asarray(samples, dtype=np.float32)))
        events = []
        while len(self.pending) >= 512 or finished and len(self.pending):
            frame, self.pending = self.pending[:512], self.pending[512:]
            padded = np.pad(frame, (0, 512 - len(frame)))
            rms = float(np.sqrt(np.mean(padded * padded)))
            probability = self.probability(padded * min(256.0, 0.1 / max(0.0001, rms)))
            if probability >= 0.6:
                if not self.active:
                    self.audio.extend(self.preroll)
                    self.preroll = []
                    self.active = True
                self.audio.extend(self.silence)
                self.silence = []
                self.audio.extend(frame.tolist())
            elif self.active:
                self.silence.extend(frame.tolist())
                if len(self.silence) >= self.silence_limit:
                    events.extend(self._close())
            else:
                self.preroll = (self.preroll + frame.tolist())[-1600:]
            while self.active and len(self.audio) >= self.chunk_size:
                events.append((self._take(self.chunk_size), False))
        if finished and self.active:
            events.extend(self._close())
        return events
