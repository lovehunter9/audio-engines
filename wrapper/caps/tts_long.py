# Shared long-read helpers: cut, pace, and join.
import queue
import re
import subprocess
import threading

from . import tts_el

# Official 8 inline events live in the speak text. Do not cut inside them.
_EVENT_RE = re.compile(
    r"\[(?:笑|咳嗽|清嗓子|叹气)\]"
    r"|\((?:laugh|cough|clears throat|sigh)\)"
)
# CJK stops need no following space; EN .!? need space/end, and "." is not a cut after a digit (3.14) or a title (Mr.).
_SENT_END = re.compile(
    r"\.\.\.|…"
    r"|[。．！？；]"
    r"|(?<!\d)[.!?](?=\s|$|[\"'”’）)\]])"
)
_CLAUSE_END = re.compile(r"[，、]|[；;](?=\s|$)|(?<!\d),(?=\s)")
_EN_ABBR = re.compile(
    r"(?:^|[\s(\[（])(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.$",
    re.I,
)
# Head fade is only anti-click into the pause. The last slice is never trimmed.
JOIN_FADE_MS = 16.0
PAUSE_MS_ZH = 520.0
PAUSE_MS_EN = 280.0
TRIM_KEEP_MS = 80.0
TRIM_THRESH = 0.008
BREAK_RE = re.compile(r"[。．！？；;，、!?,]\s*")


def collect(audio):
    import numpy as np

    a = np.asarray(audio, dtype="float32")
    if a.ndim > 1:
        a = a.reshape(-1)
    return np.clip(a, -1.0, 1.0)


def _atempo_chain(speed):
    """Build a portable atempo chain whose individual factors stay in 0.5..2.0."""
    factor = max(0.25, min(4.0, float(speed if speed is not None else 1.0)))
    parts = []
    while factor < 0.5 - 1e-9:
        parts.append(0.5)
        factor /= 0.5
    while factor > 2.0 + 1e-9:
        parts.append(2.0)
        factor /= 2.0
    if abs(factor - 1.0) >= 1e-9 or not parts:
        parts.append(factor)
    return ",".join("atempo=%.8g" % value for value in parts)


class TempoStream:
    """One pitch-preserving tempo filter for every chunk in a request."""

    _DONE = object()

    def __init__(self, sr, speed, tail_ms=10.0):
        import numpy as np

        self.sr = int(sr)
        self.speed = max(0.25, min(4.0, float(speed if speed is not None else 1.0)))
        self.tail = max(1, int(round(float(tail_ms) / 1000.0 * self.sr)))
        self.pending = np.zeros(0, dtype="float32")
        self.remainder = b""
        self.q = queue.Queue()
        self.proc = None
        if abs(self.speed - 1.0) < 0.02:
            return
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(self.sr), "-ac", "1", "-i", "pipe:0",
            "-af", _atempo_chain(self.speed),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as e:
            raise RuntimeError("ffmpeg is required for speed control: %s" % e) from e
        threading.Thread(target=self._pump, daemon=True, name="tts-tempo").start()

    def _pump(self):
        try:
            while True:
                chunk = self.proc.stdout.read(4096)
                if not chunk:
                    break
                self.q.put(chunk)
        finally:
            self.q.put(self._DONE)

    def _emit(self, arrays, final=False):
        import numpy as np

        if arrays:
            joined = np.concatenate([self.pending] + arrays)
        else:
            joined = self.pending
        if not final:
            if len(joined) <= self.tail:
                self.pending = joined
                return []
            self.pending = joined[-self.tail:].copy()
            return [joined[:-self.tail]]
        self.pending = np.zeros(0, dtype="float32")
        if not len(joined):
            return []
        out = joined.astype("float32", copy=True)
        fade = min(self.tail, len(out))
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")
        return [out, np.zeros(self.tail, dtype="float32")]

    def _drain(self, wait=0.01, until_done=False):
        import numpy as np

        arrays = []
        done = False
        while True:
            try:
                item = self.q.get(timeout=2.0 if until_done else wait)
            except queue.Empty:
                break
            if item is self._DONE:
                done = True
                break
            raw = self.remainder + item
            cut = len(raw) - (len(raw) % 4)
            if cut:
                arrays.append(np.frombuffer(raw[:cut], dtype="<f4").astype("float32", copy=True))
            self.remainder = raw[cut:]
            if not until_done:
                wait = 0.0
        if until_done and not done:
            raise RuntimeError("tempo filter did not finish")
        return arrays

    def write(self, wave, sr):
        w = collect(wave)
        if int(sr) != self.sr:
            w, _ = tts_el._resample(w, int(sr), self.sr)
        if self.proc is None:
            return [w]
        try:
            self.proc.stdin.write(w.astype("<f4").tobytes())
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            raise RuntimeError("tempo filter stopped early") from e
        return self._emit(self._drain())

    def finish(self):
        if self.proc is None:
            return []
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        arrays = self._drain(until_done=True)
        rc = self.proc.wait(timeout=15)
        if rc != 0:
            detail = (self.proc.stderr.read() or b"")[-300:].decode("utf-8", "replace")
            raise RuntimeError("tempo filter failed: %s" % (detail or rc))
        if self.remainder:
            raise RuntimeError("tempo filter returned incomplete float samples")
        return self._emit(arrays, final=True)

    def abort(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)


def pace(wave, sr, speed):
    """Whole-buffer tempo, for callers that already hold all of the audio."""
    import numpy as np

    factor = max(0.25, min(4.0, float(speed if speed is not None else 1.0)))
    w = collect(wave)
    if abs(factor - 1.0) < 0.02 or not len(w):
        return w, sr
    tempo = TempoStream(sr, factor)
    try:
        parts = tempo.write(w, sr) + tempo.finish()
    finally:
        tempo.abort()
    if not parts:
        raise RuntimeError("tempo filter produced no audio")
    return np.concatenate(parts), sr


def _hold_events(text):
    held = []

    def keep(m):
        held.append(m.group(0))
        return "\x00%d\x00" % (len(held) - 1)

    return _EVENT_RE.sub(keep, text), held


def _unhold(text, held):
    if not held:
        return text
    return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)


def _skip_en_abbr(text, match):
    """Do not treat 'Mr. Smith' as two sentences. CJK marks never hit this."""
    return match.group(0) == "." and bool(_EN_ABBR.search(text[:match.end()]))


def _cut(text, ender, skip=None):
    """Keep the delimiter on the left piece. Empty parts are dropped."""
    parts, start = [], 0
    for m in ender.finditer(text):
        if skip and skip(text, m):
            continue
        piece = text[start:m.end()]
        if piece:
            parts.append(piece)
        start = m.end()
    tail = text[start:]
    if tail:
        parts.append(tail)
    return parts


def _hard_cut(piece, limit):
    """A stretch with nothing to break on. Prefer a space in reach over the count."""
    out = []
    while len(piece) > limit:
        at = piece[:limit].rfind(" ")
        if at < limit // 2:
            at = limit
        out.append(piece[:at].strip())
        piece = piece[at:].lstrip()
    out.append(piece.strip())
    return [p for p in out if p]


def split_speak(text, limit):
    """Newlines always split. Sentence then clause only when a piece is over budget."""
    limit = int(limit)
    raw = text or ""
    if not raw.strip():
        return [raw] if raw else []
    protected, held = _hold_events(raw)
    out = []
    for line in re.split(r"\n+", protected):
        if not line.strip():
            continue
        if len(_unhold(line, held)) <= limit:
            piece = _unhold(line, held).strip()
            if piece:
                out.append(piece)
            continue
        for sent in _cut(line, _SENT_END, skip=_skip_en_abbr) or [line]:
            if len(_unhold(sent, held)) <= limit:
                piece = _unhold(sent, held).strip()
                if piece:
                    out.append(piece)
                continue
            for cl in _cut(sent, _CLAUSE_END) or [sent]:
                piece = _unhold(cl, held).strip()
                if not piece:
                    continue
                # Counting characters is the last resort: punctuation carries the prosody.
                if len(piece) <= limit:
                    out.append(piece)
                else:
                    out.extend(_hard_cut(piece, limit))
    return out


def pause_ms(text):
    if any("\u4e00" <= ch <= "\u9fff" for ch in text or ""):
        return PAUSE_MS_ZH
    return PAUSE_MS_EN


def snap(text, at, span):
    """First break at or after `at`. Only forward, so the tail can come in under the
    seconds asked for but never over them, which is what the budget was reserved for."""
    m = BREAK_RE.search(text, at, min(len(text), at + span))
    return m.end() if m else at


def tail_pair(text, wave, sr, seconds):
    """The tail of a finished slice with the text that goes with it. The text is cut by
    the share of the audio it covers, moved to a break, and the audio cut to match."""
    import numpy as np

    w = np.asarray(wave, dtype="float32").reshape(-1)
    total = len(w) / float(sr or 1)
    if not len(w) or not (text or "").strip() or total <= seconds:
        return (text or "").strip(), w
    want = len(text) * (seconds / total)
    at = snap(text, int(len(text) - want), max(4, int(want * 0.3)))
    if at <= 0 or at >= len(text):
        return text.strip(), w
    cut = max(int(len(w) * (float(at) / len(text))), len(w) - int(seconds * (sr or 1)))
    return text[at:].strip(), w[cut:]


def fade_head(wave, fade):
    import numpy as np

    n = min(int(fade), len(wave))
    if n <= 0:
        return wave
    out = np.array(wave, dtype="float32", copy=True)
    out[:n] *= np.linspace(0.0, 1.0, n, dtype="float32")
    return out


def trim_tail(wave, sr, thresh=TRIM_THRESH, keep_ms=TRIM_KEEP_MS):
    """Drop trailing silence only; keep keep_ms after the last voiced sample so the last syllable stays."""
    import numpy as np

    w = np.asarray(wave, dtype="float32").reshape(-1)
    if not len(w):
        return w
    keep = max(0, int(float(keep_ms) / 1000.0 * sr))
    loud = np.flatnonzero(np.abs(w) >= thresh)
    if not len(loud):
        return w[:max(1, keep)]
    return w[:min(len(w), int(loud[-1]) + 1 + keep)]


def join(waves, sr, fade_ms=JOIN_FADE_MS, gap_ms=PAUSE_MS_ZH):
    import numpy as np

    raw = [np.asarray(w, dtype="float32").reshape(-1) for w in waves]
    last = len(raw) - 1
    parts = []
    for i, w in enumerate(raw):
        parts.append(w if i == last else trim_tail(w, sr))
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros(0, dtype="float32"), sr
    if len(parts) == 1:
        return parts[0], sr
    fade = max(1, int(float(fade_ms) / 1000.0 * sr))
    gap = np.zeros(max(0, int(float(gap_ms) / 1000.0 * sr)), dtype="float32")
    out = parts[0]
    for wave in parts[1:]:
        out = np.concatenate([out, gap, fade_head(wave, fade)])
    return np.clip(out, -1.0, 1.0), sr


def vram(log, tag):
    """This process's own high-water mark, then reset it. On a timeslice card the whole
    card's used is every tenant's sum, so ours is the only figure we can attribute."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        log.info("vram %s: peak=%.0f MiB now=%.0f MiB reserved=%.0f MiB", tag,
                 torch.cuda.max_memory_allocated() / 2 ** 20,
                 torch.cuda.memory_allocated() / 2 ** 20,
                 torch.cuda.memory_reserved() / 2 ** 20)
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
