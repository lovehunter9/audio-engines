"""Submit complete English sentences to the original MT queue, without timed cuts."""
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import re
import textwrap

FINAL_BOUNDARY = object()


_ABBREVIATIONS = frozenset({'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'st.', 'vs.', 'etc.', 'e.g.', 'i.e.',
                          'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.', 'aug.', 'sep.', 'sept.', 'oct.', 'nov.', 'dec.',
                          'no.', 'fig.', 'inc.', 'ltd.', 'a.m.', 'p.m.'})
_END = frozenset('.!?。！？…')
_CLOSE = frozenset('"\'”’）)]】》')


def sentence_ends(text):
    """Conservative literal punctuation boundary; never rewrite source text."""
    result = []
    i = 0
    while i < len(text):
        if text[i] not in _END:
            i += 1
            continue
        start = i
        while i < len(text) and text[i] in _END:
            i += 1
        punctuation = text[start:i]
        if i == len(text):
            # The next increment can complete a decimal, ellipsis or quote.
            break
        if punctuation == '.':
            if not text[i].isspace() and text[i] not in _CLOSE:
                continue
            token = re.search(r'[^\s]+$', text[:i])
            token = token.group().lstrip('("\'“‘') if token else ''
            if token.lower() in _ABBREVIATIONS or re.fullmatch(r'(?:[A-Za-z]\.)+', token):
                continue
        elif punctuation == '..':
            continue
        while i < len(text) and text[i] in _CLOSE:
            i += 1
        result.append(i)
    return result


class SourceContinuity:
    def __init__(self):
        self.pending = ''
        self.received = hashlib.sha256()
        self.released = hashlib.sha256()
        self.input_chars = self.output_chars = 0

    def release(self, text):
        self.released.update(text.encode())
        self.output_chars += len(text)
        return text

    def feed(self, text):
        self.received.update(text.encode())
        self.input_chars += len(text)
        text = self.pending + text
        spans = sentence_ends(text)
        values = []
        start = 0
        for end in spans:
            values.append(self.release(text[start:end]))
            start = end
        self.pending = text[start:]
        return values

    def finish(self):
        text, self.pending = self.pending, ''
        return self.release(text)

    def summary(self):
        exact = self.input_chars == self.output_chars and self.received.digest() == self.released.digest()
        if self.pending or not exact:
            raise RuntimeError('ASR source continuity lost input')
        return {'state': 'raw_ASR_to_MT_continuity_exact', 'input_chars': self.input_chars,
                'output_chars': self.output_chars, 'received_sha256': self.received.hexdigest(),
                'released_sha256': self.released.hexdigest(), 'pending_chars': len(self.pending)}


class ContinuityQueue(asyncio.Queue):
    def __init__(self, idle_flush, audit_path=None):
        super().__init__()
        self.continuity = SourceContinuity()
        self.idle_flush = idle_flush
        self.audit_path = audit_path

    def put_nowait(self, item):
        if isinstance(item, str):
            for ready in self.continuity.feed(item):
                super().put_nowait(ready)
                super().put_nowait(self.idle_flush)
            return
        if item is FINAL_BOUNDARY or item is None:
            tail = self.continuity.finish()
            if tail:
                super().put_nowait(tail)
            if item is None and self.audit_path:
                self.audit_path.write_text(json.dumps(self.continuity.summary(), indent=2) + '\n')
            item = self.idle_flush if item is FINAL_BOUNDARY else None
        # An idle flush still uses the original policy for already complete
        super().put_nowait(item)


def install(server, audit_path=None):
    cls = server.SpeechSession
    original_init = cls.__init__
    original_update = cls._on_asr_update
    source = textwrap.dedent(inspect.getsource(original_update))
    old = 'self._translation_queue.put_nowait(_FLUSH)'
    if source.count(old) != 1:
        raise RuntimeError('Pinned official ASR callback differs')
    scope = dict(server.__dict__, FINAL_BOUNDARY=FINAL_BOUNDARY)
    exec(compile(source.replace(old, 'self._translation_queue.put_nowait(FINAL_BOUNDARY)', 1),
                 '<source-continuity-reset-only>', 'exec'), scope)
    final_update = scope['_on_asr_update']

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.direction == 'en2zh':
            self._translation_queue = ContinuityQueue(server._FLUSH, audit_path)

    async def update(self, value):
        callback = final_update if self.direction == 'en2zh' else original_update
        return await callback(self, value)

    cls.__init__ = initialize
    cls._on_asr_update = update


# Uvicorn imports the same pinned application after installing the queue-only
if __name__ == 'source_bridge':
    from inference import server
    install(server, Path.cwd() / 'source-continuity.json')
    app = server.app
