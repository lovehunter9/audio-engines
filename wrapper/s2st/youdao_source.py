import asyncio
from dataclasses import dataclass
import json
import time
from types import SimpleNamespace
import re
from .youdao_numbers import boundary, decimal_separator


_TRAILING = re.compile(
    r"(?:\b(?:and|or|to|of|the|a|an|by|with|for|from|is|are|was|were)|以及|而且|或者|通过|包括|如果|虽然|因为|所以|为了|以便|能够|可以|那些|一个|我们|他们|这是|将|把|被|的|第|是)$",
    re.I,
)
_ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|e\.g|i\.e)|\b[A-Z])\.$", re.I)


class SourceClauses:
    def __init__(self, direction, minimum_samples=0):
        if minimum_samples not in (0, 32000, 64000):
            raise ValueError("Unsupported translation source increment")
        self.direction = direction
        self.minimum_samples = minimum_samples if direction == "en2zh" else 0
        self.last_commit_samples = 0
        self.pending = ""
        self.received = ""
        self.committed = ""
        self.since = None

    def feed(self, text, samples, final=False):
        if text and not self.pending:
            self.since = samples
        self.pending += text
        self.received += text
        if not final and samples - self.last_commit_samples < self.minimum_samples:
            return []
        end = 0
        for match in re.finditer(r"[,，。！？!?；;.]", self.pending):
            index = match.end()
            if decimal_separator(self.pending, match.start()):
                continue
            if match.group() == "." and self.direction == "en2zh":
                if index == len(self.pending) or not self.pending[index].isspace():
                    continue
                if _ABBREVIATION.search(self.pending[:index]):
                    continue
            content = self.pending[:index].rstrip(" ,，。！？!?；;.")
            if _TRAILING.search(content):
                continue
            end = index
        complete = bool(end)
        if not end and self.since is not None and samples - self.since >= 32000:
            if self.direction == "en2zh":
                end = max((m.end() for m in re.finditer(r"\s+", self.pending)), default=0)
                content = self.pending[:end].strip()
                if len(content.split()) < 4:
                    end = 0
            else:
                end = len(self.pending)
                content = self.pending.strip()
                if len(re.findall(r"[\u3400-\u9fff]", content)) < 6:
                    end = 0
            if self.direction == "en2zh" and _TRAILING.search(content):
                end = 0
        if final:
            end = len(self.pending)
            complete = True
        if not final:
            end = boundary(self.pending, end)
        if not end:
            return []
        value, self.pending = self.pending[:end], self.pending[end:]
        self.committed += value
        self.last_commit_samples = samples
        self.since = samples if self.pending else None
        if final and (self.pending or self.committed != self.received):
            raise RuntimeError("Source clause adapter changed source content")
        return [(value, complete)]


def install(english_chunk_ms=0):
    if english_chunk_ms not in (0, 2000, 4000):
        raise ValueError("Unsupported translation source increment")
    from inference import asr, server
    from .youdao_bounded_wait import SourceIncrement
    from inference.translation import TranslationEngine

    @dataclass(frozen=True)
    class CompleteSource(SourceIncrement):
        source_samples: int
        complete_clause: bool

    native_feed = TranslationEngine.feed

    async def feed(self, text):
        events = await native_feed(self, text)
        if (
            getattr(self, "_source_complete_clause", False)
            and self.pending_source
            and not any(e.get("type") == "translation" for e in events)
        ):
            events = await self.flush()
        return events

    TranslationEngine.feed = feed

    async def on_asr(self, update):
        self._asr_latest_source_samples = update.source_samples
        if update.text:
            await self.send(
                dict(
                    type="asr",
                    text=update.text,
                    reset=update.reset,
                    source_samples=update.source_samples,
                )
            )
        if not hasattr(self, "_source_clauses"):
            self._source_clauses = SourceClauses(self.direction, english_chunk_ms * 16)
        for text, complete in self._source_clauses.feed(
            update.text,
            update.source_samples,
            final=update.reset,
        ):
            if text:
                self._translation_queue.put_nowait(
                    CompleteSource(text, time.monotonic(), update.source_samples, complete)
                )
        if update.reset:
            self._cancel_idle_timer()
            self._translation_queue.put_nowait(server._FLUSH)

    server.SpeechSession._on_asr_update = on_asr

    original_finish = asr.R2T2StreamingASR.finish

    async def read(self):
        try:
            while True:
                raw = await self._ws.recv()
                value = asr.parse_r2t2_message(raw)
                if value is None:
                    continue
                packet = json.loads(raw).get("msg", {})
                mark = packet.get("source_samples")
                if not isinstance(mark, int) or mark < getattr(self, "_s2st_seen_samples", 0):
                    raise asr.ASRError("Missing or nonmonotonic ASR sample watermark")
                self._s2st_seen_samples = mark
                if getattr(self, "_s2st_ending", False) and value.reset:
                    self._s2st_final = True
                if self.on_update is not None:
                    await self.on_update(
                        SimpleNamespace(
                            text=value.text,
                            reset=value.reset,
                            source_samples=mark,
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not getattr(self, "_s2st_final", False):
                self._s2st_error = exc

    async def finish(self):
        self._s2st_ending = True
        await original_finish(self)
        if getattr(self, "_s2st_error", None) is not None:
            raise asr.ASRError("ASR transport/inference failed: " + str(self._s2st_error))
        if not getattr(self, "_s2st_final", False):
            raise asr.ASRError("ASR closed without a final acknowledgment")

    asr.R2T2StreamingASR._read_loop = read
    asr.R2T2StreamingASR.finish = finish
