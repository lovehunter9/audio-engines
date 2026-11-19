import asyncio
import hashlib
import re
import time


def count(text, english):
    return (
        len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?", text))
        if english
        else len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))
    )


def install():
    from simulinfer.tts import Speech

    if getattr(Speech, "_micro_guard_installed", False):
        return

    async def next_text(self):
        if not hasattr(self, "_guard_received"):
            self._guard_received = ""
            self._guard_committed = ""
            self._guard_audited = False

        def accept(item):
            if item is not None:
                self._guard_received += item[0] if self.source_voice is not None else item

        if self._input_done:
            assert self._guard_received == self._guard_committed
            if not self._guard_audited:
                self._guard_audited = True
                await self.emit(
                    dict(
                        event="tts_clause_integrity",
                        identical=True,
                        received_sha256=hashlib.sha256(self._guard_received.encode()).hexdigest(),
                        committed_sha256=hashlib.sha256(self._guard_committed.encode()).hexdigest(),
                    )
                )
            return None
        if self._carry is not None:
            item, self._carry = self._carry, None
        else:
            item = await self.queue.get()
            self.queue.task_done()
            accept(item)
        if item is None:
            self._input_done = True
            return await next_text(self)
        text, snapshot = item if self.source_voice is not None else (item, None)
        started = time.monotonic()
        delay = 0 if self.s.tts_min_chars == 1 else self.s.tts_batch_ms / 1000
        deadline = asyncio.get_running_loop().time() + delay
        limit = 240 if self.language == "en" else 100
        guarded = False
        while delay and len(text) < limit:
            remaining = deadline - asyncio.get_running_loop().time()
            micro = count(text, self.language == "en") < 4
            if (
                self.queue.empty()
                and not micro
                and (remaining <= 0 or re.search(r"[。！？!?\n.]\s*$", text))
            ):
                break
            guarded = guarded or (micro and self.queue.empty())
            try:
                if not self.queue.empty():
                    extra = self.queue.get_nowait()
                elif micro:
                    extra = await self.queue.get()
                else:
                    extra = await asyncio.wait_for(self.queue.get(), max(0.001, remaining))
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break
            self.queue.task_done()
            accept(extra)
            if extra is None:
                self._input_done = True
                break
            next_value, next_snapshot = extra if self.source_voice is not None else (extra, None)
            if len(text) + len(next_value) > limit:
                self._carry = extra
                break
            text += next_value
            snapshot = next_snapshot
        self._guard_committed += text
        await self.emit(
            dict(
                event="tts_clause_commit",
                chars=len(text),
                semantic_units=count(text, self.language == "en"),
                buffered_seconds=time.monotonic() - started,
                micro_guarded=guarded,
                normal_batch_ms=self.s.tts_batch_ms,
                final=self._input_done,
            )
        )
        return text, snapshot

    Speech._next_text = next_text
    Speech._micro_guard_installed = True
