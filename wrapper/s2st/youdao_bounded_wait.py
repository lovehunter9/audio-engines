import asyncio
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class SourceIncrement:
    text: str
    arrived: float


@dataclass(frozen=True)
class Deadline:
    generation: int
    due: float


def patch_bridge(bridge, flush_marker, seconds):
    if getattr(bridge, '_bounded_wait_installed', False):
        return
    if seconds <= 0:
        raise ValueError('Pending-source deadline must be positive')
    original_cancel = bridge._cancel_idle_timer

    def cancel(self):
        original_cancel(self)
        self._pending_generation = getattr(self, '_pending_generation', 0) + 1
        self._deadline_queued = False

    def arm(self, since):
        if getattr(self, '_deadline_queued', False) or (self._idle_task is not None and not self._idle_task.done()):
            return
        generation = getattr(self, '_pending_generation', 0)
        due = since + seconds

        async def wait():
            try:
                await asyncio.sleep(max(0.0, due - time.monotonic()))
            except asyncio.CancelledError:
                return
            if generation == getattr(self, '_pending_generation', 0) and not self._ending:
                self._deadline_queued = True
                self._translation_queue.put_nowait(Deadline(generation, due))

        self._idle_task = asyncio.create_task(wait())

    async def on_asr(self, update):
        if update.text:
            arrived = time.monotonic()
            await self.send({'type': 'asr', 'text': update.text, 'reset': update.reset})
            self._translation_queue.put_nowait(SourceIncrement(update.text, arrived))
        if update.reset:
            cancel(self)
            self._translation_queue.put_nowait(flush_marker)

    async def worker(self):
        assert self._translation is not None
        pending_since = None
        try:
            while True:
                item = await self._translation_queue.get()
                try:
                    if item is None:
                        return
                    if isinstance(item, Deadline):
                        if item.generation != getattr(self, '_pending_generation', 0):
                            continue
                        self._deadline_queued = False
                        self._idle_task = None
                        await self.send(dict(type='wait_deadline', maximum_pending_seconds=seconds,
                                             queued_late_seconds=max(0.0, time.monotonic() - item.due),
                                             pending_source=self._translation.engine.pending_source))
                        events = await self._translation.flush()
                        if self._translation.engine.pending_source:
                            pending_since = time.monotonic()
                    elif item is flush_marker:
                        events = await self._translation.flush()
                    else:
                        if isinstance(item, SourceIncrement):
                            text, arrived = item.text, item.arrived
                        else:
                            text, arrived = str(item), time.monotonic()
                        if pending_since is None:
                            pending_since = arrived
                        self._translation.engine._source_increment_arrived = arrived
                        self._translation.engine._source_samples = getattr(item, 'source_samples', None)
                        self._translation.engine._source_complete_clause = getattr(item, 'complete_clause', False)
                        events = await self._translation.feed(text)
                        tail_arrivals = [event.pop('_retained_source_tail_arrived') for event in events
                                         if event.pop('_retained_source_tail', False)]
                        if tail_arrivals:
                            cancel(self)
                            pending_since = min(tail_arrivals)
                    for event in events:
                        if event.get('type') == 'translation':
                            await self.send(event)
                    await self.send({'type': 'metrics', **self._translation.engine.stats})
                    if self._translation.engine.pending_source:
                        arm(self, pending_since if pending_since is not None else time.monotonic())
                    else:
                        cancel(self)
                        pending_since = None
                finally:
                    self._translation_queue.task_done()
        finally:
            cancel(self)

    bridge._cancel_idle_timer = cancel
    bridge._on_asr_update = on_asr
    bridge._translate_worker = worker
    bridge._bounded_wait_installed = True


def install(seconds=8.0):
    from inference.server import SpeechSession, _FLUSH
    patch_bridge(SpeechSession, _FLUSH, seconds)
