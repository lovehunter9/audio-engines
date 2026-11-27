import base64
import hashlib
from pathlib import Path
import wave


class CompleteWords:
    def __init__(self, english):
        self.english = english
        self.pending = ""
        self.received = ""
        self.committed = ""

    def feed(self, text, final=False):
        self.received += text
        self.pending += text
        end = (
            len(self.pending)
            if final or not self.english
            else max((i + 1 for i, c in enumerate(self.pending) if c.isspace()), default=0)
        )
        out, self.pending = self.pending[:end], self.pending[end:]
        self.committed += out
        if final:
            assert self.committed == self.received and not self.pending
        return out


def install():
    from simulinfer.tts import Speech

    if getattr(Speech, "_validated_delivery_installed", False):
        return
    original_text, original_finish, original_synthesize = (
        Speech.text,
        Speech.finish,
        Speech._synthesize,
    )
    original_body = Speech._speech_body

    def speech_body(self, value):
        body = original_body(self, value)
        if self.s.tts_backend == "vllm_omni":
            body["seed"] = 41 + getattr(self, "_delivery_attempt", 1)
        return body

    async def text(self, value):
        state = getattr(self, "_complete_words", None)
        if state is None:
            state = self._complete_words = CompleteWords(self.language == "en")
        out = state.feed(value)
        if out:
            await original_text(self, out)

    async def finish(self):
        state = getattr(self, "_complete_words", None)
        if state is not None:
            out = state.feed("", final=True)
            if out:
                await original_text(self, out)
            await self.emit(
                {
                    "event": "tts_text_integrity",
                    "identical": state.received == state.committed,
                    "received_sha256": hashlib.sha256(state.received.encode()).hexdigest(),
                    "committed_sha256": hashlib.sha256(state.committed.encode()).hexdigest(),
                }
            )
        return await original_finish(self)

    async def synthesize(self, value):
        if self.paired:
            raise RuntimeError("Validated phrase delivery requires generated history disabled")
        public_emit, initial_bytes = self.emit, self.bytes
        for attempt in (1,):
            self._delivery_attempt = attempt
            events = []

            async def stage(event):
                if event.get("event") in ("audio", "tts_end"):
                    events.append(event)
                else:
                    await public_emit(dict(event, tts_attempt=attempt))

            self.emit = stage
            try:
                await original_synthesize(self, value)
            except BaseException as exc:
                self.bytes = initial_bytes
                self.emit = public_emit
                raw = b"".join(
                    base64.b64decode(e["data"]) for e in events if e.get("event") == "audio"
                )
                case = getattr(self, "_bench_case_dir", None)
                artifact = None
                if case is not None and raw:
                    folder = Path(case) / "failed-tts-attempts"
                    folder.mkdir(exist_ok=True)
                    artifact = folder / f"{self.seq:04d}-{attempt}.wav"
                    with wave.open(str(artifact), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(self.rate)
                        w.writeframes(raw)
                retry = False
                await public_emit(
                    {
                        "event": "tts_attempt_failed",
                        "seq": self.seq,
                        "attempt": attempt,
                        "seed": 41 + attempt,
                        "error_type": type(exc).__name__,
                        "detail": str(exc)[:500],
                        "withheld_audio_bytes": len(raw),
                        "published_audio_bytes": 0,
                        "failed_audio": None if artifact is None else str(artifact),
                        "retry": retry,
                    }
                )
                if not retry:
                    raise
                continue
            finally:
                self.emit = public_emit
            await public_emit(
                {
                    "event": "tts_piece_validated",
                    "seq": self.seq,
                    "attempt": attempt,
                    "seed": 41 + attempt,
                    "delivery": "complete phrase before publication; original chunk order retained",
                }
            )
            for event in events:
                await public_emit(event)
            return

    Speech.text, Speech.finish, Speech._synthesize = text, finish, synthesize
    Speech._speech_body = speech_body
    Speech._validated_delivery_installed = True
