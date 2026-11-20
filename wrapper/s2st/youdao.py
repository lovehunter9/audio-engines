"""Audio-only adapter for the pinned author's R2T2/T3PO speech session."""

import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx

from .gpu_queue import async_stage


class Youdao:
    separate_speech_events = True

    def __init__(
        self, directory, root, url="ws://127.0.0.1:18093/ws/simul-demo", latency_mode="high",
        source_chunk_ms=0,
    ):
        if latency_mode not in ("high", "native"):
            raise ValueError(
                "Only the validated official high/native translation policies are allowed"
            )
        self.latency_mode = latency_mode
        if source_chunk_ms not in (0, 2000, 4000):
            raise ValueError("Unsupported translation source increment")
        self.directory = Path(directory)
        self.root = Path(root)
        self.url = url
        self.socket = None
        self.feeder = None
        self.speech_text = ""
        self.completion_metadata = {"translation_source_chunk_ms": source_chunk_ms}
        self._close_lock = asyncio.Lock()
        self._eos_sent = False

    def speech_stage(self):
        return async_stage("ming", self.root / "gpu-stage-fifo")

    async def run(self, audio, emit_source):
        import websockets
        from inference.translation import join_translation_segments

        self.directory.mkdir(parents=True, exist_ok=False)
        observer = self.root / "results/asr-observer.jsonl"
        mt_observer = self.root / "results/target-stream-observer.jsonl"
        mt_offset = mt_observer.stat().st_size if mt_observer.exists() else 0
        offset = observer.stat().st_size if observer.exists() else 0
        original = bytearray()
        conditioned = 0
        committed = []
        streamed = ""
        initialized = False
        ended = False
        padding_samples = 8000
        started = time.monotonic()
        self.socket = await websockets.connect(
            self.url,
            ping_interval=None,
            max_size=4 * 1024 * 1024,
            open_timeout=120,
            close_timeout=15,
        )
        try:
            await self.socket.send(
                json.dumps(
                    {
                        "type": "init",
                        "direction": audio.stream.direction,
                        "latency_mode": self.latency_mode,
                    }
                )
            )
            deadline = time.monotonic() + 120
            loading = []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Youdao session initialization exceeded its deadline")
                initial = json.loads(await asyncio.wait_for(self.socket.recv(), remaining))
                if initial.get("type") == "loading":
                    loading.append(initial)
                    continue
                if initial.get("type") != "init_ok":
                    raise RuntimeError("Youdao session did not initialize: " + str(initial))
                break
            self.completion_metadata["initialization_loading_events"] = loading
            initialized = True

            async def feed():
                async for pcm in audio:
                    original.extend(pcm)
                    await self.socket.send(pcm)
                self._eos_sent = True
                await self.socket.send(bytes(padding_samples * 2))
                await self.socket.send(json.dumps({"type": "end"}))

            self.feeder = asyncio.create_task(feed())
            while True:
                if self.feeder.done():
                    self.feeder.result()
                remaining = 2400 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Youdao session exceeded its input/compute deadline")
                receiver = asyncio.create_task(self.socket.recv())
                try:
                    wait_seconds = min(600, remaining)
                    deadline = time.monotonic() + wait_seconds
                    if not self.feeder.done():
                        ready, _ = await asyncio.wait(
                            {receiver, self.feeder},
                            timeout=wait_seconds,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not ready:
                            raise TimeoutError(
                                "Youdao stream produced no event before its deadline"
                            )
                    if self.feeder.done():
                        self.feeder.result()
                    event = json.loads(
                        await asyncio.wait_for(receiver, max(0.001, deadline - time.monotonic()))
                    )
                finally:
                    if not receiver.done():
                        receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)
                kind = event.get("type")
                if kind in ("translation_delta", "speech_text"):
                    mark = event.get("request_source_samples")
                    if (
                        not isinstance(mark, int)
                        or isinstance(mark, bool)
                        or mark < 0
                        or mark > len(original) // 2 + (padding_samples if self._eos_sent else 0)
                    ):
                        raise RuntimeError("Youdao referenced source audio before arrival")
                if kind == "translation_delta":
                    delta = event.get("canonical_text", event.get("text", ""))
                    streamed += delta
                    if delta:
                        yield dict(
                            type="translation_delta",
                            text=delta,
                            source_samples=min(mark, len(original) // 2),
                            asr_wire_samples=mark,
                        )
                elif kind == "speech_text":
                    mark = event.get("request_source_samples")
                    if not isinstance(mark, int) or mark < conditioned:
                        raise RuntimeError("Invalid causal source-speaker watermark")
                    if mark > len(original) // 2 + (padding_samples if self._eos_sent else 0):
                        raise RuntimeError("Youdao referenced source audio before arrival")
                    end = min(mark, len(original) // 2)
                    if end < conditioned or end > len(original) // 2:
                        raise RuntimeError("Youdao referenced source audio before arrival")
                    await emit_source(bytes(original[conditioned * 2 : end * 2]))
                    conditioned = end
                    self.speech_text += event["text"]
                    if event["text"]:
                        yield dict(type="speech_text", text=event["text"])
                elif kind == "translation":
                    committed.append(event["text"])
                    value = join_translation_segments(committed, audio.stream.direction)
                    if not streamed.startswith(value):
                        raise RuntimeError(
                            "Committed translation differs from the live canonical stream"
                        )
                elif kind == "speech_text_integrity":
                    if not event.get("identical"):
                        raise RuntimeError("Youdao speech text integrity failed")
                elif kind == "error":
                    raise RuntimeError(
                        "Youdao source/translation failure: " + str(event.get("message", event))
                    )
                elif kind == "asr":
                    yield dict(
                        type="source_transcript",
                        text=event.get("text", ""),
                        source_samples=event.get("source_samples"),
                    )
                elif kind == "ended":
                    if event.get("stats", {}).get("buffer_units"):
                        raise RuntimeError("Youdao ended with uncommitted source text")
                    ended = True
                    break
            await asyncio.wait_for(asyncio.shield(self.feeder), 2)
            self.feeder.result()
            canonical = join_translation_segments(committed, audio.stream.direction)
            if (
                not canonical
                or streamed != canonical
                or join_translation_segments([self.speech_text], audio.stream.direction)
                != canonical
            ):
                raise RuntimeError(
                    "Youdao did not deliver all final translated text to shared MING"
                )
            consumed = 0
            observed_chunk_samples = set()
            if not observer.exists():
                raise RuntimeError("ASR consumption observer is absent")
            with observer.open() as stream:
                stream.seek(offset)
                for line in stream:
                    row = json.loads(line)
                    if row.get("event", "").endswith("_failed"):
                        raise RuntimeError("An original ASR model call failed")
                    consumed += row.get("input_samples", 0)
                    if row.get("input_samples", 0) > 0:
                        observed_chunk_samples.add(row["input_samples"])
            if consumed != audio.samples + padding_samples:
                raise RuntimeError(
                    "ASR did not consume the complete original input and explicit EOS padding"
                )
            if len(original) // 2 != audio.samples or not audio.exhausted:
                raise RuntimeError("Youdao input producer stopped early")
            self.completion_metadata = dict(
                asr_model_consumed_samples=consumed,
                asr_observed_chunk_samples=sorted(observed_chunk_samples),
                asr_eos_padding_samples=padding_samples,
                asr_source_consumed_samples=audio.samples,
                translation_retries=0,
                translation_latency_mode=self.latency_mode,
                speech_input_text_sha256=hashlib.sha256(self.speech_text.encode()).hexdigest(),
                speech_text_integrity_scope="Pinned official join_translation_segments(speech_text) equals canonical MT; explicit word-boundary whitespace is allowed",
            )
            if not mt_observer.exists():
                raise RuntimeError("Translation observation is absent")
            with mt_observer.open() as stream:
                stream.seek(mt_offset)
                rows = [json.loads(line) for line in stream]
            requests = [row for row in rows if row.get("event") == "translation_request"]
            responses = [row for row in rows if row.get("event") == "translation_response"]
            if not requests or len(requests) != len(responses):
                raise RuntimeError("Translation request/completion counts differ")
            self.completion_metadata.update(
                translation_quality_warnings=[
                    {key: row[key] for key in ("sequence", "reason", "text")}
                    for row in rows
                    if row.get("event") == "translation_quality_warning"
                ],
                translation_calls=len(requests),
                translation_compute_seconds=sum(
                    row["finished_monotonic_seconds"] - row["started_monotonic_seconds"]
                    for row in responses
                ),
                translation_timing_scope="Actual MT SSE after GPU gate; includes local HTTP/dispatch, not kernel-only time",
            )
            if not initialized or not ended:
                raise RuntimeError("Youdao session has no complete acknowledgment")
        finally:
            await self.close()

    async def close(self):
        async with self._close_lock:
            await self._close_owned()

    async def _close_owned(self):
        if self.feeder is not None and not self.feeder.done():
            self.feeder.cancel()
        if self.feeder is not None:
            await asyncio.gather(self.feeder, return_exceptions=True)
        if self.socket is None:
            return
        socket, self.socket = self.socket, None
        await socket.close()
        await socket.wait_closed()
        # The author's ended event precedes its final ASR/socket cleanup.
        # Wait for actual session release before public success or another request.
        deadline = time.monotonic() + 60
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
            while time.monotonic() < deadline:
                response = await client.get("http://127.0.0.1:18093/api/v1/health")
                response.raise_for_status()
                metrics = await client.get("http://127.0.0.1:18092/metrics")
                metrics.raise_for_status()
                active = {
                    line.split()[0]: float(line.split()[1])
                    for line in metrics.text.splitlines()
                    if line.startswith(
                        ("llamacpp:requests_processing ", "llamacpp:requests_deferred ")
                    )
                }
                if (
                    response.json().get("active_speech_sessions") == 0
                    and active.get("llamacpp:requests_processing") == 0
                    and active.get("llamacpp:requests_deferred") == 0
                ):
                    return
                await asyncio.sleep(0.05)
        raise RuntimeError("Youdao cleanup did not finish; the request is not complete")
