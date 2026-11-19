import asyncio
import codecs
import os
from pathlib import Path
import signal
import time

import httpx

from .config import TENCENT_TRANSLATION


def command(model, direction, binary):
    from simulinfer.translation import INSTRUCTIONS

    direction = "zh-en" if direction == "zh2en" else "en-zh"
    prompt = (
        "<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think"
        "<｜hy_User｜>" + INSTRUCTIONS[direction]
        + "<__media__><｜hy_Assistant｜><think></think>"
    )
    return [binary, *TENCENT_TRANSLATION.arguments(model), "--audio", "pcm16:stdin",
            "-n", "-1", "--temp", "0", "--seed", "42", "-p", prompt]


class Tencent:
    def __init__(self, model, directory, binary="/opt/llama/bin/llama-mtmd-cli", native_url=None):
        self.model = Path(model)
        self.directory = Path(directory)
        self.binary = binary
        self.native_url = native_url
        self.native_client = None
        self.fifo = None
        self.native_started = False
        self.process = None
        self.feeder = None
        self.logs = []
        self.consumed_samples = 0

    async def run(self, audio, emit_source):
        self.directory.mkdir(parents=True, exist_ok=False)
        events = self.directory / "translation.tsv"
        self.release = self.directory / "translation.release"
        environment = os.environ.copy()
        environment.update(
            BENCHMARK_EVENTS_PATH=str(events),
            BENCHMARK_HOLD_EXIT=str(self.release),
            BENCHMARK_REALTIME_INPUT="0",
        )
        arguments = command(self.model, audio.stream.direction, self.binary)
        if self.native_url:
            self.fifo_path = self.directory / "input.pcm"
            os.mkfifo(self.fifo_path, mode=0o600)
            self.native_client = httpx.AsyncClient(trust_env=False, timeout=90)
            self.native_started = True
            response = await self.native_client.post(
                self.native_url + "/start", json={"directory": str(self.directory), "command": arguments}
            )
            response.raise_for_status()
            # O_NONBLOCK avoids an uncancellable background open when startup fails.
            for _ in range(600):
                try:
                    self.fifo = os.open(self.fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError as exc:
                    import errno
                    if exc.errno != errno.ENXIO:
                        raise
                    await asyncio.sleep(0.05)
            else:
                raise TimeoutError("Native PCM reader did not open")
        else:
            for name in ["translation-stdout.log", "translation-stderr.log"]:
                self.logs.append((self.directory / name).open("wb"))
            self.process = await asyncio.create_subprocess_exec(
                *arguments, env=environment, stdin=asyncio.subprocess.PIPE,
                stdout=self.logs[0], stderr=self.logs[1], start_new_session=True,
            )
        source = bytearray()

        async def feed():
            async for pcm in audio:
                source.extend(pcm)
                if self.native_url:
                    view = memoryview(pcm)
                    while view:
                        try:
                            count = os.write(self.fifo, view)
                            view = view[count:]
                        except BlockingIOError:
                            await asyncio.sleep(0.005)
                else:
                    self.process.stdin.write(pcm)
                    await self.process.stdin.drain()
            if self.native_url:
                os.close(self.fifo)
                self.fifo = None
            else:
                self.process.stdin.close()
                await self.process.stdin.wait_closed()

        self.feeder = asyncio.create_task(feed())
        decoder = codecs.getincrementaldecoder("utf-8")()
        reader = None
        pending = ""
        finished = False
        resident = False
        deadline = time.monotonic() + 1800
        checked = 0.0
        try:
            while not resident:
                if time.monotonic() > deadline:
                    raise TimeoutError("Tencent session exceeded its input/compute deadline")
                if self.feeder.done():
                    self.feeder.result()
                if reader is None and events.exists():
                    reader = events.open()
                if reader is not None:
                    data = reader.read()
                    if data:
                        lines = (pending + data).split("\n")
                        pending = lines.pop()
                        for line in lines:
                            elapsed, kind, turn, consumed, token, encoded = line.split("\t")
                            samples = round(float(consumed) * 16000)
                            if kind == "audio":
                                if samples > len(source) // 2 or samples < self.consumed_samples:
                                    raise RuntimeError(
                                        "Invalid native source-consumption watermark"
                                    )
                                await emit_source(
                                    bytes(source[self.consumed_samples * 2 : samples * 2])
                                )
                                self.consumed_samples = samples
                            elif kind == "token":
                                text = decoder.decode(bytes.fromhex(encoded))
                                if text:
                                    yield {
                                        "type": "translation_delta",
                                        "text": text,
                                        "source_samples": samples,
                                        "model_seconds": int(elapsed) / 1e6,
                                    }
                            elif kind == "finished":
                                finished = True
                            elif kind == "resident_wait":
                                resident = True
                if self.process is not None and self.process.returncode is not None and not resident:
                    raise RuntimeError("Tencent worker exited before its completion marker")
                if self.native_url and not resident and time.monotonic() - checked >= 1:
                    status = await self.native_client.get(self.native_url + "/status")
                    status.raise_for_status()
                    state = status.json()
                    if state["returncode"] is not None:
                        raise RuntimeError("Native entry ended before completion: " + str(state))
                    checked = time.monotonic()
                await asyncio.sleep(0.01)
            if not self.feeder.done():
                await asyncio.wait_for(asyncio.shield(self.feeder), 2)
            self.feeder.result()
            tail = decoder.decode(b"", final=True)
            if tail:
                yield {"type": "translation_delta", "text": tail}
            if not finished or self.consumed_samples != audio.samples:
                raise RuntimeError("Tencent did not consume the complete original audio")
        except BaseException:
            await self.close()
            raise
        finally:
            if reader is not None:
                reader.close()

    async def close(self):
        if self.feeder is not None and not self.feeder.done():
            self.feeder.cancel()
        if self.feeder is not None:
            await asyncio.gather(self.feeder, return_exceptions=True)
        if self.fifo is not None:
            os.close(self.fifo)
            self.fifo = None
        release = getattr(self, "release", None)
        if release is not None:
            release.touch()
        if self.native_client is not None:
            try:
                if self.native_started:
                    response = await self.native_client.post(
                        self.native_url + "/close", json={"directory": str(self.directory)}
                    )
                    response.raise_for_status()
                    self.native_started = False
            finally:
                await self.native_client.aclose()
                self.native_client = None
        if self.process is not None:
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(self.process.wait(), 5)
                except asyncio.TimeoutError:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(self.process.wait(), 5)
                    except asyncio.TimeoutError:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        await self.process.wait()
        for log in self.logs:
            log.close()
        self.logs.clear()
