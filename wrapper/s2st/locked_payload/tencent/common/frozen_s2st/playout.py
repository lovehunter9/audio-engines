import asyncio
import json
from pathlib import Path
import shutil
import time
import wave


class OnlinePlayout:
    def __init__(
        self,
        directory,
        base_speed,
        rate=24000,
        prebuffer=1.0,
        target_queue=4.0,
        min_speed=1.0,
        publish=None,
    ):
        self.publish = publish
        self.directory = Path(directory)
        self.rate = rate
        self.base_speed = base_speed
        if not 0 < min_speed <= base_speed or prebuffer < 0 or target_queue <= 0:
            raise ValueError("Invalid playout policy")
        self.prebuffer = prebuffer
        self.target_queue = target_queue
        self.min_speed = min_speed
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            from imageio_ffmpeg import get_ffmpeg_exe

            self.ffmpeg = get_ffmpeg_exe()
        self.pending = {}
        self.end = None
        self.first = None
        self.rows = []
        self.bytes_in = 0
        self.bytes_out = 0
        self.writer = wave.open(str(self.directory / "playout.wav"), "wb")
        self.writer.setnchannels(1)
        self.writer.setsampwidth(2)
        self.writer.setframerate(rate)
        self.log = (self.directory / "playout-events.jsonl").open("w")

    def audio(self, seq, pcm):
        self.pending.setdefault(seq, bytearray()).extend(pcm)
        self.bytes_in += len(pcm)

    async def complete(self, seq, epoch):
        pcm = self.pending.pop(seq)
        now = time.monotonic()
        queued = self.prebuffer if self.end is None else max(0.0, self.end - now)
        speed = min(self.base_speed, max(self.min_speed, 1.1 + 0.05 * (queued - self.target_queue)))
        identity = abs(speed - self.base_speed) < 1e-9
        if identity:
            output = bytes(pcm)
        else:
            process = await asyncio.create_subprocess_exec(
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-f",
                "s16le",
                "-ar",
                str(self.rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-af",
                f"atempo={speed / self.base_speed:.10f}",
                "-f",
                "s16le",
                "-ar",
                str(self.rate),
                "-ac",
                "1",
                "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                output, error = await asyncio.wait_for(process.communicate(bytes(pcm)), 15)
            except BaseException:
                process.kill()
                await process.wait()
                raise
            if process.returncode or not output or len(output) % 2:
                raise RuntimeError(
                    "Online pitch-preserving transform failed: "
                    + error.decode(errors="replace")[:300]
                )
        ready = time.monotonic()
        initial = self.end is None
        if initial:
            self.end = ready + self.prebuffer
            self.first = self.end
        start = max(self.end, ready)
        gap = start - self.end
        self.writer.writeframes(b"\0" * (round(gap * self.rate) * 2))
        self.writer.writeframes(output)
        duration = len(output) / (2 * self.rate)
        self.end = start + duration
        self.bytes_out += len(output)
        row = dict(
            seq=seq,
            speed=speed,
            transform_seconds=ready - now,
            ready_seconds=ready - epoch,
            start_seconds=start - epoch,
            duration=duration,
            gap_seconds=gap,
            input_bytes=len(pcm),
            output_bytes=len(output),
            identity_passthrough=identity,
        )
        self.rows.append(row)
        if self.publish is not None:
            await self.publish(output, row)
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()

    def finish(self, epoch, source_seconds, expected_bytes):
        self.writer.close()
        self.log.close()
        if self.first is None:
            return dict(
                complete=False,
                scope="Online transform failed before first scheduled phrase",
                input_bytes=self.bytes_in,
                expected_bytes=expected_bytes,
            )
        assert not self.pending and self.bytes_in == expected_bytes
        return dict(
            scope="Live input and online transform; FIFO playout schedule, not physical audio-device measurement",
            complete=True,
            first_audio_seconds=self.first - epoch,
            playback_tail_seconds=self.end - epoch - source_seconds,
            max_gap_seconds=max(row["gap_seconds"] for row in self.rows),
            total_gap_seconds=sum(row["gap_seconds"] for row in self.rows),
            transform_seconds=sum(row["transform_seconds"] for row in self.rows),
            phrases=len(self.rows),
            input_bytes=self.bytes_in,
            output_bytes=self.bytes_out,
            min_speed=min(row["speed"] for row in self.rows),
            max_speed=max(row["speed"] for row in self.rows),
        )
