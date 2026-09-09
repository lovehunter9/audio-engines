# How much audio one request may bring, for the caps that hold a whole clip at once.
#
# Every cap here decodes the upload in full before it does anything with it, so the memory a
# request costs is set by the caller, not by the engine. Unbounded, a long enough upload gets the
# container OOM-killed, and the caller sees a dropped connection: no status, no message, and
# nothing to distinguish it from a crash. A 413 says the same thing in a form a client can act on,
# and it says it before the memory is spent.
#
# Two bounds, because they catch different mistakes. Bytes catch an upload that is too big to keep
# on disk at all, and are checked while it arrives. Seconds catch a small file that decodes into
# hours -- a compressed container is a fraction of its PCM -- and are checked off the header, so
# the refusal still costs nothing.
import asyncio
import logging
import os
import subprocess

from fastapi import HTTPException

from .audioio import probe_seconds, spill_upload, unlink

log = logging.getLogger("audio-limits")

MIB = 1 << 20


def ffprobe_seconds(path):
    """Duration through ffmpeg's own demuxers, or None when they cannot measure it either."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return round(float(out.stdout.strip()), 3)
        log.info("ffprobe could not measure %s: %s", path, (out.stderr or "").strip()[:200])
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        log.info("ffprobe unusable (%s); duration stays unknown", e)
    return None


def duration(path):
    """Seconds, preferring the header readers and falling back to ffprobe.

    The two disagree by construction, and that gap is the reason for the fallback: probe_seconds
    reads through soundfile and the stdlib wave module, while the engines decode through ffmpeg.
    A container the first pair cannot parse may be one an engine reads happily, and that is
    exactly the file that would slip past the length check.

    None survives as an answer. ffprobe may be absent from an image, and a file neither reader can
    measure is one the engine almost certainly cannot decode either -- it fails on its own, before
    any memory is spent.
    """
    seconds = probe_seconds(path)
    if seconds is not None:
        return seconds
    return ffprobe_seconds(path)


class Bounds:
    """One cap's upload limits, claimed from ENGINE_ARGS so a deployment can move them.

    A deployment that knows its own memory ceiling raises or lowers these; the defaults are
    picked per cap from what that engine does with a clip, and are documented where they are
    passed in.
    """

    def __init__(self, args, seconds, megabytes=1024):
        self.seconds = args.number("--max-audio-seconds", seconds)
        self.megabytes = args.number("--max-upload-mb", megabytes)

    async def spill(self, file, what="this engine holds the whole clip in memory"):
        """The upload on disk within these bounds, as (path, seconds).

        Raises 413 rather than returning, because there is nothing for the caller to do with a
        refusal except see it. `seconds` may be None: a container nothing here can measure is
        passed through rather than refused -- an engine may still read it -- and then bills as
        unmeasured, which is why the miss is logged.
        """
        path, size, over = await spill_upload(file, max_bytes=int(self.megabytes * MIB))
        if over:
            raise HTTPException(
                status_code=413,
                detail="upload is over %.0f MiB; this engine accepts at most that"
                       % self.megabytes)
        # Header reads and ffprobe are both blocking, and this runs on the event loop.
        seconds = await asyncio.to_thread(duration, path)
        if seconds is not None and seconds > self.seconds:
            unlink(path)
            raise HTTPException(
                status_code=413,
                detail="audio is %.0fs; %s and accepts at most %.0fs"
                       % (seconds, what, self.seconds))
        if seconds is None:
            log.warning("could not read the duration of %s (%d bytes): it is neither billed "
                        "nor length-checked", os.path.basename(path), size)
        return path, seconds
