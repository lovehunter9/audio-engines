# Intel streaming: the talker hands over each frame's codes as it makes them and the codec decodes a sliding window.
import functools
import queue
import threading
import time

import numpy as np

# The codec decoder is causal but its sliding attention reaches 72 frames back. With 25 frames of left
# context a window drifts several dB from the whole-sentence decode; filling the rest of the bucket the
# window is padded to anyway brings it within a fraction of a dB at no extra cost.
CONTEXT_FLOOR = 25


class _Stopped(Exception):
    """The reader hung up; raised inside the talker loop to end generation at the next frame."""


class _Generated(Exception):
    """model.generate returned. Raised so the pipeline skips its whole-sentence decode."""


class FrameTap:
    """Routes the frames of the one generation that runs under _gen_lock to its stream."""

    def __init__(self):
        self.sink = None


def install(ov):
    """Wrap the talker's forward and the model's generate once per loaded checkpoint."""
    model = ov.model
    tap = getattr(model, "_olares_frame_tap", None)
    if tap is not None:
        return tap
    tap = FrameTap()
    talker = model.talker
    talker_forward = talker.forward
    model_generate = model.generate

    @functools.wraps(talker_forward)
    def forward(*args, **kwargs):
        out = talker_forward(*args, **kwargs)
        sink = tap.sink
        if sink is not None:
            codes = out.hidden_states[1]
            if codes is not None:
                sink.frame(codes)
        return out

    @functools.wraps(model_generate)
    def generate(*args, **kwargs):
        sink = tap.sink
        if sink is not None:
            prompt = kwargs.get("voice_clone_prompt") or {}
            ref = (prompt.get("ref_code") or [None])[0]
            if ref is not None:
                sink.context(ref)
        result = model_generate(*args, **kwargs)
        if sink is not None:
            raise _Generated()
        return result

    talker.forward = forward
    model.generate = generate
    model._olares_frame_tap = tap
    return tap


def _rows(codes):
    if hasattr(codes, "detach"):
        codes = codes.detach().cpu().numpy()
    return np.asarray(codes, dtype=np.int64)


class _Sink:
    def __init__(self):
        self.items = queue.Queue()
        self.stop = threading.Event()

    def frame(self, codes):
        if self.stop.is_set():
            raise _Stopped()
        self.items.put(("frame", _rows(codes).reshape(-1)))

    def context(self, codes):
        self.items.put(("context", list(_rows(codes))))


def window(history, done, bucket):
    """The codes to decode for frames done.. and how many leading frames are only context."""
    new = len(history) - done
    span = bucket(new + CONTEXT_FLOOR)
    start = max(0, len(history) - span)
    return np.stack(history[start:], axis=1)[None], done - start


def stream(generate, decode, upsample, chunk_size, tap, bucket, log=None):
    """Run generate on a worker thread and yield each chunk_size frames of audio as they land.

    Closing the generator stops the talker at its next frame and waits for the worker, so the
    caller's _gen_lock is free again when close() returns.
    """
    chunk = max(1, int(chunk_size or 1))
    sink = _Sink()

    def work():
        try:
            generate()
            sink.items.put(("end", None))
        except (_Generated, _Stopped):
            sink.items.put(("end", None))
        except BaseException as exc:  # noqa: BLE001 - handed to the reader
            sink.items.put(("error", exc))
        finally:
            tap.sink = None

    tap.sink = sink
    worker = threading.Thread(target=work, name="tts-ov-stream", daemon=True)
    started = time.perf_counter()
    first = None
    chunks = 0
    history = []
    done = 0
    worker.start()
    try:
        while True:
            kind, value = sink.items.get()
            if kind == "context":
                history.extend(value)
                done = len(history)
                continue
            if kind == "error":
                raise value
            if kind == "frame":
                history.append(value)
                if len(history) - done < chunk:
                    continue
            elif len(history) == done:
                break
            codes, lead = window(history, done, bucket)
            audio = decode(codes)[lead * upsample:(codes.shape[-1]) * upsample]
            done = len(history)
            chunks += 1
            if first is None:
                first = time.perf_counter() - started
            yield audio
            if kind == "end":
                break
    finally:
        sink.stop.set()
        worker.join()
        if log is not None and first is not None:
            log.info("streamed %d chunks, first %.2fs, total %.2fs",
                     chunks, first, time.perf_counter() - started)
