# TTS and zero-shot cloning on audio.cpp: the model runs in a child engine, this is the translation.
#
# What the engine wants and what our contract promises differ in four places, and each one is why
# this file exists rather than a proxy:
#   * reference audio: it reads a server-side path, our callers send a data: URL or an upload;
#   * response formats: it returns WAV (PCM while streaming), we promise wav/flac/mp3/opus/pcm;
#   * `voice`: an unmatched name is read as a cached voice id and rejected deep in the model, so a
#     family without built-in voices must refuse it up front instead of 500ing;
#   * `instructions`: on a text_prefix family the field is silently dropped, so it is folded into
#     the text — a caller asking for a voice would otherwise get the default one with a 200.
#
# Which routes exist is decided by what the engine advertises for this family, not by this file:
# streaming is mounted only where the loader has a streaming mode, cloning only where it clones.
import base64
import io
import json
import logging
import os
import struct
import threading

from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import StreamingResponse

from .. import acpp
from .. import hfgate
from .. import tasks
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-acpp-tts")

_runtime = Runtime("OpenBMB/VoxCPM2", engine=None, spec=None, sample_rate=0, channels=1)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
# Only needed when several specs in the image claim the same weight filenames; normally matched.
FAMILY = str(_args.text("--family", "") or "")
# "" lets the model detect the language; a caller may still pass `language` per request.
DEFAULT_LANGUAGE = str(_args.text("--language", "") or "")
# Weight load plus the first synthesis, which is where CUDA graphs get built.
BOOT_TIMEOUT_S = 2400.0

# Audio out of the child is always WAV, or raw PCM while streaming; the rest is ours to encode.
_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),  # raw little-endian int16, no container
}
# The engine decodes reference audio by extension, so an upload keeps its own.
_REF_SUFFIX = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
               "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
               "audio/x-flac": ".flac", "audio/ogg": ".ogg", "audio/opus": ".opus",
               "audio/aac": ".aac", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
               "audio/webm": ".webm", "video/mp4": ".mp4", "video/webm": ".webm",
               "video/quicktime": ".mov", "video/x-matroska": ".mkv"}

# Sampling knobs: ours on the left, the engine's spelling on the right.
_GEN_KEYS = (("temperature", "temperature", float), ("top_p", "top_p", float),
             ("top_k", "top_k", int), ("repetition_penalty", "repetition_penalty", float),
             ("guidance_scale", "guidance_scale", float),
             ("num_inference_steps", "num_inference_steps", int), ("seed", "seed", int),
             ("max_new_tokens", "max_tokens", int), ("max_tokens", "max_tokens", int))

_state = _runtime.state
# Reference clips outlive one call only while a task runs, so they live under the engine's reach.
_REF_DIR = "/tmp/acpp/refs"


def _capabilities():
    """The model's capability facts, without loading anything.

    Resolved before build_app so the route table and /api/engine-spec agree from the first
    request: llm-init reads the contract while weights are still downloading, and a spec that
    grows routes later would have lied to it.
    """
    weights, spec = acpp.resolve_weights(MODEL_REPO, family=FAMILY, token=HF_TOKEN)
    log.info("%s: weights=%s %s", MODEL_NAME, weights, spec)
    if spec.api_endpoints:
        log.info("%s advertises %s", spec.family, ", ".join(spec.api_endpoints))
    if spec.can_design and not spec.instructions_policy:
        # Worth a warning because the failure is silent: a text_prefix family drops the field and
        # answers 200 in the default voice, so the caller never learns its request was ignored.
        log.warning("%s does voice design, but the loader's instructions policy could not be read "
                    "(audiocpp_cli --list-loaders gave nothing); `instructions` will be sent as a "
                    "field, which some families ignore without saying so", spec.family)
    return weights, spec


def _boot():
    """Start the child engine and prove it can synthesize before anyone is told we are ready."""
    try:
        weights, spec = _state.get("weights"), _state.get("spec")
        if spec is None:
            weights, spec = _capabilities()
            _state.update(weights=weights, spec=spec)
        engine = acpp.Engine(spec=spec, weights_dir=weights, args=_args)
        engine.start()
        _state["engine"] = engine
        _warmup(engine, spec)
        _state.update(ready=True, error=None)
        log.info("%s ready: family=%s mode=%s sample_rate=%d clone=%s stream=%s", MODEL_NAME,
                 spec.family, spec.run_mode, _state["sample_rate"], spec.can_clone, spec.streams)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("audio.cpp TTS failed to start: %s", e)


def _warmup(engine, spec):
    """One real synthesis before readiness: it builds the CUDA graphs the first caller would
    otherwise pay for, and it is the only place the output sample rate can be learned.

    Failing here fails the boot on purpose. A TTS engine that cannot say one word is not ready,
    and finding that out now beats every caller finding out with a 500.
    """
    body = {"model": acpp.MODEL_ID, "input": "Warm up.", "response_format": "wav"}
    wav = engine.post("/v1/audio/speech", body).content
    rate, channels = _wav_facts(wav)
    _state.update(sample_rate=rate, channels=channels)
    log.info("warmup produced %d bytes of %d Hz audio", len(wav), rate)


def _wav_facts(data):
    """(sample_rate, channels) off a RIFF header. The engine's real rate, not one we assumed."""
    if len(data) < 36 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise acpp.EngineError("the engine did not return WAV audio", status=502)
    # Walk the chunks: some encoders put LIST before fmt, so a fixed offset is not safe.
    pos = 12
    while pos + 8 <= len(data):
        name = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if name == b"fmt " and pos + 8 + 16 <= len(data):
            channels = struct.unpack("<H", data[pos + 10:pos + 12])[0]
            rate = struct.unpack("<I", data[pos + 12:pos + 16])[0]
            return int(rate), int(channels)
        pos += 8 + size + (size & 1)
    raise acpp.EngineError("the engine's WAV has no fmt chunk", status=502)


def _pcm_of(wav):
    """The samples inside a RIFF container, so a pcm/stream caller gets no header."""
    pos = 12
    while pos + 8 <= len(wav):
        name = wav[pos:pos + 4]
        size = struct.unpack("<I", wav[pos + 4:pos + 8])[0]
        if name == b"data":
            return wav[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
    return b""


def _wav_stream_header(rate, channels=1, bits=16):
    """A RIFF header with unknown length: the sizes are only known once generation ends."""
    byte_rate = rate * channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate,
                          channels * bits // 8, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF - 36))


def _transcode(wav, fmt):
    """WAV from the engine into the format the caller asked for."""
    if fmt == "wav":
        return wav
    if fmt == "pcm":
        return _pcm_of(wav)
    import soundfile as sf

    container, subtype, _mime = _FORMATS[fmt]
    try:
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
        buf = io.BytesIO()
        sf.write(buf, audio, rate, format=container, subtype=subtype)
        return buf.getvalue()
    except Exception as e:
        # libsndfile writes mp3 only from 1.1 on, and the base image's version is upstream's
        # choice; ffmpeg is in the image for exactly this kind of gap.
        log.info("soundfile could not write %s (%s); using ffmpeg", fmt, e)
        return _ffmpeg(wav, fmt)


def _ffmpeg(wav, fmt):
    import subprocess

    codec = {"mp3": ("-f", "mp3"), "opus": ("-f", "ogg", "-c:a", "libopus"),
             "flac": ("-f", "flac")}.get(fmt)
    if codec is None:
        raise HTTPException(status_code=400, detail="cannot encode %s here" % fmt)
    proc = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
                           *codec, "pipe:1"], input=wav, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(status_code=500, detail="encoding to %s failed: %s"
                            % (fmt, (proc.stderr or b"").decode("utf-8", "replace")[:200]))
    return proc.stdout


class _ref_file:
    """The engine reads reference audio from a path, so a data: URL lands on disk for the call."""

    def __init__(self, data, suffix=".wav"):
        self._data, self._suffix, self._path = data, suffix, None

    def __enter__(self):
        import tempfile

        os.makedirs(_REF_DIR, exist_ok=True)
        fd, self._path = tempfile.mkstemp(suffix=self._suffix, prefix="ref-", dir=_REF_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(self._data)
        return self._path

    def __exit__(self, *exc):
        try:
            os.unlink(self._path)
        except OSError:
            pass
        return False


def build_app(supports):
    app = FastAPI(title="audio-audiocpp (TTS)")
    mount_metrics(app)

    # Resolved here, not in the load thread: the routes below and the self-report both depend on
    # it, and both have to be right before the first request.
    spec, weights, spec_error = None, None, None
    try:
        weights, spec = _capabilities()
        _state.update(weights=weights, spec=spec)
    except Exception as e:
        spec_error = str(e)
        _state["error"] = spec_error
        log.error("cannot resolve what this model can do: %s", e)

    # Nothing is warned about here: the child engine takes every flag no cap claimed, in its own
    # spelling. Logged so a typo is visible as "audio.cpp rejected --foo" rather than silence.
    leftovers = _args.passthrough()
    if leftovers:
        log.info("forwarding %d unclaimed flag(s) to audio.cpp: %s",
                 len(leftovers), " ".join(leftovers))

    has_tts = "tts" in supports
    has_clone = "tts_clone" in supports and (spec is None or spec.can_clone)
    # Advertised only where the loader really streams: MOSS-TTS-Nano has no streaming decode, and
    # a socket that answers "not supported" on every frame is worse than no socket.
    streams = bool(spec and spec.streams)
    speech_mode = "tts" if has_tts else "tts_clone"

    # What this instance does not mount, so /api/engine-spec can say so instead of overpromising.
    withheld = []
    if not streams:
        why = ("this model has no streaming mode" if spec
               else "capabilities unknown: %s" % spec_error)
        withheld.append(("WS", "/v1/audio/speech/stream", why))
    if "tts_clone" in supports and not has_clone:
        withheld.append(("POST", "/v1/audio/speech/clone",
                         "this model does not do reference-audio cloning"))

    register(app, model_name=MODEL_NAME, module="audiocpp_tts", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=lambda: _state["sample_rate"] or None,
             # The family of the weights that actually loaded, not one this module picked.
             family=lambda: _state["spec"].family if _state["spec"] else None,
             withheld=withheld)

    def _engine():
        if not _state["ready"] or _state["engine"] is None:
            raise HTTPException(status_code=503,
                                detail=_state["error"] or "engine not ready")
        return _state["engine"]

    def _headers(mode, fmt):
        # Channels is not cosmetic: a pcm caller gets no container to read it from, and these
        # models are not all mono (MOSS-TTS-Nano returns stereo at 48 kHz).
        return {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": mode, "X-Audio-Format": fmt,
                "X-Audio-Sample-Rate": str(_state["sample_rate"]),
                "X-Audio-Channels": str(_state["channels"])}

    def _check(payload):
        """Validate our body shape and settle the response format. Returns the format."""
        if not str(payload.get("input") or "").strip():
            raise HTTPException(status_code=400, detail="input (the text to speak) is required")
        fmt = str(payload.setdefault("response_format", "wav")).strip().lower()
        payload["response_format"] = fmt
        if fmt not in _FORMATS:
            raise HTTPException(status_code=400, detail="response_format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        if payload.get("voice"):
            # Refused rather than dropped: the engine would take the name for a cached voice id
            # and fail inside the model, and there is no voice list to check it against.
            raise HTTPException(
                status_code=400,
                detail="%s has no built-in voices, so `voice` cannot be honoured; clone one "
                       "instead with ref_audio (a data: URL) or POST /v1/audio/speech/clone"
                       % MODEL_NAME)
        return fmt

    def _ref_of(payload):
        """(bytes, suffix) for the reference clip, or (None, None) for the model's own voice."""
        ref = payload.get("ref_audio")
        if not ref:
            return None, None
        if not has_clone:
            raise HTTPException(status_code=400,
                                detail="%s does not clone from reference audio" % MODEL_NAME)
        if not isinstance(ref, str) or not ref.startswith("data:"):
            raise HTTPException(status_code=400,
                                detail="ref_audio must be a data: URL (base64 reference audio)")
        head, _, b64 = ref.partition(",")
        mime = head[5:].split(";")[0] or "audio/wav"
        try:
            return base64.b64decode(b64), _REF_SUFFIX.get(mime, ".bin")
        except Exception:
            raise HTTPException(status_code=400, detail="ref_audio is not valid base64")

    def _engine_body(payload, ref_path, stream=False):
        """Our OpenAI-shaped body as the engine's own."""
        text = str(payload["input"])
        instructions = str(payload.get("instructions") or "").strip()
        body = {"model": acpp.MODEL_ID}
        if instructions and spec is not None and spec.prefixes_instructions:
            # The family's policy is text_prefix: sent as a field it would be dropped in silence.
            text = "(%s)%s" % (instructions.rstrip("()"), text)
        elif instructions:
            body["instructions"] = instructions
        body["input"] = text
        language = str(payload.get("language") or DEFAULT_LANGUAGE).strip()
        if language:
            body["language"] = language
        for ours, theirs, cast in _GEN_KEYS:
            if payload.get(ours) is None:
                continue
            try:
                body[theirs] = cast(payload[ours])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="%s must be a number" % ours)
        if ref_path:
            body["voice_ref"] = ref_path
            ref_text = str(payload.get("ref_text") or "").strip()
            if ref_text:
                body["reference_text"] = ref_text
        if stream:
            # Raw PCM rather than SSE: the bytes are relayed as they arrive, with no base64 hop.
            body.update(stream=True, stream_format="audio", response_format="pcm")
        else:
            body["response_format"] = "wav"
        return body

    def _synthesize_blocking(payload, ref_path):
        """One synthesis. Returns the engine's WAV bytes."""
        engine = _engine()
        try:
            return engine.post("/v1/audio/speech", _engine_body(payload, ref_path)).content
        except acpp.EngineError as e:
            # The engine answers most caller mistakes with a 500; a bad request is still a 400.
            status = e.status if e.status in (400, 413, 429, 503) else 502
            raise HTTPException(status_code=status, detail=str(e))

    async def _stream_response(payload, mode, fmt, ref):
        if fmt not in ("pcm", "wav"):
            raise HTTPException(status_code=400,
                                detail="stream supports response_format pcm or wav; "
                                       "%s only exists as a whole file" % fmt)
        if not streams:
            raise HTTPException(status_code=400,
                                detail="%s has no streaming mode; drop stream to synthesize the "
                                       "whole clip" % MODEL_NAME)
        engine = _engine()
        data, suffix = ref
        rate = _state["sample_rate"]

        def body():
            if fmt == "wav":
                yield _wav_stream_header(rate, _state["channels"])
            ctx = _ref_file(data, suffix) if data else None
            path = ctx.__enter__() if ctx else None
            try:
                with engine.stream("/v1/audio/speech",
                                   _engine_body(payload, path, stream=True)) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise HTTPException(status_code=502,
                                            detail=acpp._engine_message(response))
                    for chunk in response.iter_bytes():
                        if chunk:
                            yield chunk
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)

        return StreamingResponse(body(), media_type=_FORMATS[fmt][2], headers=_headers(mode, fmt))

    async def _synthesize(payload, async_, mode):
        fmt = _check(payload)
        _engine()
        ref = _ref_of(payload)
        if payload.get("stream"):
            return await _stream_response(payload, mode, fmt, ref)
        data, suffix = ref

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="synthesis")
            if data:
                with _ref_file(data, suffix) as path:
                    wav = _synthesize_blocking(payload, path)
            else:
                wav = _synthesize_blocking(payload, None)
            body = _transcode(wav, fmt)
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(body, _FORMATS[fmt][2], suffix="." + fmt,
                                headers=_headers(mode, fmt))

        return await tasks.dispatch(async_, mode, MODEL_NAME, _work, fail="speech synthesis failed")

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON in the OpenAI /v1/audio/speech shape")
        # async travels in the query string here, so the body stays exactly OpenAI's.
        return await _synthesize(payload, request.query_params.get("async"), speech_mode)

    @app.post("/v1/audio/speech/batch")
    async def speech_batch(request: Request):
        # One engine, one queue: items run back to back, the win is one dispatch, not parallelism.
        _engine()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON for /v1/audio/speech/batch")
        items = payload.get("items")
        if not isinstance(items, list) or not items or len(items) > 32:
            raise HTTPException(status_code=400, detail="items must be a non-empty array (1–32)")
        shared = {k: v for k, v in payload.items() if k != "items"}
        rows = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="items[%d] must be an object" % i)
            row = dict(shared)
            row.update(item)
            row.pop("stream", None)
            _check(row)
            # Settled before the task exists: an unusable reference is a bad request like any other.
            rows.append((row, _ref_of(row)))
        async_ = request.query_params.get("async")

        def _work(ctx):
            n = len(rows)
            ctx.progress(ratio=0.0, stage="batch", done=0, total=n)
            out = []
            for i, (row, (data, suffix)) in enumerate(rows):
                ctx.checkpoint()
                if data:
                    with _ref_file(data, suffix) as path:
                        wav = _synthesize_blocking(row, path)
                else:
                    wav = _synthesize_blocking(row, None)
                fmt = row["response_format"]
                out.append({"index": i, "format": fmt, "sample_rate": _state["sample_rate"],
                            "audio": base64.b64encode(_transcode(wav, fmt)).decode("ascii")})
                ctx.progress(ratio=(i + 1) / n, stage="batch", done=i + 1, total=n)
            return {"model": MODEL_NAME, "items": out}

        return await tasks.dispatch(async_, speech_mode, MODEL_NAME, _work,
                                    fail="speech batch failed")

    if streams:
        @app.websocket("/v1/audio/speech/stream")
        async def speech_stream(ws: WebSocket):
            """Incremental text in, audio out — the one thing POST /v1/audio/speech cannot do.

            A caller relaying an LLM's tokens does not have the sentence yet when it wants audio
            started. Text arrives in `input.text` frames and is synthesized a sentence at a time;
            `input.done` flushes whatever is left.
            """
            import asyncio

            await ws.accept()
            if not _state["ready"]:
                await ws.send_text(json.dumps({"type": "error",
                                               "message": _state["error"] or "engine not ready"}))
                await ws.close()
                return
            cfg = {"response_format": "pcm", "language": DEFAULT_LANGUAGE, "ref_audio": None}
            pending = ""
            work = asyncio.Queue(maxsize=256)
            FLUSHED = object()

            async def flush(text):
                text = text.strip()
                if not text:
                    return
                payload = dict(cfg)
                payload["input"] = text
                data, suffix = _ref_of(payload)
                ctx = _ref_file(data, suffix) if data else None
                path = ctx.__enter__() if ctx else None
                try:
                    body = _engine_body(payload, path, stream=True)
                    # The child's stream is blocking, so it is drained off the event loop.
                    def pump(queue, loop):
                        try:
                            with _engine().stream("/v1/audio/speech", body) as response:
                                if response.status_code >= 400:
                                    response.read()
                                    raise acpp.EngineError(acpp._engine_message(response))
                                for chunk in response.iter_bytes():
                                    if chunk:
                                        asyncio.run_coroutine_threadsafe(
                                            queue.put(("audio", chunk)), loop).result()
                        except Exception as e:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("error", e)), loop).result()
                        finally:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("done", None)), loop).result()

                    loop = asyncio.get_running_loop()
                    chunks = asyncio.Queue(maxsize=16)
                    threading.Thread(target=pump, args=(chunks, loop), daemon=True).start()
                    while True:
                        kind, value = await chunks.get()
                        if kind == "done":
                            break
                        if kind == "error":
                            raise value
                        await ws.send_bytes(value)
                finally:
                    if ctx:
                        ctx.__exit__(None, None, None)

            def split_sentences(buf):
                """Emit complete sentences, keep the tail. Latency is one sentence, not one turn."""
                out, start = [], 0
                for i, ch in enumerate(buf):
                    if ch in "。！？!?\n" or (ch == "." and i + 1 < len(buf) and buf[i + 1] == " "):
                        out.append(buf[start:i + 1])
                        start = i + 1
                return out, buf[start:]

            async def speaker():
                """Speak queued sentences in order; a failure on one is reported, not fatal."""
                while True:
                    item = await work.get()
                    if item is FLUSHED:
                        await ws.send_text(json.dumps({"type": "session.done"}))
                        continue
                    try:
                        await flush(item)
                    except Exception as e:
                        log.warning("speech/stream sentence failed: %s", e)
                        detail = getattr(e, "detail", None) or str(e)
                        await ws.send_text(json.dumps({"type": "error", "message": detail}))

            worker = asyncio.create_task(speaker())
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    raw = msg.get("text")
                    if raw is None:
                        continue
                    try:
                        frame = json.loads(raw)
                    except Exception:
                        await ws.send_text(json.dumps({"type": "error",
                                                       "message": "frames must be JSON"}))
                        continue
                    kind = frame.get("type")
                    if kind == "session.config":
                        for key in ("response_format", "language", "ref_audio", "ref_text",
                                    "instructions", "temperature", "top_p", "top_k", "seed"):
                            if frame.get(key) is not None:
                                cfg[key] = frame[key]
                        if frame.get("voice"):
                            await ws.send_text(json.dumps(
                                {"type": "error",
                                 "message": "%s has no built-in voices; send ref_audio to clone"
                                            % MODEL_NAME}))
                        if str(cfg["response_format"]).lower() != "pcm":
                            await ws.send_text(json.dumps(
                                {"type": "error",
                                 "message": "the socket carries raw pcm frames; use POST "
                                            "/v1/audio/speech for a container format"}))
                            cfg["response_format"] = "pcm"
                        await ws.send_text(json.dumps(
                            {"type": "session.ready", "model": MODEL_NAME,
                             "sample_rate": _state["sample_rate"], "format": "pcm"}))
                    elif kind == "input.text":
                        pending += str(frame.get("text") or "")
                        ready, pending = split_sentences(pending)
                        for sentence in ready:
                            await work.put(sentence)
                    elif kind == "input.done":
                        await work.put(pending)
                        pending = ""
                        await work.put(FLUSHED)
                    elif kind == "session.close":
                        await ws.close()
                        return
                    else:
                        await ws.send_text(json.dumps({"type": "error",
                                                       "message": "unknown frame type %r" % kind}))
            except WebSocketDisconnect:
                pass
            except Exception as e:
                log.exception("speech/stream failed: %s", e)
                try:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
                    await ws.close()
                except Exception:
                    pass
            finally:
                worker.cancel()

    if has_clone:
        @app.post("/v1/audio/speech/clone")
        async def clone(file: UploadFile = File(...),
                        text: str = Form(..., alias="input"),
                        ref_text: str = Form(default=""),
                        language: str = Form(default=""),
                        instructions: str = Form(default=""),
                        fmt: str = Form(default="wav", alias="response_format"),
                        async_: str = Form(default=None, alias="async")):
            _engine()
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="file (the reference audio) is empty")
            mime = file.content_type or "audio/wav"
            payload = {"input": text, "response_format": fmt,
                       "ref_audio": "data:%s;base64,%s"
                                    % (mime, base64.b64encode(data).decode("ascii"))}
            # A missing optional means "unset"; sending "" would override the engine's default.
            if ref_text:
                payload["ref_text"] = ref_text
            if language:
                payload["language"] = language
            if instructions:
                payload["instructions"] = instructions
            return await _synthesize(payload, async_, "tts_clone")

    return app


def run(supports):
    # Ready means the child engine loaded AND synthesized once, so the deadline covers both.
    _runtime.serve(supports, _boot, build_app, "audio.cpp engine", timeout_s=BOOT_TIMEOUT_S)
