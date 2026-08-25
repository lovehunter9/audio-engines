# Offline STT and streaming STT on audio.cpp: the model runs in a child engine, this translates.
#
# What the engine wants and what our contract promises differ in three places:
#   * OpenAI clients upload multipart; the engine's JSON path wants a server-side WAV path, and
#     its own multipart only accepts WAV — so anything else is transcoded here;
#   * there is no native batch route, so `segments` on the same POST is how batch lands, same
#     as qwen; a `/transcriptions/batch` would be a path the engine does not have;
#   * live capture is chunked PCM + SSE on `/v1/audio/transcriptions/live`. DEMO/gateway speak
#     `WS /v1/audio/stream`, so that socket is a translation onto /live, not a third protocol.
#
# Which streaming routes exist is decided by the loader, not this file: a family without a
# streaming mode withholds both /live and the socket rather than answering "not supported"
# on every frame.
import array
import asyncio
import io
import json
import logging
import math
import os
import queue
import struct
import threading
import time
import wave

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from .. import acpp
from .. import hfgate
from .. import tasks
from ..audioio import probe_seconds
from ..batch import parse_segments
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-acpp-stt")

_runtime = Runtime("mistralai/Voxtral-Mini-4B-Realtime-2602", engine=None, spec=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
FAMILY = str(_args.text("--family", "") or "")
BOOT_TIMEOUT_S = 2400.0

_state = _runtime.state
_AUDIO_DIR = "/tmp/acpp/audio"
_infer_lock = threading.Lock()
_stream_busy = threading.Event()
_keep_stop = threading.Event()

# Form/JSON fields that belong to the engine's request.options, not to our wiring.
_OPTION_KEYS = ("audio_chunk_mode", "audio_chunk_duration_sec")


def _flag(v):
    if isinstance(v, bool):
        return v
    return tasks.truthy(v)


def _capabilities():
    """The model's capability facts, without loading anything.

    Resolved before build_app so the route table and /api/engine-spec agree from the first
    request: llm-init reads the contract while weights are still downloading.
    """
    weights, spec = acpp.resolve_weights(MODEL_REPO, family=FAMILY, token=HF_TOKEN)
    log.info("%s: weights=%s %s", MODEL_NAME, weights, spec)
    if spec.api_endpoints:
        log.info("%s advertises %s", spec.family, ", ".join(spec.api_endpoints))
    return weights, spec


def _boot():
    try:
        weights, spec = _state.get("weights"), _state.get("spec")
        if spec is None:
            weights, spec = _capabilities()
            _state.update(weights=weights, spec=spec)
        engine = acpp.Engine(spec=spec, weights_dir=weights, args=_args)
        engine.start()
        _state["engine"] = engine
        try:
            _warmup(engine)
        except Exception as e:
            log.warning("boot warmup failed: %s", e)
        _state.update(ready=True, error=None)
        if spec.family == "voxtral_realtime":
            threading.Thread(target=_keep_gpu, name="acpp-gpu-keep", daemon=True).start()
            log.info("voxtral gpu keep every %.0fs: first CUDA graph after a lock looks idle to HAMi",
                     _KEEP_S)
        log.info("%s ready: family=%s mode=%s stream=%s", MODEL_NAME,
                 spec.family, spec.run_mode, spec.streams)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("audio.cpp STT failed to start: %s", e)


def _warmup_wav(seconds=0.25):
    """A short voiced-band tone: silence can make an encoder skip the pass and warm nothing."""
    buf = io.BytesIO()
    rate = 16000
    n = max(1, int(rate * float(seconds)))
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / rate)))
                          for i in range(n))
        w.writeframes(frames)
    return buf.getvalue()


def _warmup(engine):
    seconds = _DECODE_S if getattr(_state.get("spec"), "family", "") == "voxtral_realtime" else 0.25
    wav = _warmup_wav(seconds)
    with _audio_file(wav) as path:
        body = {"model": acpp.MODEL_ID, "audio": path}
        doc = engine.post("/v1/audio/transcriptions", body).json()
    log.info("warmup transcribed %d bytes -> %r", len(wav), (doc.get("text") or "")[:80])


class _audio_file:
    """The engine's JSON path reads a server-side file, so an upload lands on disk for the call."""

    def __init__(self, data, suffix=".wav"):
        self._data, self._suffix, self._path = data, suffix, None

    def __enter__(self):
        import tempfile

        os.makedirs(_AUDIO_DIR, exist_ok=True)
        fd, self._path = tempfile.mkstemp(suffix=self._suffix, prefix="stt-", dir=_AUDIO_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(self._data)
        return self._path

    def __exit__(self, *exc):
        try:
            os.unlink(self._path)
        except OSError:
            pass
        return False


def _to_wav(raw, filename=""):
    """WAV bytes the engine will accept. Its multipart path only takes WAV; we always send WAV."""
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw
    import subprocess

    suffix = os.path.splitext(filename or "")[1] or ".bin"
    with _audio_file(raw, suffix) as src:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", src,
             "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
            capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(status_code=400, detail="could not decode audio as WAV: %s"
                            % (proc.stderr or b"").decode("utf-8", "replace")[:200])
    return proc.stdout


def _options_of(payload):
    """Engine request.options: pass through what the family actually reads, drop the rest."""
    src = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    out = dict(src)
    for key in _OPTION_KEYS:
        if payload.get(key) is None or payload.get(key) == "":
            continue
        val = payload[key]
        if key == "audio_chunk_duration_sec":
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="%s must be a number" % key)
        else:
            out[key] = val
    return out


def _live_query(cfg):
    """Query string for /live: wiring fields plus whatever the family reads as options."""
    params = {"model": acpp.MODEL_ID, "sample_rate": str(cfg.get("sample_rate") or 16000),
              "channels": "1", "sample_format": "s16le"}
    language = str(cfg.get("language") or "").strip()
    if language:
        params["language"] = language
    for key, val in (cfg.get("options") or {}).items():
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            params[key] = "true" if val else "false"
        else:
            params[key] = str(val)
    return params


# HAMi time-slice releases a process lock after ~5–15 s of 0% GPU util. A long /live
# that only decodes in bursts (Voxtral) looks idle and then never gets the card back.
# The platform WS therefore does offline POSTs,
# but NOT on a metronome: energy VAD cuts on pauses (how people actually talk).
# While a breath is still going we decode the current utterance every ~2 s so the
# GPU stays touched and the DEMO can show a growing interim line. Pause (or the
# 30 s safety cap) finalizes. audio_chunk_duration_sec on start is the cap, not a sentence.
_DECODE_S = 2.0
# Pause-cut only. The cap is a safety lid for a lecture that never breathes,
# not a metronome.
_VAD_MAX_S = 30.0
_VAD_MIN_S = 0.4
_VAD_HANG_S = 0.45
_VAD_FRAME_S = 0.02
_VAD_SPEECH_RMS = 400.0
_HOP_MIN_S = 1.0
_HOP_MAX_S = 30.0
_WS_HOP_S = _VAD_MAX_S
_KEEP_S = 2.0
_LIVE_PUNCT = "。．.！!？?…"
_WINDOW_FAMILIES = ("voxtral_realtime",)


def _window_live(spec):
    return bool(spec) and spec.family in _WINDOW_FAMILIES


def _pcm16_wav(pcm, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate) or 16000)
        w.writeframes(pcm)
    return buf.getvalue()


def _pad_pcm(pcm, rate, seconds):
    need = max(0, int(rate * float(seconds)) * 2)
    if need <= len(pcm):
        return pcm
    return pcm + bytes(need - len(pcm))


def _transcribe_pcm(pcm, cfg):
    payload = {"language": cfg.get("language") or ""}
    payload.update(cfg.get("options") or {})
    rate = cfg.get("sample_rate") or 16000
    if getattr(_state.get("spec"), "family", "") == "voxtral_realtime":
        pcm = _pad_pcm(pcm, rate, _DECODE_S)
    return (_transcribe_blocking(payload, _pcm16_wav(pcm, rate)).get("text") or "")


def _vad_max_s(cfg):
    opts = cfg.get("options") or {} if isinstance(cfg, dict) else {}
    raw = opts.get("audio_chunk_duration_sec")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = 0.0
    if val <= 0:
        val = _VAD_MAX_S
    return min(_HOP_MAX_S, max(_HOP_MIN_S, val))


def _hop_s(cfg):
    """Back-compat alias: start.audio_chunk_duration_sec is the utterance cap."""
    return _vad_max_s(cfg)


def _frame_rms(frame):
    n = len(frame) - (len(frame) % 2)
    if n < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(bytes(frame[:n]))
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _strip_live(text):
    return (text or "").strip().rstrip(_LIVE_PUNCT + " \t")


def _join_asr(parts, family=""):
    """Join committed utterances. Live (in-progress) text is appended by the pump,
    with trailing punct stripped so the DEMO keeps it as the interim line.

    Voxtral auto-detects language per utterance, so each stays its own line.
    """
    pieces = []
    for part in parts:
        piece = (part or "").strip()
        if piece:
            pieces.append(piece)
    if family == "voxtral_realtime":
        return "\n".join(pieces)
    out = ""
    for piece in pieces:
        if out and out[-1].isascii() and piece[0].isascii() and not out[-1].isspace():
            out += " "
        out += piece
    return out


def _with_live(parts, live, family):
    committed = _join_asr(parts, family)
    piece = _strip_live(live)
    if not piece:
        return committed
    if family == "voxtral_realtime":
        return piece if not committed else committed + "\n" + piece
    return committed + piece


def _session_warm(seconds):
    """Dummy transcribe so HAMi sees GPU activity and the CUDA graph stays captured."""
    wav = _warmup_wav(seconds)
    t0 = time.monotonic()
    with _audio_file(wav) as path:
        with _infer_lock:
            doc = _engine().post("/v1/audio/transcriptions",
                                 {"model": acpp.MODEL_ID, "audio": path})
    elapsed_ms = (time.monotonic() - t0) * 1000
    text = ""
    try:
        text = (doc.json().get("text") or "")[:40]
    except Exception:
        pass
    log.info("gpu keep %.2fs audio in %.0fms -> %r", seconds, elapsed_ms, text)


def _keep_gpu():
    """Voxtral's first graph after lock_ok reports 0% util; HAMi steals the lock in ~15 s.
    A 0.25 s dummy does not even run the 4B encoder. Keep a 2 s voiced clip — the same
    length as the first live decode — so the first caption reuses that graph."""
    while True:
        if not _stream_busy.is_set() and _state.get("ready"):
            try:
                _session_warm(_DECODE_S)
            except Exception as e:
                log.warning("gpu keep failed: %s", e)
        if _keep_stop.wait(_KEEP_S):
            return


def _engine_body(path, payload):
    body = {"model": acpp.MODEL_ID, "audio": path}
    language = str(payload.get("language") or "").strip()
    if language:
        body["language"] = language
    opts = _options_of(payload)
    if opts:
        body["options"] = opts
    return body


def _parse_engine(response):
    try:
        if callable(getattr(response, "json", None)):
            return response.json()
        return json.loads(response.content)
    except (ValueError, TypeError):
        raise acpp.EngineError("the engine did not return JSON", status=502)


def _transcribe_blocking(payload, wav):
    engine = _engine()
    try:
        with _audio_file(wav) as path:
            with _infer_lock:
                return _parse_engine(engine.post("/v1/audio/transcriptions",
                                                 _engine_body(path, payload)))
    except acpp.EngineError as e:
        status = e.status if e.status in (400, 413, 429, 503) else 502
        raise HTTPException(status_code=status, detail=str(e))


def _slice_wav(wav, start_s, end_s):
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    lo = max(0, int(float(start_s) * rate))
    hi = min(len(audio), int(float(end_s) * rate))
    if hi <= lo:
        return b""
    buf = io.BytesIO()
    sf.write(buf, audio[lo:hi], rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _format_result(doc, response_format):
    text = str((doc or {}).get("text") or "")
    if response_format in ("text", "srt", "vtt"):
        return Response(content=text, media_type="text/plain")
    out = {"text": text}
    if isinstance(doc, dict) and "timing" in doc:
        out["timing"] = doc["timing"]
    return out


def _engine():
    if not _state["ready"] or _state["engine"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")
    return _state["engine"]


def _sse_events(raw):
    """Yield parsed `data:` payloads from an SSE byte stream, skipping `[DONE]`."""
    buf = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    for block in buf.split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except ValueError:
                continue


def build_app(supports):
    app = FastAPI(title="audio-audiocpp (STT)")
    mount_metrics(app)

    spec, weights, spec_error = None, None, None
    try:
        weights, spec = _capabilities()
        _state.update(weights=weights, spec=spec)
    except Exception as e:
        spec_error = str(e)
        _state["error"] = spec_error
        log.error("cannot resolve what this model can do: %s", e)

    leftovers = _args.passthrough()
    if leftovers:
        log.info("forwarding %d unclaimed flag(s) to audio.cpp: %s",
                 len(leftovers), " ".join(leftovers))

    has_stt = "stt" in supports
    has_stream = "stt_stream" in supports
    streams = bool(spec and spec.streams)

    withheld = []
    if has_stream and not streams:
        why = ("this model has no streaming mode" if spec
               else "capabilities unknown: %s" % spec_error)
        withheld.append(("POST", "/v1/audio/transcriptions/live", why))
        withheld.append(("WS", "/v1/audio/stream", why))

    register(app, model_name=MODEL_NAME, module="audiocpp_stt", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=has_stt,
             family=lambda: _state["spec"].family if _state["spec"] else None,
             withheld=withheld)

    async def _from_payload(payload, async_flag, wav=None, filename=""):
        if _flag(payload.get("stream")) and tasks.truthy(async_flag):
            raise HTTPException(status_code=400, detail="stream=true cannot be combined with async")
        fmt = str(payload.get("response_format") or "json").strip().lower()
        _engine()
        if wav is None:
            audio = payload.get("audio") or payload.get("audio_path") or payload.get("file")
            if isinstance(audio, str) and audio.startswith("data:"):
                import base64

                head, _, b64 = audio.partition(",")
                try:
                    wav = _to_wav(base64.b64decode(b64), filename)
                except Exception:
                    raise HTTPException(status_code=400, detail="audio data: URL is not valid base64")
            elif isinstance(audio, str) and audio and os.path.isfile(audio):
                # Native JSON: a server-side path. Forward it; the engine reads it itself.
                if _flag(payload.get("stream")):
                    return _stream_file(payload, audio)
                return await _dispatch_path(payload, audio, async_flag, fmt)
            else:
                raise HTTPException(status_code=400, detail="file (audio) is required")
        else:
            wav = _to_wav(wav, filename)
        if _flag(payload.get("stream")):
            return _stream_bytes(payload, wav)
        segments = payload.get("segments")
        if segments:
            if isinstance(segments, str):
                segs = parse_segments(segments)
            elif isinstance(segments, list):
                segs = segments
            else:
                raise HTTPException(status_code=400, detail="`segments` must be a JSON array")

            def _work_batch(ctx):
                out = []
                ctx.progress(stage="transcribe", done=0, total=len(segs))
                for i, seg in enumerate(segs, 1):
                    ctx.checkpoint()
                    try:
                        piece = _slice_wav(wav, seg.get("start") or 0, seg.get("end") or 0)
                        if not piece:
                            out.append({"text": ""})
                        else:
                            ctx.meter(input_seconds=probe_seconds(piece))
                            out.append({"text": _transcribe_blocking(payload, piece).get("text") or ""})
                    except tasks.Cancelled:
                        raise
                    except HTTPException as e:
                        out.append({"error": e.detail})
                    except Exception as e:
                        out.append({"error": "stt failed: %s" % e})
                    finally:
                        ctx.progress(done=i, total=len(segs))
                return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}

            return await tasks.dispatch(async_flag, "stt", MODEL_NAME, _work_batch,
                                        fail="transcription failed")

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="transcribe")
            ctx.meter(input_seconds=probe_seconds(wav))
            doc = _transcribe_blocking(payload, wav)
            ctx.progress(ratio=1.0, stage="done")
            return _format_result(doc, fmt)

        return await tasks.dispatch(async_flag, "stt", MODEL_NAME, _work,
                                    fail="transcription failed")

    async def _dispatch_path(payload, path, async_flag, fmt):
        def _work(ctx):
            ctx.progress(ratio=0.0, stage="transcribe")
            ctx.meter(input_seconds=probe_seconds(path))
            try:
                doc = _parse_engine(_engine().post("/v1/audio/transcriptions",
                                                   _engine_body(path, payload)))
            except acpp.EngineError as e:
                status = e.status if e.status in (400, 413, 429, 503) else 502
                raise HTTPException(status_code=status, detail=str(e))
            ctx.progress(ratio=1.0, stage="done")
            return _format_result(doc, fmt)

        return await tasks.dispatch(async_flag, "stt", MODEL_NAME, _work,
                                    fail="transcription failed")

    def _stream_path(payload, path, hold=None):
        if not streams:
            if hold is not None:
                hold.__exit__(None, None, None)
            raise HTTPException(status_code=400,
                                detail="%s has no streaming mode; drop stream to transcribe "
                                       "the whole clip" % MODEL_NAME)
        body = _engine_body(path, payload)
        body["stream"] = True

        def pump():
            try:
                with _engine().stream("/v1/audio/transcriptions", body) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise HTTPException(status_code=502,
                                            detail=acpp._engine_message(response))
                    for chunk in response.iter_bytes():
                        if chunk:
                            yield chunk
            except acpp.EngineError as e:
                raise HTTPException(status_code=502, detail=str(e))
            finally:
                if hold is not None:
                    hold.__exit__(None, None, None)

        return StreamingResponse(pump(), media_type="text/event-stream")

    def _stream_file(payload, path):
        return _stream_path(payload, path)

    def _stream_bytes(payload, wav):
        hold = _audio_file(wav)
        path = hold.__enter__()
        return _stream_path(payload, path, hold=hold)

    if has_stt:
        @app.post("/v1/audio/transcriptions")
        async def transcriptions(request: Request):
            ctype = (request.headers.get("content-type") or "").lower()
            async_flag = request.query_params.get("async")
            if "application/json" in ctype:
                try:
                    payload = dict(await request.json())
                except Exception:
                    raise HTTPException(status_code=400, detail="body must be JSON")
                async_flag = async_flag or payload.get("async")
                return await _from_payload(payload, async_flag)
            form = await request.form()
            upload = form.get("file")
            raw = await upload.read() if hasattr(upload, "read") else None
            if not raw:
                raise HTTPException(status_code=400, detail="file (audio) is required")
            payload = {k: form.get(k) for k in ("language", "response_format", "stream",
                                                "segments", "async") + _OPTION_KEYS
                       if form.get(k) not in (None, "")}
            # A JSON `options` object can arrive as a form string.
            raw_opts = form.get("options")
            if raw_opts:
                try:
                    payload["options"] = json.loads(raw_opts) if isinstance(raw_opts, str) else raw_opts
                except ValueError:
                    raise HTTPException(status_code=400, detail="options must be JSON")
            async_flag = async_flag or payload.pop("async", None)
            filename = getattr(upload, "filename", "") or ""
            return await _from_payload(payload, async_flag, wav=raw, filename=filename)

    if has_stream and streams:
        @app.post("/v1/audio/transcriptions/live")
        async def live(request: Request):
            """Native live ingest: chunked PCM in, SSE out. Query params pass through."""
            _engine()
            params = {k: v for k, v in request.query_params.items()}
            params["model"] = acpp.MODEL_ID
            chunks = queue.Queue()
            done = object()

            async def fill():
                try:
                    async for piece in request.stream():
                        if piece:
                            chunks.put(piece)
                finally:
                    chunks.put(done)

            def sync_iter():
                while True:
                    item = chunks.get()
                    if item is done:
                        return
                    yield item

            def pump():
                try:
                    with _engine().stream_live("/v1/audio/transcriptions/live",
                                              sync_iter(), params) as response:
                        if response.status_code >= 400:
                            response.read()
                            raise HTTPException(status_code=response.status_code,
                                                detail=acpp._engine_message(response))
                        for chunk in response.iter_bytes():
                            if chunk:
                                yield chunk
                except acpp.EngineError as e:
                    status = e.status if e.status in (400, 413, 429, 503) else 502
                    raise HTTPException(status_code=status, detail=str(e))

            fill_task = asyncio.create_task(fill())

            def gen():
                try:
                    yield from pump()
                finally:
                    fill_task.cancel()

            return StreamingResponse(gen(), media_type="text/event-stream")

        @app.websocket("/v1/audio/stream")
        async def stream(ws: WebSocket):
            """Platform WS shape, translated onto the engine's /live ingest."""
            await ws.accept()
            if not _state["ready"]:
                await ws.send_text(json.dumps({"type": "error",
                                               "detail": _state["error"] or "model not ready"}))
                await ws.close()
                return
            await ws.send_text(json.dumps({"type": "ready"}))
            cfg = {"sample_rate": 16000, "language": "", "options": {}}
            incoming = queue.Queue()
            outgoing = queue.Queue()
            END = object()

            def sync_iter():
                while True:
                    item = incoming.get()
                    if item is END:
                        return
                    yield item

            def pump_live():
                params = _live_query(cfg)
                try:
                    with _engine().stream_live("/v1/audio/transcriptions/live",
                                              sync_iter(), params) as response:
                        if response.status_code >= 400:
                            response.read()
                            outgoing.put(("error", acpp._engine_message(response)))
                            return
                        acc, buf = "", b""
                        for chunk in response.iter_bytes():
                            if not chunk:
                                continue
                            buf += chunk
                            while b"\n\n" in buf:
                                block, buf = buf.split(b"\n\n", 1)
                                for event in _sse_events(block + b"\n\n"):
                                    kind = event.get("type")
                                    if kind == "transcript.text.delta":
                                        acc += str(event.get("delta") or "")
                                        outgoing.put(("partial", acc))
                                    elif kind == "transcript.text.done":
                                        outgoing.put(("final", str(event.get("text") or acc)))
                                    elif kind == "error":
                                        outgoing.put(("error", event.get("message")
                                                      or event.get("error") or json.dumps(event)))
                except Exception as e:
                    outgoing.put(("error", str(e)))
                finally:
                    outgoing.put(("closed", None))

            def pump_windows():
                rate = int(cfg.get("sample_rate") or 16000)
                max_s = _vad_max_s(cfg)
                family = getattr(_state.get("spec"), "family", "") or ""
                frame_n = max(2, int(rate * _VAD_FRAME_S) * 2)
                hang_n = max(frame_n, int(_VAD_HANG_S * rate) * 2)
                min_n = max(frame_n, int(_VAD_MIN_S * rate) * 2)
                max_n = max(min_n, int(max_s * rate) * 2)
                decode_n = max(frame_n, int(_DECODE_S * rate) * 2)
                buf = bytearray()
                scan = 0
                utt0 = None
                last_loud = 0
                last_decode = 0
                parts = []
                live = ""
                acc = ""

                def transcribe(piece):
                    try:
                        return _transcribe_pcm(piece, cfg)
                    except Exception as e:
                        log.warning("window transcribe failed: %s", e)
                        return None

                def emit():
                    nonlocal acc
                    acc = _with_live(parts, live, family)
                    if acc:
                        outgoing.put(("partial", acc))

                def commit(end):
                    nonlocal utt0, live, last_decode, acc
                    start = utt0
                    utt0 = None
                    live = ""
                    last_decode = end
                    piece = bytes(buf[start:end])
                    if len(piece) < min_n:
                        return
                    text = transcribe(piece)
                    if text and text.strip():
                        parts.append(text)
                        emit()

                def decode_live(now):
                    nonlocal last_decode, live
                    if utt0 is None or now - last_decode < decode_n:
                        return
                    piece = bytes(buf[utt0:now])
                    if len(piece) < min_n:
                        return
                    last_decode = now
                    text = transcribe(piece)
                    if text is None:
                        return
                    live = text
                    emit()

                try:
                    _stream_busy.set()
                    while True:
                        item = incoming.get()
                        if item is END:
                            break
                        buf.extend(item)
                        while scan + frame_n <= len(buf):
                            rms = _frame_rms(buf[scan:scan + frame_n])
                            end = scan + frame_n
                            if rms >= _VAD_SPEECH_RMS:
                                if utt0 is None:
                                    utt0 = scan
                                    last_decode = scan
                                last_loud = end
                            if utt0 is not None:
                                uttered = last_loud - utt0
                                silence = end - last_loud
                                if uttered >= max_n:
                                    commit(last_loud)
                                elif uttered >= min_n and silence >= hang_n:
                                    commit(last_loud)
                                else:
                                    decode_live(end)
                            scan = end
                    if utt0 is not None:
                        commit(len(buf))
                    elif buf and not parts:
                        text = transcribe(bytes(buf))
                        if text and text.strip():
                            parts.append(text)
                            acc = _join_asr(parts, family)
                    outgoing.put(("final", acc or _join_asr(parts, family)))
                except Exception as e:
                    outgoing.put(("error", str(e)))
                finally:
                    _stream_busy.clear()
                    outgoing.put(("closed", None))

            def pump():
                if _window_live(_state.get("spec")):
                    pump_windows()
                else:
                    pump_live()

            async def emit(kind, value):
                if kind == "partial":
                    await ws.send_text(json.dumps({"type": "partial", "text": value}))
                elif kind == "final":
                    await ws.send_text(json.dumps({"type": "final", "text": value}))
                elif kind == "error":
                    await ws.send_text(json.dumps({"type": "error", "detail": value}))

            async def drain():
                loop = asyncio.get_running_loop()
                while True:
                    kind, value = await loop.run_in_executor(None, outgoing.get)
                    if kind == "closed":
                        return
                    await emit(kind, value)

            worker = None
            drainer = None
            started = False
            total = 0
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    text = msg.get("text")
                    if text is not None:
                        try:
                            obj = json.loads(text)
                        except Exception:
                            obj = {}
                        kind = obj.get("type")
                        if kind == "start":
                            cfg["language"] = str(obj.get("language") or "")
                            cfg["sample_rate"] = int(obj.get("sample_rate") or 16000)
                            try:
                                cfg["options"] = _options_of(obj)
                            except HTTPException as e:
                                await emit("error", e.detail)
                                break
                            if not started:
                                started = True
                                worker = threading.Thread(target=pump, daemon=True)
                                worker.start()
                                drainer = asyncio.create_task(drain())
                            continue
                        if kind in ("stop", "done", "finish"):
                            break
                        continue
                    data = msg.get("bytes")
                    if not data:
                        continue
                    if not started:
                        started = True
                        worker = threading.Thread(target=pump, daemon=True)
                        worker.start()
                        drainer = asyncio.create_task(drain())
                    total += len(data)
                    incoming.put(data)
                incoming.put(END)
                if drainer is not None:
                    await asyncio.wait_for(drainer, timeout=60.0)
                await ws.send_text(json.dumps({
                    "type": "closed",
                    "audio_seconds": round(total / 2.0 / max(cfg["sample_rate"], 1), 3),
                }))
                await ws.close()
            except WebSocketDisconnect:
                incoming.put(END)
            except Exception as e:
                log.exception("stream error: %s", e)
                incoming.put(END)
                try:
                    await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                    await ws.close()
                except Exception:
                    pass
            finally:
                if drainer is not None and not drainer.done():
                    drainer.cancel()

    elif has_stream and not streams:
        pass

    return app


def run(supports):
    _runtime.serve(supports, _boot, build_app, "audio.cpp engine", timeout_s=BOOT_TIMEOUT_S,
                   disable_ws_ping=True)
