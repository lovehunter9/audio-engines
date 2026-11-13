# The official streaming speech agent serves both whole recordings and live PCM.
import asyncio
import logging
import os
import threading

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from .. import hfgate
from .. import tasks
from ..audioio import decode_mono, pcm16_to_float32, resample_linear, seconds
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-s3-s2st")

_runtime = Runtime(model=None, dtype=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
SOURCE_LANG = str(_args.text("--source-lang", "English") or "English")
TARGET_LANG = str(_args.text("--target-lang", "Chinese") or "Chinese")
CHUNK_MS = _args.count("--chunk-size-ms", 160)
DTYPE = _args.text("--dtype", "bf16")
THRESHOLD = (
    _args.number("--decision-threshold", 0.5) if _args.given("--decision-threshold") else None
)

IN_RATE = 16000
OUT_RATE = 24000
BOOT_TIMEOUT_S = 3600.0

_LANGS = {
    "en": "eng",
    "english": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    "zh": "cmn",
    "zh-cn": "cmn",
    "chinese": "cmn",
}


_state = _runtime.state
_lock = threading.Lock()


def _lang(raw, field):
    name = str(raw or "").strip()
    code = _LANGS.get(name.lower(), name)
    supported = _state.get("languages", {"cmn", "eng"})
    if code not in supported:
        raise HTTPException(status_code=400, detail="unsupported %s: %s" % (field, name))
    return code


def _pair(src, tgt):
    source = _lang(src or SOURCE_LANG, "source_lang")
    target = _lang(tgt or TARGET_LANG, "target_lang")
    if source == target:
        raise HTTPException(status_code=400, detail="source_lang and target_lang must differ")
    return source, target


def _boot():
    try:
        from .s3_core import Engine

        engine = Engine(_runtime._hub_spec()[0], dtype=DTYPE, threshold=THRESHOLD)
        _state.update(
            engine=engine, languages=engine.languages, dtype=DTYPE, ready=True, error=None
        )
        log.info("S3-S2ST loaded (%s)", DTYPE)
    except Exception as error:
        _state["error"] = hfgate.explain(_runtime._hub_spec()[0], error)
        log.exception("S3-S2ST failed to load")


def _new_states(text_only=False):
    from types import SimpleNamespace

    return SimpleNamespace(
        source=[], processed_samples=0, upstream=None, text_only=text_only, has_text=False
    )


def _drain(states, source, target, finished, checkpoint=None):
    engine = _state["engine"]
    with _lock:
        if states.upstream is None:
            if not states.text_only and target not in engine.speech_languages:
                raise HTTPException(
                    status_code=400, detail="target has text output only; use response_format=text"
                )
            states.upstream = engine.session(target, text_only=states.text_only)
        events = []
        done = False
        size = max(1600, int(IN_RATE * CHUNK_MS / 1000))
        while len(states.source) - states.processed_samples >= size or finished:
            if checkpoint is not None:
                checkpoint()
            start = states.processed_samples
            end = min(start + size, len(states.source))
            last = bool(finished and end == len(states.source))
            pieces, done = engine.push(states.upstream, states.source[start:end], target, last)
            states.processed_samples = end
            for kind, value in pieces:
                if kind == "text" and value:
                    if (
                        states.has_text
                        and target not in ("cmn", "cmn_Hant", "jpn")
                        and value[0] not in ",.!?:;，。！？：；"
                    ):
                        value = " " + value
                    states.has_text = True
                    events.append((kind, value))
                elif kind == "samples" and not states.text_only:
                    pcm = _pcm16(value)
                    if pcm:
                        events.append(("pcm", pcm))
            if last or done:
                break
        if states.processed_samples:
            states.source = states.source[states.processed_samples :]
            states.processed_samples = 0
        return events, done


def _pcm16(samples):
    import numpy as np

    audio = np.asarray(samples, dtype="float32")
    if audio.size == 0:
        return b""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _samples(data):
    wav = decode_mono(data, IN_RATE)
    audio = wav.squeeze(0).detach().float().cpu().numpy()
    return audio.tolist()


def _join(events):
    import base64
    import io

    import numpy as np
    import soundfile as sf

    texts = [payload for kind, payload in events if kind == "text"]
    chunks = []
    for kind, payload in events:
        if kind != "pcm" or not payload:
            continue
        chunks.append(np.frombuffer(payload, dtype="<i2").astype("float32") / 32768.0)
    audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, OUT_RATE, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()
    return "".join(texts), audio, wav, base64.b64encode(wav).decode("ascii")


async def _send_events(ws, events, finished):
    for kind, payload in events:
        if kind == "text" and payload:
            await ws.send_json({"text": payload, "finished": False})
        elif kind == "pcm" and payload:
            await ws.send_bytes(payload)
    if finished:
        await ws.send_json({"text": "", "finished": True, "sample_rate": OUT_RATE})


def build_app(supports):
    app = FastAPI(title="audio-s3-s2st (S3-S2ST)")
    mount_metrics(app)
    register(
        app,
        model_name=MODEL_NAME,
        module="s3_s2st",
        served=supports,
        repo=MODEL_REPO,
        is_ready=lambda: _state["ready"],
        error=lambda: _state["error"],
        task_api=True,
        sample_rate=OUT_RATE,
    )
    _args.warn_unclaimed(log)

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    @app.post("/v1/audio/translations")
    async def translations(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(default=None),
        response_format: str = Form(default="json"),
        source_lang: str = Form(default=None),
        target_lang: str = Form(default=None),
    ):
        """Whole utterance through the speech agent. JSON returns text and 24 kHz wav."""
        _require_ready()
        fmt = str(response_format or "json").strip().lower()
        if fmt not in ("json", "text", "wav", "pcm"):
            raise HTTPException(
                status_code=400, detail="response_format must be json, text, wav, or pcm"
            )
        source, target = _pair(source_lang, target_lang)
        if fmt != "text" and target not in _state["engine"].speech_languages:
            raise HTTPException(
                status_code=400, detail="target has text output only; use response_format=text"
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="file is empty")

        def _work(ctx):
            samples = _samples(data)
            ctx.meter(input_seconds=seconds(samples, IN_RATE))
            ctx.progress(ratio=0.0, stage="translate")
            states = _new_states(text_only=fmt == "text")
            states.source = samples
            events, _done = _drain(states, source, target, True, ctx.checkpoint)
            text, audio, wav, b64 = _join(events)
            ctx.meter(output_seconds=seconds(audio, OUT_RATE))
            ctx.progress(ratio=1.0, stage="done")
            headers = {"X-Audio-Sample-Rate": str(OUT_RATE)}
            if fmt == "text":
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse(text)
            if fmt == "wav":
                return tasks.Binary(wav, "audio/wav", headers=headers, suffix=".wav")
            if fmt == "pcm":
                pcm = b"".join(payload for kind, payload in events if kind == "pcm")
                return tasks.Binary(pcm, "audio/L16", headers=headers, suffix=".pcm")
            return {"text": text, "sample_rate": OUT_RATE, "audio": b64}

        return await tasks.dispatch(
            request.query_params.get("async"),
            "audio_s2s",
            MODEL_NAME,
            _work,
            fail="translation failed",
        )

    @app.websocket("/v1/audio/translations/stream")
    async def translations_stream(ws: WebSocket):
        """Binary frames are little-endian int16 PCM at 16 kHz. `{"event":"end"}` finishes.

        The server speaks at 24 kHz: text frames are `{"text","finished"}`, and audio frames
        are little-endian int16 PCM. The first frame says which rate that PCM uses.
        """
        await ws.accept()
        if not _state["ready"]:
            await ws.send_json({"error": _state["error"] or "engine not ready"})
            await ws.close()
            return
        await ws.send_json(
            {"event": "ready", "sample_rate": OUT_RATE, "input_sample_rate": IN_RATE}
        )
        source, target = _pair(None, None)
        states = _new_states()
        rate = IN_RATE
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text is not None:
                    source, target, states, rate, stop = _ws_text(
                        text, source, target, states, rate
                    )
                    if stop:
                        events, _done = await asyncio.to_thread(
                            _drain, states, source, target, True
                        )
                        await _send_events(ws, events, finished=True)
                        break
                    continue
                data = msg.get("bytes")
                if not data:
                    continue
                if len(data) % 2:
                    await ws.send_json({"error": "pcm must be little-endian int16"})
                    continue
                seg = pcm16_to_float32(data)
                if rate != IN_RATE:
                    seg = resample_linear(seg, rate, IN_RATE)
                states.source.extend(seg.tolist())
                events, _done = await asyncio.to_thread(_drain, states, source, target, False)
                await _send_events(ws, events, finished=False)
        except WebSocketDisconnect:
            return
        except HTTPException as e:
            await ws.send_json({"error": e.detail})
            await ws.close()
        except Exception:
            log.exception("S3-S2ST session failed")
            await ws.send_json({"error": "translation failed"})
            await ws.close(code=1011)

    return app


def _ws_text(text, source, target, states, rate):
    import json

    try:
        obj = json.loads(text)
    except Exception:
        raise HTTPException(status_code=400, detail="text frames must be JSON")
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="text frames must be a JSON object")
    event = str(obj.get("event") or obj.get("type") or "").strip().lower()
    stop = event in ("end", "stop", "done", "finish")
    if not stop and (obj.get("source_lang") or obj.get("target_lang")):
        source, target = _pair(obj.get("source_lang") or source, obj.get("target_lang") or target)
        if states.upstream is not None or states.processed_samples or states.source:
            states = _new_states(text_only=states.text_only)
    if obj.get("response_format") is not None:
        fmt = str(obj["response_format"]).lower()
        if fmt not in ("text", "pcm"):
            raise HTTPException(
                status_code=400, detail="stream response_format must be text or pcm"
            )
        text_only = fmt == "text"
        if text_only != states.text_only:
            if states.upstream is not None or states.source:
                raise HTTPException(
                    status_code=400, detail="set response_format before sending audio"
                )
            states = _new_states(text_only=text_only)
    if obj.get("sample_rate"):
        try:
            rate = int(obj["sample_rate"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="sample_rate must be an integer")
        if rate <= 0:
            raise HTTPException(status_code=400, detail="sample_rate must be positive")
    return source, target, states, rate, stop


def run(supports):
    _pair(SOURCE_LANG, TARGET_LANG)
    _runtime.serve(
        supports, _boot, build_app, "S3-S2ST engine", disable_ws_ping=True, timeout_s=BOOT_TIMEOUT_S
    )
