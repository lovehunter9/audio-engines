"""Meta SeamlessStreaming through its original streaming speech agent."""
import asyncio
import base64
import io
import json
import logging
import os

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from .. import hfgate, tasks
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-seamless")
_runtime = Runtime(engine=None)
_args = EngineArgs()
_SOURCE_CHUNK_MS = _args.count("--source-chunk-ms", 1280 if os.environ.get("AUDIO_BASE") == "seamlessov" else 320)


def language(raw):
    value = str(raw).strip().lower()
    return {"en": "eng", "english": "eng", "zh": "cmn", "zh-cn": "cmn", "chinese": "cmn",
            "ja": "jpn", "ko": "kor", "fr": "fra", "de": "deu", "es": "spa", "ru": "rus",
            "pt": "por", "ar": "arb", "hi": "hin", "it": "ita"}.get(value, value)


def _boot():
    try:
        from huggingface_hub import snapshot_download
        from .seamless_native import Engine
        repo, include, exclude = _runtime._hub_spec()
        snapshot = snapshot_download(repo, local_files_only=True, allow_patterns=include, ignore_patterns=exclude)
        _runtime.state.update(engine=Engine(snapshot, backend="openvino" if os.environ.get("AUDIO_BASE") == "seamlessov" else "cuda", source_chunk_ms=_SOURCE_CHUNK_MS), ready=True, error=None)
    except Exception as error:
        _runtime.state["error"] = hfgate.explain(_runtime.model_repo, error)
        log.exception("Official Seamless runtime initialization failed")


def build_app(supports):
    app = FastAPI(title="SeamlessStreaming official runtime")
    state = _runtime.state
    mount_metrics(app)
    register(app, model_name=_runtime.model_name, module="seamless", served=supports,
             repo=_runtime.model_repo, is_ready=lambda: state["ready"],
             error=lambda: state["error"], task_api=True, sample_rate=16000)
    _args.warn_unclaimed(log)

    def require():
        if not state["ready"]:
            raise HTTPException(503, state["error"] or "engine not ready")
        return state["engine"]

    @app.post("/v1/audio/translations")
    async def translate(request: Request, file: UploadFile = File(...),
                        source_lang: str = Form("eng"), target_lang: str = Form("cmn"),
                        response_format: str = Form("json"), model: str = Form(None)):
        engine = require()
        source, target = language(source_lang), language(target_lang)
        if target not in engine.target_languages:
            raise HTTPException(400, "Unsupported speech target language")
        if response_format not in ("json", "text", "wav", "pcm"):
            raise HTTPException(400, "response_format must be json, text, wav, or pcm")
        data = await file.read()
        if not data:
            raise HTTPException(400, "file is empty")
        def work(ctx):
            import soundfile as sf
            row, wav = engine.translate(data, source, target)
            samples, rate = sf.read(io.BytesIO(wav), dtype="int16")
            ctx.meter(output_seconds=len(samples)/rate)
            if response_format == "wav":
                return tasks.Binary(wav, "audio/wav", suffix=".wav")
            if response_format == "pcm":
                return tasks.Binary(samples.astype("<i2").tobytes(), "audio/L16", suffix=".pcm")
            if response_format == "text":
                from fastapi.responses import PlainTextResponse
                return PlainTextResponse(row["text"])
            return {"text": row["text"], "sample_rate": rate,
                    "audio": base64.b64encode(wav).decode(), "elapsed_seconds": row["elapsed_s"]}
        return await tasks.dispatch(request.query_params.get("async"), "audio_s2s", _runtime.model_name,
                                    work, fail="official translation failed")

    @app.websocket("/v1/audio/translations/stream")
    async def stream(ws: WebSocket):
        import numpy as np
        await ws.accept()
        engine = None
        locked = False
        try:
            engine = require()
            locked = engine.lock.acquire(blocking=False)
            if not locked:
                raise HTTPException(409, "Another translation session is active")
            session = await asyncio.to_thread(engine.new_session, "cmn")
            await ws.send_json({"event": "ready", "input_sample_rate": 16000, "sample_rate": 16000})
            started = False
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                finished = False
                samples = []
                if message.get("text") is not None:
                    value = json.loads(message["text"])
                    if not isinstance(value, dict):
                        raise ValueError("Text frames must contain a JSON object")
                    finished = value.get("event", value.get("type")) in ("end", "stop", "done", "finish")
                    if value.get("sample_rate", 16000) != 16000:
                        raise ValueError("Input PCM must use 16000 Hz")
                    if value.get("target_lang"):
                        if started:
                            raise ValueError("Cannot change language after audio input")
                        session = await asyncio.to_thread(engine.new_session, language(value["target_lang"]))
                    if not finished:
                        continue
                elif message.get("bytes"):
                    data = message["bytes"]
                    if len(data) % 2:
                        raise ValueError("PCM frames must contain little-endian int16 samples")
                    samples = np.frombuffer(data, dtype="<i2").astype(np.float32)/32768.
                    started = True
                events = await asyncio.to_thread(engine.push, session, samples, finished)
                for kind, payload in events:
                    if kind == "text":
                        await ws.send_json({"text": payload, "finished": False})
                    else:
                        await ws.send_bytes(payload)
                if finished:
                    await ws.send_json({"text": "", "finished": True, "sample_rate": 16000})
                    break
        except WebSocketDisconnect:
            pass
        except Exception as error:
            await ws.send_json({"error": error.detail if isinstance(error, HTTPException) else str(error)})
            await ws.close()
        finally:
            if locked:
                engine.lock.release()
    return app


def run(supports):
    _runtime.serve(supports, _boot, build_app, "SeamlessStreaming official runtime",
                   disable_ws_ping=True, timeout_s=3600)
