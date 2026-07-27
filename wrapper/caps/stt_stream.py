# Qwen3-ASR on one in-process vLLM load, serving BOTH offline stt and WebSocket stt_stream.
import os
import json
import asyncio
import logging
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
import uvicorn

from .. import tasks
from .. import watchdog
from ..gpu import mount_metrics
from ..contract import register

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-stt-stream")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (_src or MODEL_NAME)
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))
GPU_UTIL = float(os.environ.get("VLLM_GPU_UTIL", "0.45") or 0.45)
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "32") or 32)
# Holds ONE unit of work (a <=540s offline chunk or a ~240s streaming window); chart-derived.
MAX_MODEL_LEN = int(os.environ.get("STREAM_MAX_LEN", "16384") or 16384)
UNFIXED_CHUNK_NUM = int(os.environ.get("STREAM_UNFIXED_CHUNK_NUM", "2") or 2)
UNFIXED_TOKEN_NUM = int(os.environ.get("STREAM_UNFIXED_TOKEN_NUM", "5") or 5)
CHUNK_SIZE_SEC = float(os.environ.get("STREAM_CHUNK_SIZE_SEC", "2.0") or 2.0)
DEFAULT_STEP_MS = int(os.environ.get("STREAM_STEP_MS", "500") or 500)
# Finalize + re-init this often, so a long session never overflows vLLM's ~8192-token encoder cache.
ROLL_SEC = float(os.environ.get("STREAM_ROLL_SEC", "240") or 240)

_state = {"ready": False, "error": None, "asr": None}
# vLLM's generate is blocking and not concurrency-safe, so all inference shares one lock.
_infer_lock = asyncio.Lock()
# The same engine is also driven by the task worker (offline stt), which lives on another thread.
_gpu = threading.Lock()


def _gated(fn, *a):
    with _gpu:
        return fn(*a)


def _p(msg):
    print("[stream] " + msg, flush=True)


def _capture_kw():
    # Capture is where startup wedges holding the vGPU lock; inference is batch 1, so 4 shapes do.
    if tasks.truthy(os.environ.get("VLLM_ENFORCE_EAGER")):
        _p("VLLM_ENFORCE_EAGER is set: skipping CUDA graphs entirely")
        return {"enforce_eager": True}
    raw = os.environ.get("VLLM_CAPTURE_SIZES", "1,2,4,8")
    sizes = [int(s) for s in raw.replace(" ", "").split(",") if s]
    if not sizes:
        return {}
    try:
        from vllm.config import CompilationConfig

        fields = set(getattr(CompilationConfig, "model_fields", None) or {})
    except Exception as e:
        _p("WARN cannot inspect vLLM CompilationConfig (%s); leaving capture sizes alone" % e)
        return {}
    for name in ("cudagraph_capture_sizes", "capture_sizes"):
        if name in fields:
            return {"compilation_config": {name: sizes}}
    _p("WARN CompilationConfig has no capture-size field; leaving capture sizes alone")
    return {}


def _load_blocking():
    # On the MAIN thread before uvicorn: vLLM installs signal handlers, so a daemon thread fails.
    _p("importing qwen_asr ...")
    from qwen_asr import Qwen3ASRModel

    # qwen-asr silence-splits at this window; 540s keeps one call inside the ~600s encoder cache.
    try:
        import qwen_asr.inference.qwen3_asr as _qasr_mod
        import qwen_asr.inference.utils as _qasr_utils

        _win = int(os.environ.get("OFFLINE_MAX_INPUT_SEC", "540") or 540)
        _qasr_utils.MAX_ASR_INPUT_SECONDS = _win
        _qasr_mod.MAX_ASR_INPUT_SECONDS = _win
        _p("patched qwen-asr MAX_ASR_INPUT_SECONDS -> %ds" % _win)
    except Exception as e:
        _p("WARN could not patch MAX_ASR_INPUT_SECONDS (%s)" % e)
    _p("constructing Qwen3ASRModel.LLM(model=%s, gpu_util=%.2f, max_model_len=%d) ..."
       % (MODEL_REPO, GPU_UTIL, MAX_MODEL_LEN))
    # These are vLLM kwargs qwen-asr forwards; a build that takes fewer of them gets less.
    _kw = dict(model=MODEL_REPO, gpu_memory_utilization=GPU_UTIL, max_new_tokens=MAX_NEW_TOKENS)
    _cap = _capture_kw()
    if _cap:
        _p("graph capture tuning: %s" % _cap)
    _attempts = [dict(_kw, max_model_len=MAX_MODEL_LEN, **_cap)] if _cap else []
    _attempts += [dict(_kw, max_model_len=MAX_MODEL_LEN), dict(_kw)]
    asr = None
    for _i, _try in enumerate(_attempts, 1):
        try:
            asr = Qwen3ASRModel.LLM(**_try)
            break
        # ValueError = pydantic rejected a field; OOM is a RuntimeError and must NOT be retried.
        except (TypeError, ValueError) as e:
            if _i == len(_attempts):
                raise
            _p("LLM() rejected %s (%s); retrying with fewer kwargs"
               % (sorted(set(_try) - set(_kw)), e))
    _state["asr"] = asr
    _state["ready"] = True
    _p("engine READY: %s (gpu_util=%.2f)" % (MODEL_REPO, GPU_UTIL))
    log.info("qwen-asr streaming engine loaded: %s (gpu_util=%.2f)", MODEL_REPO, GPU_UTIL)


def _decode_to_16k_mono(raw, filename):
    import tempfile
    import librosa

    suffix = os.path.splitext(filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        y, _sr = librosa.load(path, sr=16000, mono=True)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    return y.astype("float32")


def _offline_transcribe(audio):
    # Native offline transcription on the same load; max_tokens is raised then restored.
    asr = _state["asr"]
    off_max = int(os.environ.get("OFFLINE_MAX_TOKENS", "4096") or 4096)
    sp = getattr(asr, "sampling_params", None)
    old = getattr(sp, "max_tokens", None) if sp is not None else None
    try:
        if sp is not None:
            sp.max_tokens = off_max
        results = asr.transcribe(audio=(audio, 16000), language=None, return_time_stamps=False)
    finally:
        if sp is not None and old is not None:
            sp.max_tokens = old
    r = results[0] if results else None
    t = getattr(r, "text", None) if r is not None else None
    if t is None and isinstance(r, dict):
        t = r.get("text")
    return (t or "").strip()


def _pcm16_to_f32(buf):
    import numpy as np

    if not buf:
        return np.zeros((0,), dtype="float32")
    return np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0


def _resample_linear(seg, src_sr):
    import numpy as np

    if src_sr == 16000 or seg.shape[0] == 0:
        return seg.astype("float32", copy=False)
    dur = seg.shape[0] / float(src_sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype="float32")
    xo = np.linspace(0.0, dur, num=seg.shape[0], endpoint=False)
    xn = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(xn, xo, seg).astype("float32")


def build_app(supports):
    has_stt = "stt" in supports
    has_stream = "stt_stream" in supports
    app = FastAPI(title="audio-stt-stream (Qwen3-ASR)")
    mount_metrics(app)

    endpoints = []
    if has_stt:
        endpoints.append({"method": "POST", "path": "/v1/audio/transcriptions",
                          "description": "Offline transcription (single / batch segments; %s)"
                                         % tasks.ASYNC_HINT})
    if has_stream:
        endpoints.append({"method": "WS", "path": "/v1/audio/stream",
                          "description": "Streaming ASR (WebSocket)"})

    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"], task_api=has_stt)

    if has_stt:
        @app.post("/v1/audio/transcriptions")
        async def transcriptions(file: UploadFile = File(...),
                                 model: str = Form(None),
                                 language: str = Form(None),
                                 response_format: str = Form("json"),
                                 segments: str = Form(None),
                                 async_: str = Form(None, alias="async")):
            if not _state["ready"]:
                raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
            raw = await file.read()
            audio = await asyncio.to_thread(_decode_to_16k_mono, raw, file.filename)
            # BATCH mode (opt-in): `segments` JSON [{start,end}] — slice + transcribe each.
            if segments:
                import json as _json

                try:
                    segs = _json.loads(segments)
                except Exception as e:
                    raise HTTPException(status_code=400, detail="invalid `segments` json: %s" % e)
                if not isinstance(segs, list):
                    raise HTTPException(status_code=400, detail="`segments` must be a JSON array")

                def _work_batch(ctx):
                    out = []
                    ctx.progress(stage="transcribe", done=0, total=len(segs))
                    for i, seg in enumerate(segs, 1):
                        ctx.checkpoint()
                        try:
                            lo = max(0, int(float(seg.get("start") or 0) * 16000))
                            hi = min(len(audio), int(float(seg.get("end") or 0) * 16000))
                            if hi <= lo:
                                out.append({"text": ""})
                            else:
                                out.append({"text": _offline_transcribe(audio[lo:hi])})
                        except tasks.Cancelled:
                            raise
                        except Exception as e:
                            out.append({"error": "stt failed: %s" % e})
                        finally:
                            ctx.progress(done=i, total=len(segs))
                    return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}

                return await tasks.dispatch(async_, "stt", MODEL_NAME, _work_batch,
                                            fail="transcription failed", gate=_gpu)

            # SINGLE mode.
            def _work(ctx):
                ctx.progress(ratio=0.0, stage="transcribe")
                text = _offline_transcribe(audio)
                ctx.progress(ratio=1.0, stage="done")
                if response_format in ("text", "srt", "vtt"):
                    return Response(content=text, media_type="text/plain")
                return {"text": text}

            return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                        fail="transcription failed", gate=_gpu)

    if has_stream:
        @app.websocket("/v1/audio/stream")
        async def stream(ws: WebSocket):
            import numpy as np

            await ws.accept()
            if not _state["ready"]:
                await ws.send_text(json.dumps({"type": "error",
                                               "detail": _state["error"] or "model not ready"}))
                await ws.close()
                return
            asr = _state["asr"]
            sample_rate = 16000
            step_ms = DEFAULT_STEP_MS
            language = None

            def _new_state():
                return asr.init_streaming_state(
                    unfixed_chunk_num=UNFIXED_CHUNK_NUM,
                    unfixed_token_num=UNFIXED_TOKEN_NUM,
                    chunk_size_sec=CHUNK_SIZE_SEC,
                )

            # prefix = text finalized by earlier rolls; samples = audio fed to the current state.
            S = {"st": _new_state(), "prefix": "", "samples": 0}
            roll_samples = max(16000, int(ROLL_SEC * 16000))
            pending = np.zeros((0,), dtype="float32")
            await ws.send_text(json.dumps({"type": "ready"}))

            def _join(a, b):
                if not a:
                    return b
                if not b:
                    return a
                # Space only between two ASCII words (CJK needs none).
                if a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
                    return a + " " + b
                return a + b

            def _full_text():
                return _join(S["prefix"], getattr(S["st"], "text", "") or "")

            async def _emit(kind):
                await ws.send_text(json.dumps({
                    "type": kind,
                    "text": _full_text(),
                    "language": getattr(S["st"], "language", None) or language,
                }))

            async def _roll():
                # Fold the finalized text into prefix and start fresh, resetting encoder-cache use.
                async with _infer_lock:
                    await asyncio.to_thread(_gated, asr.finish_streaming_transcribe, S["st"])
                S["prefix"] = _join(S["prefix"], getattr(S["st"], "text", "") or "")
                S["st"] = _new_state()
                S["samples"] = 0

            async def _feed(cur):
                # Backstop: if the cache overflows despite the proactive roll, roll and retry once.
                try:
                    async with _infer_lock:
                        await asyncio.to_thread(_gated, asr.streaming_transcribe, cur, S["st"])
                except Exception as e:
                    msg = str(e).lower()
                    if "encoder cache" in msg or "exceeds" in msg or "pre-allocated" in msg:
                        log.warning("encoder-cache overflow; rolling session and retrying: %s", e)
                        await _roll()
                        async with _infer_lock:
                            await asyncio.to_thread(_gated, asr.streaming_transcribe,
                                                    cur, S["st"])
                    else:
                        raise
                S["samples"] += int(cur.shape[0])

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
                        t = obj.get("type")
                        if t == "start":
                            language = obj.get("language") or None
                            sample_rate = int(obj.get("sample_rate") or 16000)
                            step_ms = int(obj.get("step_ms") or DEFAULT_STEP_MS)
                            continue
                        if t in ("stop", "done", "finish"):
                            break
                        continue
                    data = msg.get("bytes")
                    if not data:
                        continue
                    seg = _resample_linear(_pcm16_to_f32(data), sample_rate)
                    pending = np.concatenate([pending, seg]) if pending.size else seg
                    step = max(1, int(round(step_ms / 1000.0 * 16000)))
                    while pending.shape[0] >= step:
                        cur, pending = pending[:step], pending[step:]
                        await _feed(cur)
                        await _emit("partial")
                        # Proactive roll at a safe point so we never approach the cap.
                        if S["samples"] >= roll_samples:
                            await _roll()
                            await _emit("partial")
                # flush tail + finalize
                if pending.size:
                    await _feed(pending)
                async with _infer_lock:
                    await asyncio.to_thread(_gated, asr.finish_streaming_transcribe, S["st"])
                await _emit("final")
                await ws.close()
            except WebSocketDisconnect:
                pass
            except Exception as e:
                log.exception("stream error: %s", e)
                try:
                    await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                    await ws.close()
                except Exception:
                    pass

    return app


def run(supports):
    _p("stt_stream starting; model=%s port=%s supports=%s" % (MODEL_REPO, PORT, supports))
    # Armed first: this load blocks the main thread, so only another thread can time it out.
    watchdog.arm(lambda: _state["ready"], lambda: _state["error"], "qwen-asr vLLM")
    try:
        _load_blocking()
    except Exception as e:
        _state["error"] = str(e)
        _p("engine load FAILED: %s" % e)
        log.exception("engine load failed: %s", e)
    app = build_app(supports)
    _p("starting uvicorn on :%s (ready=%s)" % (PORT, _state["ready"]))
    # No server-initiated WS keepalive: bursty inference lags Pong and drops a healthy session.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL,
                ws_ping_interval=None, ws_ping_timeout=None)
