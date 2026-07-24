# Qwen3-ASR served via qwen-asr's in-process vLLM engine (Qwen3ASRModel.LLM),
# loaded ONCE. Serves offline stt (asr.transcribe — genuine offline API, NOT a
# streaming fake) and streaming stt_stream (asr.streaming_transcribe over a
# WebSocket) off the SAME load. Ported from the tested stream.py; deps baked at
# build time (no runtime pip); contract surface via wrapper.gpu + wrapper.contract.
import os
import sys
import json
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
import uvicorn

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
# max_model_len holds ONE unit of work (offline chunk <=540s+4096 out, or a
# ~240s rolled streaming window), DERIVED from the GPU quota by the chart.
MAX_MODEL_LEN = int(os.environ.get("STREAM_MAX_LEN", "16384") or 16384)
UNFIXED_CHUNK_NUM = int(os.environ.get("STREAM_UNFIXED_CHUNK_NUM", "2") or 2)
UNFIXED_TOKEN_NUM = int(os.environ.get("STREAM_UNFIXED_TOKEN_NUM", "5") or 5)
CHUNK_SIZE_SEC = float(os.environ.get("STREAM_CHUNK_SIZE_SEC", "2.0") or 2.0)
DEFAULT_STEP_MS = int(os.environ.get("STREAM_STEP_MS", "500") or 500)
# Roll (finalize + re-init) every ROLL_SEC so a long session never overflows
# vLLM's ~8192-token audio encoder cache; keeps memory bounded, transcript monotonic.
ROLL_SEC = float(os.environ.get("STREAM_ROLL_SEC", "240") or 240)

_state = {"ready": False, "error": None, "asr": None}
# vLLM's offline generate is blocking + not concurrency-safe; serialize all
# inference across connections behind one lock.
_infer_lock = asyncio.Lock()


def _p(msg):
    print("[stream] " + msg, flush=True)


def _load_blocking():
    # Construct the qwen-asr vLLM engine on the MAIN process/thread BEFORE uvicorn
    # (vLLM installs signal handlers + spawns workers at construction, so a daemon
    # thread fails); health comes up only after.
    _p("importing qwen_asr ...")
    from qwen_asr import Qwen3ASRModel

    # Offline long-audio: qwen-asr silence-splits at MAX_ASR_INPUT_SECONDS and
    # seamlessly join-merges, but the ~8192-token encoder cache (~600s) is the real
    # ceiling; lower the window to 540s so asr.transcribe(whole) fits in one call.
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
    # gpu_memory_utilization + max_model_len are vLLM LLM kwargs qwen-asr forwards;
    # retry without max_model_len if a build doesn't forward it.
    _kw = dict(model=MODEL_REPO, gpu_memory_utilization=GPU_UTIL, max_new_tokens=MAX_NEW_TOKENS)
    try:
        asr = Qwen3ASRModel.LLM(max_model_len=MAX_MODEL_LEN, **_kw)
    except TypeError as e:
        _p("LLM() rejected max_model_len (%s); retrying without it" % e)
        asr = Qwen3ASRModel.LLM(**_kw)
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


async def _offline_transcribe(audio):
    # Native offline transcription on the SAME loaded model: whole clip to
    # asr.transcribe() (qwen-asr internally splits at 540s + seamlessly merges).
    # max_tokens temporarily raised for offline chunks then restored (serialised).
    asr = _state["asr"]
    off_max = int(os.environ.get("OFFLINE_MAX_TOKENS", "4096") or 4096)

    def _run():
        sp = getattr(asr, "sampling_params", None)
        old = getattr(sp, "max_tokens", None) if sp is not None else None
        try:
            if sp is not None:
                sp.max_tokens = off_max
            return asr.transcribe(audio=(audio, 16000), language=None, return_time_stamps=False)
        finally:
            if sp is not None and old is not None:
                sp.max_tokens = old

    async with _infer_lock:
        results = await asyncio.to_thread(_run)
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
                          "description": "Offline transcription (single / batch segments)"})
    if has_stream:
        endpoints.append({"method": "GET", "path": "/v1/audio/stream",
                          "description": "Streaming ASR (WebSocket)"})

    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

    if has_stt:
        @app.post("/v1/audio/transcriptions")
        async def transcriptions(file: UploadFile = File(...),
                                 model: str = Form(None),
                                 language: str = Form(None),
                                 response_format: str = Form("json"),
                                 segments: str = Form(None)):
            if not _state["ready"]:
                raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
            raw = await file.read()
            audio = await asyncio.to_thread(_decode_to_16k_mono, raw, file.filename)
            # BATCH mode (opt-in): `segments` JSON [{start,end}] — slice + transcribe each.
            if segments:
                import json as _json

                try:
                    _segs = _json.loads(segments)
                except Exception as e:
                    raise HTTPException(status_code=400, detail="invalid `segments` json: %s" % e)
                if not isinstance(_segs, list):
                    raise HTTPException(status_code=400, detail="`segments` must be a JSON array")
                _out = []
                for _seg in _segs:
                    try:
                        _a = float(_seg.get("start") or 0)
                        _b = float(_seg.get("end") or 0)
                        _lo = max(0, int(_a * 16000))
                        _hi = min(len(audio), int(_b * 16000))
                        if _hi <= _lo:
                            _out.append({"text": ""})
                            continue
                        _t = await _offline_transcribe(audio[_lo:_hi])
                        _out.append({"text": _t})
                    except Exception as e:
                        _out.append({"error": "stt failed: %s" % e})
                return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": _out}
            # SINGLE mode.
            text = await _offline_transcribe(audio)
            if response_format in ("text", "srt", "vtt"):
                return Response(content=text, media_type="text/plain")
            return {"text": text}

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

            # prefix = text finalized by previous rolls; samples = audio fed to
            # the current state (drives the proactive roll).
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
                # Finalize current segment, fold text into prefix, start a fresh
                # state so encoder-cache usage resets to ~zero.
                async with _infer_lock:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, S["st"])
                S["prefix"] = _join(S["prefix"], getattr(S["st"], "text", "") or "")
                S["st"] = _new_state()
                S["samples"] = 0

            async def _feed(cur):
                # Backstop: if the encoder cache overflows despite the proactive
                # roll, roll and retry the chunk once.
                try:
                    async with _infer_lock:
                        await asyncio.to_thread(asr.streaming_transcribe, cur, S["st"])
                except Exception as e:
                    msg = str(e).lower()
                    if "encoder cache" in msg or "exceeds" in msg or "pre-allocated" in msg:
                        log.warning("encoder-cache overflow; rolling session and retrying: %s", e)
                        await _roll()
                        async with _infer_lock:
                            await asyncio.to_thread(asr.streaming_transcribe, cur, S["st"])
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
                    await asyncio.to_thread(asr.finish_streaming_transcribe, S["st"])
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
    try:
        _load_blocking()
    except Exception as e:
        _state["error"] = str(e)
        _p("engine load FAILED: %s" % e)
        log.exception("engine load failed: %s", e)
    app = build_app(supports)
    _p("starting uvicorn on :%s (ready=%s)" % (PORT, _state["ready"]))
    # Disable server-initiated WS keepalive: bursty offloaded inference can lag on
    # Pong past the 20s ping timeout and drop a healthy session with 1011.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL,
                ws_ping_interval=None, ws_ping_timeout=None)
