# Voxtral Mini 4B Realtime on mainline vLLM. Child `vllm serve` owns /v1/realtime;
# this wrapper is the platform face (WS /v1/audio/stream, POST /v1/audio/transcriptions).
#
# Not vLLM-Omni: one engine process, official Mistral+vLLM recipe
# (`--tokenizer-mode mistral`). audio.cpp hop-slicing is what made this model look
# like a language-guessing metronome; do not put that path back.
import asyncio
import atexit
import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from .. import hfgate
from .. import tasks
from ..audioio import pcm16_to_float32, resample_linear
from ..batch import parse_segments
from ..contract import EngineArgs, register
from ..gpu import memory_fraction, mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-voxtral")

_runtime = Runtime("mistralai/Voxtral-Mini-4B-Realtime-2602", engine=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
PORT = _runtime.port
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Chart-intrinsic: tokenizer-mode is required, eager avoids HAMi graph-capture wedges,
# 8192 tokens is ~10 min at 80 ms/token. 16384 aborts EngineCore on a 16 Gi quota: the
# weights take 8.43 GiB and vLLM pads this model to 6 extra KV layers, leaving 3.82 GiB
# of KV where 16384 wants 4.02 GiB. The chart overrides this per card.
DEFAULT_ENGINE_ARGS = "--tokenizer-mode mistral --enforce-eager --max-model-len 8192"
CHILD_PORT = 8001
BOOT_TIMEOUT_S = 1800.0
# vLLM's APIServer survives a dead EngineCore, so process liveness alone reports a failed
# load as "still loading" for the whole boot timeout while the weights stay in VRAM.
CHILD_FATAL_MARKERS = (
    "EngineCore failed to start",
    "Engine core initialization failed",
    "EngineDeadError",
)
REQUEST_TIMEOUT_S = 1800.0
# Official client sends 4 KiB; 80 ms at 16 kHz PCM16 is 2560 B. Either is fine.
APPEND_BYTES = 4096

_state = _runtime.state


def _raw_args():
    return (os.environ.get("ENGINE_ARGS") or "").strip() or DEFAULT_ENGINE_ARGS


def _model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def vllm_argv(model_path, model_name, port=None):
    """Child argv. Host/port/model/served-model-name stay ours so we cannot collide with :8000."""
    args = EngineArgs(_raw_args())
    args.text("--host", "")
    args.text("--port", "")
    args.text("--model", "")
    args.text("--served-model-name", "")
    util = args.number("--gpu-memory-utilization", 0)
    leftover = args.passthrough()
    bin_ = shutil.which("vllm") or "vllm"
    argv = [bin_, "serve", model_path,
            "--host", "127.0.0.1",
            "--port", str(port or CHILD_PORT),
            "--served-model-name", model_name]
    argv.extend(leftover)
    if util > 0:
        argv.extend(["--gpu-memory-utilization", str(util)])
    else:
        frac = memory_fraction()
        if frac is not None:
            argv.extend(["--gpu-memory-utilization", "%.3f" % frac])
    return argv


def _float_to_pcm16(samples):
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype="float32") * 32767.0, -32768, 32767)
    return clipped.astype("<i2").tobytes()


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


def fold_event(acc, event):
    """Apply one upstream Realtime event. Returns (acc, emit_kind, emit_text).

    `partial` text is cumulative (platform WS). `delta` is the increment, kept on the
    event so file-upload SSE can relay it as transcript.text.delta.
    """
    kind = str((event or {}).get("type") or "")
    if kind in ("transcription.delta", "transcript.text.delta"):
        piece = str(event.get("delta") or "")
        if not piece:
            full = str(event.get("text") or "")
            piece = full[len(acc):] if full.startswith(acc) else full
        acc = acc + piece
        return acc, "partial", acc
    if kind in ("transcription.done", "transcript.text.done"):
        text = str(event.get("text") or acc)
        return text, "final", text
    if kind == "error":
        msg = event.get("message") or event.get("error") or json.dumps(event)
        return acc, "error", str(msg)
    return acc, None, None


class Child:
    """vllm serve on 127.0.0.1:CHILD_PORT. Logs are the only account of a load failure."""

    def __init__(self, argv, port):
        self.argv = argv
        self.port = port
        self.base_url = "http://127.0.0.1:%d" % port
        self.ws_url = "ws://127.0.0.1:%d/v1/realtime" % port
        self._proc = None
        self._log_tail = []
        self._client = None
        self._fatal = None

    @property
    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def log_tail(self):
        return "\n".join(self._log_tail[-40:])

    def start(self, timeout_s=BOOT_TIMEOUT_S):
        env = os.environ.copy()
        env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
        log.info("starting %s", " ".join(self.argv))
        self._proc = subprocess.Popen(
            self.argv, cwd="/app", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        threading.Thread(target=self._pump_logs, daemon=True).start()
        atexit.register(self.stop)
        import httpx

        self._client = httpx.Client(base_url=self.base_url, timeout=REQUEST_TIMEOUT_S)
        try:
            self._await_models(timeout_s)
            self._assert_realtime()
        except Exception:
            self.stop()  # otherwise a half-loaded engine holds its weights in VRAM
            raise

    def _pump_logs(self):
        for line in self._proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            self._log_tail.append(line)
            del self._log_tail[:-80]
            if self._fatal is None:
                for marker in CHILD_FATAL_MARKERS:
                    if marker in line:
                        self._fatal = marker
                        break
            log.info("[vllm] %s", line)

    def _await_models(self, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._fatal:
                raise RuntimeError("vllm serve reported %s during startup:\n%s"
                                   % (self._fatal, self.log_tail()))
            if not self.alive:
                raise RuntimeError("vllm serve exited with %s during startup:\n%s"
                                   % (self._proc.returncode, self.log_tail()))
            try:
                r = self._client.get("/v1/models", timeout=5.0)
                if r.status_code == 200:
                    log.info("vllm serve is up on %s", self.base_url)
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("vllm serve did not answer /v1/models within %.0fs:\n%s"
                           % (timeout_s, self.log_tail()))

    def _assert_realtime(self):
        """Fail the load if this vLLM has no Realtime route instead of hanging the first WS."""
        import httpx

        try:
            r = self._client.get("/openapi.json", timeout=10.0)
            paths = (r.json() or {}).get("paths") or {}
            if r.status_code == 200 and paths and "/v1/realtime" not in paths:
                raise RuntimeError(
                    "this vLLM has no /v1/realtime (need >= 0.16, documented from 0.20). "
                    "Do not fall back to hop-sliced POST. Tail:\n%s" % self.log_tail())
        except RuntimeError:
            raise
        except (httpx.HTTPError, ValueError, TypeError):
            pass
        joined = "\n".join(self._log_tail)
        if "realtime" in joined.lower() and "not registered" in joined.lower():
            raise RuntimeError("vLLM did not register /v1/realtime:\n%s" % self.log_tail())

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _load():
    path = _model_path()
    argv = vllm_argv(path, MODEL_NAME, CHILD_PORT)
    child = Child(argv, CHILD_PORT)
    child.start()
    _state["engine"] = child
    _state["ready"] = True
    log.info("engine READY: %s via vllm serve (realtime)", MODEL_REPO)


async def _open_realtime():
    import websockets

    child = _state.get("engine")
    if child is None:
        raise RuntimeError("vllm child is not up")
    ws = await websockets.connect(child.ws_url, max_size=None)
    try:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
    except Exception:
        await ws.close()
        raise
    if first.get("type") not in ("session.created", "ready"):
        await ws.close()
        raise RuntimeError("unexpected realtime hello: %s" % first)
    await ws.send(json.dumps({"type": "session.update", "model": MODEL_NAME}))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    return ws


async def _send_pcm(ws, pcm):
    for i in range(0, len(pcm), APPEND_BYTES):
        chunk = pcm[i:i + APPEND_BYTES]
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }))


async def _drain_until(ws, want_final, timeout_s=REQUEST_TIMEOUT_S):
    acc, final = "", ""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, remaining))
        except asyncio.TimeoutError:
            break
        try:
            event = json.loads(raw)
        except (TypeError, ValueError):
            continue
        acc, kind, text = fold_event(acc, event)
        if kind == "error":
            raise RuntimeError(text)
        if kind == "final":
            final = text
            if want_final:
                return acc, final
        if kind == "partial" and not want_final:
            return acc, final
    return acc, final


async def _transcribe_pcm(pcm):
    ws = await _open_realtime()
    try:
        await _send_pcm(ws, pcm)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        acc, final = await _drain_until(ws, want_final=True)
        return (final or acc or "").strip()
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def _offline_transcribe(audio):
    pcm = _float_to_pcm16(audio)
    return asyncio.run(_transcribe_pcm(pcm))


def build_app(supports):
    has_stt = "stt" in supports
    has_stream = "stt_stream" in supports
    app = FastAPI(title="audio-voxtral-realtime")
    mount_metrics(app)
    register(app, model_name=MODEL_NAME, module="voxtral_realtime", served=supports,
             repo=MODEL_REPO, is_ready=lambda: _state["ready"], error=lambda: _state["error"],
             task_api=has_stt, family="voxtral_realtime", sample_rate=16000)

    if has_stt:
        @app.post("/v1/audio/transcriptions")
        async def transcriptions(file: UploadFile = File(...),
                                 model: str = Form(None),
                                 language: str = Form(None),
                                 response_format: str = Form("json"),
                                 stream: str = Form(None),
                                 segments: str = Form(None),
                                 async_: str = Form(None, alias="async")):
            if not _state["ready"]:
                raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
            raw = await file.read()
            audio = await asyncio.to_thread(_decode_to_16k_mono, raw, file.filename)
            want_sse = str(stream or "").strip().lower() in ("1", "true", "yes")

            if segments:
                segs = parse_segments(segments)

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
                                            fail="transcription failed")

            if want_sse:
                async def events():
                    pcm = _float_to_pcm16(audio)
                    ws = await _open_realtime()
                    acc = ""
                    try:
                        await _send_pcm(ws, pcm)
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.commit", "final": True,
                        }))
                        while True:
                            raw_ev = await asyncio.wait_for(ws.recv(), timeout=REQUEST_TIMEOUT_S)
                            try:
                                event = json.loads(raw_ev)
                            except (TypeError, ValueError):
                                continue
                            prev = acc
                            acc, kind, text = fold_event(acc, event)
                            if kind == "partial":
                                delta = event.get("delta") or (text[len(prev):] if text.startswith(prev) else text)
                                yield "event: transcript.text.delta\ndata: %s\n\n" % json.dumps(
                                    {"type": "transcript.text.delta", "delta": delta})
                            elif kind == "final":
                                yield "event: transcript.text.done\ndata: %s\n\n" % json.dumps(
                                    {"type": "transcript.text.done", "text": text})
                                return
                            elif kind == "error":
                                yield "event: error\ndata: %s\n\n" % json.dumps({"message": text})
                                return
                    finally:
                        try:
                            await ws.close()
                        except Exception:
                            pass

                return StreamingResponse(events(), media_type="text/event-stream")

            def _work(ctx):
                ctx.progress(ratio=0.0, stage="transcribe")
                text = _offline_transcribe(audio)
                ctx.progress(ratio=1.0, stage="done")
                if response_format in ("text", "srt", "vtt"):
                    return Response(content=text, media_type="text/plain")
                return {"text": text}

            return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                        fail="transcription failed")

    if has_stream:
        @app.websocket("/v1/audio/stream")
        async def stream(ws: WebSocket):
            await ws.accept()
            if not _state["ready"]:
                await ws.send_text(json.dumps({"type": "error",
                                               "detail": _state["error"] or "model not ready"}))
                await ws.close()
                return
            await ws.send_text(json.dumps({"type": "ready"}))
            sample_rate = 16000
            upstream = None
            acc = ""
            total = 0
            pending = b""
            stopping = asyncio.Event()
            got_done = asyncio.Event()

            async def emit(kind, text):
                await ws.send_text(json.dumps({"type": kind, "text": text}))

            async def pump_upstream():
                nonlocal acc
                try:
                    async for raw_ev in upstream:
                        try:
                            event = json.loads(raw_ev)
                        except (TypeError, ValueError):
                            continue
                        acc, kind, text = fold_event(acc, event)
                        if kind == "error":
                            await emit("error", text)
                            got_done.set()
                            return
                        if kind in ("partial", "final"):
                            await emit("partial", text)
                            if kind == "final" and stopping.is_set():
                                got_done.set()
                                return
                except Exception:
                    got_done.set()

            reader = None
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
                            # Language is auto. Official asr.md has no --language for Voxtral.
                            sample_rate = int(obj.get("sample_rate") or 16000)
                            continue
                        if kind in ("stop", "done", "finish"):
                            break
                        continue
                    data = msg.get("bytes")
                    if not data:
                        continue
                    if sample_rate != 16000:
                        data = _float_to_pcm16(resample_linear(pcm16_to_float32(data), sample_rate))
                    pending += data
                    total += len(data) // 2
                    if upstream is None:
                        upstream = await _open_realtime()
                        reader = asyncio.create_task(pump_upstream())
                    while len(pending) >= APPEND_BYTES:
                        chunk, pending = pending[:APPEND_BYTES], pending[APPEND_BYTES:]
                        await upstream.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }))
                if pending and upstream is not None:
                    await upstream.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pending).decode("ascii"),
                    }))
                if upstream is not None:
                    stopping.set()
                    await upstream.send(json.dumps({
                        "type": "input_audio_buffer.commit", "final": True,
                    }))
                    try:
                        await asyncio.wait_for(got_done.wait(), timeout=60)
                    except asyncio.TimeoutError:
                        log.warning("realtime session did not send transcription.done in 60s")
                await emit("final", acc)
                await ws.send_text(json.dumps({
                    "type": "closed",
                    "audio_seconds": round(total / 16000.0, 3),
                }))
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
            finally:
                if reader is not None:
                    reader.cancel()
                if upstream is not None:
                    try:
                        await upstream.close()
                    except Exception:
                        pass

    return app


def run(supports):
    log.info("voxtral realtime starting; model=%s port=%s supports=%s",
             MODEL_REPO, PORT, supports)

    def load():
        try:
            _load()
        except Exception as e:
            _state["error"] = hfgate.explain(MODEL_REPO, e)
            log.exception("engine load failed: %s", e)

    def build(served):
        return build_app(served)

    _runtime.serve(
        supports,
        load,
        build,
        "voxtral vLLM realtime",
        load_on_main=False,
        disable_ws_ping=True,
        timeout_s=BOOT_TIMEOUT_S,
    )
