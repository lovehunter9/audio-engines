import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from anyio import CancelScope

from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime
from .config import (
    MING_MODEL_REF,
    TENCENT_MODEL_REF,
    YOUDAO_MODEL_REF,
    R2T2_MODEL_REF,
    VAD_MODEL_REF,
)
from .pipeline import run as run_pipeline
from .processes import Worker, Workers
from .protocol import Input, Stream
from .tencent import Tencent
from .youdao import Youdao


async def consume_file(source, pcm):
    for offset in range(0, len(pcm), 5120):
        await source.feed(pcm[offset : offset + 5120])
    await source.finish()


def build_app(supports):
    runtime = Runtime()
    args = EngineArgs()
    family = "tencent" if os.environ.get("AUDIO_BASE") == "s2st-tencent" else "youdao"
    translation_override = args.text("--translation-model")
    speech_override = args.text("--speech-model")
    recognition_override = args.text("--recognition-model") if family == "youdao" else None
    vad_override = args.text("--vad-model") if family == "youdao" else None
    latency_mode = args.text("--translation-latency", "high") if family == "youdao" else None
    if family == "youdao" and latency_mode not in ("high", "native"):
        raise ValueError("Unsupported official translation latency policy")
    asr_chunk_ms = args.count("--asr-chunk-ms", 2000) if family == "youdao" else None
    if family == "youdao" and asr_chunk_ms not in (800, 1600, 2000, 4000):
        raise ValueError("Unsupported ASR delivery increment")
    asr_chunk_ms_zh = args.count("--asr-chunk-ms-zh", asr_chunk_ms) if family == "youdao" else None
    if family == "youdao" and asr_chunk_ms_zh not in (800, 1600, 2000, 4000):
        raise ValueError("Unsupported Chinese ASR delivery increment")
    cache = Path(args.text("--cache-dir", "/data/s2st"))
    binary = args.text("--translation-binary", "/opt/llama/bin/llama-mtmd-cli")
    args.warn_unclaimed(__import__("logging").getLogger("audio-s2st"))
    workers = Workers(cache / "workers")
    exclusive = asyncio.Lock()
    model = None
    speech_url = "http://127.0.0.1:18091/v1"

    async def boot():
        nonlocal model
        try:
            from huggingface_hub import snapshot_download

            if translation_override:
                model = Path(translation_override)
            else:
                repo, include, exclude = runtime._hub_spec()
                snapshot = await asyncio.to_thread(
                    snapshot_download,
                    repo,
                    revision=TENCENT_MODEL_REF if family == "tencent" else YOUDAO_MODEL_REF,
                    local_files_only=True,
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                    allow_patterns=include,
                    ignore_patterns=exclude,
                )
                candidates = list(
                    Path(snapshot).glob("*Q4_K_M.gguf" if family == "tencent" else "*Q6_K.gguf")
                )
                if len(candidates) != 1:
                    raise RuntimeError("Exactly one official translation GGUF file is required")
                model = candidates[0]
            speech = (
                Path(speech_override)
                if speech_override
                else Path(
                    await asyncio.to_thread(
                        snapshot_download,
                        "inclusionAI/Ming-3.0-tts-0.5B",
                        revision=MING_MODEL_REF,
                        local_files_only=True,
                        cache_dir=os.environ.get("HF_HUB_CACHE"),
                    )
                )
            )
            if not model.is_file() or not speech.is_dir():
                raise RuntimeError(
                    "All cascade weights must already exist in the local Common cache"
                )
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("CUDA_CACHE_PATH", str(cache / "cuda"))
            python = sys.executable
            environment = {
                "PYTHONPATH": str(Path(__file__).resolve().parents[2])
                + ":"
                + os.environ.get("PYTHONPATH", ""),
                "HF_HUB_OFFLINE": "1",
                "OMP_NUM_THREADS": "4",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                "TRITON_CACHE_DIR": str(cache / "triton"),
                "CUDA_CACHE_PATH": str(cache / "cuda"),
                "XDG_CACHE_HOME": str(cache / "xdg"),
                "NUMBA_CACHE_DIR": str(cache / "numba"),
                "MPLCONFIGDIR": str(cache / "matplotlib"),
            }
            for key in ("CUDA_HOME", "CUDA_PATH"):
                if os.environ.get(key):
                    environment[key] = os.environ[key]
            if os.environ.get("LD_LIBRARY_PATH"):
                environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
            if family == "tencent":
                await workers.start(
                    Worker(
                        "ming",
                        (
                            python,
                            "-m",
                            "wrapper.s2st.tencent_worker",
                            "--model",
                            str(speech),
                            "--translation-model",
                            str(model),
                            "--cache",
                            str(cache / "ming"),
                            "--compile-flow",
                        ),
                        "http://127.0.0.1:18091/internal/tencent/health",
                        environment=environment,
                        timeout=1800,
                    )
                )
                # A short synthetic input verifies the actual full-GPU co-residency at startup.
                warm = Input(Stream("zh2en"))
                await warm.feed(bytes(51200))
                await warm.finish()
                with tempfile.TemporaryDirectory(prefix="warmup-", dir=cache) as directory:
                    producer = Tencent(
                        model,
                        Path(directory) / "native",
                        binary=binary,
                        native_url="http://127.0.0.1:18091/internal/tencent",
                    )

                    async def discard(pcm):
                        pass

                    try:
                        async for event in producer.run(warm, discard):
                            pass
                    finally:
                        await producer.close()
            else:

                async def snapshot(repo, revision, override=None):
                    if override:
                        value = Path(override)
                    else:
                        value = Path(
                            await asyncio.to_thread(
                                snapshot_download,
                                repo,
                                revision=revision,
                                local_files_only=True,
                                cache_dir=os.environ.get("HF_HUB_CACHE"),
                            )
                        )
                    if not value.is_dir():
                        raise RuntimeError("Required local snapshot is absent: " + repo)
                    return value

                recognition = await snapshot(
                    "netease-youdao/Confucius4-R2T2", R2T2_MODEL_REF, recognition_override
                )
                vad = (
                    Path(vad_override)
                    if vad_override
                    else await snapshot("FireRedTeam/FireRedVAD", VAD_MODEL_REF) / "Stream-VAD"
                )
                if not (vad / "model.pth.tar").is_file() or not (vad / "cmvn.ark").is_file():
                    raise RuntimeError("The official Stream-VAD snapshot is incomplete")
                environment["PYTHONPATH"] = "/opt/asr-private:" + environment["PYTHONPATH"]
                await workers.start(
                    Worker(
                        "ming",
                        (
                            python,
                            "-m",
                            "wrapper.s2st.youdao_worker",
                            "--model",
                            str(speech),
                            "--translation-model",
                            str(model),
                            "--recognition-model",
                            str(recognition),
                            "--vad-model",
                            str(vad),
                            "--cache",
                            str(cache / "ming"),
                            "--root",
                            str(cache),
                            "--asr-chunk-ms",
                            str(asr_chunk_ms),
                            "--asr-chunk-ms-zh",
                            str(asr_chunk_ms_zh),
                            "--compile-flow",
                        ),
                        "http://127.0.0.1:18091/internal/youdao/health",
                        environment=environment,
                        timeout=1800,
                    )
                )
                api_environment = dict(
                    environment,
                    CUDA_VISIBLE_DEVICES="",
                    PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                    TRANSLATION_BACKEND="openai",
                    VLLM_BASE_URL="http://127.0.0.1:18092/v1",
                    VLLM_MODEL="Confucius4-T3PO",
                    ASR_WS_URL="ws://127.0.0.1:18094/asr_stream_api_v1",
                    ASR_SECRET_KEY="test0102",
                    ASR_RECV_TIMEOUT_SECONDS="600",
                    LATENCY_MODE=latency_mode,
                    HOST="127.0.0.1",
                    PORT="18093",
                    FORCE_BREAK_THRESHOLD="200",
                    HISTORY_WINDOW="128",
                    MAX_NEW_TOKENS="256",
                    FORCE_MAX_NEW_TOKENS="256",
                )
                await workers.start(
                    Worker(
                        "youdao-api",
                        (python, "-m", "wrapper.s2st.youdao_api", "--root", str(cache)),
                        "http://127.0.0.1:18093/api/v1/health",
                        environment=api_environment,
                        timeout=180,
                    )
                )
            runtime.state["ready"] = True
        except Exception as exc:
            runtime.state.update(ready=False, error=str(exc))
            await workers.close()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(boot())
        try:
            yield
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await workers.close()

    app = FastAPI(
        title=("Hy-SimulMT" if family == "tencent" else "Confucius4")
        + " + MING speech translation",
        lifespan=lifespan,
    )

    def ready():
        if not runtime.state["ready"]:
            return False
        try:
            workers.assert_alive()
        except RuntimeError as exc:
            runtime.state.update(ready=False, error=str(exc))
        return runtime.state["ready"]

    register(
        app,
        model_name=runtime.model_name,
        module="s2st",
        served=supports,
        is_ready=ready,
        error=lambda: runtime.state["error"],
        repo=runtime.model_repo,
        sample_rate=24000,
    )
    mount_metrics(app)

    def available():
        if not ready():
            raise HTTPException(503, runtime.state["error"] or "loading cascade")
        workers.assert_alive()
        if exclusive.locked():
            raise HTTPException(429, "one active speech translation session is supported")

    async def session(source, emit, *, streaming=False):
        terminal = None

        async def deliver(event):
            nonlocal terminal
            if event["type"] == "done":
                terminal = event
            else:
                await emit(event)

        async with exclusive:
            workers.assert_alive()
            with tempfile.TemporaryDirectory(prefix="session-", dir=cache) as directory:
                producer = (
                    Tencent(
                        model,
                        Path(directory) / "native",
                        binary=binary,
                        native_url="http://127.0.0.1:18091/internal/tencent",
                    )
                    if family == "tencent"
                    else Youdao(Path(directory) / "native", cache, latency_mode=latency_mode)
                )
                await run_pipeline(source, producer, deliver, speech_url, streaming=streaming)
        # A terminal acknowledgement means native cleanup and the session lock are released.
        if terminal is None:
            raise RuntimeError("S2ST ended without its completion record")
        await emit(terminal)

    @app.post("/v1/audio/speech-translations")
    async def translate(
        file: UploadFile = File(...), direction: str = Form(...), stream: bool = Form(False)
    ):
        available()
        try:
            specification = Stream(direction)
            raw = await file.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("audio upload exceeds 64 MiB")
            with wave.open(io.BytesIO(raw), "rb") as reader:
                if (
                    reader.getframerate() != 16000
                    or reader.getnchannels() != 1
                    or reader.getsampwidth() != 2
                ):
                    raise ValueError("input must be mono 16 kHz PCM16 WAV")
                pcm = reader.readframes(reader.getnframes())
            if not pcm or len(pcm) / 32000 > 900:
                raise ValueError("input duration must be between zero and 900 seconds")
        except (ValueError, wave.Error, EOFError) as exc:
            raise HTTPException(400, str(exc)) from exc
        source = Input(specification)
        if stream:

            async def generate():
                queue = asyncio.Queue(maxsize=32)

                async def emit(event):
                    await queue.put(event)

                async def perform():
                    feeder = asyncio.create_task(consume_file(source, pcm))
                    try:
                        await session(source, emit, streaming=True)
                        await feeder
                    finally:
                        feeder.cancel()
                        await asyncio.gather(feeder, return_exceptions=True)

                task = asyncio.create_task(perform())
                try:
                    while True:
                        item = asyncio.create_task(queue.get())
                        done, _ = await asyncio.wait(
                            {task, item}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if item in done:
                            event = item.result()
                            yield json.dumps(event, ensure_ascii=False) + "\n"
                            if event["type"] == "done":
                                await task
                                break
                        elif task in done:
                            item.cancel()
                            await asyncio.gather(item, return_exceptions=True)
                            task.result()
                            if queue.empty():
                                raise RuntimeError("S2ST ended without its completion record")
                except Exception as exc:
                    yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
                finally:
                    with CancelScope(shield=True):
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

            return StreamingResponse(generate(), media_type="application/x-ndjson")
        audio = bytearray()
        final = None

        async def collect(event):
            nonlocal final
            if event["type"] == "audio":
                audio.extend(base64.b64decode(event["pcm_base64"]))
            elif event["type"] == "done":
                final = event

        async def complete_response():
            # Establish headers before inference so Common's header timeout does not cap WAV length.
            feeder = asyncio.create_task(consume_file(source, pcm))
            try:
                await session(source, collect)
                await feeder
                output = io.BytesIO()
                with wave.open(output, "wb") as writer:
                    writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    writer.writeframes(audio)
                yield json.dumps(
                    dict(
                        final,
                        audio_base64=base64.b64encode(output.getvalue()).decode(),
                        audio_format="wav",
                        sample_rate=24000,
                    ),
                    ensure_ascii=False,
                )
            except Exception as exc:
                yield json.dumps({"type": "error", "message": str(exc)})
            finally:
                with CancelScope(shield=True):
                    feeder.cancel()
                    await asyncio.gather(feeder, return_exceptions=True)

        return StreamingResponse(complete_response(), media_type="application/json")

    @app.websocket("/v1/audio/speech-translations/stream")
    async def translate_stream(socket: WebSocket):
        await socket.accept()
        task = None
        source = None
        try:
            available()
            init = await asyncio.wait_for(socket.receive_json(), 30)
            source = Input(Stream(init["direction"]), spool_directory=cache)

            async def emit(event):
                await socket.send_json(event)

            await socket.send_json({"type": "ready", "sample_rate": 16000, "format": "pcm_s16le"})
            task = asyncio.create_task(session(source, emit, streaming=True))
            while True:
                receiving = asyncio.create_task(socket.receive())
                done, _ = await asyncio.wait({task, receiving}, return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    receiving.cancel()
                    await asyncio.gather(receiving, return_exceptions=True)
                    task.result()
                    break
                packet = receiving.result()
                if packet["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                if packet.get("bytes") is not None:
                    if source.samples / 16000 > 900:
                        raise ValueError("session exceeds 900 seconds")
                    await source.feed(packet["bytes"])
                elif json.loads(packet.get("text") or "{}") == {"type": "end"}:
                    await source.finish()
                    await task
                    break
                else:
                    raise ValueError("expected PCM16 bytes or an end record")
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as exc:
            try:
                await socket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            with CancelScope(shield=True):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if source is not None:
                    source.close()
                try:
                    await socket.close()
                except Exception:
                    pass

    return app


def main(supports):
    import uvicorn

    uvicorn.run(
        build_app(supports),
        host="0.0.0.0",
        port=int(os.getenv("ENGINE_PORT", "8000")),
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )
