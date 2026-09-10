# Streaming ASR on one in-process vLLM load (e.g. Qwen3-ASR etc.), serving BOTH offline stt
# and WebSocket stt_stream.
import os
import json
import asyncio
import logging
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import Response

from .. import hfgate
from .. import tasks
from ..batch import parse_segments
from ..gpu import mount_metrics, memory_fraction
from ..contract import register, EngineArgs
from ..audioio import pcm16_to_float32, resample_linear
from ..runtime import Runtime

log = logging.getLogger("audio-stt-stream")

_runtime = Runtime(asr=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
PORT = _runtime.port

_args = EngineArgs()
# vLLM wants a share of the whole card; the platform hands out a quota, so derive one from it.
GPU_UTIL = _args.number("--gpu-memory-utilization", memory_fraction() or 0.45)
# Holds ONE unit of work; the chart sizes it per machine type, since unified memory needs less.
MAX_MODEL_LEN = _args.count("--max-model-len", 8192)
# Capture is where startup wedges holding the vGPU lock; inference is batch 1, so 4 shapes do.
ENFORCE_EAGER = _args.switch("--enforce-eager")
# One generate() for the whole batch instead of one per span. Off by default: this changes
# where a mid-batch cancellation can land, so the two paths have to be comparable in the
# same image before either becomes the default.
BATCH_ONE_SHOT = _args.switch("--batch-one-shot")
# Ending a span that has started repeating, rather than folding the repetition out of the text
# afterwards. A 2.2 second clip was measured producing 4096 tokens and six characters of
# transcript: qwen-asr's parse_asr_output collapses a repeated pattern (threshold 20), so the
# transcript reads correctly and only the clock suffers -- and once spans are batched, the whole
# batch waits for that one. vLLM's scheduler can end such a request instead.
#
# Off unless asked for, in three steps. Absent is what an engine already deployed does today,
# so merging this changes nothing for anybody; the bare flag turns it on without anyone having
# to know a threshold; a JSON value merges into the built-in ones for whoever re-measured.
#
# The thresholds are built in rather than required because neither is a deployment's choice.
# They have to be read together: min_pattern_size must be at least 2, since "对对对" is real
# Mandarin speech and stopping there drops the rest of the span; min_count must clear the 20
# the downstream folding uses, since at 10 the request stops one repetition short of that
# threshold and what is left survives into the transcript.
#
# 🔴 These numbers belong to the MODEL, not to the machine. They were measured against
# qwen3-asr, which is the only model this module serves (catalog.FAMILIES maps stt_stream to
# qwen3-asr, and one instance is one model). A different card does not change them; a new
# qwen3-asr release can, so re-measure on a model upgrade rather than assuming they carry
# over. The override exists so that a re-measured value can ship without a new image.
#
# The cap below is what protection looks like where the detector cannot run: vLLM 0.16 on
# arm64 has no RepetitionDetectionParams, the lazy import says so and serving continues, and
# there a runaway span has nothing bounding it. It substitutes for the detector rather than
# adding to it -- 13.0s with both against 13.2s for the detector alone -- and its headroom is
# a guess, 12 tokens per audio second over a measured 3.4, which a faster-talking corpus would
# turn into truncated speech. So it applies on exactly one condition: protection was asked for
# and this build cannot supply it. Asking for nothing gets the stock budget on both arches.
TOKENS_PER_AUDIO_SEC = 12
TOKENS_FLOOR = 64
REPETITION_DEFAULTS = {"min_pattern_size": 2, "max_pattern_size": 20, "min_count": 20}
REPETITION_OVERRIDE = (_args.text("--repetition-detection", "") or "").strip()
# switch() reads the bare form and text() the JSON one; a JSON value is not "on" to switch(),
# so both have to be consulted to answer "was it asked for at all".
REPETITION_ON = _args.switch("--repetition-detection") or bool(REPETITION_OVERRIDE)
_args.warn_unclaimed(log)

MAX_NEW_TOKENS = 32
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 2.0
DEFAULT_STEP_MS = 500
# Finalize + re-init this often, so a long session never overflows vLLM's ~8192-token encoder cache.
ROLL_SEC = 240.0
_CAPTURE_SIZES = (1, 2, 4, 8)
# qwen-asr silence-splits at this window; 540s keeps one call inside the ~600s encoder cache.
OFFLINE_MAX_INPUT_SEC = 540
OFFLINE_MAX_TOKENS = 4096

_state = _runtime.state
# One slot each, filled on first use. Two rather than one because the backstop asks what the
# build CAN do and the detector asks what was REQUESTED: see _detector_class, _repetition_params.
_repdet_class = []
_repdet_cache = []
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
    if ENFORCE_EAGER:
        _p("--enforce-eager given: skipping CUDA graphs entirely")
        return {"enforce_eager": True}
    sizes = list(_CAPTURE_SIZES)
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

    try:
        import qwen_asr.inference.qwen3_asr as _qasr_mod
        import qwen_asr.inference.utils as _qasr_utils

        _qasr_utils.MAX_ASR_INPUT_SECONDS = OFFLINE_MAX_INPUT_SEC
        _qasr_mod.MAX_ASR_INPUT_SECONDS = OFFLINE_MAX_INPUT_SEC
        _p("patched qwen-asr MAX_ASR_INPUT_SECONDS -> %ds" % OFFLINE_MAX_INPUT_SEC)
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
    _warmup()
    _state["ready"] = True
    _p("engine READY: %s (gpu_util=%.2f)" % (MODEL_REPO, GPU_UTIL))
    log.info("qwen-asr streaming engine loaded: %s (gpu_util=%.2f)", MODEL_REPO, GPU_UTIL)


def _warmup():
    """One throwaway transcription before we report ready, so no caller pays the cold-start cost.

    vLLM answers as soon as its constructor returns, but the first real inference still compiles
    and captures graphs: measured at 74s on a time-sliced card against 0.3s once warm. llm-init
    allows an upstream 60s to produce response headers, so without this the first offline caller
    reads a 502 where a transcript belongs. Streaming pays the same cost, only spread across an
    already-open socket where nothing times out -- and either path warms the other, so warming
    the simpler one here covers both.
    """
    import numpy as np

    t0 = time.time()
    try:
        # A voiced-band tone, not silence: the encoder may skip a silent clip and warm nothing.
        n = 16000
        tone = (0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000.0)).astype("float32")
        _offline_transcribe(tone)
        _p("warmup transcription took %.0fs" % (time.time() - t0))
    except Exception as e:
        # Serviceable either way; failing here only means the first caller pays after all.
        _p("WARN warmup transcription failed after %.0fs: %s" % (time.time() - t0, e))


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


def _detector_class():
    """The detector class this vLLM has, or None. Probed once, and says so once.

    Asked independently of the flags, because the backstop below turns on what this build
    CANNOT do rather than on what was requested.

    Not imported at module load: a vLLM without RepetitionDetectionParams must still serve,
    and the failure has to be one log line rather than an engine that will not start.
    """
    if not _repdet_class:
        try:
            from vllm.sampling_params import RepetitionDetectionParams

            _repdet_class.append(RepetitionDetectionParams)
        except Exception as e:
            # An older vLLM has no such class. Serving without the detector is what this
            # engine did before, so say it once and carry on rather than refusing to start.
            _p("WARN repetition detection unavailable in this vLLM (%s)" % e)
            _repdet_class.append(None)
    return _repdet_class[0]


def _repetition_params():
    """The configured detector, or None when it was not asked for or cannot run."""
    if not REPETITION_ON:
        return None
    if not _repdet_cache:
        cls = _detector_class()
        if cls is None:
            _repdet_cache.append(None)
        else:
            try:
                params = dict(REPETITION_DEFAULTS)
                if REPETITION_OVERRIDE:
                    params.update(json.loads(REPETITION_OVERRIDE))
                _repdet_cache.append(cls(**params))
                _p("repetition detection: %s%s"
                   % (params, " (overridden)" if REPETITION_OVERRIDE else ""))
            except Exception as e:
                # A malformed override is the operator's typo, not a reason to stop serving.
                _p("WARN --repetition-detection ignored (%s)" % e)
                _repdet_cache.append(None)
    return _repdet_cache[0]


def _apply_repetition(sp):
    """Sets the detector for one call and returns how to put the old value back, or None.

    None when nothing was set, so a build that will not take the attribute does not then
    fail again inside a finally block -- where the failure would replace the transcript.
    """
    params = _repetition_params()
    if params is None:
        return None
    old = getattr(sp, "repetition_detection", None)
    try:
        sp.repetition_detection = params
    except Exception as e:
        _p("WARN could not set repetition_detection (%s)" % e)
        return None

    def restore():
        sp.repetition_detection = old

    return restore


def _token_budget(seconds):
    # Two conditions, and both have to hold before the stock budget is replaced: protection
    # was asked for, and this build cannot supply it. Asking for nothing therefore leaves the
    # budget exactly where a deployment already has it, on either arch -- which is the whole
    # point of the flag being off by default. Keying this on "is the detector running" instead
    # would mean turning the detector off silently turned this on, so there would be no way
    # left to ask for the engine's own behaviour.
    if not REPETITION_ON or _detector_class() is not None:
        return OFFLINE_MAX_TOKENS
    return max(TOKENS_FLOOR,
               min(OFFLINE_MAX_TOKENS, int(seconds * TOKENS_PER_AUDIO_SEC) + TOKENS_FLOOR))


def _offline_transcribe(audio):
    # Native offline transcription on the same load; max_tokens is raised then restored.
    asr = _state["asr"]
    sp = getattr(asr, "sampling_params", None)
    old = getattr(sp, "max_tokens", None) if sp is not None else None
    restore_rep = None
    try:
        if sp is not None:
            sp.max_tokens = _token_budget(len(audio) / 16000.0)
            restore_rep = _apply_repetition(sp)
        results = asr.transcribe(audio=(audio, 16000), language=None, return_time_stamps=False)
    finally:
        if sp is not None and old is not None:
            sp.max_tokens = old
        if restore_rep is not None:
            restore_rep()
    r = results[0] if results else None
    t = getattr(r, "text", None) if r is not None else None
    if t is None and isinstance(r, dict):
        t = r.get("text")
    return (t or "").strip()


def _offline_transcribe_many(clips):
    # qwen-asr's transcribe() takes a list and hands the whole list to vLLM in one
    # generate() call: max_inference_batch_size defaults to -1 on the LLM factory, and
    # chunk_list yields the list unsplit for any non-positive size. Feeding it one clip at
    # a time is what kept vLLM from ever batching -- a 40 minute meeting arrived as four
    # HTTP requests and left as 214 single-sequence generate calls.
    asr = _state["asr"]
    sp = getattr(asr, "sampling_params", None)
    old = getattr(sp, "max_tokens", None) if sp is not None else None
    restore_rep = None
    try:
        if sp is not None:
            # One SamplingParams covers the whole call, so the budget follows the
            # longest clip in the batch.
            sp.max_tokens = _token_budget(max(len(c) for c in clips) / 16000.0)
            restore_rep = _apply_repetition(sp)
        results = asr.transcribe(audio=[(c, 16000) for c in clips],
                                 language=None, return_time_stamps=False)
    finally:
        if sp is not None and old is not None:
            sp.max_tokens = old
        if restore_rep is not None:
            restore_rep()
    texts = []
    for r in (results or []):
        t = getattr(r, "text", None)
        if t is None and isinstance(r, dict):
            t = r.get("text")
        texts.append((t or "").strip())
    if len(texts) != len(clips):
        raise RuntimeError("transcribe returned %d results for %d clips"
                           % (len(texts), len(clips)))
    return texts


def build_app(supports):
    has_stt = "stt" in supports
    has_stream = "stt_stream" in supports
    app = FastAPI(title="audio-stt-stream (Qwen3-ASR)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="stt_stream", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=has_stt)

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
                segs = parse_segments(segments)

                def _work_batch(ctx):
                    ctx.progress(stage="transcribe", done=0, total=len(segs))
                    if BATCH_ONE_SHOT:
                        # Slice every span first, then hand the whole list over once. The
                        # spans in one request are independent -- the caller batches them
                        # precisely because nothing downstream depends on their order.
                        out = [None] * len(segs)
                        spans = []
                        for i, seg in enumerate(segs):
                            lo = max(0, int(float(seg.get("start") or 0) * 16000))
                            hi = min(len(audio), int(float(seg.get("end") or 0) * 16000))
                            if hi <= lo:
                                out[i] = {"text": ""}
                            else:
                                ctx.meter(input_seconds=(hi - lo) / 16000.0)
                                spans.append((i, audio[lo:hi]))
                        if spans:
                            # The only checkpoint there can be: one generate() covers the
                            # whole batch, so a cancellation arriving mid-call is not seen
                            # until it returns. That is the cost of this path.
                            ctx.checkpoint()
                            try:
                                texts = _offline_transcribe_many([c for _, c in spans])
                                for (i, _), t in zip(spans, texts):
                                    out[i] = {"text": t}
                            except tasks.Cancelled:
                                raise
                            except Exception as e:
                                # No per-span outcome exists when the single call fails, so
                                # every span carries the same error and the caller retries
                                # the batch -- which is what it already does today.
                                for i, _ in spans:
                                    out[i] = {"error": "stt failed: %s" % e}
                        ctx.progress(done=len(segs), total=len(segs))
                        return {"model": MODEL_NAME, "mode": "stt", "batch": True, "results": out}
                    out = []
                    for i, seg in enumerate(segs, 1):
                        ctx.checkpoint()
                        try:
                            lo = max(0, int(float(seg.get("start") or 0) * 16000))
                            hi = min(len(audio), int(float(seg.get("end") or 0) * 16000))
                            if hi <= lo:
                                out.append({"text": ""})
                            else:
                                ctx.meter(input_seconds=(hi - lo) / 16000.0)
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
                ctx.meter(input_seconds=len(audio) / 16000.0)
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

            # prefix = text finalized by earlier rolls; samples resets on a roll, total never does.
            S = {"st": _new_state(), "prefix": "", "samples": 0, "total": 0}
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
                S["total"] += int(cur.shape[0])

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
                    seg = resample_linear(pcm16_to_float32(data), sample_rate)
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
                # We consumed the audio, so the closing frame — not the caller — reports its length.
                await ws.send_text(json.dumps({
                    "type": "closed",
                    "audio_seconds": round(S["total"] / 16000.0, 3),
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

    return app


def run(supports):
    _p("stt_stream starting; model=%s port=%s supports=%s" % (MODEL_REPO, PORT, supports))

    def load():
        try:
            _load_blocking()
        except Exception as e:
            _state["error"] = hfgate.explain(MODEL_REPO, e)
            _p("engine load FAILED: %s" % e)
            log.exception("engine load failed: %s", e)

    def build(served):
        app = build_app(served)
        _p("starting uvicorn on :%s (ready=%s)" % (PORT, _state["ready"]))
        return app

    # No server-initiated WS keepalive: bursty inference lags Pong and drops a healthy session.
    _runtime.serve(
        supports,
        load,
        build,
        "qwen-asr vLLM",
        load_on_main=True,
        disable_ws_ping=True,
    )
