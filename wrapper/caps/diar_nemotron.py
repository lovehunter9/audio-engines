# Nemotron 3 Diarization on a slim CUDA image: NeMo's Sortformer, not the NGC training container.
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import decode, pcm16_to_float32, resample_linear, unlink
from ..limits import Bounds
from ..runtime import Runtime
from . import diar_stream as stream

log = logging.getLogger("audio-diar-nemotron")

_runtime = Runtime(model=None, device="cpu")
MODEL_NAME = _runtime.model_name
# MODEL_SOURCE carries llm-init flags (--include ...) after the repo id.
MODEL_REPO = (_runtime.model_repo.split() or [""])[0]
PORT = _runtime.port

# Official input-buffer configs, 80 ms frames: (chunk, right context, fifo, update, speaker cache). Latency excludes compute.
_PRESETS = {
    "high": (340, 40, 40, 300, 264),       # 30.4s, the offline profile
    "low": (9, 4, 264, 222, 264),          # 1.04s
    "verylow": (6, 2, 264, 222, 264),      # 0.64s
    "ultralow": (3, 1, 264, 222, 264),     # 0.32s, lowest recommended
}

_args = EngineArgs()
_preset = (_args.text("--latency-preset", "high") or "high").strip().lower()
if _preset not in _PRESETS:
    log.warning("unknown latency preset %r; falling back to high", _preset)
    _preset = "high"
_base = _PRESETS[_preset]
CHUNK_LEN = _args.count("--chunk-len", _base[0])
RIGHT_CONTEXT = _args.count("--right-context", _base[1])
FIFO_LEN = _args.count("--fifo-len", _base[2])
UPDATE_PERIOD = _args.count("--update-period", _base[3])
SPKCACHE_LEN = _args.count("--spkcache-len", _base[4])
# The model marks word-level activity; 1.25s bridges it to turn-level on AMI, offline and streaming alike.
MIN_DURATION_OFF = _args.number("--min-duration-off", 1.25)
MIN_DURATION_ON = _args.number("--min-duration-on", 0.0)
# Claimed before warn_unclaimed. Offline memory does not grow with length, so the clip cap is 0.
BOUNDS = Bounds(_args, seconds=0, megabytes=0)
_args.warn_unclaimed(log)
try:
    MIN_DURATION_OFF = stream._seconds(MIN_DURATION_OFF, 1.25, "--min-duration-off")
    MIN_DURATION_ON = stream._seconds(MIN_DURATION_ON, 0.0, "--min-duration-on")
except ValueError as e:
    log.warning("%s; using 1.25 / 0", e)
    MIN_DURATION_OFF, MIN_DURATION_ON = 1.25, 0.0
_state = _runtime.state
_infer_lock = threading.Lock()


def _p(msg):
    print("[diar_nemotron] " + msg, flush=True)


def _bind_stream():
    # The socket runner lives in diar_stream and reads that module's globals.
    stream._state = _state
    stream._infer_lock = _infer_lock
    stream.MIN_DURATION_OFF = MIN_DURATION_OFF
    stream.MIN_DURATION_ON = MIN_DURATION_ON
    stream.PER_SPEAKER = True
    stream.BOUNDED_PREDS = True


def _find_nemo():
    cache = (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
             or "/cache/hf/hub")
    # Only this repo's snapshots: the hub cache is shared, and another model's .nemo would load silently.
    repo = "models--" + MODEL_REPO.replace("/", "--")
    hits = sorted(glob.glob(os.path.join(cache, repo, "snapshots", "*", "*.nemo")))
    return hits[0] if hits else None


def _load():
    _bind_stream()
    try:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        path = _find_nemo()
        if not path:
            raise RuntimeError(
                "no %s .nemo in the shared cache; llm-init downloads the weights and this "
                "engine loads them offline. Check that the model finished downloading and "
                "that HF_HUB_CACHE points at the same volume llm-init wrote to." % MODEL_REPO)
        _p("restoring Sortformer from cached .nemo: %s" % path)
        model = SortformerEncLabelModel.restore_from(
            restore_path=path, map_location=dev, strict=False)
        model.eval()
        sm = model.sortformer_modules
        sm.chunk_len = CHUNK_LEN
        sm.chunk_right_context = RIGHT_CONTEXT
        sm.fifo_len = FIFO_LEN
        sm.spkcache_update_period = UPDATE_PERIOD
        sm.spkcache_len = SPKCACHE_LEN
        check = getattr(model, "_check_streaming_parameters", None)
        if check is None:
            check = getattr(sm, "_check_streaming_parameters", None)
        if check is not None:
            try:
                check()
            except Exception as e:
                log.warning("streaming-parameter check skipped: %s", e)
        _state["n_spk"] = int(getattr(sm, "n_spk", 8) or 8)
        _state["subsampling"] = int(getattr(sm, "subsampling_factor", 8) or 8)
        _state["streaming_ok"] = bool(
            hasattr(sm, "init_streaming_state") and hasattr(sm, "streaming_feat_loader")
            and hasattr(model, "forward_streaming_step") and hasattr(model, "preprocessor"))
        # CUDA image leaves AUDIO_BASE=nemotron, so this import and the method swap never run there.
        ov_dev = None
        if (os.environ.get("AUDIO_BASE") or "").strip() == "nemotron-ov":
            from .diar_nemotron_ov import install as _install_ov

            ov_dev = _install_ov(model)
            # OpenVINO's "GPU" is not a torch device, so speaker-cache tensors stay on CPU.
            dev = "cpu"
        _state["model"], _state["device"], _state["ready"] = model, dev, True
        _p("streaming API present=%s n_spk=%d preset=%s (chunk=%d rc=%d fifo=%d up=%d cache=%d)"
           % (_state["streaming_ok"], _state["n_spk"], _preset,
              CHUNK_LEN, RIGHT_CONTEXT, FIFO_LEN, UPDATE_PERIOD, SPKCACHE_LEN))
        if ov_dev is None:
            _p("engine READY: %s on %s" % (MODEL_REPO, dev))
        else:
            _p("engine READY: %s on OpenVINO %s" % (MODEL_REPO, ov_dev))
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        _p("engine load FAILED: %s" % e)
        log.exception("engine load failed: %s", e)


def _mono16k(path):
    waveform, sr = decode(path)
    arr = waveform.detach().float().cpu().numpy()
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return resample_linear(arr, int(sr))


# Offline feeds this much decoded audio per step; the streamer keeps only its bounded window.
BLOCK_SEC = 60.0
# OpenVINO mel runs once per meeting, capped at one hour, because this checkpoint does not normalize per window.
OV_BLOCK_SEC = 3600.0


def _pcm_blocks(path, seconds=BLOCK_SEC):
    # Mono 16k float32 blocks straight out of ffmpeg, so no clip is ever decoded whole.
    if not shutil.which("ffmpeg"):
        yield _mono16k(path)
        return
    n = int(16000 * seconds) * 2
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
                              "-ar", "16000", "-f", "s16le", "-"],
                             stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                data = p.stdout.read(n)
                if not data:
                    break
                yield pcm16_to_float32(data)
        finally:
            p.stdout.close()
            rc = p.wait()
        if rc != 0:
            err.seek(0)
            raise RuntimeError("ffmpeg could not decode the upload: %s"
                               % err.read().decode("utf-8", "replace").strip()[-300:])


def _diarize_chunked(ctx, path, seconds, keep_scores=False):
    # Bounded streaming keeps memory flat. keep_scores is only for exclusive reconstruction.
    if (os.environ.get("AUDIO_BASE") or "").strip() == "nemotron-ov":
        from . import diar_nemotron_ov as ov_mod
        ov_mod.reset_clock()
        block_sec = OV_BLOCK_SEC
    else:
        ov_mod = None
        block_sec = BLOCK_SEC
    st = stream._Streamer(bounded=True, keep_scores=keep_scores)
    fed = 0
    held = None
    t_dec = time.perf_counter()
    for block in _pcm_blocks(path, seconds=block_sec):
        if ov_mod is not None:
            ov_mod.add("decode", time.perf_counter() - t_dec)
        # Hold the last block and step it as final, so one meeting pays for one mel.
        if held is not None:
            ctx.checkpoint()
            st.feed(held)
            fed += int(held.shape[0])
            with _infer_lock:
                st.step()
            if seconds:
                ctx.progress(ratio=min(0.99, fed / 16000.0 / seconds), stage="diarize")
        held = block
        t_dec = time.perf_counter()
    if held is not None:
        ctx.checkpoint()
        st.feed(held)
        fed += int(held.shape[0])
    with _infer_lock:
        st.step(final=True)
    if ov_mod is not None:
        _p("timing " + ov_mod.summary())
    scores = st.scores() if keep_scores else None
    return st.segments(), fed / 16000.0, scores, st.frame_sec


def _diarize_whole(path):
    # Only when this NeMo build lacks the step API: the model reads the clip as one array.
    audio = _mono16k(path)
    with _infer_lock:
        res = _state["model"].diarize(audio=[audio], batch_size=1, sample_rate=16000)
    return stream._parse_segments(res[0] if res else []), float(audio.shape[0]) / 16000.0


def build_app(supports):
    app = FastAPI(title="audio-nemotron (Nemotron 3 Diarization)")
    mount_metrics(app)
    register(app, model_name=MODEL_NAME, module="diar_nemotron", served=supports,
             repo=MODEL_REPO, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/diarization")
    async def diarize(file: UploadFile = File(...), num_speakers: str = Form(default=None),
                      exclusive: str = Form(default=None),
                      exclusive_method: str = Form(default=None),
                      min_duration_off: str = Form(default=None),
                      min_duration_on: str = Form(default=None),
                      async_: str = Form(default=None, alias="async")):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        try:
            off = stream._seconds(min_duration_off, MIN_DURATION_OFF, "min_duration_off")
            on = stream._seconds(min_duration_on, MIN_DURATION_ON, "min_duration_on")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        want_exclusive = tasks.truthy(exclusive)
        method = (exclusive_method or "").strip()
        if method and not want_exclusive:
            raise HTTPException(status_code=400, detail="exclusive_method requires exclusive=1")
        if want_exclusive and method not in ("", "score_aware_v1"):
            raise HTTPException(status_code=400,
                                detail="exclusive_method must be score_aware_v1")
        if want_exclusive and not _state.get("streaming_ok"):
            raise HTTPException(status_code=400,
                                detail="exclusive diarization needs the streaming step API")
        if num_speakers:
            log.info("num_speakers is a pyannote knob; this checkpoint ignores it")
        path, seconds = await BOUNDS.spill(file, "this deployment caps clip length")

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="diarize")
            if _state.get("streaming_ok"):
                raw, dur, scores, frame_sec = _diarize_chunked(
                    ctx, path, seconds, keep_scores=want_exclusive)
            else:
                raw, dur = _diarize_whole(path)
                scores, frame_sec = None, None
            ctx.meter(input_seconds=dur)
            if want_exclusive:
                ordinary = stream._tidy(raw, off, on)
                segs, evidence = stream.exclusive_timeline(scores, frame_sec, off, on)
            else:
                ordinary, evidence = None, None
                segs = stream._tidy(raw, off, on)
            speakers = sorted({s["speaker"] for s in segs})
            ctx.progress(ratio=1.0, stage="done")
            body = {"model": MODEL_NAME, "mode": "diar", "device": _state["device"],
                    "num_speakers": len(speakers), "speakers": speakers,
                    "num_segments": len(segs), "segments": segs, "exclusive": want_exclusive,
                    "min_duration_off": off, "min_duration_on": on}
            if want_exclusive:
                body["exclusive_method"] = "score_aware_v1"
                body["nonexclusive_segments"] = ordinary
                body["exclusive_evidence"] = {
                    "method": "score_aware_v1",
                    "frame_step_ms": round(frame_sec * 1000.0, 3),
                    "segments": evidence,
                }
            return body

        # The lock is taken per chunk, so an hours-long job does not stall live sockets.
        return await tasks.dispatch(async_, "diar", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="diarization failed")

    @app.websocket("/v1/audio/diarize/stream")
    async def diarize_stream(ws: WebSocket):
        _bind_stream()
        await ws.accept()
        if not _state["ready"]:
            await ws.send_text(json.dumps({"type": "error",
                                           "detail": _state["error"] or "model not ready"}))
            await ws.close()
            return
        await ws.send_text(json.dumps({"type": "ready"}))
        try:
            if _state.get("streaming_ok"):
                try:
                    await stream._run_streaming(ws)
                    return
                except stream._FallbackToWindow as fb:
                    _p("streaming self-test failed (%s); using window .diarize() fallback" % fb)
            await stream._run_window(ws)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("diar_nemotron stream error: %s", e)
            try:
                await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                await ws.close()
            except Exception:
                pass

    return app


def run(supports):
    _p("diar_nemotron starting; model=%s port=%s preset=%s" % (MODEL_REPO, PORT, _preset))
    extra = {}
    if (os.environ.get("AUDIO_BASE") or "").strip() == "nemotron-ov":
        # First boot exports the IRs. The CUDA image does not take this branch, so its 1800s stays.
        extra["timeout_s"] = 7200.0
    _runtime.serve(supports, _load, build_app, "nemotron diarization",
                   disable_ws_ping=True, **extra)
