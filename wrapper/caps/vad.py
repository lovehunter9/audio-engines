# Voice activity detection with Silero VAD.
import asyncio
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics
from ..contract import register
from ..audioio import decode_mono
from ..runtime import Runtime

log = logging.getLogger("audio-vad")

_runtime = Runtime(model=None, get_ts=None)
MODEL_NAME = _runtime.model_name
SR = 16000


# Defaults tuned for music/singing, each overridable per-request via form.
VAD_THRESHOLD = 0.3
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 200
VAD_MIN_SPEECH_MS = 250
VAD_MAX_SPEECH_S = 30

_state = _runtime.state


def _load():
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps

        _state["model"] = load_silero_vad()
        _state["get_ts"] = get_speech_timestamps
        _state["ready"] = True
        log.info("silero-vad loaded; ready")
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_NAME, e)
        log.exception("vad load failed: %s", e)


def build_app(supports):
    app = FastAPI(title="audio-vad (silero)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="vad", served=supports,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/vad")
    async def vad(
        file: UploadFile = File(...),
        threshold: str = Form(default=None),
        min_silence_ms: str = Form(default=None),
        speech_pad_ms: str = Form(default=None),
        min_speech_ms: str = Form(default=None),
        max_speech_s: str = Form(default=None),
        async_: str = Form(default=None, alias="async"),
    ):
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not ready")
        data = await file.read()

        def _decode():
            return decode_mono(data, SR).squeeze(0).contiguous()

        try:
            wav = await asyncio.to_thread(_decode)
        except Exception as e:
            raise HTTPException(status_code=400, detail="audio decode failed: %s" % e)

        def _pick(form_val, default):
            if form_val is None or form_val == "":
                return default
            try:
                return float(form_val)
            except Exception:
                return default

        thr = _pick(threshold, VAD_THRESHOLD)
        max_s = _pick(max_speech_s, VAD_MAX_SPEECH_S)
        kw = {
            "sampling_rate": SR,
            "threshold": thr,
            "min_silence_duration_ms": int(_pick(min_silence_ms, VAD_MIN_SILENCE_MS)),
            "speech_pad_ms": int(_pick(speech_pad_ms, VAD_SPEECH_PAD_MS)),
            "min_speech_duration_ms": int(_pick(min_speech_ms, VAD_MIN_SPEECH_MS)),
        }
        # silero treats max_speech_duration_s=inf as "no cap"; only pass a finite cap.
        if max_s and max_s > 0:
            kw["max_speech_duration_s"] = max_s
        log.info("vad params: %s", {k: v for k, v in kw.items() if k != "sampling_rate"})

        def _work(ctx):
            ctx.meter(input_seconds=int(wav.shape[-1]) / SR)
            ctx.progress(ratio=0.0, stage="inference")
            # silero get_speech_timestamps returns [{'start','end'}] in SAMPLES.
            ts = _state["get_ts"](wav, _state["model"], **kw)
            segs = [{"start": round(t["start"] / SR, 3), "end": round(t["end"] / SR, 3)}
                    for t in ts]
            speech = round(sum(s["end"] - s["start"] for s in segs), 3)
            ctx.progress(ratio=1.0, stage="done")
            return {
                "model": MODEL_NAME,
                "mode": "vad",
                "sampling_rate": SR,
                "duration": round(int(wav.shape[-1]) / SR, 3),
                "speech_seconds": speech,
                "num_segments": len(segs),
                "params": {k: v for k, v in kw.items() if k != "sampling_rate"},
                "segments": segs,
            }

        return await tasks.dispatch(async_, "vad", MODEL_NAME, _work, fail="vad failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "silero-vad")
