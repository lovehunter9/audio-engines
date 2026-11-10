# SimulS2ST-Omni simultaneous speech-to-speech. One in-process agent, no child engine.
#
# The streaming policy is the upstream speech agent. A finished utterance is that same policy
# with source_finished set. Text is the transcript of the same pass, not a second model.
#
# Languages are English and Chinese, and the two sides differ, so one of them is Chinese.
# The pair is per request: the system prompt is rebuilt under the generation lock. Weights stay.
import asyncio
import logging
import os
import threading

from fastapi import (
    FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect,
)

from .. import hfgate
from .. import tasks
from ..audioio import decode_mono, pcm16_to_float32, resample_linear, seconds
from ..contract import EngineArgs, register
from ..gpu import mount_metrics
from ..runtime import Runtime

log = logging.getLogger("audio-simuls2st")

_runtime = Runtime(model=None, dtype=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
LATENCY = _args.count("--latency-multiplier", 2)
BEAMS = _args.count("--num-beams", 1)
MIN_START = _args.number("--min-start-sec", 1.0)
SOURCE_LANG = str(_args.text("--source-lang", "English") or "English")
TARGET_LANG = str(_args.text("--target-lang", "Chinese") or "Chinese")
if not 1 <= LATENCY <= 8:
    log.warning("--latency-multiplier %s is outside 1..8; using 2", LATENCY)
    LATENCY = 2
if not 1 <= BEAMS <= 4:
    log.warning("--num-beams %s is outside 1..4; using 1", BEAMS)
    BEAMS = 1

IN_RATE = 16000
OUT_RATE = 24000
BOOT_TIMEOUT_S = 3600.0

# The simultaneous checkpoint is trained on this pair. Anything else is a 400, not a guess.
_LANGS = {
    "en": "English", "english": "English", "en-us": "English", "en-gb": "English", "eng": "English",
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese", "chinese": "Chinese",
    "cn": "Chinese", "zhs": "Chinese", "zht": "Chinese",
}

_state = _runtime.state
_lock = threading.Lock()


def _model_path():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _lang(raw, field):
    text = str(raw or "").strip()
    name = _LANGS.get(text.lower())
    if name is None and text in ("English", "Chinese"):
        name = text
    if name not in ("English", "Chinese"):
        raise HTTPException(status_code=400,
                            detail="%s must be English or Chinese" % field)
    return name


def _pair(src, tgt):
    source = _lang(src or SOURCE_LANG, "source_lang")
    target = _lang(tgt or TARGET_LANG, "target_lang")
    if source == target:
        raise HTTPException(status_code=400,
                            detail="source_lang and target_lang must differ, "
                                   "and one of them must be Chinese")
    return source, target


def _boot():
    try:
        _load()
        log.info("%s loaded: latency_multiplier=%d thinker_beams=%d %s -> %s out=%dHz",
                 MODEL_NAME, LATENCY, BEAMS, SOURCE_LANG, TARGET_LANG, OUT_RATE)
        _state.update(ready=True, error=None)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("simuls2st engine failed to start: %s", e)


def _load():
    import sys
    import torch
    from argparse import Namespace

    import tempfile

    root = os.environ.get("SIMULS2ST_ROOT", "/opt/simuls2st")
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.agents.simuleval_omni_talker_s2st_agent import OmniTalkerStreamingS2STAgent

    if not torch.cuda.is_available():
        raise RuntimeError("SimulS2ST CUDA engine requires a CUDA device")
    snap = _model_path()
    needed = (
        os.path.join(snap, "offline"),
        os.path.join(snap, "simuls2st_adapter"),
        os.path.join(snap, "voicebox", "voicebox.safetensors"),
        os.path.join(snap, "voicebox", "vocos.safetensors"),
        os.path.join(snap, "voicebox", "voicebox_config.json"),
        os.path.join(snap, "dualcodec", "dualcodec.safetensors"),
        os.path.join(snap, "dualcodec", "w2v_bert_stats.pt"),
        os.path.join(snap, "w2v"),
    )
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise RuntimeError("snapshot %s is missing %s" % (snap, ", ".join(missing)))
    os.environ["SIMULS2ST_MODEL_ROOT"] = snap
    os.environ["SIMULS2ST_W2V_BERT_PATH"] = os.path.join(snap, "w2v")
    dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    side = tempfile.mkdtemp(prefix="simuls2st-")
    agent = OmniTalkerStreamingS2STAgent(Namespace(
        model_name_or_path=os.path.join(snap, "offline"),
        checkpoint_path=os.path.join(snap, "simuls2st_adapter"),
        adapter_name_or_path=os.path.join(snap, "simuls2st_adapter"),
        device="cuda:0",
        dtype=dtype,
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        latency_multiplier=LATENCY,
        min_start_sec=float(MIN_START),
        min_final_chunk_sec=0.32,
        thinker_max_new_tokens=50,
        thinker_do_sample=False,
        thinker_temperature=1.0,
        thinker_top_p=0.8,
        thinker_top_k=20,
        thinker_num_beams=BEAMS,
        thinker_repetition_penalty=1.1,
        thinker_no_repeat_ngram_size=0,
        no_thinker_kv_cache=False,
        talker_max_new_tokens=125,
        talker_do_sample=False,
        talker_temperature=1.0,
        talker_top_p=0.8,
        talker_top_k=20,
        talker_repetition_penalty=1.4,
        talker_no_repeat_ngram_size=0,
        no_talker_kv_cache=False,
        enable_wait_silence_decode=False,
        use_prompt_mel=True,
        prompt_source_chunks=2,
        prompt_generated_chunks=1,
        prompt_source_sec=4.0,
        prompt_generated_sec=2.0,
        enable_history_window=True,
        history_window_turns=30,
        history_overlap_turns=8,
        voicebox_path=None,
        vocos_path=None,
        voicebox_config_path=None,
        codec_model_path=None,
        codec_stats_path=None,
        output=side,
        source="",
    ))
    _state.update(agent=agent, dtype=dtype)


def _bind(agent, source, target):
    if agent.source_lang == source and agent.target_lang == target:
        return
    agent.source_lang = source
    agent.target_lang = target
    agent.system_instruction = agent._build_system_instruction()
    agent._system_token_ids = agent.tokenizer(
        agent.system_instruction, add_special_tokens=False).input_ids


def _new_states():
    states = _state["agent"].build_states()
    states.source = []
    states.source_sample_rate = IN_RATE
    states.source_finished = False
    return states


def _pcm16(samples):
    import numpy as np

    audio = np.asarray(samples, dtype="float32")
    if audio.size == 0:
        return b""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _drain(states, source, target, finished):
    """Run the official speech policy until it needs more audio, or the utterance is done.

    A write carries 24 kHz samples. The translated words are on the state, not on the action.
    ReadAction after a consumed chunk means the model waited; audio still in `states.source`
    has to be offered. ReadAction with no progress means it needs more input.
    """
    agent = _state["agent"]
    with _lock:
        _bind(agent, source, target)
        states.source_sample_rate = IN_RATE
        states.source_finished = bool(finished)
        events = []
        done = False
        steps = 0
        limit = max(8, len(states.source) // IN_RATE + 8)
        while steps < limit:
            steps += 1
            before = states.processed_samples
            n_text = len(getattr(states, "pred_text_chunks", []) or [])
            action = agent.policy(states)
            for piece in (getattr(states, "pred_text_chunks", []) or [])[n_text:]:
                if piece:
                    events.append(("text", piece))
            if type(action).__name__ == "WriteAction":
                content = getattr(action, "content", None)
                samples = getattr(content, "content", None)
                if samples:
                    pcm = _pcm16(samples)
                    if pcm:
                        events.append(("pcm", pcm))
                if getattr(action, "finished", False):
                    done = True
                    break
            if states.processed_samples == before:
                break
        else:
            log.warning("policy stopped after %d steps with %d samples", steps, len(states.source))
        return events, done


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
    app = FastAPI(title="audio-simuls2st (SimulS2ST-Omni)")
    mount_metrics(app)
    register(app, model_name=MODEL_NAME, module="simuls2st", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=OUT_RATE)
    _args.warn_unclaimed(log)

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    @app.post("/v1/audio/translations")
    async def translations(request: Request, file: UploadFile = File(...),
                           model: str = Form(default=None),
                           response_format: str = Form(default="json"),
                           source_lang: str = Form(default=None),
                           target_lang: str = Form(default=None)):
        """Whole utterance through the speech agent. JSON returns text and 24 kHz wav."""
        _require_ready()
        fmt = str(response_format or "json").strip().lower()
        if fmt not in ("json", "text", "wav", "pcm"):
            raise HTTPException(status_code=400,
                                detail="response_format must be json, text, wav, or pcm")
        source, target = _pair(source_lang, target_lang)
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="file is empty")

        def _work(ctx):
            samples = _samples(data)
            ctx.meter(input_seconds=seconds(samples, IN_RATE))
            ctx.progress(ratio=0.0, stage="translate")
            states = _new_states()
            states.source = samples
            events, _done = _drain(states, source, target, True)
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

        return await tasks.dispatch(request.query_params.get("async"), "audio_s2s", MODEL_NAME,
                                    _work, fail="translation failed")

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
        await ws.send_json({
            "event": "ready", "sample_rate": OUT_RATE, "input_sample_rate": IN_RATE})
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
                        text, source, target, states, rate)
                    if stop:
                        events, _done = await asyncio.to_thread(
                            _drain, states, source, target, True)
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
        if states.processed_samples or states.source:
            states = _new_states()
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
    _runtime.serve(supports, _boot, build_app, "SimulS2ST-Omni engine", timeout_s=BOOT_TIMEOUT_S)
