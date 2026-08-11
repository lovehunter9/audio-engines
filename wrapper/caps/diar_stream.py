# Streaming speaker diarization on NVIDIA Streaming Sortformer; wire protocol in the README.
import os
import json
import glob
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .. import hfgate
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..audioio import pcm16_to_float32, resample_linear
from ..runtime import Runtime

log = logging.getLogger("audio-diar-stream")

_runtime = Runtime(
    "diar-streaming-sortformer",
    default_repo="nvidia/diar_streaming_sortformer_4spk-v2.1",
    model=None,
    device="cpu",
)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
PORT = _runtime.port

# The card's presets in 80ms frames; `high` (~10s) resolves adjacent turns and is ~18x cheaper.
_PRESETS = {
    "low": (6, 7, 188, 144, 188),
    "high": (124, 1, 124, 124, 188),
    "veryhigh": (340, 40, 40, 300, 188),
}

_args = EngineArgs()
_preset = (_args.text("--latency-preset", "high") or "high").strip().lower()
if _preset not in _PRESETS:
    log.warning("unknown latency preset %r; falling back to high", _preset)
    _preset = "high"
_base = _PRESETS[_preset]

# Sortformer's own streaming attributes, each free to deviate from the preset it started at.
CHUNK_LEN = _args.count("--chunk-len", _base[0])
RIGHT_CONTEXT = _args.count("--right-context", _base[1])
FIFO_LEN = _args.count("--fifo-len", _base[2])
UPDATE_PERIOD = _args.count("--update-period", _base[3])
SPKCACHE_LEN = _args.count("--spkcache-len", _base[4])
_args.warn_unclaimed(log)

STEP_SEC = 2.0   # partial every N s of audio
MIN_SEC = 1.0    # fallback: minimum to work on
# Fallback only: diarize just the last WINDOW_SEC, or a session would cost O(n^2) and grow forever.
WINDOW_SEC = 60.0
OVERLAP_SEC = 12.0

_state = _runtime.state
# Sortformer's diarize() is blocking and not concurrency-safe, so inference is serialized.
_infer_lock = asyncio.Lock()


def _p(msg):
    print("[diar_stream] " + msg, flush=True)


def _find_nemo():
    # Locate the llm-init-downloaded .nemo in the shared HF cache (offline load).
    cache = (os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
             or "/cache/hf/hub")
    repo = "models--" + MODEL_REPO.replace("/", "--")
    for pat in (os.path.join(cache, repo, "snapshots", "*", "*.nemo"),
                os.path.join(cache, "**", "*.nemo")):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def _load():
    try:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        path = _find_nemo()
        if path:
            _p("restoring Sortformer from cached .nemo: %s" % path)
            model = SortformerEncLabelModel.restore_from(
                restore_path=path, map_location=dev, strict=False)
        else:
            _p("no cached .nemo found; from_pretrained(%s) (needs network/token)" % MODEL_REPO)
            model = SortformerEncLabelModel.from_pretrained(MODEL_REPO, map_location=dev)
        model.eval()
        sm = model.sortformer_modules
        sm.chunk_len = CHUNK_LEN
        sm.chunk_right_context = RIGHT_CONTEXT
        sm.fifo_len = FIFO_LEN
        sm.spkcache_update_period = UPDATE_PERIOD
        sm.spkcache_len = SPKCACHE_LEN
        try:
            sm._check_streaming_parameters()
        except Exception as e:
            log.warning("streaming-parameter check skipped: %s", e)
        # Probe the incremental API; if any piece is missing, the window .diarize() path takes over.
        _state["n_spk"] = int(getattr(sm, "n_spk", 4) or 4)
        _state["subsampling"] = int(getattr(sm, "subsampling_factor", 8) or 8)
        _state["streaming_ok"] = bool(
            hasattr(sm, "init_streaming_state") and hasattr(sm, "streaming_feat_loader")
            and hasattr(model, "forward_streaming_step") and hasattr(model, "preprocessor"))
        _state["model"], _state["device"], _state["ready"] = model, dev, True
        _p("streaming API present=%s n_spk=%d subsampling=%d (methods: %s)"
           % (_state["streaming_ok"], _state["n_spk"], _state["subsampling"],
              ",".join(sorted(m for m in ("init_streaming_state", "streaming_feat_loader")
                              if hasattr(sm, m)))))
        _p("engine READY: %s on %s (chunk=%d rc=%d fifo=%d up=%d cache=%d)"
           % (MODEL_REPO, dev, CHUNK_LEN, RIGHT_CONTEXT, FIFO_LEN, UPDATE_PERIOD, SPKCACHE_LEN))
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        _p("engine load FAILED: %s" % e)
        log.exception("engine load failed: %s", e)


def _spk_name(spk):
    # Normalize speaker labels to spk_<n> ("speaker_2" / 2 / "2" -> "spk_2").
    s = str(spk).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return ("spk_" + digits) if digits else (s or "spk_0")


def _parse_segments(raw):
    # diarize() yields "start end speaker"; tolerate tuples/objects so a NeMo bump can't break us.
    out = []
    for seg in (raw or []):
        s = e = spk = None
        if isinstance(seg, str):
            parts = seg.split()
            if len(parts) >= 3:
                s, e, spk = parts[0], parts[1], " ".join(parts[2:])
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            s, e, spk = seg[0], seg[1], seg[2]
        else:
            s = getattr(seg, "start", None)
            e = getattr(seg, "end", None)
            spk = getattr(seg, "speaker", None) or getattr(seg, "label", None)
        if s is None or e is None:
            continue
        try:
            out.append({"start": round(float(s), 3), "end": round(float(e), 3),
                        "speaker": _spk_name(spk)})
        except Exception:
            continue
    return out


def _merge_segments(segs, gap=0.8):
    # Collapse adjacent same-speaker turns, bridging short gaps, to keep the timeline clean.
    s2 = sorted(segs, key=lambda x: (x["start"], x["end"]))
    out = []
    for s in s2:
        if out and out[-1]["speaker"] == s["speaker"] and s["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append({"start": round(float(s["start"]), 3),
                        "end": round(float(s["end"]), 3), "speaker": s["speaker"]})
    return out


def _preds_to_segments(preds, frame_sec, thr=0.5):
    # preds [T, n_spk] of activity per 80ms frame; AOSC pins column k to one speaker, so no remap.
    out = []
    if preds is None or getattr(preds, "size", 0) == 0:
        return out
    T = preds.shape[0]
    S = preds.shape[1] if preds.ndim > 1 else 1
    act = preds >= thr
    for s in range(S):
        col = act[:, s]
        i = 0
        while i < T:
            if col[i]:
                j = i
                while j + 1 < T and col[j + 1]:
                    j += 1
                out.append({"start": round(i * frame_sec, 3),
                            "end": round((j + 1) * frame_sec, 3),
                            "speaker": "spk_%d" % s})
                i = j + 1
            else:
                i += 1
    return _merge_segments(out)


def _diarize(buf):
    # Fallback path: float32 mono @16k in, segments relative to buf[0] out, O(n) per call.
    model = _state["model"]
    res = model.diarize(audio=[buf], batch_size=1, sample_rate=16000)
    return _parse_segments(res[0] if res else [])


class _FallbackToWindow(Exception):
    # Raised by the pre-flight self-test, so the switch happens before any client audio arrives.
    pass


async def _send_closed(ws, total_samples):
    # We consumed the audio, so the closing frame — not the caller — reports its length.
    await ws.send_text(json.dumps({
        "type": "closed",
        "audio_seconds": round(total_samples / 16000.0, 3),
    }))


async def _run_streaming(ws):
    # Features must span a CONTIGUOUS window: per-block normalization splits one speaker into two.
    import numpy as np
    import torch

    model = _state["model"]
    sm = model.sortformer_modules
    device = _state["device"]
    n_spk = _state["n_spk"]
    sub = _state["subsampling"]
    frame_sec = sub * 0.01                      # 80ms output frame (sub * 10ms hop)
    hop = 160                                   # samples per 10ms preprocessor frame @16k
    CS = max(1, int(sm.chunk_len) * sub)        # feat frames committed per chunk
    CRs = int(getattr(sm, "chunk_right_context", 0)) * sub   # feat frames of right context
    chunk_audio = CS * hop                       # samples committed per chunk
    rc_frames = CRs
    step_samples = max(8000, int(STEP_SEC * 16000))          # process/emit cadence
    # Chunk-aligned window: enough for stable normalization plus left context, trimmed each step.
    keep_chunks = max(8, int(WINDOW_SEC * 16000) // chunk_audio)
    sample_rate = 16000
    buf = np.zeros((0,), dtype="float32")
    ss = {"state": None, "preds": None}
    st = {"base_chunks": 0, "committed": 0}      # chunks dropped from buf front / total fed

    def _init_stream():
        ss["state"] = sm.init_streaming_state(batch_size=1, async_streaming=True, device=device)
        ss["preds"] = torch.zeros((1, 0, n_spk), device=device)
        st["base_chunks"] = 0
        st["committed"] = 0

    def _process(final=False):
        # Feed each new chunk that has full right context; offset stays 0 since state carries history.
        nonlocal buf
        if buf.size < hop:
            return
        sig = torch.as_tensor(buf, dtype=torch.float32, device=device).unsqueeze(0)
        slen = torch.tensor([sig.shape[1]], device=device)
        with torch.inference_mode():
            proc, proc_len = model.preprocessor(input_signal=sig, length=slen)
            T = int(proc.shape[2])
            zoff = torch.zeros((proc.shape[0],), dtype=torch.long, device=device)
            loader = sm.streaming_feat_loader(feat_seq=proc, feat_seq_length=proc_len,
                                              feat_seq_offset=zoff)
            local_fed = st["committed"] - st["base_chunks"]   # leading chunks already fed
            for _tup in loader:
                i, chunk_feat, feat_lengths, lo, ro = _tup[0], _tup[1], _tup[2], _tup[3], _tup[4]
                if i < local_fed:
                    continue                       # already committed in a previous step
                if not final and ((i + 1) * CS + rc_frames > T):
                    break                          # right context not fully arrived yet
                ss["state"], ss["preds"] = model.forward_streaming_step(
                    processed_signal=chunk_feat, processed_signal_length=feat_lengths,
                    streaming_state=ss["state"], total_preds=ss["preds"],
                    left_offset=lo, right_offset=ro)
                st["committed"] += 1
        # Trim behind the committed frontier, chunk-aligned so the feature grid never shifts.
        max_base = st["committed"] - keep_chunks
        if max_base > st["base_chunks"]:
            drop = max_base - st["base_chunks"]
            buf = buf[drop * chunk_audio:]
            st["base_chunks"] = max_base

    def _selftest():
        # Run the pipeline on silence, then reset, so the probe cannot shift the real session.
        nonlocal buf
        _init_stream()
        buf = np.zeros((CS * 3 * hop,), dtype="float32")
        _process(final=True)
        buf = np.zeros((0,), dtype="float32")
        _init_stream()

    def _segments():
        p = ss["preds"]
        if p is None or p.shape[1] == 0:
            return []
        arr = p[0].detach().float().cpu().numpy()
        return _preds_to_segments(arr, frame_sec)

    async def _send(kind):
        segs = await asyncio.to_thread(_segments)
        speakers = sorted({s["speaker"] for s in segs})
        await ws.send_text(json.dumps({"type": kind, "segments": segs, "speakers": speakers}))

    # Pre-flight: prove the streaming API works BEFORE consuming client audio.
    async with _infer_lock:
        try:
            await asyncio.to_thread(_selftest)
        except Exception as e:
            raise _FallbackToWindow("%s: %s" % (type(e).__name__, e))
    _p("streaming path active (chunk=%d frames rc=%d keep=%d chunks frame=%.3fs n_spk=%d)"
       % (CS, rc_frames, keep_chunks, frame_sec, n_spk))

    new_samples = 0
    total_new = 0
    last_emit = 0
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
                sample_rate = int(obj.get("sample_rate") or 16000)
                continue
            if t in ("stop", "done", "finish"):
                break
            continue
        data = msg.get("bytes")
        if not data:
            continue
        seg = resample_linear(pcm16_to_float32(data), sample_rate)
        buf = np.concatenate([buf, seg]) if buf.size else seg
        new_samples += int(seg.shape[0])
        total_new += int(seg.shape[0])
        if new_samples >= step_samples:
            new_samples = 0
            async with _infer_lock:
                await asyncio.to_thread(_process)
            if total_new - last_emit >= step_samples:
                last_emit = total_new
                await _send("partial")
    # Feed the remaining chunks, accepting rc=0 on the trailing one.
    async with _infer_lock:
        await asyncio.to_thread(_process, True)
    await _send("final")
    await _send_closed(ws, total_new)
    await ws.close()


async def _run_window(ws):
    # Fallback: re-diarize the active window each step — slower, but never "no output".
    import numpy as np

    sample_rate = 16000
    buf = np.zeros((0,), dtype="float32")   # only the active window, [base_offset, now]
    since = 0                                # samples fed since last diarize
    total = 0                                # whole session, for the closing report
    step_samples = max(16000, int(STEP_SEC * 16000))
    min_samples = max(8000, int(MIN_SEC * 16000))
    window_samples = max(step_samples * 2, int(WINDOW_SEC * 16000))
    overlap_samples = min(window_samples - step_samples, max(0, int(OVERLAP_SEC * 16000)))
    base_offset = 0.0        # ABS time (s) of buf[0]
    committed = []           # frozen ABS canonical segments from rolled-off audio
    overlap_ref = []         # ABS canonical segs over the retained tail (remap ref)
    win_map = {}             # local spk -> canonical spk for the CURRENT window
    canon = {"n": 0}         # next canonical speaker index

    def _relabel(local_abs):
        # Window labels are AOSC-stable, so after a roll anchor them by max overlap with the tail.
        order, by_spk = [], {}
        for s in local_abs:
            if s["speaker"] not in by_spk:
                order.append(s["speaker"])
            by_spk.setdefault(s["speaker"], []).append(s)
        used = set(win_map.values())
        for L in order:
            if L in win_map:
                continue
            C = None
            if overlap_ref:
                best = 0.0
                cand_set = {r["speaker"] for r in overlap_ref if r["speaker"] not in used}
                for cand in cand_set:
                    ov = 0.0
                    for a in by_spk[L]:
                        for r in overlap_ref:
                            if r["speaker"] != cand:
                                continue
                            ov += max(0.0, min(a["end"], r["end"]) - max(a["start"], r["start"]))
                    if ov > best:
                        best, C = ov, cand
            if C is None:
                C = "spk_%d" % canon["n"]
                canon["n"] += 1
            win_map[L] = C
            used.add(C)
        return [{"start": s["start"], "end": s["end"], "speaker": win_map[s["speaker"]]}
                for s in local_abs]

    async def _current():
        # Diarize only the active window, returning canonical absolute-time segments.
        if buf.shape[0] < min_samples:
            return []
        async with _infer_lock:
            local = await asyncio.to_thread(_diarize, buf)   # times relative to buf[0]
        local_abs = [{"start": s["start"] + base_offset, "end": s["end"] + base_offset,
                      "speaker": s["speaker"]} for s in local]
        return _relabel(local_abs)

    def _roll(cur):
        # Freeze turns before the boundary, keep the last OVERLAP_SEC, stash its labels to remap by.
        nonlocal buf, base_offset, overlap_ref, win_map, committed
        if buf.shape[0] < window_samples or overlap_samples <= 0:
            return
        roll_point = buf.shape[0] - overlap_samples
        boundary = base_offset + roll_point / 16000.0
        for s in cur:
            if s["start"] < boundary:
                committed.append({"start": s["start"], "end": min(s["end"], boundary),
                                  "speaker": s["speaker"]})
        committed = _merge_segments(committed)
        overlap_ref = [{"start": max(s["start"], boundary), "end": s["end"],
                        "speaker": s["speaker"]} for s in cur if s["end"] > boundary]
        base_offset = boundary
        buf = buf[roll_point:]
        win_map = {}   # rebuilt on the next _current() via overlap_ref

    async def _send(kind, cur):
        full = _merge_segments(committed + cur)
        speakers = sorted({s["speaker"] for s in full})
        await ws.send_text(json.dumps({"type": kind, "segments": full, "speakers": speakers}))

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
                sample_rate = int(obj.get("sample_rate") or 16000)
                continue
            if t in ("stop", "done", "finish"):
                break
            continue
        data = msg.get("bytes")
        if not data:
            continue
        seg = resample_linear(pcm16_to_float32(data), sample_rate)
        buf = np.concatenate([buf, seg]) if buf.size else seg
        since += int(seg.shape[0])
        total += int(seg.shape[0])
        if since >= step_samples:
            since = 0
            cur = await _current()
            await _send("partial", cur)
            _roll(cur)   # roll AFTER emitting, reusing the just-computed segments
    await _send("final", await _current())
    await _send_closed(ws, total)
    await ws.close()


def build_app(supports):
    app = FastAPI(title="audio-diar-stream (Streaming Sortformer)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="diar_stream", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"])

    @app.websocket("/v1/audio/diarize/stream")
    async def diarize_stream(ws: WebSocket):
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
                    await _run_streaming(ws)
                    return
                except _FallbackToWindow as fb:
                    _p("streaming self-test failed (%s); using window .diarize() fallback" % fb)
            await _run_window(ws)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.exception("diar_stream error: %s", e)
            try:
                await ws.send_text(json.dumps({"type": "error", "detail": str(e)}))
                await ws.close()
            except Exception:
                pass

    return app


def run(supports):
    _p("diar_stream starting; model=%s port=%s" % (MODEL_REPO, PORT))
    # No server-initiated WS keepalive: bursty inference lags Pong and drops a healthy session.
    _runtime.serve(
        supports,
        _load,
        build_app,
        "streaming sortformer",
        disable_ws_ping=True,
    )
