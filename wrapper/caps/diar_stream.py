# Streaming / online speaker diarization via NVIDIA Streaming Sortformer on the
# NeMo runtime. Ported from the tested diar_stream.py; deps baked at build time;
# contract surface via wrapper.gpu + wrapper.contract.
#
# It is its own capability (NOT diar + a flag) because streaming diar must keep
# speaker labels CONSISTENT across time at bounded latency, which Sortformer
# solves end-to-end with an Arrival-Order Speaker Cache (AOSC) numbering
# speakers by arrival time (deterministic on a given prefix, so spk_0/spk_1/..
# are stable across steps).
#
# UNBOUNDED LENGTH via TWO PATHS:
#   1. TRUE STREAMING (primary) keeps ONE streaming_state and feeds only
#      NEWLY-arrived chunk-aligned audio through the incremental API
#      (init_streaming_state + streaming_feat_loader + forward_streaming_step);
#      total_preds accumulates per-80ms-frame activity for the WHOLE session,
#      compute is O(1)/chunk, AOSC keeps column k == one speaker so there is no
#      window/overlap/relabel and memory stays bounded.
#   2. BOUNDED WINDOW (_run_window fallback) re-runs .diarize() over the last
#      WINDOW_SEC, COMMITs older turns, and carries an OVERLAP_SEC tail so the
#      next window's labels are remapped by time overlap — slower (O(n)/step)
#      but correct, so diar_stream never regresses to "no output".
# The path is chosen ONCE per connection BEFORE any client audio (the self-test
# runs on silence). Both use ONLY public NeMo API on SortformerEncLabelModel +
# sortformer_modules (no monkey-patching).
#
# Wire protocol on WS /v1/audio/diarize/stream: client sends optional
# {"type":"start","sample_rate":16000}, then BINARY PCM16LE mono chunks, then
# {"type":"stop"} (or close); server sends {"type":"ready"},
# {"type":"partial"|"final","segments":[{"start","end","speaker"}],"speakers":[...]},
# {"type":"error"}. Fusing these turns with ASR text is the CONSUMER's job.
import os
import json
import glob
import asyncio
import logging
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from ..gpu import mount_metrics
from ..contract import register

LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").lower()
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("audio-diar-stream")

MODEL_NAME = os.environ.get("MODEL_NAME", "diar-streaming-sortformer")
_src = os.environ.get("MODEL_SOURCE", "")
MODEL_REPO = _src[5:] if _src.startswith("hf://") else (
    _src or "nvidia/diar_streaming_sortformer_4spk-v2.1")
PORT = int(os.environ.get("WRAPPER_PORT", "8000"))

# Streaming knobs (frames = 80ms). Defaults follow the card's HIGH-LATENCY preset
# (~10s input latency, RTF~0.005). Rationale: transcription (a separate stt_stream
# engine) stays real-time; for SPEAKERS accuracy matters more than immediacy, and a
# bigger chunk lets the model attend over a whole ~10s span at once -> it resolves
# RAPID adjacent turns (e.g. male->female->male) far better than the 480ms
# low-latency chunk, which commits each fragment with too little context and smears
# them onto the dominant speaker. NVIDIA's own DER table shows this preset is better
# for conversational/meeting audio (CALLHOME 4spk 12.44 -> 11.72) AND ~18x cheaper.
# Override per host via DIAR_* env:
#   low latency (1.04s, live-first):  chunk_len=6   rc=7  fifo=188 update=144 cache=188
#   high latency (10s, accuracy):     chunk_len=124 rc=1  fifo=124 update=124 cache=188
#   very high latency (30.4s):        chunk_len=340 rc=40 fifo=40  update=300 cache=188
CHUNK_LEN = int(os.environ.get("DIAR_CHUNK_LEN", "124") or 124)
RIGHT_CONTEXT = int(os.environ.get("DIAR_RIGHT_CONTEXT", "1") or 1)
FIFO_LEN = int(os.environ.get("DIAR_FIFO_LEN", "124") or 124)
UPDATE_PERIOD = int(os.environ.get("DIAR_UPDATE_PERIOD", "124") or 124)
SPKCACHE_LEN = int(os.environ.get("DIAR_SPKCACHE_LEN", "188") or 188)
# Emit cadence: send a partial update every this many seconds of NEW audio (both paths).
STEP_SEC = float(os.environ.get("DIAR_STREAM_STEP_SEC", "2.0") or 2.0)
# (Fallback-only) need a little audio before the first .diarize() pass is meaningful.
MIN_SEC = float(os.environ.get("DIAR_STREAM_MIN_SEC", "1.0") or 1.0)
# (Fallback-only) UNBOUNDED-LENGTH ROLLING WINDOW: re-running .diarize() on the WHOLE
# session buffer is O(n) per step -> O(n^2) over a session, and buffer + GPU work grow
# without limit. So only ever diarize the last WINDOW_SEC (bounded compute AND memory),
# COMMIT the older turns, and carry an OVERLAP_SEC tail across each roll so the NEW
# window's labels can be remapped onto the previous window's by time-overlap.
WINDOW_SEC = float(os.environ.get("DIAR_STREAM_WINDOW_SEC", "60") or 60)
OVERLAP_SEC = float(os.environ.get("DIAR_STREAM_OVERLAP_SEC", "12") or 12)

_state = {"ready": False, "error": None, "model": None, "device": "cpu"}
# Sortformer's diarize() is blocking and not concurrency-safe; serialize inference.
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
        # Configure the streaming behaviour (card: "Setting up Streaming Configuration").
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
        # Probe the TRUE incremental-streaming API: if present we feed only NEW audio
        # per step (O(1)/chunk, real-time); if ANY piece is missing we fall back to the
        # bounded-window .diarize() path. n_spk / subsampling_factor drive
        # preds->timestamps (each output frame = subsampling_factor * 10ms hop = 80ms).
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
        _state["error"] = str(e)
        _p("engine load FAILED: %s" % e)
        log.exception("engine load failed: %s", e)


def _spk_name(spk):
    # Normalize speaker labels to spk_<n> ("speaker_2" / 2 / "2" -> "spk_2").
    s = str(spk).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return ("spk_" + digits) if digits else (s or "spk_0")


def _parse_segments(raw):
    # The card's diarize() yields items like "start end speaker" (str) across
    # versions; tolerate tuples/objects too so a NeMo minor bump can't break the
    # wire format.
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
    # Collapse adjacent same-speaker turns (short gaps bridged) so the committed
    # timeline and the seam between committed+live stay clean.
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
    # preds: numpy [T, n_spk] per-frame speaker-activity probabilities (each frame
    # spans frame_sec = 80ms) from the streaming model; threshold each speaker column
    # and collapse contiguous active frames into ABS [start,end] turns (AOSC keeps
    # column k == one speaker so labels are globally stable, NO remap).
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
    # FALLBACK path only (used when the streaming API is absent): buf = float32 mono
    # @16k; returns parsed segments over the buffer (times RELATIVE to buf[0]);
    # callers keep buf bounded to WINDOW_SEC. This is the OLD per-step re-diarize
    # (O(n)/call), kept solely as a safety net if a NeMo version lacks
    # forward_streaming_step.
    model = _state["model"]
    res = model.diarize(audio=[buf], batch_size=1, sample_rate=16000)
    return _parse_segments(res[0] if res else [])


def _pcm16_to_f32(data):
    import numpy as np

    if not data:
        return np.zeros((0,), dtype="float32")
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0


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


class _FallbackToWindow(Exception):
    # Raised by the streaming path's pre-flight self-test if the incremental API is
    # not usable on this NeMo build, so the handler switches to the window path
    # BEFORE any real audio is consumed (no data lost).
    pass


async def _run_streaming(ws):
    # TRUE incremental streaming, done the way the model was validated: the key to
    # QUALITY is matching NeMo's reference path — extract log-mel over a CONTIGUOUS
    # span (consistent per-feature normalization) and let streaming_feat_loader hand
    # each chunk its left/right neighbour context (per-tiny-block extraction
    # re-normalizes every ~2s and drops seam context, so the SAME speaker splits and
    # DIFFERENT speakers merge). So we keep a rolling WINDOW of recent audio,
    # re-extract features over the WHOLE window each step, run the loader over it, and
    # only feed forward_streaming_step for chunks that are NEW (not yet committed) and
    # already have full right-context in the window (committed chunks are SKIPPED but
    # still give new chunks left context + shared normalization). total_preds
    # accumulates per-80ms-frame activity for the whole session. Model compute stays
    # O(1)/chunk, feature extraction O(window)/step (cheap log-mel), memory bounded.
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
    # Rolling window kept for extraction: long enough for stable normalization + left
    # context, chunk-aligned, re-extracted each step and trimmed to keep_chunks from
    # the committed frontier.
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
        # Feed every NEW chunk in the current window that has full right context (or
        # all remaining chunks when final); feat_seq_offset is ALWAYS 0 (state carries
        # history).
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
        # Trim the window to keep_chunks behind the committed frontier (keeps left
        # context + a normalization window; chunk-aligned so the grid never shifts).
        max_base = st["committed"] - keep_chunks
        if max_base > st["base_chunks"]:
            drop = max_base - st["base_chunks"]
            buf = buf[drop * chunk_audio:]
            st["base_chunks"] = max_base

    def _selftest():
        # Exercise the whole pipeline on ~2 chunks of silence then RESET so the probe
        # doesn't shift the real session; any error here => fall back to window mode.
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
        seg = _resample_linear(_pcm16_to_f32(data), sample_rate)
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
    # Final: feed all remaining chunks (accept rc=0 on the trailing chunk), emit final.
    async with _infer_lock:
        await asyncio.to_thread(_process, True)
    await _send("final")
    await ws.close()


async def _run_window(ws):
    # FALLBACK path: the ORIGINAL bounded-window per-step .diarize() implementation,
    # used only if the streaming self-test fails — slower (re-diarizes the active
    # window each step) but correct so diar_stream never regresses to "no output".
    import numpy as np

    sample_rate = 16000
    # Rolling window: `buf` holds only the ACTIVE window ([base_offset, now]).
    buf = np.zeros((0,), dtype="float32")
    since = 0                                # samples fed since last diarize
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
        # Map this window's LOCAL speaker labels (already in ABS time) to stable
        # SESSION-canonical labels: within a window Sortformer's arrival-order (AOSC)
        # labels are stable so win_map is sticky; after a roll, anchor the new window's
        # labels to the previous window's via max overlap with overlap_ref (the carried
        # tail), unmatched speakers get a fresh id.
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
        # Diarize ONLY the active window; return canonical ABS segments for it.
        if buf.shape[0] < min_samples:
            return []
        async with _infer_lock:
            local = await asyncio.to_thread(_diarize, buf)   # times relative to buf[0]
        local_abs = [{"start": s["start"] + base_offset, "end": s["end"] + base_offset,
                      "speaker": s["speaker"]} for s in local]
        return _relabel(local_abs)

    def _roll(cur):
        # Once the window is full, freeze turns before the roll boundary into
        # `committed`, keep only the last OVERLAP_SEC of audio, and stash the tail's
        # canonical segments as the remap reference for the next window.
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
        seg = _resample_linear(_pcm16_to_f32(data), sample_rate)
        buf = np.concatenate([buf, seg]) if buf.size else seg
        since += int(seg.shape[0])
        if since >= step_samples:
            since = 0
            cur = await _current()
            await _send("partial", cur)
            _roll(cur)   # roll AFTER emitting, reusing the just-computed segments
    await _send("final", await _current())
    await ws.close()


def build_app(supports):
    app = FastAPI(title="audio-diar-stream (Streaming Sortformer)")
    mount_metrics(app)

    endpoints = [{"method": "WS", "path": "/v1/audio/diarize/stream",
                  "description": "Streaming speaker diarization (WebSocket)"}]
    register(app, model_name=MODEL_NAME, mode="audio", supports=supports,
             endpoints=endpoints, is_ready=lambda: _state["ready"],
             error=lambda: _state["error"])

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
    threading.Thread(target=_load, daemon=True).start()
    app = build_app(supports)
    _p("diar_stream starting; model=%s port=%s" % (MODEL_REPO, PORT))
    # Bursty offloaded inference + a client-paced feed, so disable server-initiated
    # WS keepalive to avoid dropping a healthy session on 1011.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL,
                ws_ping_interval=None, ws_ping_timeout=None)
