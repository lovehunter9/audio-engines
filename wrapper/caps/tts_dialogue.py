# Multi-speaker dialogue synthesis: SoulX-Podcast in this process, the wrapper is the only server.
#
# Upstream publishes no PyPI package and no setup.py, so bases/soulx/deps.Dockerfile clones it at
# a pinned commit into /opt/soulx and puts that on PYTHONPATH. Nothing here adjusts sys.path: the
# package is imported as an ordinary top-level `soulxpodcast`, which is also the only thing its
# own absolute imports will tolerate.
#
# Three properties of the model shape the API below, and none of them are choices made here:
#
#   * Every speaker needs a reference clip. This is a zero-shot cloning architecture with no
#     preset voices at all, so there is no "just synthesize it" path — a request without
#     reference audio cannot be served, only rejected.
#   * A turn is not an independent utterance. forward_longform carries context across the whole
#     script, which is the entire point: prosody, turn-taking and back-channelling only work
#     because turn N sees turns 1..N-1. Splitting a script into N requests loses exactly that.
#   * process_single_input asserts one script per call, so there is no batch endpoint. A script
#     is already the unit of batching here.
#
# There is likewise no streaming: SoulX has no incremental decode path.
import base64
import contextlib
import io
import logging
import os
import tempfile
import threading
import time

import numpy as np

from fastapi import FastAPI, HTTPException, Request

from .. import hfgate
from .. import tasks
from ..gpu import mount_metrics
from ..contract import register, EngineArgs
from ..runtime import Runtime

log = logging.getLogger("audio-tts-dialogue")

_runtime = Runtime("Soul-AILab/SoulX-Podcast-1.7B", model=None, dataset=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
HF_TOKEN = os.environ.get("HF_TOKEN") or None

_args = EngineArgs()
# Halves the flow decoder only; the LLM stage stays bf16, which is why it is a flag not a dtype.
FP16_FLOW = _args.switch("--fp16-flow", False)
# Seeded once at load and again per request: sampling is stateful across turns.
SEED = _args.count("--seed", 1988)

# The HiFiGAN vocoder decodes at 24 kHz for this checkpoint.
OUT_SR = 24000
# Reference audio is resampled internally, so its rate does not matter; it is held whole per speaker.
MAX_SPEAKERS = 4
BOOT_TIMEOUT_S = 2400.0

_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/L16"),  # raw little-endian int16, no container
}

# Upstream reports an unreadable reference clip two ways: a TypeError, or a WARNING plus None.
_REF_UNREADABLE = ("the model could not read the reference audio — no decoder for it, or it is "
                   "near silent (upstream logged a warning and returned nothing)")

# audioread's ffmpeg backend goes by extension, and a browser uploads whatever the file really is.
_REF_SUFFIX = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
               "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/flac": ".flac",
               "audio/x-flac": ".flac", "audio/ogg": ".ogg", "audio/opus": ".opus",
               "audio/aac": ".aac", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
               "audio/webm": ".webm", "video/mp4": ".mp4", "video/webm": ".webm",
               "video/quicktime": ".mov", "video/x-matroska": ".mkv"}

_state = _runtime.state
# One dataset object, rewritten per request: two concurrent scripts would read each other's speakers.
_gen_lock = threading.Lock()


class _temp_ref:
    """The dataloader opens reference audio by path, so an uploaded clip lands on disk first."""

    def __init__(self, data, suffix=".wav"):
        self._data, self._suffix, self._path = data, suffix, None

    def __enter__(self):
        fd, self._path = tempfile.mkstemp(suffix=self._suffix, prefix="dlg-ref-")
        with os.fdopen(fd, "wb") as f:
            f.write(self._data)
        return self._path

    def __exit__(self, *exc):
        try:
            if self._path:
                os.unlink(self._path)
        except OSError:
            pass


def _model_path():
    """The pre-downloaded snapshot: llm-init fetches the weights before this process ever starts."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=HF_TOKEN)


def _load():
    from soulxpodcast.utils.infer_utils import initiate_model

    path = _model_path()
    log.info("loading %s from %s (fp16_flow=%s seed=%d)", MODEL_NAME, path, FP16_FLOW, SEED)
    # llm_engine is pinned to "hf": upstream's optional vllm branch is not in this image.
    return initiate_model(SEED, path, "hf", FP16_FLOW)


def _warmup_ref():
    """A throwaway reference clip for warmup: two seconds of noise at speaking level.

    Real speech would be better conditioning, but shipping an audio fixture in the image to warm
    a cache is not worth it — this exercises the same code path (tokenizer, speaker embedding,
    mel, flow, vocoder), which is all warmup is for.

    The level is not cosmetic. Upstream's audio_volume_normalize returns early — still a numpy
    array, skipping the torch.from_numpy that every other exit performs — when ten or fewer
    samples exceed 0.01, and everything downstream of it expects a tensor. This clip was noise at
    1e-3, so it took that branch on every boot: torch.stft asked the array for .dim(), upstream
    logged the AttributeError as a WARNING and handed back None, and the only thing that reached
    a log was a subscript on None two frames later.
    """
    import soundfile as sf

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(OUT_SR * 2) * 0.1).astype(np.float32)
    assert int((np.abs(audio) > 0.01).sum()) > 10, \
        "warmup clip is below upstream's normalizer threshold; see this docstring"
    buf = io.BytesIO()
    sf.write(buf, audio, OUT_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _warmup():
    """Spend the first-call cost here: lazy kernels, allocator growth, onnx session setup.

    Failures are deliberately NOT caught. This drives the same _synthesize_blocking every request
    drives, so whatever breaks here breaks all of them — treating it as advisory once shipped an
    arm64 image that loaded, logged one warning, reported ready, and then failed every call.
    Letting it reach _boot puts the reason in /v1/models, where an install can see it.
    """
    t0 = time.time()
    with _gen_lock, _temp_ref(_warmup_ref()) as path:
        _synthesize_blocking(
            turns=[(0, "你好。")],
            refs=[(path, "你好。")],
            seed=None,
        )
    log.info("warmup synthesis took %.0fs", time.time() - t0)


def _boot():
    try:
        model, dataset = _load()
        _state.update(model=model, dataset=dataset)
        log.info("%s loaded: sample_rate=%d max_speakers=%d", MODEL_NAME, OUT_SR, MAX_SPEAKERS)
        _warmup()
        _state.update(ready=True, error=None)
    except Exception as e:
        _state["error"] = hfgate.explain(MODEL_REPO, e)
        log.exception("tts_dialogue engine failed to start: %s", e)


def _explain(e):
    """Some exceptions carry everything in the type and nothing in str() — say the type too."""
    msg = str(e).strip()
    return msg if msg else type(e).__name__


def _synthesize_blocking(turns, refs, seed):
    """Run one script. `turns` is [(speaker_index, text)], `refs` is [(wav_path, ref_text)].

    Returns a list of float32 mono arrays, one per turn, in script order. Caller holds _gen_lock.
    """
    from soulxpodcast.utils.commons import set_all_random_seed
    from soulxpodcast.utils.infer_utils import process_single_input

    if seed is not None:
        set_all_random_seed(seed)

    # The "[S1]" prefix is upstream's real interface; indices are 0-based here and 1-based there.
    target_text_list = ["[S%d]%s" % (spk + 1, text) for spk, text in turns]
    try:
        data = process_single_input(
            _state["dataset"],
            target_text_list,
            [path for path, _ in refs],
            [text for _, text in refs],
            False,  # use_dialect_prompt: needs a separate dialect checkpoint, see build_app's docstring
            None,
        )
    except TypeError as e:
        # Upstream turns any read failure into a WARNING plus None; its WARNING names which one.
        if "subscriptable" not in str(e):
            raise
        raise RuntimeError("%s: %s" % (_REF_UNREADABLE, _explain(e))) from e
    if data is None:
        # The same failure on the path where upstream returns rather than raises.
        raise RuntimeError(_REF_UNREADABLE)
    results = _state["model"].forward_longform(**data)
    out = []
    for wav in results["generated_wavs"]:
        if hasattr(wav, "detach"):
            wav = wav.detach().float().cpu().numpy()
        # Each turn comes back as [1, T]; flattened because they are concatenated end to end.
        out.append(np.ascontiguousarray(np.asarray(wav, dtype=np.float32).reshape(-1)))
    return out


def _encode(audio, sr, fmt):
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    if fmt == "pcm":
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    container, subtype, _mime = _FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=container, subtype=subtype)
    return buf.getvalue()


def build_app(supports):
    app = FastAPI(title="audio-tts-dialogue (SoulX-Podcast)")
    mount_metrics(app)

    register(app, model_name=MODEL_NAME, module="tts_dialogue", served=supports, repo=MODEL_REPO,
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True,
             sample_rate=OUT_SR)

    # No child engine takes the leftovers here, so a mistyped flag would otherwise vanish silently.
    _args.warn_unclaimed(log)

    def _require_ready():
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")

    def _headers(fmt, turns):
        return {"X-Audio-Model": MODEL_NAME, "X-Audio-Mode": "tts_dialogue",
                "X-Audio-Format": fmt, "X-Audio-Sample-Rate": str(OUT_SR),
                "X-Audio-Turns": str(turns)}

    def _speakers(payload):
        """Decode the per-speaker reference clips. Returns [(bytes, suffix, ref_text)]."""
        speakers = payload.get("speakers")
        if not isinstance(speakers, list) or not speakers:
            raise HTTPException(status_code=400,
                                detail="speakers must be a non-empty array; %s is a zero-shot "
                                       "cloning model with no preset voices, so every speaker "
                                       "needs its own reference clip" % MODEL_NAME)
        if len(speakers) > MAX_SPEAKERS:
            raise HTTPException(status_code=400,
                                detail="at most %d speakers per script" % MAX_SPEAKERS)
        out = []
        for i, spk in enumerate(speakers):
            if not isinstance(spk, dict):
                raise HTTPException(status_code=400, detail="speakers[%d] must be an object" % i)
            ref = spk.get("ref_audio")
            if not isinstance(ref, str) or not ref.startswith("data:"):
                raise HTTPException(status_code=400,
                                    detail="speakers[%d].ref_audio must be a data: URL "
                                           "(base64 reference audio)" % i)
            head, _, b64 = ref.partition(",")
            mime = head[5:].split(";")[0] or "audio/wav"
            try:
                data = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400,
                                    detail="speakers[%d].ref_audio is not valid base64" % i)
            if not data:
                raise HTTPException(status_code=400,
                                    detail="speakers[%d].ref_audio is empty" % i)
            # The transcript is the prompt half of the pair; a wrong one degrades the voice silently.
            text = str(spk.get("ref_text") or "").strip()
            if not text:
                raise HTTPException(status_code=400,
                                    detail="speakers[%d].ref_text is required: it must be the "
                                           "verbatim transcript of that reference clip" % i)
            out.append((data, _REF_SUFFIX.get(mime, ".bin"), text))
        return out

    def _turns(payload, n_speakers):
        """Validate the script. Returns [(speaker_index, text)]."""
        turns = payload.get("turns")
        if not isinstance(turns, list) or not turns:
            raise HTTPException(status_code=400,
                                detail="turns must be a non-empty array of "
                                       "{speaker, text} objects")
        out = []
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise HTTPException(status_code=400, detail="turns[%d] must be an object" % i)
            text = str(turn.get("text") or "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="turns[%d].text is empty" % i)
            try:
                spk = int(turn.get("speaker", 0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400,
                                    detail="turns[%d].speaker must be an integer index into "
                                           "speakers[]" % i)
            if not 0 <= spk < n_speakers:
                raise HTTPException(status_code=400,
                                    detail="turns[%d].speaker is %d, but %d speaker(s) were "
                                           "supplied" % (i, spk, n_speakers))
            out.append((spk, text))
        return out

    def _check(payload):
        """Validate one body and settle its response format. Returns (turns, speakers, fmt, seed)."""
        speakers = _speakers(payload)
        turns = _turns(payload, len(speakers))
        fmt = str(payload.get("response_format") or "wav").strip().lower()
        if fmt not in _FORMATS:
            raise HTTPException(status_code=400, detail="response_format must be one of %s"
                                                       % ", ".join(sorted(_FORMATS)))
        seed = payload.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="seed must be an integer")
        return turns, speakers, fmt, seed

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        """Synthesize a multi-speaker script in one pass.

        Body: `speakers` is an array of {ref_audio, ref_text}, one entry per voice; `turns` is the
        script as [{speaker, text}] where `speaker` indexes into `speakers`. Returns the whole
        conversation as one audio file, or with `per_turn: true` a JSON array of per-turn clips.

        The script is synthesized as a unit — later turns are conditioned on earlier ones, which
        is what makes the turn-taking sound like a conversation rather than concatenated lines.

        Dialects (Sichuanese, Henanese, Cantonese) are not exposed: upstream gates them behind a
        separate dialect checkpoint plus a several-thousand-token few-shot prompt, and this
        deployment serves the Mandarin weights.
        """
        _require_ready()
        try:
            payload = dict(await request.json())
        except Exception:
            raise HTTPException(status_code=400,
                                detail="body must be JSON with `speakers` and `turns`")
        turns, speakers, fmt, seed = _check(payload)
        per_turn = bool(payload.get("per_turn"))

        def _work(ctx):
            ctx.progress(ratio=0.0, stage="synthesis")
            with contextlib.ExitStack() as stack:
                paths = [(stack.enter_context(_temp_ref(data, suffix)), text)
                         for data, suffix, text in speakers]
                with _gen_lock:
                    clips = _synthesize_blocking(turns, paths, seed)
            if per_turn:
                items = [{"index": i, "speaker": spk, "format": fmt, "sample_rate": OUT_SR,
                          "audio": base64.b64encode(_encode(clip, OUT_SR, fmt)).decode("ascii")}
                         for i, ((spk, _text), clip) in enumerate(zip(turns, clips))]
                ctx.progress(ratio=1.0, stage="done")
                return {"model": MODEL_NAME, "sample_rate": OUT_SR, "turns": items}
            body = _encode(np.concatenate(clips) if clips else np.zeros(0, np.float32),
                           OUT_SR, fmt)
            ctx.progress(ratio=1.0, stage="done")
            return tasks.Binary(body, _FORMATS[fmt][2], suffix="." + fmt,
                                headers=_headers(fmt, len(turns)))

        return await tasks.dispatch(request.query_params.get("async"), "tts_dialogue", MODEL_NAME,
                                    _work, fail="dialogue synthesis failed")

    return app


def run(supports):
    # Ready means loaded AND warmed, so the deadline has to cover both or it kills a healthy boot.
    _runtime.serve(supports, _boot, build_app, "SoulX-Podcast engine", timeout_s=BOOT_TIMEOUT_S)
