# MOSS-Transcribe-Diarize on SGLang-Omni: the same stt and diar answers as diar_asr, served by a
# child `sgl-omni serve` instead of transformers in this process.
#
# The child owns the GPU. This process keeps the llm-init contract and never touches the device,
# because a device context opened here would take its own slice of the GPU quota.
#
# Memory is sized by the longest recording the engine accepts, not by a share of the card. MOSS
# reads the whole recording in one generation, so a recording needs KV for all of its input
# (about 13.3 tokens a second of audio) plus all of its transcript (up to about 11 a second in a
# dense meeting), and a request that does not fit is refused by SGLang before it starts. The KV
# pool is therefore capped at what --max-audio-seconds needs; what the card has beyond that stays
# free for the encoder and prefill activations, which grow with the recording as well.
import contextlib
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .. import tasks
from .. import watchdog
from ..audioio import unlink
from ..contract import EngineArgs, cache_dir, register
from ..gpu import mount_metrics
from ..limits import Bounds
from ..runtime import Runtime
from .diar_text import MOSS_PROMPT, joined_text, parse_moss, turns

log = logging.getLogger("audio-diar-sgl")

ENGINE_BIN = os.environ.get("SGL_OMNI_BIN") or "sgl-omni"
RUN_DIR = os.environ.get("RUN_DIR") or "/run/llm-init"
# /tmp is the app's data volume in the chart, so compiled kernels survive a restart.
ENGINE_HOME = os.environ.get("SGL_OMNI_HOME") or "/tmp/sgl-omni-home"
XPU = (os.environ.get("AUDIO_BASE") or "").strip().endswith("-xpu")

# Measured on AMI meetings: 13.2 input tokens a second plus the instruction, and at most 10.5
# transcript tokens a second. The margins make the estimate an upper bound.
INPUT_TOKENS_PER_SECOND = 13.5
OUTPUT_TOKENS_PER_SECOND = 12.0
PROMPT_TOKENS = 128
# SGLang refuses a request whose input + max_new_tokens + this exceeds the pool.
POOL_MARGIN = 16 + 64

_runtime = Runtime(proc=None, device="xpu" if XPU else "cuda", pool=None, max_seconds=None)
MODEL_NAME = _runtime.model_name
MODEL_REPO = _runtime.model_repo
_state = _runtime.state

_args = EngineArgs()
MAX_NEW_TOKENS = _args.count("--max-new-tokens", 32768)
# 30 minutes fits a 16 GiB quota with room for the activations; raise it with the quota.
BOUNDS = Bounds(_args, seconds=1800)
USE_DRAFT = _args.switch("--draft", not XPU)
MEM_FRACTION = _args.text("--mem-fraction-static")
ENGINE_PORT = _args.count("--engine-port", _runtime.port + 1)
_passthrough = _args.passthrough()
_args.warn_unreadable(log)

_KV_POOL = re.compile(r"KV Cache is allocated\..*#tokens: (\d+)")
_stopping = threading.Event()
_exit = os._exit
_EXIT_GRACE_S = 2.0


def pool_tokens(seconds):
    """The KV pool a recording of this many seconds needs, transcript included."""
    return int(math.ceil(seconds * (INPUT_TOKENS_PER_SECOND + OUTPUT_TOKENS_PER_SECOND))) \
        + PROMPT_TOKENS + POOL_MARGIN


def seconds_for(pool):
    """The longest recording a pool of this many tokens holds."""
    room = pool - PROMPT_TOKENS - POOL_MARGIN
    return max(0.0, room / (INPUT_TOKENS_PER_SECOND + OUTPUT_TOKENS_PER_SECOND))


def output_budget(pool, seconds, prompt_chars=0):
    """max_new_tokens for one recording: everything the pool has left after the input."""
    need_in = int(math.ceil(seconds * INPUT_TOKENS_PER_SECOND)) + PROMPT_TOKENS + prompt_chars
    return max(1, min(MAX_NEW_TOKENS, pool - need_in - POOL_MARGIN))


def _snapshot(repo):
    """The newest snapshot directory of a repo in the shared HF cache, or None."""
    snapshots = os.path.join(cache_dir(repo), "snapshots")
    try:
        found = [os.path.join(snapshots, r) for r in os.listdir(snapshots)]
    except OSError:
        return None
    found = [p for p in found if os.path.isdir(p)]
    return max(found, key=os.path.getmtime) if found else None


def _repos():
    out = []
    for seg in (os.environ.get("MODEL_SOURCE") or "").split(","):
        tokens = seg.strip().split()
        if tokens:
            repo = tokens[0][5:] if tokens[0].startswith("hf://") else tokens[0]
            out.append(repo.split("@")[0])
    return out


def model_paths():
    """(model directory, draft directory or None).

    llm-init writes both into its run directory; the cache layout is the fallback for a pod that
    does not mount it.
    """
    repos = _repos()
    main, draft = None, None
    with contextlib.suppress(OSError):
        with open(os.path.join(RUN_DIR, "model_path")) as f:
            main = f.read().strip() or None
    with contextlib.suppress(OSError):
        with open(os.path.join(RUN_DIR, "extra_model_path")) as f:
            draft = next((line.strip() for line in f if line.strip()), None)
    if main is None and repos:
        main = _snapshot(repos[0])
    if draft is None and len(repos) > 1:
        draft = _snapshot(repos[1])
    return main, draft


def wait_for_draft(timeout_s=3600.0, interval_s=5.0):
    """Block until llm-init has written extra_model_path, when it is going to.

    The wrapper starts once the main snapshot exists, but llm-init fetches the draft after it and
    writes extra_model_path only when every download is done. Starting earlier would run the
    engine without DFlash for the life of the pod.
    """
    if XPU or not USE_DRAFT or len(_repos()) < 2 or not os.path.isdir(RUN_DIR):
        return
    path = os.path.join(RUN_DIR, "extra_model_path")
    started = time.monotonic()
    while not os.path.exists(path):
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            log.warning("no %s after %.0fs; starting without the draft", path, elapsed)
            return
        _state["waiting"] = "waiting for the draft model download"
        time.sleep(interval_s)
    _state["waiting"] = None


def engine_argv(model, draft):
    argv = [ENGINE_BIN, "serve", "--model-path", model, "--host", "127.0.0.1",
            "--port", str(ENGINE_PORT),
            "--asr.engine.max_total_tokens", str(pool_tokens(BOUNDS.seconds)),
            "--asr.engine.max_running_requests", "4", "--asr.engine.cuda_graph_max_bs", "4"]
    if MEM_FRACTION:
        argv += ["--mem-fraction-static", MEM_FRACTION]
    if XPU:
        argv += ["--asr.engine.enable_torch_compile", "false",
                 "--asr.engine.attention_backend", "intel_xpu"]
    elif USE_DRAFT and draft:
        argv += ["--asr.factory.enable_async_decode", "true",
                 "--asr.engine.speculative_algorithm", "DFLASH",
                 "--asr.engine.speculative_draft_model_path", draft]
    return argv + _passthrough


def engine_env():
    env = dict(os.environ, HOME=ENGINE_HOME, HF_HUB_OFFLINE="1")
    os.makedirs(ENGINE_HOME, exist_ok=True)
    return env


class _Child:
    """The sgl-omni server process, in its own process group so it can be stopped as a whole.

    sgl-omni forks a scheduler and stage workers. Signalling only the leader leaves them holding
    the GPU, which is how a stopped benchmark server once answered the next run's requests.
    """

    def __init__(self):
        self.proc = None
        self.pool = None
        self.pool_seen = threading.Event()

    def start(self, argv, env):
        log.info("starting engine: %s", " ".join(argv))
        self.proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._reap, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            sys.stderr.write(line)
            if self.pool is None:
                m = _KV_POOL.search(line)
                if m:
                    self.pool = int(m.group(1))
                    self.pool_seen.set()

    def _reap(self):
        rc = self.proc.wait()
        _state["ready"] = False
        _state["error"] = "engine process exited with code %d" % rc
        if _stopping.is_set():
            log.info("engine process exited with code %d during shutdown", rc)
            return
        log.error("engine process exited with code %d; exiting so the container is restarted", rc)
        self.stop()
        time.sleep(_EXIT_GRACE_S)
        _exit(watchdog.EXIT_CODE)

    def stop(self, grace_s=20.0):
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except OSError:
            return
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except OSError:
                return
            time.sleep(0.5)
        log.warning("engine process group %d outlived SIGTERM by %.0fs; killing it", pgid, grace_s)
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


_child = _Child()


def _healthy(client):
    try:
        return client.get("http://127.0.0.1:%d/health" % ENGINE_PORT, timeout=5).status_code == 200
    except Exception:
        return False


def _load():
    import httpx

    try:
        wait_for_draft()
        model, draft = model_paths()
        if not model:
            raise RuntimeError("no local snapshot of %s" % MODEL_REPO)
        if USE_DRAFT and not XPU and not draft:
            log.warning("no draft weights in MODEL_SOURCE or %s; running without DFlash", RUN_DIR)
        _child.start(engine_argv(model, draft), engine_env())
        with httpx.Client() as client:
            while not _healthy(client):
                if not _child.alive():
                    return
                time.sleep(2)
        pool = _child.pool
        if pool is None:
            raise RuntimeError("engine is up but never reported its KV pool size")
        seconds = min(BOUNDS.seconds, seconds_for(pool))
        if seconds < BOUNDS.seconds:
            log.warning("KV pool holds %d tokens, enough for %.0fs of audio rather than the %.0fs "
                        "--max-audio-seconds asks for; raise the GPU quota or "
                        "--mem-fraction-static", pool, seconds, BOUNDS.seconds)
        _state.update(pool=pool, max_seconds=seconds, ready=True)
        log.info("engine ready: KV pool %d tokens, recordings up to %.0fs", pool, seconds)
    except Exception as e:
        _state["error"] = str(e)
        log.exception("engine load failed: %s", e)


def _events(response):
    """The JSON events of an SSE response, in order."""
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def _transcribe(path, seconds, prompt, ctx):
    """(segments, truncated) of one whole recording."""
    import httpx

    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")
    instruction = MOSS_PROMPT + ("热词提示：" + prompt if prompt else "")
    budget = output_budget(_state["pool"], seconds or 0.0, len(prompt or ""))
    form = {"stream": "true", "response_format": "json", "temperature": "0",
            "max_new_tokens": str(budget), "prompt": instruction}
    started = time.monotonic()
    text, final, finish = [], None, None
    ctx.progress(ratio=0.0, stage="transcribe")
    with open(path, "rb") as f, httpx.Client(timeout=httpx.Timeout(30.0, read=600.0)) as client:
        files = {"file": (os.path.basename(path), f, "application/octet-stream")}
        with client.stream("POST", "http://127.0.0.1:%d/v1/audio/transcriptions" % ENGINE_PORT,
                           data=form, files=files) as r:
            if r.status_code != 200:
                detail = r.read().decode("utf-8", "replace")[:500]
                raise HTTPException(status_code=400 if r.status_code < 500 else 500,
                                    detail="engine refused the recording: %s" % detail)
            last = 0.0
            for event in _events(r):
                kind = event.get("type")
                if kind == "transcript.text.delta":
                    text.append(event.get("delta") or "")
                    now = time.monotonic()
                    if now - last >= 1.0:
                        last = now
                        ctx.checkpoint()
                        segs = parse_moss("".join(text))
                        if segs and seconds:
                            ctx.progress(ratio=segs[-1]["end"] / seconds, stage="transcribe")
                elif kind == "transcript.text.done":
                    final, finish = event.get("text"), event.get("finish_reason")
                elif kind == "error":
                    raise RuntimeError((event.get("error") or {}).get("message") or "engine error")
    segments = parse_moss(final if final is not None else "".join(text))
    truncated = finish == "length"
    log.info("diarized stt complete model=%s audio_seconds=%.1f budget=%d segments=%d speakers=%d "
             "truncated=%s duration_seconds=%.1f", MODEL_NAME, seconds or 0.0, budget,
             len(segments), len({s["speaker"] for s in segments}), truncated,
             time.monotonic() - started)
    return segments, truncated


async def _accept(file):
    """The upload on disk, refused before a task exists when the pool cannot hold it."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "engine not ready")
    path, seconds = await BOUNDS.spill(file, "this engine holds the whole recording in its KV cache")
    if seconds is not None and seconds > _state["max_seconds"]:
        unlink(path)
        raise HTTPException(status_code=413, detail=(
            "audio is %.0fs; this engine's KV cache holds at most %.0fs"
            % (seconds, _state["max_seconds"])))
    return path, seconds


def _stt_body(segments, truncated, seconds, response_format):
    text = joined_text(segments)
    rf = (response_format or "json").lower()
    if rf == "text":
        return PlainTextResponse(text)
    if rf in ("diarized_json", "verbose_json", "verbose"):
        return {"task": "transcribe", "duration": round(seconds or 0.0, 3), "text": text,
                "segments": segments, "truncated": truncated}
    return {"text": text}


def _diar_body(segments, truncated, exclusive):
    segs = turns(segments, exclusive)
    speakers = sorted({s["speaker"] for s in segs})
    body = {"model": MODEL_NAME, "mode": "diar", "device": _state["device"],
            "num_speakers": len(speakers), "speakers": speakers,
            "num_segments": len(segs), "segments": segs, "exclusive": exclusive,
            "truncated": truncated}
    if exclusive:
        body["nonexclusive_segments"] = turns(segments, False)
    return body


@contextlib.asynccontextmanager
async def _lifespan(_app):
    yield
    _stopping.set()
    _child.stop()


def build_app(supports):
    app = FastAPI(title="audio-diar-sgl (speaker-attributed transcription on SGLang-Omni)",
                  lifespan=_lifespan)
    mount_metrics(app, nvml_fallback=True, torch_probe=False)

    register(app, model_name=MODEL_NAME, module="diar_sgl", served=supports, repo=MODEL_REPO,
             model_format="safetensors", quantization="bf16",
             is_ready=lambda: _state["ready"], error=lambda: _state["error"], task_api=True)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form(default=None),
                             language: str = Form(default=None),
                             response_format: str = Form(default="json"),
                             prompt: str = Form(default=None),
                             segments: str = Form(default=None),
                             async_: str = Form(default=None, alias="async")):
        if segments:
            raise HTTPException(status_code=400, detail=(
                "this model transcribes whole recordings with speakers; send no segments"))
        path, seconds = await _accept(file)

        def _work(ctx):
            ctx.meter(input_seconds=seconds)
            segs, truncated = _transcribe(path, seconds, prompt, ctx)
            return _stt_body(segs, truncated, seconds, response_format)

        return await tasks.dispatch(async_, "stt", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="transcription failed")

    @app.post("/v1/audio/diarization")
    async def diarization(file: UploadFile = File(...), exclusive: str = Form(default=None),
                          num_speakers: str = Form(default=None),
                          min_speakers: str = Form(default=None),
                          max_speakers: str = Form(default=None),
                          async_: str = Form(default=None, alias="async")):
        if num_speakers or min_speakers or max_speakers:
            raise HTTPException(status_code=400, detail=(
                "this model counts speakers itself; num_speakers, min_speakers and "
                "max_speakers are not supported"))
        want_exclusive = tasks.truthy(exclusive) if exclusive is not None else False
        path, seconds = await _accept(file)

        def _work(ctx):
            ctx.meter(input_seconds=seconds)
            segs, truncated = _transcribe(path, seconds, None, ctx)
            return _diar_body(segs, truncated, want_exclusive)

        return await tasks.dispatch(async_, "diar", MODEL_NAME, _work,
                                    cleanup=lambda: unlink(path), fail="diarization failed")

    return app


def run(supports):
    _runtime.serve(supports, _load, build_app, "sgl-omni engine", timeout_s=3600.0)
