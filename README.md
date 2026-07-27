# audio-engines

Engine images for Olares audio model bases. Each image bakes an inference
engine **plus** a thin wrapper that implements the small contract `llm-init`
needs, so the base charts stay minimal (deploy one ready image + `llm-init`
reverse-proxy) and audio is just "another reverse-proxied engine" — never a
special case in `llm-init` or the chart.

One image per **engine family** (deps are mutually incompatible across families,
so they cannot share an image). One installed instance = one model; the
capabilities it exposes are chosen at clone time via `MODEL_SUPPORTS`.

## Engine contract

Every image MUST expose, on the engine port (default `8000`):

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | Self-report `{data:[{id, mode, supports, endpoints}]}`. `id` MUST equal `MODEL_NAME`. Returns **200 only when the model is loaded**, `503` while loading — `llm-init` polls this for liveness/readiness (`WaitAlive`/`Ready`) and the dashboard renders `endpoints[]`. |
| `GET /metrics` | Prometheus text with the generic gauges `gpu_present`, `gpu_mem_used_bytes`, `gpu_mem_total_bytes`, `gpu_util_ratio` (NOT `audio_*`). `llm-init` relays these for the GPU UI. |
| `GET /health` \| `/healthz` \| `/readyz` | The engine's own health (200 ready / 503 loading). |
| `POST\|GET /v1/audio/*` | The actual capability endpoints; an unsupported op returns its own `404`. |

`llm-init` does the model **download** (into the shared HF cache) and writes a
sentinel; the engine container waits for that sentinel, then serves **offline**
from the cache. Nothing probes the engine via k8s — `llm-init` gates `/v1/*`
until the engine's `/v1/models` is alive. Since no probe watches the engine, a
load that hangs forever would go unnoticed, so `wrapper/watchdog.py` exits
non-zero after `LOAD_TIMEOUT_S` (default 1800) if the model is neither ready nor
failed, letting k8s rebuild the container. A load that failed *with a reason* is
left alone: `/v1/models` reporting the reason beats a crash loop.

## Tasks: any-length audio without a long-lived request

Every hop in front of the engine gives up on a silent request long before a long
clip is done (`llm-init` 60 s for the response header, the platform's Envoy 300 s
for the whole stream), so any capability whose work can outlast that also accepts
**`async=1`** as a form field:

| Request | Answer |
|---|---|
| without `async` | exactly as before: the result, on the same request |
| `async=1` | `202 {"task":{id, cap, model, status, poll, result_url}}` |

| Endpoint | Purpose |
|---|---|
| `GET /v1/audio/tasks/{id}` | `status` (`queued`→`running`→`succeeded`\|`failed`\|`canceled`), `progress` (`ratio`, `stage`, `done`/`total`), and the JSON `result` once it succeeded |
| `GET /v1/audio/tasks/{id}/result` | the result: audio bytes for enhance (with the same headers the sync path sends), otherwise the same JSON |
| `DELETE /v1/audio/tasks/{id}` | cancel a running task at its next checkpoint, or drop a finished one's result |
| `GET /v1/audio/tasks` | every live task, for a human debugging the instance |

It is a form field and not a header on purpose: the gateway forwards audio bodies
verbatim but not arbitrary headers, so a field is what actually survives the trip.

Both paths run on **one worker thread** — one instance owns one model on one
(time-sliced) GPU — so a sync request now queues behind whatever is running,
exactly as it already did behind the per-cap inference lock. The event loop stays
free either way, which is what keeps `/v1/models` and `/metrics` answering while
a three-hour clip is being processed. `stt_stream` / `diar_stream` keep their
WebSocket, which never had this problem; a base that only streams (nemo) mounts
no task API at all.

Batching is unrelated and unchanged: `segments`, pyannote's batch sizes and
faster-whisper's `BatchedInferencePipeline` are throughput and quality levers,
while `async=1` only decides who waits.

What tasks deliberately do NOT do: survive a restart (they live in memory, and a
poll after a pod restart is a `404` the caller should treat as "resubmit"),
outlive `TASK_TTL_S` (default 1800 s, after which results are reclaimed), or
queue without bound (`TASK_QUEUE_MAX`, default 32, then `503`). Results larger
than a JSON blob go to a temp file rather than the heap, and because the charts
mount `/tmp` from a host volume, the runner sweeps `upload-*` / `task-*` files a
previous run stranded when it starts.

## Layout

```
wrapper/                 shared Python package (used by every base)
  gpu.py                 generic gpu_* /metrics
  contract.py            /v1/models + /health surface (load-gated)
  tasks.py               the one worker, and the async=1 task API
  watchdog.py            exit non-zero if the model never loads
  audioio.py             decoding shared by the torch caps
  app.py                 entrypoint: dispatch by AUDIO_BASE + MODEL_SUPPORTS
  caps/                  capability implementations
bases/<base>/Dockerfile  FROM the engine base image + build-time deps + wrapper
bases/<base>/append.env  instead of a Dockerfile, for a base too big to unpack
scripts/append-image.sh  registry-level build for those (crane, no unpack)
.github/workflows/       build-image.yml (shared) + one <base>-ci.yml per base
Makefile                 local hand-build to a personal registry (dev phase)
```

Every image bakes `AUDIO_BASE` and a `/usr/local/bin/audio-python` pointing at
the interpreter that owns the deps (NeMo, for one, keeps them in a venv), so
`wrapper/app.py` can route the same cap name to the right engine and every
chart's sentinel shell can run the identical `exec audio-python -m wrapper.app`.

When those deps live in a **venv**, `audio-python` must be a wrapper script
(`#!/bin/sh` + `exec <venv>/bin/python "$@"`), never a symlink: CPython decides
it is inside a venv by looking for `pyvenv.cfg` **next to the executable**, and a
symlink in `/usr/local/bin` has none, so it starts against the system prefix,
the venv's `site-packages` never enter `sys.path`, and the wrapper dies on
`import fastapi` — while a build check run against the interpreter's real path
passes. For the same reason the build's final import check must go **through**
`audio-python`: verify the exact command the chart executes.

## Bases

| Base | Image | Capabilities | Engine / runtime | Status |
|---|---|---|---|---|
| `qwen` | `beclab/audio-qwen` | `stt`, `stt_stream`, `align` | qwen-asr in-process vLLM (`Qwen3ASRModel.LLM`) | validated |
| `fasterwhisper` | `beclab/audio-fasterwhisper` | `stt` (+ `/v1/audio/translations`) | faster-whisper (CTranslate2) | validated |
| `pyannote` | `beclab/audio-pyannote` | `vad`, `diar`, `speaker_embed`, `enhance` | pyannote / speechbrain / silero (torch) | validated |
| `nemo` | `beclab/audio-nemo` | `diar_stream` | NVIDIA NeMo | validated |

`stt` means different engines on different bases (`qwen-asr` vs CTranslate2),
which is why routing is keyed on `AUDIO_BASE` and not on the capability alone.
Within a base, capabilities that need DIFFERENT models (`align` vs the `stt`
pair; each of the four pyannote caps) are separate clones — the wrapper serves
the first match and says so in the log.

> Keep this table in sync whenever a base is added or its capabilities change.

### Per-base notes worth knowing before editing

**`qwen`.** One vLLM load serves `stt` (task-based) and `stt_stream` (WebSocket),
so the two are kept off each other with a plain `threading.Lock` the task worker
also takes — an `asyncio` lock cannot span the worker thread. vLLM is asked to
capture only a handful of CUDA graph shapes (`VLLM_CAPTURE_SIZES`, default
`1,2,4,8`) because inference here is always batch 1 and capture is where startup
has been seen to wedge holding the vGPU lock; the field name is probed off
`CompilationConfig` rather than assumed, and `VLLM_ENFORCE_EAGER=1` skips graphs
altogether if that ever needs to be ruled out.

**`fasterwhisper`.** Three traps, all in the single `RUN`:

- The image ships several pythons; only one owns `faster_whisper`, and the web
  layer must be installed into **that** one.
- That install needs `--ignore-installed`, because the venv is built with
  `--system-site-packages`: pip sees `fastapi` in the system python, says
  "already satisfied", installs nothing, and the venv still cannot import it.
- `transformers>=4.56` is a floor, not a preference: the image's `ctranslate2`
  calls `from_pretrained(dtype=...)`, and older `transformers` forwards that
  unknown kwarg into the model constructor and dies. Only the CT2 **converter**
  path uses it (`openai/whisper-large-v3` and other transformers-format
  checkpoints, converted once into the shared cache on first load);
  `faster_whisper` itself does not.

The base's CUDA-matched `ctranslate2` + `faster_whisper` must never be replaced
by a generic wheel, so nothing else is touched. This is also the one base that
raises `LOAD_TIMEOUT_S` (to 5400, in the Dockerfile): that one-time conversion is
legitimate multi-GB work and must not look like a hung load to the watchdog.

**`pyannote`.** Both diarization stages default to `batch_size=1`, and
segmentation slides a 10 s window at a 1 s hop, so a 3 h clip becomes ~12 000
tiny forward passes with the GPU idle in between — hence `DIAR_SEG_BATCH` /
`DIAR_EMB_BATCH` (chart-derived from the GPU quota, `auto` = 1 on CPU) and the
one-way fallback to 1 on CUDA OOM. `enhance` windows long clips
(`ENHANCE_CHUNK_S`) with an overlap-add crossfade and can answer
`format=wav|flac|ogg`, degrading to FLAC then WAV if libsndfile lacks the codec.

**`nemo` (`diar_stream`).** Its own capability rather than `diar` plus a flag,
because streaming diarization has to keep speaker labels consistent over time at
bounded latency; Sortformer does that with an Arrival-Order Speaker Cache, so
`spk_0`/`spk_1`/… stay stable across steps. Two paths, chosen once per
connection by a self-test on silence, before any client audio: the incremental
API (`init_streaming_state` + `streaming_feat_loader` + `forward_streaming_step`,
O(1) per chunk) and, if any piece of it is missing on this NeMo build, a bounded
`WINDOW_SEC` re-`diarize()` that commits older turns and remaps labels across an
`OVERLAP_SEC` tail — slower, but never "no output". Only public NeMo API, no
monkey-patching.

Wire protocol on `WS /v1/audio/diarize/stream`: the client sends an optional
`{"type":"start","sample_rate":16000}`, then **binary PCM16LE mono** chunks, then
`{"type":"stop"}` (or just closes). The server sends `{"type":"ready"}`, then
`{"type":"partial"|"final","segments":[{start,end,speaker}],"speakers":[…]}`, or
`{"type":"error"}`. Fusing these turns with ASR text is the consumer's job.

Latency presets, in 80 ms frames, via `DIAR_*` env — the default is **high
latency**, because transcription streams from a separate `stt_stream` engine and
speaker accuracy matters more than immediacy here; a ~10 s chunk also resolves
rapid adjacent turns far better than a 480 ms one and is ~18x cheaper (NVIDIA's
own CALLHOME 4spk DER: 12.44 -> 11.72):

| Preset | `CHUNK_LEN` | `RIGHT_CONTEXT` | `FIFO_LEN` | `UPDATE_PERIOD` | `SPKCACHE_LEN` |
|---|---|---|---|---|---|
| low (1.04 s, live-first) | 6 | 7 | 188 | 144 | 188 |
| **high (10 s, accuracy)** | **124** | **1** | **124** | **124** | **188** |
| very high (30.4 s) | 340 | 40 | 40 | 300 | 188 |

## Build

**CI (release).** Each `bases/<base>/**` change (or a `v*` tag, or a manual
`workflow_dispatch`) runs `.github/workflows/<base>-ci.yml`, which builds
`linux/amd64` and pushes `beclab/audio-<base>` (`:latest` + `:sha-<short>`, or
`:<tag>` on a release tag). PRs build only. Uses the `DOCKERHUB_USERNAME` /
`DOCKERHUB_PASS` repo secrets (login identity with push access to the `beclab`
org; the namespace is hardcoded, not derived from the username).

**Bases too big to unpack.** `docker build` unpacks the whole base image to run
even a single `COPY`, and `nemo`'s upstream (25.7 GB compressed, ~55 GB unpacked)
exceeds a runner's entire disk. Such a base declares `bases/<base>/append.env`
instead of a Dockerfile; CI and `make build-push` then both call
`scripts/append-image.sh`, which uses crane to push the wrapper as one small
layer and cross-repo-mount the rest inside the registry — seconds, no unpack, no
disk. The cost is that nothing can be installed or checked at build time, so the
upstream image must already carry every import.

**Dev.** Same recipe, different destination — no separate dev build path.
Either dispatch the workflow with `namespace` + `image_tag` to publish
`<ns>/audio-<base>:<tag>` from a feature branch, or build by hand:

```bash
export REGISTRY=docker.io/<your-namespace>   # required, no default
make build-push BASE=qwen TAG=dev1           # -> <your-namespace>/audio-qwen:dev1
```

A hand build behind an HTTP proxy needs the proxy passed through to the
builder; see the `EXTRA` hook in the `Makefile`.

## Adding a new engine base

1. `bases/<base>/Dockerfile` — `FROM` the engine's base image, `pip install` all
   deps at **build time** (no runtime pip), assert the imports so a broken base
   fails the build, `COPY wrapper /app/wrapper`, set `ENV AUDIO_BASE=<base>`,
   symlink `audio-python`, default `CMD ["audio-python", "-m", "wrapper.app"]`.
   If the upstream image is too big to unpack, write `append.env` instead (see
   `bases/nemo/append.env`) — but only then, since it gives up build-time deps.
2. `wrapper/caps/<cap>.py` — implement the capabilities as `build_app(supports)`
   + `run(supports)`; expose `/v1/audio/*` and wire `wrapper.gpu.mount_metrics` +
   `wrapper.contract.register` so the contract above is satisfied.
3. Add the base to `ROUTES` in `wrapper/app.py`.
4. `.github/workflows/<base>-ci.yml` — copy `qwen-ci.yml` and change the paths,
   `base` and `repo`; the build itself is the shared `build-image.yml`.
5. Update the **Bases** table above.

## Branching

`main` starts empty; work lands on `feat/audio-stt` first and is promoted to
`main` later. Because CI publishes only from `main` / tags, dev images are built
locally (see above) until then.
