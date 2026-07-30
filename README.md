# audio-engines

Engine images for Olares audio model bases. Each image bakes an inference
engine **plus** a thin wrapper that implements the small contract `llm-init`
needs, so the base charts stay minimal (deploy one ready image + `llm-init`
reverse-proxy) and audio is just "another reverse-proxied engine" — never a
special case in `llm-init` or the chart.

One image per **engine family** (deps are mutually incompatible across families,
so they cannot share an image). One installed instance = one model; the
capabilities it exposes are chosen at clone time via `MODEL_SUPPORTS`.

`MODEL_SUPPORTS` is a CSV of `supports_*` keys — the same vocabulary `llm-init`
and the gateway use for every other capability (`supports_stt,supports_stt_stream`),
so both containers of a chart read one value in one format. The prefix is dropped
inside the engine, where a capability is just `stt`, and in the `capabilities` the
model list reports. A key the image does not implement is a **startup failure**,
not a silently ignored line.

## Engine contract

Every image MUST expose, on the engine port (default `8000`):

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | The standard model list, exactly like any other engine: Ollama `models[]` (with `capabilities`, `details`) merged with OpenAI `data[]`. `id` / `name` MUST equal `MODEL_NAME`; capabilities are the coarse `audio` plus the fine-grained keys this instance serves. Returns **200 only when the model is loaded**, `503` while loading — `llm-init` polls it for liveness/readiness (`WaitAlive`/`Ready`). |
| `GET /api/engine-spec` | Internal, never load-gated: `implements` (what the image can do), `declares` (what `MODEL_SUPPORTS` asked for), `serves` (what this process mounted) and the capability `endpoints[]`, each with `available` and, when false, the `reason`. `llm-init` merges it into `/api/endpoints`. |
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
  contract.py            /v1/models + /api/engine-spec + /health surface
  catalog.py             the one table: base -> caps -> endpoints
  tasks.py               the one worker, and the async=1 task API
  watchdog.py            exit non-zero if the model never loads
  audioio.py             decoding shared by the torch caps
  app.py                 entrypoint: dispatch by AUDIO_BASE + MODEL_SUPPORTS
  caps/                  capability implementations
tests/                   GPU-free checks: every engine stubbed, wiring asserted
bases/<base>/deps.Dockerfile  FROM the engine image + the deps; rebuilt rarely
bases/<base>/append.env  how to append the wrapper onto that deps image
scripts/deps-image.sh    the deps ref: <same repo>:deps-<hash of the recipe>
scripts/append-image.sh  the actual build: deps image + one wrapper layer
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
altogether if that ever needs to be ruled out. Deps `FROM` is arch-selected:
amd64 keeps the validated cu129 / v0.23 image; arm64 uses the general aarch64
CUDA track (`vllm …:v0.16.0-cu130`) plus an **arm64-only** post-install patch that
moves `qwen-asr`'s `_get_data_parser` onto `ProcessingInfo.get_data_parser` /
`build_data_parser` (v0.15.1 lacks `configs.qwen3_asr` and cannot load ASR).
Wrapper code is identical across arches.

**`fasterwhisper`.** Arch-selected deps (`base-amd64` / `base-arm64`):

- **amd64** keeps `harveyff-whisper-webui` and its three traps in that stage's
  single `RUN`: pick the python that owns `faster_whisper`; pip
  `--ignore-installed` for the web layer (venv `--system-site-packages`);
  `transformers>=4.56` floor for the CT2 converter's `from_pretrained(dtype=...)`.
  The base's CUDA-matched `ctranslate2` must never be replaced by a generic wheel.
- **arm64**: PyPI `ctranslate2` aarch64 wheels are CPU-only. Build CT2 from source
  (`WITH_CUDA`/`WITH_CUDNN`) on `nvidia/cuda:*-cudnn-devel` (cu130 track), keep the
  wheel, install CUDA torch (wrapper `device=auto` keys off `torch.cuda`), then
  `faster-whisper`, then force-reinstall the CUDA CT2 wheel. Build asserts
  `get_supported_compute_types("cuda")` is non-empty.

This is also the one base that raises `LOAD_TIMEOUT_S` (to 5400, via its
`append.env`): CT2 conversion is legitimate multi-GB work and must not look like
a hung load to the watchdog.

**`pyannote`.** amd64 keeps the maximsachs CUDA image; arm64 builds CUDA
torch/torchaudio from the PyTorch `cu130` index (same general aarch64 CUDA track
as `qwen`) and re-asserts `torch.version.cuda` after pip. Wrapper behavior is
unchanged from the amd64-tested tree: Silero `vad` stays on Silero's CPU JIT
path; `diar` / `speaker_embed` / `enhance` select `cuda` when visible. Both
diarization stages default to `batch_size=1`, and segmentation slides a 10 s
window at a 1 s hop, so a 3 h clip becomes ~12 000 tiny forward passes with the
GPU idle in between — hence `DIAR_SEG_BATCH` / `DIAR_EMB_BATCH` (chart-derived
from the GPU quota, `auto` = 1 on CPU) and the one-way fallback to 1 on CUDA
OOM. `enhance` windows long clips (`ENHANCE_CHUNK_S`) with an overlap-add
crossfade and can answer `format=wav|flac|ogg`, degrading to FLAC then WAV if
libsndfile lacks the codec.

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
`linux/amd64` **and** `linux/arm64` (native runners: `ubuntu-latest` +
`ubuntu-24.04-arm`), merges them into one multi-arch tag on `beclab/audio-<base>`
(`:latest` + `:sha-<short>`, or `:<tag>` on a release tag). PRs lint only. Uses
the `DOCKERHUB_USERNAME` / `DOCKERHUB_PASS` repo secrets (login identity with push
access to the destination namespace; the namespace is an input, default `beclab`).

### Deps once, wrapper in seconds

Every base is split in two, because the wrapper changes daily and the deps
almost never do:

- `bases/<base>/deps.Dockerfile` — `FROM` the upstream engine image, install the
  deps, and **assert the imports** so a broken base fails the build. Published as
  `<same repo>:deps-<hash of that file>`. The tag being the recipe's content hash
  is what makes a stale deps image impossible: edit the recipe and the tag the
  build asks for simply does not exist yet, so CI builds it, once.
- `scripts/append-image.sh` — puts the wrapper on top of that deps image with
  crane: one small layer plus a config edit, pushed **inside the registry**.
  Nothing is pulled, unpacked or run, so it takes seconds and no disk.

That matters beyond CI time. Olares nodes pull through a mirror that syncs from
Docker Hub, and a full rebuild used to publish gigabytes of new layer digests
(pip is not reproducible) which the mirror then had to copy before an install
could even start — hours. Appending changes ~100 KB, so a new tag is usable
almost immediately.

`nemo` is the same shape with one difference: its upstream (25.7 GB compressed,
~55 GB unpacked, more than a runner's whole disk) needs nothing installed on top,
so its `append.env` names that image directly and no `deps.Dockerfile` exists.

The tradeoff is real but narrow: the import assertions now run when the deps
recipe changes, not on every wrapper change. `tests/` covers the wrapper half of
that (it imports every capability module against stubbed engines), and a missing
*engine* dep can only appear when the recipe changes, which is exactly when the
assertions run again.

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

1. `bases/<base>/deps.Dockerfile` — `FROM` the engine's base image, `pip install`
   all deps at **build time** (no runtime pip), assert the imports so a broken
   base fails the build, and leave `/usr/local/bin/audio-python` pointing at the
   interpreter that owns them. Plus `bases/<base>/append.env` for the platform and
   any extra `ENV` (`AUDIO_BASE`, `PYTHONPATH`, `WRAPPER_PORT` and the `CMD` are
   set for you). Nothing else: no `COPY wrapper`, no `LABEL`.
2. `wrapper/caps/<cap>.py` — implement the capabilities as `build_app(supports)`
   + `run(supports)`; expose `/v1/audio/*` and wire `wrapper.gpu.mount_metrics` +
   `wrapper.contract.register` so the contract above is satisfied.
3. Add the base to `BASES` in `wrapper/catalog.py`, its capability endpoints to
   `_MOUNTS` and its family to `FAMILIES`; that table is what the engine spec,
   the `/v1/models` details and the entrypoint dispatch all read.
4. `.github/workflows/<base>-ci.yml` — copy `qwen-ci.yml` and change the paths,
   `base` and `repo`; the build itself is the shared `build-image.yml`.
5. Update the **Bases** table above.

## Branching

`main` starts empty; work lands on `feat/audio-stt` first and is promoted to
`main` later. Because CI publishes only from `main` / tags, dev images are built
locally (see above) until then.
