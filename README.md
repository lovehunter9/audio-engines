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
| `GET /api/engine-spec` | Internal, never load-gated: the versioned account of this engine's proxied data plane. `llm-init` merges it into `/api/endpoints`. |
| `GET /metrics` | Prometheus text with the generic gauges `gpu_present`, `gpu_mem_used_bytes`, `gpu_mem_total_bytes`, `gpu_util_ratio` (NOT `audio_*`). `llm-init` relays these for the GPU UI. |
| `GET /health` \| `/healthz` \| `/readyz` | The engine's own health (200 ready / 503 loading). |
| `POST\|GET /v1/audio/*` | The actual capability endpoints; an unsupported op returns its own `404`. |

Engine-spec v1 has `schema_version: 1`, a non-empty `model`,
`implements` (what the image can do), `declares` (what `MODEL_SUPPORTS`
asked for), `serves` (what this process mounted), and a non-empty
`endpoints[]`. Each usable endpoint has `method`, `path`, `available`, and
optionally `capability`, `description`, `reason`, or `deprecated`. `base` is
an audio extension naming the engine family; it is not required by the shared
v1 contract, so OCR legitimately omits it.

A structurally valid v1 report with at least one usable endpoint row is
authoritative for `llm-init`'s proxied data-plane catalog: undeclared static
proxy rows are removed. Reports with an unknown or missing version can still
relay well-formed endpoint rows for compatibility, but cannot remove the
static fallback; malformed v1 reports are ignored. If the engine is
unavailable, the static `MODEL_MODE=audio` / `MODEL_MODE=ocr` task directory
remains visible. A reported `model` that differs from configured `MODEL_NAME`
is added to `/api/endpoints` `reasons` as a diagnostic and does not by itself
change `available`.

`llm-init` does the model **download** (into the shared HF cache) and writes a
sentinel; the engine container waits for that sentinel, then serves **offline**
from the cache. Nothing probes the engine via k8s — `llm-init` gates `/v1/*`
until the engine's `/v1/models` is alive. Since no probe watches the engine, a
load that hangs forever would go unnoticed, so `wrapper/watchdog.py` exits
non-zero after 1800 s (`fasterwhisper` asks for 5400, since it may convert to CT2
first) if the model is neither ready nor failed, letting k8s rebuild the container. A load that failed *with a reason* is
left alone: `/v1/models` reporting the reason beats a crash loop.

### Tuning: `ENGINE_ARGS` and nothing else

`ENGINE_ARGS` is the one channel for every knob, per `llm-init`'s contract (for
audio it is read on the engine container, never by `llm-init`). Charts set it,
`wrapper/contract.py`'s `EngineArgs` parses it, each cap claims the flags it
understands under the **upstream's own spelling** — `--gpu-memory-utilization`,
`--max-model-len`, `--beam-size`, `--compute-type` — and whatever is left over is
handed to a child engine's argv, or logged as ignored where there is no child.

A new knob is therefore a new flag, never a new env. The only envs an engine reads
are the platform's own (`MODEL_NAME`, `MODEL_SOURCE`, `MODEL_SUPPORTS`,
`ENGINE_PORT`, `ENGINE_ARGS`, `REQUIRED_GPU_MEMORY`, `HF_*`, `LOG_LEVEL`) plus
`AUDIO_BASE`, which the image bakes in. Values with only one right answer are
constants in the code, and where the environment already knows the answer the code
derives it: GPU utilization comes from `REQUIRED_GPU_MEMORY` measured against the
card CUDA actually reports, so no chart hardcodes a card's size.

## Tasks: any-length audio without a long-lived request

Every hop in front of the engine gives up on a silent request long before a long
clip is done (`llm-init` 60 s for the response header, the platform's Envoy 300 s
for the whole stream), so any capability whose work can outlast that also accepts
**`async=1`** as a form field:

| Request | Answer |
|---|---|
| without `async` | exactly as before: the result, on the same request |
| `async=1` | `202 {"task":{id, kind, cap, model, status, poll, result_url}}` |

It is a form field and not a header on purpose: the gateway forwards audio bodies
verbatim but not arbitrary headers, so a field is what actually survives the trip.

The four query endpoints — `GET /v1/tasks`, `GET /v1/tasks/{id}`,
`GET /v1/tasks/{id}/result`, `DELETE /v1/tasks/{id}` — and the task document they
answer with are **not audio's own**: they are the cross-engine async task contract
that OCR and, later, image implement identically. It is specified in llm-init's
[`docs/api/openapi.yaml`](https://github.com/beclab/llm-init/blob/main/docs/api/openapi.yaml);
the `async-tasks` tag and `Task` schema are the contract's stable locating terms.
`wrapper/tasks.py` is this engine's implementation of it, with `kind: "audio"`.

The paths this engine shipped first, `/v1/audio/tasks*`, stay mounted as aliases of
the same runner and are advertised as `deprecated` in `/api/engine-spec`.

`GET /v1/tasks` takes the contract's `?status=` and `?limit=` (default 100, finished
tasks are the only ones a limit drops, `truncated: true` when it did).

What is audio-specific is what a result looks like: enhance answers audio bytes
with the same headers the sync path sends, everything else answers JSON.

## Duration: the only thing downstream can bill

Audio has no tokens, and the gateway in front of this engine streams the payload
through without decoding it — so nothing outside this process can say how long a
clip was. Every capability therefore reports what it measured, on the synchronous
response and on the task document alike:

| Header | Task field | Reported by |
|---|---|---|
| `X-Audio-Input-Duration-Seconds` | `input_duration_seconds` | the capabilities that consume audio: `stt`, `stt_stream`, `align`, `vad`, `diar`, `speaker_embed`, `enhance` |
| `X-Audio-Output-Duration-Seconds` | `output_duration_seconds` | the capabilities that produce it: `tts`, `tts_clone`, `tts_dialogue`, `sound_fx`, `enhance` |

The number is the audio the job actually handled, summed over a batch: a
transcription of five slices reports their total, not the file's length, and a
slice whose `end` runs past the recording reports what existed rather than what
was asked for. A capability reports through `ctx.meter(...)`, which is additive
precisely so a batch loop can call it per item.

**A missing header means unmeasured, never zero.** A WebSocket session and a
chunked reply both write their headers before the length is known, and Router
records those calls with an `audio_unmetered` tag rather than a price of zero.

**Only a task that succeeded carries the fields.** Several capabilities measure
the input before they begin, so a job that fails or is canceled has a number
that describes audio with no result — and `…/result` answers `409`, so nobody
could see what they were charged for. The document omits it.

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
outlive 1800 s (after which results are reclaimed), or queue without bound
(32 deep, then `503`). Results larger
than a JSON blob go to a temp file rather than the heap, and because the charts
mount `/tmp` from a host volume, the runner sweeps `upload-*` / `task-*` files a
previous run stranded when it starts.

## Layout

```
wrapper/                 shared Python package (used by every base)
  gpu.py                 generic gpu_* /metrics
  contract.py            /v1/models + /api/engine-spec + /health surface
  catalog.py             the one table: base -> caps -> endpoints
  runtime.py             model/source resolution, load lifecycle, watchdog + uvicorn
  batch.py               validation of the shared segments JSON form field
  tasks.py               the one worker, and the async=1 task API
  watchdog.py            exit non-zero if the model never loads
  audioio.py             temp uploads, PCM/numpy helpers, and torch-cap decoding
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
| `qwen3tts` | `beclab/audio-qwen3tts` | `tts`, `tts_clone` | faster-qwen3-tts (in-process) | in progress |
| `dasheng` | `beclab/audio-dasheng` | `sound_fx` | Dasheng-AudioGen diffusion (in-process transformers) | in progress |
| `soulx` | `beclab/audio-soulx` | `tts_dialogue` | SoulX-Podcast (in-process, cloned at build) | in progress |
| `crispasr` | `beclab/audio-crispasr` | `tts` | Voxtral-4B-TTS on CrispASR ggml (in-process, amd64 only) | in progress |
| `audiocpp` | `beclab/audio-audiocpp` | `tts`, `tts_clone`, `stt`, `stt_stream` | audio.cpp `audiocpp_server` (child process, GGUF) | in progress |

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
capture only the shapes `1,2,4,8` because inference here is always batch 1 and
capture is where startup has been seen to wedge holding the vGPU lock; the field
name is probed off `CompilationConfig` rather than assumed, and `--enforce-eager`
skips graphs altogether if that ever needs to be ruled out. Deps `FROM` is
arch-selected:
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

This is also the one base that raises the load deadline (to 5400 s, from the cap
itself): CT2 conversion is legitimate multi-GB work and must not look like a hung
load to the watchdog.

**`pyannote`.** amd64 keeps the maximsachs CUDA image; arm64 builds CUDA
torch/torchaudio from the PyTorch `cu130` index (same general aarch64 CUDA track
as `qwen`) and re-asserts `torch.version.cuda` after pip. Wrapper behavior is
unchanged from the amd64-tested tree: Silero `vad` stays on Silero's CPU JIT
path; `diar` / `speaker_embed` / `enhance` select `cuda` when visible. Both
diarization stages default to `batch_size=1`, and segmentation slides a 10 s
window at a 1 s hop, so a 3 h clip becomes ~12 000 tiny forward passes with the
GPU idle in between — hence `--segmentation-batch-size` / `--embedding-batch-size`
(`auto` sizes them to `REQUIRED_GPU_MEMORY`, 1 on CPU) and the one-way fallback to
1 on CUDA OOM. `enhance` windows long clips (120 s, shrinking to 60 or 30 on a
small slice) with an overlap-add crossfade and can answer `format=wav|flac|ogg`,
degrading to FLAC then WAV if libsndfile lacks the codec.

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

**`dasheng`.** A diffusion transformer, so almost none of the reflexes from the
other bases apply: no KV cache, no autoregression, and `generate()` denoises a
whole list in one pass — batch here is a real forward-pass win, not just one
round trip instead of N. Two consequences shape the cap. First, duration is not
a parameter: a `DurationPredictor` reads it off the text, so callers steer
length by describing it. Second, the dtype cast happens after loading rather
than through `torch_dtype=`, because upstream's `from_pretrained` override
forwards only `local_files_only` to `_load_external_models` — the text encoder
and codec would otherwise stay fp32 while the backbone went half, costing ~3.8 GB
and leaving a dtype seam where `generate()` hands a `content.dtype` latent to
`audio_tokenizer.decode()`. The config names two more repos to pre-download,
`google/flan-t5-large` and `mispeech/dashengtokenizer`, so `MODEL_SOURCE` lists
three. Captions are English-only (the text encoder saw nothing else); translation
is the caller's job and deliberately not done here.

**`soulx`.** The only base whose engine code is not a dependency: upstream ships
no PyPI package and no `setup.py`, so `deps.Dockerfile` clones the repo at a
pinned SHA into `/opt/soulx` and puts it on `PYTHONPATH`. Nothing is copied into
this repository, and a future fix belongs in a `.patch` applied after checkout,
not in a vendored tree. Two traps are worth knowing. First, the audio tokenizer
is **not** in the model repo: `s3tokenizer.load_model()` fetches a ~480 MB onnx
from ModelScope over plain `urllib`, obeying neither `HF_HUB_OFFLINE` nor
llm-init, so it is baked into the image and `XDG_CACHE_HOME` is pinned to keep
the cache findable under whatever `HOME` the pod gets. Second, the API shape is
dictated by the model, not chosen: every speaker needs a reference clip (there
are no preset voices), a script is synthesized in one `forward_longform` call so
later turns are conditioned on earlier ones, and `process_single_input` asserts
one script per call — hence one endpoint, no batch route and no streaming.

**`crispasr`.** The only base with no Python ML framework at all. Voxtral-4B-TTS's
reference stack is vLLM-Omni, whose two-stage design gives each stage its own
CUDA context and KV pool — the reference run peaks at ~20.5 GiB on a 24 GB card,
and HF transformers has no merged support — so the model would own a card that
has to be shared with an ASR engine, an LLM and an embedder. CrispASR
reimplements all three stages as ggml graphs, which puts Q8_0 at ~4.3 GB of
weights and is the entire reason this base exists. ggml links CUDA itself, so the
image carries no torch and the `/metrics` gauges are read through NVML in the cap
rather than through `gpu.py`. Four things follow from the packaging. The CUDA
build is x86_64-only upstream, so the CI passes a single-arch `slices` and the
Dockerfile refuses anything else — an arm64 image would install cleanly and then
be CPU. The wheel also ships a CPU ggml backend, but that is not a fallback:
`libcrispasr.so` lists `libggml-cuda.so.0` as a hard `DT_NEEDED` and that needs
`libcuda.so.1`, so without a driver the loader fails before any Python runs —
this engine must always be given a GPU. The build therefore checks
`libggml-cuda.so`'s own `DT_NEEDED` with `readelf` rather than importing the
package, which would fail on a correct image on a driverless runner. The published
checkpoint ships no audio encoder, so cloning is impossible and `tts_clone` is
never served — reference audio is refused with a 400 rather than quietly answered
in a preset voice. And the binding exposes whole-utterance synthesis only, so
`stream=1` is sentence-scoped, which is why this base mounts no WebSocket route.
Output is watermarked: `synthesize()` marks its audio and the unmarked variant
needs an explicit EU AI Act Art. 50 attestation, which is deliberately not given.

**`audiocpp`.** The only base whose model does not run in this process. audio.cpp is a
C++ engine with its own HTTP face, so `deps.Dockerfile` starts `FROM` upstream's daily
multi-arch image (pinned by date+revision) and adds nothing but Python — no compile step,
and no torch, which is why `wrapper/gpu.py` grew an NVML path for `/metrics`. Upstream's
`ENTRYPOINT` is a `cli|server|perf|parity` multiplexer, so it is cleared: left in place it
answers the appended command with "Unknown command" and the wrapper never starts.

`wrapper/acpp.py` owns the child (config, process, wire) and knows nothing about TTS or STT,
because one image covers 47 model families across TTS, ASR, alignment and music; a new task is a
new `caps/` file, not a rewrite. What a given model can do is **asked of the engine** rather
than configured: `audiocpp_cli --list-loaders --json` reports each family's tasks, its modes
per task, and its `instructions_policy`, and the packaged `/app/model_specs/*.json` supply the
weight filenames used to work out which family the downloaded GGUF even is. Routes follow from
those facts, so two models on this one base legitimately differ — VoxCPM2 mounts the streaming
TTS socket, MOSS-TTS-Nano has no streaming decode and withholds that one route, Voxtral Realtime
mounts `/v1/audio/transcriptions/live` plus the platform WS, which is what
`register(withheld=...)` exists to state in `/api/engine-spec` instead of overpromising.

TTS translations (the reason `audiocpp_tts` exists rather than a proxy): reference audio is a
server-side **path** to the engine but a data: URL or upload to our callers; it returns WAV
only, while the contract promises five formats; `voice` must be refused up front, because these
families have no built-in speakers and an unmatched name is read as a cached voice id and dies
inside the model; and on a `text_prefix` family `instructions` has to be folded into the text,
since sent as a field it is dropped in silence and the caller gets the default voice with a 200.
A streaming-capable model is configured `mode=streaming` even for plain requests — the server
collects its own stream and returns one buffer — so both shapes come from one loaded copy.
VoxCPM2 then requires `options.retry_badcase=false` on every request (including warmup); the
cap injects it, callers never see the field.

STT translations (`audiocpp_stt`): OpenAI clients upload multipart, the engine's JSON path
wants a server-side WAV, and its own multipart only accepts WAV, so anything else is transcoded
here. There is no native `/transcriptions/batch`, so batch is `segments` on the same POST (same
as qwen). `stream=true` on that POST is output-SSE of an already-uploaded file; live capture is
a different transport (`POST /v1/audio/transcriptions/live`, chunked PCM in, SSE out) and is
exposed as-is. `WS /v1/audio/stream` is the platform shape DEMO/gateway already speak,
translated onto `/live`, not a third protocol of the model. That translation is duplex
(chunked PCM written on one thread, SSE read on another): httpx's HTTP/1.1 client would
hold every partial until `stop`. SenseVoice's extra knobs (`language`, `enable_itn`,
`keep_tags`, `audio_chunk_*`) travel as request options and as `/live` query params from
the WS `start` frame, not as invented capability keys.

**`audio_llm` and `audio_s2s` are reserved, not served.** No base implements them:
the open models that do are, as of 2026-08, either research-licensed or too heavy
to share a card with the rest of these. The capability names stay declared in
llm-init and the gateway so nothing downstream has to change when a base for them
does land.

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
   any extra `ENV` (`AUDIO_BASE`, `PYTHONPATH`, `ENGINE_PORT` and the `CMD` are
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
