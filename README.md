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
| `GET /api/engine-capacity` | Internal, never load-gated: reports `{\"max_concurrency\": 1}` because all inference uses the same single worker. It does not report queue depth or current load. |
| `GET /metrics` | Prometheus text with the generic gauges `gpu_present`, `gpu_mem_used_bytes`, `gpu_mem_total_bytes`, `gpu_util_ratio` (NOT `audio_*`). `llm-init` relays these for the GPU UI. |
| `GET /health` \| `/healthz` \| `/readyz` | The engine's own health (200 ready / 503 loading). |
| `POST\|GET /v1/audio/*` | The actual capability endpoints; an unsupported op returns its own `404`. |

Engine-spec v2 has `schema_version: 2`, a non-empty `model`,
`implements` (what the image can do), `declares` (what `MODEL_SUPPORTS`
asked for), `serves` (what this process mounted), and a non-empty
`endpoints[]`. Each usable endpoint has `method`, `path`, `available`, and
the stable `operation_id`, `protocol`, `transport`, sync/async flags,
input/output modalities, parameters, formats, sample rates, limits and
resource scope. Optional `capability`, `description`, `reason`, `deprecated`
and extension fields remain additive. `base` is
an audio extension naming the engine family; it is not required by the shared
contract, so OCR legitimately omits it. Model Console continues accepting v1
reports while Router clients migrate to the v2 operation directory.

A structurally valid recognized report with at least one usable endpoint row is
authoritative for `llm-init`'s proxied data-plane catalog: undeclared static
proxy rows are removed. Reports with an unknown or missing version can still
relay well-formed endpoint rows for compatibility, but cannot remove the
static fallback; malformed v1 reports are ignored. If the engine is
unavailable, the static `MODEL_MODE=audio` / `MODEL_MODE=ocr` task directory
remains visible. A reported `model` that differs from configured `MODEL_NAME`
is added to `/api/endpoints` `reasons` as a diagnostic and does not by itself
change `available`.

The Intel `ov` image reports the same `schema_version: 2` as main (and
`base: qwen` for the same routes). Dispatch stays on `AUDIO_BASE=ov`.
Charts that still pin `llm-init` **v1.7.12** only treat schema 1 as
authoritative; pair `ov` with a llm-init that accepts v2.

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

Two flags are claimed by every capability that holds a whole clip at once — `diar`
(both engines), `speaker_embed`, `enhance`, `vad`. `--max-upload-mb` (default 1024)
bounds the request while it is still arriving; `--max-audio-seconds` bounds what it
decodes into, read off the header rather than by decoding it, and defaults to four
hours everywhere except `speaker_embed`, which embeds the whole clip in one pass and
stops at thirty minutes. Past either, the answer is `413`. Without them a long enough
upload is an OOM kill, which reaches the caller as a dropped connection rather than
as something it can act on. A deployment that knows its own memory ceiling moves
them; see `wrapper/limits.py`.

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
the same runner and are advertised as `deprecated` in `/api/engine-spec`, except
on FireRed and Breeze: those two only mount `/v1/tasks`.

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
| `qwen` | `beclab/audio-qwen` | `stt`, `stt_stream`, `align` | qwen-asr transformers; one package, two loads | in progress |
| `ov` | `beclab/audio-ov` | `stt`, `stt_stream`, `align` | OpenVINO GenAI ASR + ForcedAligner (Intel iGPU / Arc, amd64) | in progress |
| `fasterwhisper` | `beclab/audio-fasterwhisper` | `stt` (+ `/v1/audio/translations`) | faster-whisper (CTranslate2) | validated |
| `pyannote` | `beclab/audio-pyannote` | `vad`, `diar`, `speaker_embed`, `enhance` | pyannote / speechbrain / silero (torch) | validated |
| `speakrs` | `beclab/audio-speakrs` | `diar` | speakrs: pyannote community-1 in Rust, on ONNX Runtime (child process) | validated |
| `speakrs-ov` | `beclab/audio-speakrs-ov` | `diar` | the same speakrs, on ONNX Runtime's OpenVINO provider for Intel GPUs (amd64 only) | in progress |
| `nemo` | `beclab/audio-nemo` | `diar_stream` | NVIDIA NeMo | validated |
| `qwen3tts` | `beclab/audio-qwen3tts` | `tts`, `tts_clone` | faster-qwen3-tts (in-process) | in progress |
| `dasheng` | `beclab/audio-dasheng` | `sound_fx` | Dasheng-AudioGen diffusion (in-process transformers) | in progress |
| `soulx` | `beclab/audio-soulx` | `tts_dialogue` | SoulX-Podcast (in-process, cloned at build) | in progress |
| `crispasr` | `beclab/audio-crispasr` | `tts` | Voxtral-4B-TTS on CrispASR ggml (in-process, amd64 only) | in progress |
| `firered` | `beclab/audio-firered` | `tts`, `tts_clone`, `tts_design` | FireRedTTS3-Instruct in-process (ElevenLabs voice_id) | in progress |
| `breeze` | `beclab/audio-breeze` | `tts`, `tts_clone`, `tts_design` | Breeze TTS 2 in-process (ElevenLabs voice_id) | in progress |

`stt` means different engines on different bases (`qwen-asr` vs CTranslate2 vs
OpenVINO GenAI), which is why routing is keyed on `AUDIO_BASE` and not on the
capability alone.
Within a base, capabilities that need DIFFERENT models (each of the four
pyannote caps, and Qwen ASR vs Align) are separate clones — the wrapper
serves the first match and says so in the log.

> Keep this table in sync whenever a base is added or its capabilities change.

### Forced alignment: four flags, and what each one buys

`align` runs its own instance — its own checkpoint, so it always does — and computes
how many spans ride in one model call. What an operator may set is four flags, and
each one is here because it has a direction a deployment can want:

| flag | default | raising it |
|---|---|---|
| `--align-batch` | **on** | groups spans instead of sending one a call; a call holds more of the card. ⚠️ The output is NOT byte-identical: on the meeting corpus this branch measured its throughput against, 226 of 17550 timestamps moved (1.29%) -- median one 80 ms cell, largest 2240 ms -- with 0 spans differing in text or length. Grouping changes what a call is padded to, and the timestamps follow. Independent of `--no-kv-cache`: the cost factors follow that flag on their own, so either flag alone is priced correctly. With the cache on a position costs about five times as much, so the same grant buys a correspondingly smaller group. 🔴 **On by default as of this branch**, so a deployment that sets nothing gets grouping and the timestamp differences above; write `--align-batch off` for one span a call, which is what shipped before. |
| `--no-kv-cache` | off | stops the checkpoint building a cache nothing reads: a position costs 28 kB instead of 138 kB. Six span shapes, every output digest unchanged. ⚠️ **That is a price a position, not a saving on every request.** It buys room where a call's peak is set by the language model -- a group of short spans, which is what `--align-batch` creates. A single long span is bounded by the encoder, which holds no cache, and there it saves nothing: measured on the shipped 4 GiB grant, a 30 s span occupied 238 MB with the cache and 238 MB without, because the count of positions rises by as much as their price falls |
| `--gpu-budget-fraction` | `0.5` | the share of what the grant leaves that one call may reach for. Lower is slower; higher is faster and more likely to have a call refused by the card. A refusal is not a `503` — the engine splits the group and retries, and the caller is not told; it shows as an `ooms` count in the telemetry and a `WARNING` in the log |
| `--max-span-seconds` | `300` | the longest span the model can answer for. The output head is 5000 classes on an 80 ms grid, so 400 s is the last timestamp it can express — past it the argmax saturates and the answer is wrong rather than missing |

The batch response carries two booleans that answer different questions: `batch` means the request used the multi-segment form, which predates this flag and is true either way; `grouped` says whether `--align-batch` put several spans into one model call.

#### What these four accept

| flag | accepted | anything else |
|---|---|---|
| `--align-batch`, `--no-kv-cache` | `1`, `true`, `yes`, `on`. The bare flag counts as on, and so does `--align-batch=` with nothing after the `=` -- an unset chart value renders that way, and it means the flag was written, not that it was turned off | the default is used, and startup logs `is not on or off; using <default>`. ⚠️ This changed with the default: while every switch defaulted off a bad value landed on the default anyway, so nothing was said |
| `--gpu-budget-fraction`, `--max-span-seconds` | a number | the default is used, and startup logs which flag could not be read and what it fell back to |

⚠️ `off`, `0`, `no` and `false` all turn a switch off and are not a mistake, so they say
nothing. It is the value in neither list -- `--align-batch enable` -- that falls back and warns.

⚠️ This is how the shared `ENGINE_ARGS` parser reads any flag, not something `align` does
differently; the table is here because these are the four flags this section is about.


Everything else that moves a number is a constant. `--budget-positions` and
`--group-slack-positions` exist so a sweep has one variable, and live in the bench
chart's injected copy rather than here; the sweep found 20, 50 and 100 to be one
result on the clock, so exposing the threshold would ask an operator to tune a
number the measurement says is flat, in a unit nobody owns.

**A group the card refuses is split and retried, and nothing is remembered.** Every
member of it is innocent — the problem is how many rode together — so the group
halves, both halves go again, and the request completes. The budget is not lowered
and the next request is sized the same way. An earlier version remembered the failed
cost as a ceiling that only ever fell, which recovered just as well and left the
process at one span a call for its whole life, at about a tenth of the throughput,
with a complete response and an error count of zero. **The splitting was never the
problem; the memory was.**

The caller is not told. A prediction that was wrong about this machine is ours to
find, not theirs to handle — it is a `WARNING` in the log and an `ooms` count in the
telemetry, and what it points at is `--gpu-budget-fraction` and the two factors. A
group that fails for any other reason is one member the library could not read, so
each member goes alone and only the bad one comes back an error. A group of one that
fails is that span, which retrying cannot change.

`GET /v1/audio/align/telemetry` is what this engine has seen of the card and the
container it shares — read-only, a bounded ring of recent requests, and
`gpu.outside` (free plus our own reserved pool) is the field that moves only when
another container does.

### Per-base notes worth knowing before editing

**`qwen`.** One CUDA-torch image. `qwen-asr` is a single PyPI package: ASR
(`Qwen3ASRModel.from_pretrained`) and Align (`Qwen3ForcedAligner.from_pretrained`)
are two loads of different checkpoints, same recipe as the four pyannote caps
on one image. Offline `stt` and WebSocket `stt_stream` share one ASR load,
kept off each other with a `threading.Lock`. Official streaming is gated to
vLLM; this wrapper runs the same 2s accumulate + prefix-rollback state machine
on `model.generate`. Chart `ENGINE_ARGS` that named vLLM flags are still parsed
so they do not show up as leftovers. `qwen-asr` is installed `--no-deps`;
gradio / flask / sox stay out.

**`ov`.** Intel iGPU and discrete Arc share this image. There is no vLLM on it:
`stt_stream.py` branches on `AUDIO_BASE=ov` and loads `openvino_genai.ASRPipeline`
on device `GPU` when `OLARES_GPU_MODE` starts with `intel` (a written `--device`
still wins; quota `0` infers `CPU`). Converted IR is
preferred (`openvino/` next to the HF snapshot); a missing IR is exported once
with `optimum-cli` and reused. `--batch-max-spans` above 1 sends the group to
one `generate()` (this image patches GenAI to accept a list of waveforms);
encode+decode inside that call stays serial — a stacked decoder copied the
first clip onto every span. `stt_stream` keeps the same WebSocket contract
(`partial` / `final`) but **does not transcribe until the client stops** —
OpenVINO streams decoder tokens after the utterance, which is the Intel
tradeoff against vLLM's incremental encoder cache. Do not emit live partials
while audio is still arriving. amd64 only: Intel GPU is x86_64, and CI passes a
single-arch `slices` like `crispasr`. Align stays on the `qwen` base.

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
1 on CUDA OOM.

`diar` takes three more knobs per request, because which answer is the right one
depends on what the caller does with it. `exclusive=1` returns pyannote 4's
non-overlapping diarization: a caller who cuts the audio along these turns and
transcribes each piece otherwise sends the overlapping seconds twice and gets the
same words back twice. `min_duration_off` (a pause shorter than this is filled
rather than ending the turn) and `clustering_threshold` (higher clusters less
eagerly, so fewer speakers) are hyper-parameters the pipeline reads off itself
mid-run, so they are set on it for the length of one job and put back after —
which is sound only because the task runner runs one job at a time. The same two
can be pinned image-wide with `--min-duration-off` / `--clustering-threshold`, and
`--exclusive` moves the default. The response echoes all three, a pipeline default
included, and says `exclusive: false` when the build has no non-overlapping output
to give. `speaker_centroids` rides along when the clustering's rows can be named
with confidence; it is for looking at, not for matching against `speaker_embed`
vectors, which come from a different model in a different space.

`enhance` windows long clips (120 s, shrinking to 60 or 30 on a
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

Latency presets, in 80 ms frames, via `ENGINE_ARGS --latency-preset
low|high|veryhigh` (each of the five attributes is also settable on its own:
`--chunk-len`, `--right-context`, `--fifo-len`, `--update-period`,
`--spkcache-len`) — the default is **high latency**, because transcription streams from a separate `stt_stream` engine and
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

**`firered` / `breeze`.** Two bases, one HTTP surface (`wrapper/caps/tts_el.py`).
Boss requirement is one llm-init instance that lists voices, clones, designs, and
speaks — ElevenLabs `voice_id` first (`GET /v1/voices`, `POST /v1/voices/add`,
`POST /v1/text-to-voice/design` then create, `POST /v1/text-to-speech/{voice_id}`).
OpenAI `/v1/audio/*` aliases and `/v1/audio/tasks` are not mounted on these two.
Long jobs use `async=1` + `/v1/tasks`. Neither model ships a preset
catalog: premade ids are design prompts that freeze a sample onto the instance
PVC the first time they are spoken, then replay through clone. FireRed loads
**Instruct only** (`FireRedTTS3Instruct`); Base is a second 8.5 GiB checkpoint
that cannot stay resident and is `--exclude`d from `MODEL_SOURCE`. Breeze is the
single 3B. Both images clone upstream at a pinned SHA (same `.pth` trick as
`soulx`) and run official PyTorch in-process. `flash_attn` is not built: Instruct
accepts SDPA. Clone always needs the reference transcript (`description` /
`ref_text`).

**`audio_llm` and `audio_s2s` are reserved, not served.** No base implements them:
the open models that do are, as of 2026-08, either research-licensed or too heavy
to share a card with the rest of these. The capability names stay declared in
llm-init and the gateway so nothing downstream has to change when a base for them
does land.

## Build

**CI (release).** Listing bases (`qwen`, `fasterwhisper`,
`pyannote`, `firered`, `breeze`) and `speakrs` keep live triggers. Other
`<base>-ci.yml` files stay, with `push` / `pull_request` / `workflow_dispatch`
commented out, so a wrapper edit or `v*` tag does not rebuild them. Each live
`bases/<base>/**` change (or a `v*` tag, or a manual `workflow_dispatch`) runs
`.github/workflows/<base>-ci.yml`, which builds
`linux/amd64` **and** `linux/arm64` (native runners: `ubuntu-latest` +
`ubuntu-24.04-arm`), merges them into one multi-arch tag on `beclab/audio-<base>`
(`:latest` + `:sha-<short>`, or `:<tag>` on a release tag). PRs lint only. Uses
the `DOCKERHUB_USERNAME` / `DOCKERHUB_PASS` repo secrets (login identity with push
access to the destination namespace; the namespace is an input, default `beclab`).

The lint job that gates every PR byte-compiles the wrapper, runs `tests/` and runs
`ruff check wrapper tests`, blocking, against the narrow rule set in `ruff.toml` —
mistakes, not style; the excluded families are house style and the file says which
and why. It installs **ffmpeg**, without which the FireRed/Breeze tempo assertions
skip themselves. The wrapper targets **Python 3.10** at the oldest (`X | Y`
annotations in `caps/tts_el.py`); the images all run newer, and CI lints and tests
on 3.11.

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
4. `.github/workflows/<base>-ci.yml` — copy `qwen-ci.yml` and change the
   paths, `base` and `repo`; the build itself is the shared `build-image.yml`.
   Park a non-listing base by commenting out its `on` triggers (leave `speakrs`
   live).
5. Update the **Bases** table above.
