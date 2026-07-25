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
until the engine's `/v1/models` is alive.

## Layout

```
wrapper/                 shared Python package (used by every base)
  gpu.py                 generic gpu_* /metrics
  contract.py            /v1/models + /health surface (load-gated)
  app.py                 entrypoint: dispatch by MODEL_SUPPORTS
  caps/                  capability implementations
bases/<base>/Dockerfile  FROM the engine base image + build-time deps + wrapper
.github/workflows/       one <base>-ci.yml per base (workflow_dispatch + push/tags)
Makefile                 local hand-build to a personal registry (dev phase)
```

## Bases

| Base | Image | Capabilities | Engine / runtime | Status |
|---|---|---|---|---|
| `qwen` | `beclab/audio-qwen` | `stt`, `stt_stream`, `align` | qwen-asr in-process vLLM (`Qwen3ASRModel.LLM`) | active |
| `fasterwhisper` | `beclab/audio-fasterwhisper` | `stt` | faster-whisper (CTranslate2) | planned |
| `pyannote` | `beclab/audio-pyannote` | `vad`, `diar`, `speaker_embed`, `enhance` | pyannote / speechbrain / silero (torch) | planned |
| `nemo` | `beclab/audio-nemo` | `diar_stream` | NVIDIA NeMo | planned |

> Keep this table in sync whenever a base is added or its capabilities change.

## Build

**CI (release).** Each `bases/<base>/**` change (or a `v*` tag, or a manual
`workflow_dispatch`) runs `.github/workflows/<base>-ci.yml`, which builds
`linux/amd64` and pushes `beclab/audio-<base>` (`:latest` + `:sha-<short>`, or
`:<tag>` on a release tag). PRs build only. Uses the `DOCKERHUB_USERNAME` /
`DOCKERHUB_PASS` repo secrets (login identity with push access to the `beclab`
org; the namespace is hardcoded, not derived from the username).

**Dev.** Same Dockerfile, different destination — no separate dev build path.
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
   deps at **build time** (no runtime pip), `COPY wrapper /app/wrapper`, default
   `CMD ["python3", "-m", "wrapper.app"]`.
2. `wrapper/caps/<cap>.py` — implement the capabilities; expose `/v1/audio/*` and
   wire `wrapper.gpu.mount_metrics` + `wrapper.contract.register` so the contract
   above is satisfied.
3. Teach `wrapper/app.py` to dispatch the new `MODEL_SUPPORTS` values.
4. `.github/workflows/<base>-ci.yml` — copy `qwen-ci.yml`, change the paths and
   the `DH_REPO` namespace.
5. Update the **Bases** table above.

## Branching

`main` starts empty; work lands on `feat/audio-stt` first and is promoted to
`main` later. Because CI publishes only from `main` / tags, dev images are built
locally (see above) until then.
