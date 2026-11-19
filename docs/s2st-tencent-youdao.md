# Tencent/Ming and Youdao/Ming CUDA engines

Status: Tencent packaging candidate; real-GPU transport validation and the formal benchmark are in progress. Youdao deployment remains gated; no published image or successful Olares deployment is claimed.

The clean integration branch starts at main `82b054b225c7f538742bec23eff8b20aa69ffe65`.
The previous `codex/s2st-models` work is archived unchanged at `8b1b664` (its first commit is `8ac8585`). It has no uncommitted changes to preserve. No code from its S3 experiment is imported here.

## Product boundary

Each engine owns all its resident subprocesses and exposes one S2ST model. Internal loopback services are implementation details; independently scheduled Olares applications are not part of this design. The Tencent cascade owns Hy-SimulMT GGUF and MING; the Youdao cascade owns R2T2, T3PO GGUF and MING. Only zh2en and en2zh are in scope. AMD64 and ARM64 share a CUDA workflow and image manifest. Intel will use a separate OpenVINO backend behind the same session contract; it is not implemented in this draft.

No process may download weights while serving. llm-init prepares all snapshots; ready is false until every required worker has loaded and passed its placement and health checks. Concurrency is one session on a 24 GB device. Worker death, incomplete model turns and uncommitted EOS text fail the request, with no hidden synthesis retry.

## Frozen upstreams

- Tencent SimulInfer: `7efcb18c521a7025b37d4bc2230a1df8ade83950`.
- Tencent audio GGUF llama.cpp fork: `d38e66aac7bb18101f417338b795e4546a5a1a59`.
- MING: `028c1eace193e7acdb256453da89b6e9207e052b`.
- Youdao T3PO: `0ac9d0c806f8f139ebaa83d394ffbd5057ff3ae3`.
- Youdao R2T2: `62757213befa282e7fbfe1bfde3c29577781527c`.

The official Tencent `serve_llamacpp.py` exists at the frozen revision (Git blob `ceccfbfd979c213c42e6a91e2eccb212d9c648bd`). Its HTTP bridge uses repeated chunked audio completion and explicitly does not preserve encoder state. The feasibility baseline uses the official native CLI with persistent encoder state. Switching to that HTTP bridge is a behavior change and requires an equivalence/performance test before it can replace the baseline.

## Required integration changes, separately audited

- MING substitutes the optional upstream TTS service. Source-speaker-only conditioning uses the official CAMPPlus CUDA encoder, CFG 1, temperature 0, seed 0, native text preparation length 24, and native EOS checks. No generated-audio history is fed back.
- English TTS receives full words; a completed Youdao translation segment must release its final word immediately. The raw model translation remains byte-for-byte intact.
- Only unambiguous sports scores in sports context are rendered as `two to zero` for speech. Dates, negative numbers, decimal numbers and the original translation are unchanged.
- The Youdao pending-source deadline uses official feed/flush serially. Continuous ASR arrivals do not reset the deadline. Final input always drains ASR, translation and TTS in order.
- A causal bounded playout queue applies pitch-preserving rate adjustment. Raw synthesis and delivered audio are both retained in benchmark evidence. Offline trace replay is not a live playback measurement.

## Benchmark gate

The full benchmark has not started. The 4-second Youdao deadline comparison is rejected: it committed a partial name as `was` and introduced a premature first/second-appearance contradiction. The 8-second high-quality candidate completes the full validation recordings but still has an observed 12.969-second maximum playout gap; this latency is disclosed and awaits product acceptance before the full run. Complete-segment delivery and the newly exposed Chinese cadence case remain under review. Until they pass, image publishing and numerical results advertised as final remain disabled. The input manifest, revisions, worker dependencies, model parameters and playout policy must be frozen before the full benchmark. ASR and quality scorers run only after performance inference releases the GPU.

## Additional native compatibility changes retained from feasibility

- R2T2 keeps its official model and BF16 forward in vLLM 0.14.0. Deployment fixes its CUDA memory budget at 0.20, context at 4096, concurrency at one and input increments at 4000 ms. These are serving parameters, not a replacement recognizer.
- R2T2's rolling audio/text window uses actual per-chunk sample counts. Fixed text-chunk trimming becomes misaligned when deployment uses larger increments; the ledger removes the corresponding audio and text together. The original method hash and generated method source are retained in the experiment evidence.
- The R2T2 WebSocket handler passes the actual residual PCM to the original recognizer before its original finish call. Finalization also runs when its current buffer is empty but accumulated audio still needs final commitment. The separate 500 ms EOS silence is logged outside the original input denominator.
- T3PO keeps the official prompt and feed/flush protocol. Its OpenAI `min_tokens=1` is adapted to a nonempty llama.cpp GBNF result without changing that prompt. `repetition_penalty` uses llama.cpp's parameter spelling. A wrong-language response can trigger one explicitly logged same-prompt English-constrained attempt; this is separate from the disabled TTS retry policy.
- Youdao's three heavy CUDA stages share the verified process FIFO gate. This avoids ASR/MT repeatedly overtaking queued synthesis. Waiting and held time are recorded separately; model weights remain resident on CUDA.
- The R2T2 streaming VAD keeps its official CUDA neural weights. Tencent’s source-speaker snapshots use ordinary peak normalization; no extra neural VAD is invoked by that path. File decoding, text processing, ordinary resampling, PCM serialization and pitch-preserving audio-rate filters use CPU.

Effective dependency records must come from imported modules, not only distribution metadata. In particular, the validated MING overlay and the R2T2 venv have different Torch/Transformers stacks. They must not be merged by an unpinned pip resolution. ARM64 CUDA wheels are checked separately; a CPU-only ARM wheel is not an acceptable build fallback.


## Tencent transport and deployment evidence

The official native CLI was compared with file input and with the PCM16 stdin wrapper, on two unspliced 40-second source prefixes in opposite directions. All six runs matched token IDs, UTF-8 output bytes, audio turns, consumed samples, EOG values and context positions; timing was excluded from this equivalence assertion. The patch adds PCM input, telemetry and process-residency control; encoder, decoder, sampler, prompt and turn sizes are retained.

Both checkpoint revisions are fixed: Tencent GGUF `8d4e9381419ac17d17e29359a297109afce02066` and MING `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0`. The engine reads already provisioned Common snapshots without serving-time downloads. CUDA has one workflow with AMD64 and ARM64 slices. Required Triton, CUDA headers and compilation tools remain present for the validated compiled MING flow.

`audio_s2s` is the existing llm-init capability. The new routes are POST `/v1/audio/speech-translations` (WAV input, complete JSON/WAV or incremental NDJSON) and WebSocket `/v1/audio/speech-translations/stream` (live PCM16 input). Their public protocol tests cover audio before input end, end-of-input drain, invalid frames and disconnection cleanup. Nonstreaming output does not insert delivery-wait silence.

The local PTL chart is experimental version 0.1.0 and uses llm-init v1.7.26 with Common Hugging Face storage and Common run state. Existing chart versions and other applications are unchanged. The report's primary total S2ST RTF includes each session's model initialization; steady-pipeline and pure translation RTF are secondary fields, not substitutes.

Non-streaming WAV requests establish HTTP headers before model work, then return a single complete JSON body with `type: done` and the complete WAV. This avoids the Common proxy response-header deadline for recordings whose processing exceeds 60 seconds. A failure after headers returns an explicit `type: error` record without audio; callers must verify the terminal type. Streaming NDJSON and WebSocket delivery are unchanged.


## Dependency repair packaging

The incremental CUDA dependency image reuses the immutable AMD64/ARM64 parent `deps-7bc3825f4572`, preserving the verified native binaries and neural stack byte-for-byte. Only `imageio-ffmpeg==0.8.1`, already used by the working host, is added for the official `TempoStream` transport. The image build actually converts PCM through that transport; an import-only probe cannot detect this lazy dependency. `bases/s2st-tencent/vendor.Dockerfile` retains the complete source-build recipe for bootstrapping a replacement parent. No model weights, decoding parameters or shared MING code change in this repair.

## Olares same-process native entry

The first installed two-process candidate exposed alternating roughly ten-second GPU waits in realtime input, and an early terminal-ack race. The new native entry compiles only the pinned official CLI translation unit against the unchanged published native libraries. Its PCM stream is a private FIFO instead of process stdin; model arguments, encoder, decoder, sampler, prompt, turn sizes and MING's seven trusted files are unchanged. Two original 40-second recordings in opposite directions match the standalone entry's complete token IDs, UTF-8 bytes, consumed samples, turns, EOG and context positions. This is an inference equivalence check, not a measured realtime performance result.

MING and the native entry run in the same GPU PID. The public server and loopback control plane do not disable the Olares preload or change its scheduling policy. Native sessions are serialized, owned by their cache directory, and drained before terminal success is acknowledged. A failed cleanup is an explicit request failure. The new packaging adds only a small compiled entry; no CUDA kernel or neural dependency rebuild is required. Actual installed realtime revalidation is still required.


## Bounded WebSocket upload storage

The installed image8 full ASAP cohort retained one failed long recording: Linkerd reported TCP user-timeout while the inference-facing 16-frame receive queue applied prolonged upload backpressure. Neither GPU worker nor Pod restarted; completed MING phrases recorded native EOS. This failure is retained as 19/20, not retried into a passing old cohort.

WebSocket input now drains valid frames into a private temporary file, bounded by the existing 900-second mono PCM16 limit (28.8 MB). The native reader still processes samples in order and source-speaker conditioning advances only with native consumption. This changes upload storage, not inference, TTS settings or Olares/mesh timeouts. EOF drains the complete recording; disconnect closes the spool after task cancellation. A ten-MiB burst regression verifies complete ingestion before a slow consumer starts, and checks PCM identity, sample count, digest, duration enforcement and cleanup. A fresh installed full cohort remains required.
