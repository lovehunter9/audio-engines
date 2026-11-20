# Tencent/MING and Youdao/MING CUDA engines

## Current validation scope

Tencent is packaged and installed on olarestest003. The development image `s2st-tencent-dev-20261120-9` completed all 20 original RealSI recordings both with ASAP upload and with 1x realtime input: 20/20 successful in each cohort, no synthesis retries, all source samples consumed. Independent generated-audio ASR and automatic text/audio scoring ran only after timed inference released the GPU. This is a public-data local benchmark, not a reproduction of Tencent's GEMBA 84.7 score or an untouched held-out set.

The ASAP input-weighted total S2ST RTF is 0.7565 (worst recording 0.9285). Realtime input has a 17.20-second maximum FIFO playback gap and 25.53-second maximum playback tail. The longest gap occurs during continuous source speech. Throughput and protocol completion have passed; uninterrupted listening has not. The complete original source, raw model translation, actual delivered PCM and arrival-paced replay are retained. No silence is removed to disguise waiting.

Youdao deployment remains gated on upstream quality and cadence validation. There is no published Youdao image or successful Olares deployment yet. Experimental host results do not establish compatibility with Olares time slicing. Only zh2en and en2zh are in scope. Intel is unimplemented and must use a separate OpenVINO backend behind the same session contract.

## Product boundary

One application exposes one S2ST engine. Its internal translation, recognition and speech services are owned implementation details, not separately scheduled Olares applications. Tencent owns Hy-SimulMT GGUF plus MING; Youdao will own R2T2, T3PO GGUF and the exact same MING implementation and settings.

All weights must already exist in llm-init's Common cache. Serving never downloads weights. Readiness waits for model loading, placement and health checks. Concurrency is one session. Worker death, abnormal native EOS, incomplete source consumption or uncommitted translated text fail the request. There is no hidden synthesis retry or replacement by reference text.

## Frozen upstreams and models

- Tencent SimulInfer: `7efcb18c521a7025b37d4bc2230a1df8ade83950`.
- Tencent audio GGUF llama.cpp fork: `d38e66aac7bb18101f417338b795e4546a5a1a59`.
- MING source: `028c1eace193e7acdb256453da89b6e9207e052b`; snapshot: `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0`.
- Tencent GGUF snapshot: `8d4e9381419ac17d17e29359a297109afce02066`.
- Youdao T3PO: `0ac9d0c806f8f139ebaa83d394ffbd5057ff3ae3`.
- Youdao R2T2: `62757213befa282e7fbfe1bfde3c29577781527c`.

The pinned Tencent `serve_llamacpp.py` HTTP bridge explicitly does not preserve encoder state. This engine instead embeds the official persistent-state native CLI. It does not substitute repeated stateless audio completions for the validated native path.

## Shared MING contract

The downstream uses source-speaker-only conditioning with the official CAMPPlus CUDA encoder, CFG 1, temperature 0, seed 0, native text preparation length 24 and normal EOS checks. Generated speech is never fed back as conditioning. English receives complete words. Unambiguous sports scores are rendered for speech in sports context; the raw translation remains intact. Dates and decimal values are not rewritten by this score renderer.

Tencent supplies committed incremental text directly to the shared Speech implementation. No extra complete-sentence gate is required. The actual full Tencent cohorts produced 528 MING requests with ASAP input and 1911 with realtime input; concatenated request text exactly matches every corresponding raw translation, including whitespace. All native fragments ended normally. These checks do not prove pronunciation or audible completeness; both cohorts were independently transcribed and remain available for listening.

A causal bounded playout queue applies pitch-preserving rate adjustment. Raw synthesis and delivered audio are distinct evidence. The arrival-paced WAV reconstructs the recorded FIFO schedule; it is not a physical sound-card recording. Nonstreaming output contains the complete audio without artificial delivery-wait silence.

## Required native and transport adaptations

1. The official native CLI gains bounded PCM16 input, telemetry, private-FIFO session ownership and a shared-library lifecycle entry. Model arguments, encoder, decoder, sampler, prompt and turn sizes remain unchanged. Two original unspliced 40-second recordings in opposite directions match standalone native token IDs, UTF-8 output, consumed samples, audio turns, EOG values and context positions. Timing is excluded from that equivalence assertion.
2. MING and the native entry share one GPU PID. The first installed two-process candidate exposed alternating roughly ten-second GPU waits. This lifecycle change retains the Olares preload and scheduling policy. Native cleanup and lock release finish before terminal success is acknowledged; cleanup failure is explicit.
3. WebSocket upload drains valid frames to a private temporary file bounded by the 900-second mono PCM16 limit (28.8 MB). The native reader still consumes samples in order, and speaker conditioning advances with native consumption. EOF drains all input; disconnect cancels work and closes the spool.
4. Nonstreaming HTTP establishes response headers before long model work, then returns one complete JSON/WAV result. Errors after headers return an explicit terminal error without successful audio; callers must inspect the terminal type. NDJSON and WebSocket remain incremental.

The image8 full ASAP cohort failed one long WebSocket upload because prolonged inference backpressure hit Linkerd's TCP user-timeout. No Pod restart or OOM occurred. That 19/20 cohort remains failed. A separate HTTP diagnostic and then a fresh image9 474-second WebSocket regression succeeded; image9 subsequently completed the two new full 20-recording cohorts above. No old failed row was retried into a passing old cohort.

## Youdao experimental adaptations under review

The frozen host candidate retains official feed/flush and autoregressive recognition. Its rolling audio/text ledger uses actual sample counts, and EOF delivers residual PCM before original finalization, including the empty-buffer case. Official CUDA neural VAD remains. Added EOS padding is separately logged and excluded from the source denominator.

Source number tails, including Chinese scales, digit/scale mixtures and decimal points, wait for the actual remaining number or unit before append-only MT. No source text is rewritten; EOF flushes all remaining characters. This prevents a reproduced split of 七十万 into 七十 and 万 that had caused a tenfold magnitude error. Physical VAD markers never force textual commitment: the original ASR reset is the final-text acknowledgement. A removed early-flush adapter had submitted half-words and half-numbers even with reset=false. The original VAD neural computation remains intact. The age of pending source text, rather than repeated ASR arrivals, controls the bounded translation deadline. Incremental T3PO SSE preserves the official prompt and latency policy. Its OpenAI minimum-one-token setting is adapted to llama.cpp's nonempty grammar, and repetition-penalty spelling is translated. Wrong-language responses and abnormal completions fail explicitly; an older wrong-language retry experiment is not the selected path.

The extra Youdao-only complete-sentence gate and speech-oriented prompt are removed. ASR delivery uses the original recognizer with 2000 ms increments, actual rolling audio/text sample alignment, final residual drain and explicit 500 ms EOS padding. Native T3PO keeps its original high policy. Only trailing English function words are held until the next translated content; the full canonical translation and exact speech characters are preserved and EOF flushes any remainder. This upstream boundary addresses a reproduced native-EOS failure on an isolated dangling preposition. It does not change MING or add retries.

The recognizer uses its official HF neural forward through an audited dispatch of the original streaming algorithms. A private Transformers 4.57/tokenizers namespace isolates ASR from MING's unchanged Transformers 4.46/tokenizers stack. Four controlled MING outputs match exact PCM both with the namespace and with all three GPU residents loaded. Full original live health/sport/science regression and deployed-product quality remain separate gates; a build/import pass is not a neural acceptance result. The product worker owns ASR/VAD, the original native T3PO server and unchanged MING in one GPU PID; the separate bridge process is CPU only. Older four-process prototypes are not the selected deployment.

## Packaging and deployment evidence

CUDA has one workflow and one multi-architecture manifest containing AMD64 and ARM64. Development image9 compressed layers are 3.493 GiB on AMD64 and 3.646 GiB on ARM64. AMD64 has the installed 003 inference results above; ARM64 passed build/import checks but has no GPU hardware inference validation.

The actual stack is Torch 2.13.0+cu130, CUDA 13.0.3 and torchaudio 2.11.0+cu130. The immutable multi-architecture dependency parent preserves native binaries and the validated neural stack. Required Triton, CUDA headers and compiler tools remain for compiled MING flow; no weights are embedded. `vendor.Dockerfile` retains the full source bootstrap recipe. The TempoStream dependency repair adds the already validated imageio-ffmpeg version and executes real PCM conversion during image build.

The experimental PTL chart remains version 0.1.0 with llm-init v1.7.26, Common Hugging Face storage and Common run state. No formal submission is made. POST `/v1/audio/speech-translations` accepts WAV for complete JSON/WAV or incremental NDJSON; WebSocket `/v1/audio/speech-translations/stream` accepts live mono PCM16. Nonstreaming, incremental delivery, input drain, invalid frames and disconnect cleanup are covered. Inference-related model initialization is included in per-request total RTF; service cold start is separately recorded.

The private T3PO server has both local and prebuilt Web UI assets disabled using upstream CMake options. CI validates the empty-UI target before CUDA compilation; the public product console is llm-init. This avoids the upstream b0/latest front-end bundle mismatch without changing inference.

Youdao accepts the official high/default-native policy through ENGINE_ARGS --translation-latency high|native. The selected policy is recorded in the completion evidence. Low/forced-short-latency is not selected, and MING settings are independent of this upstream policy.
