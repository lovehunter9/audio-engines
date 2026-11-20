# Tencent/MING and Youdao/MING CUDA engines

## Selected deployments and acceptance scope

Both engines expose one Olares application and one active S2ST session. Their recognition, translation and speech services are internal implementation details. All weighted neural computation runs on CUDA in one GPU PID. Normal audio decoding, PCM transport and text processing can run on CPU. The Olares time-slicing policy and injected preload are unchanged.

- Tencent: `s2st-tencent-dev-20261120-9`; complete installed 20-recording ASAP and 20-recording realtime cohorts passed without inference retries.
- Youdao: `s2st-youdao-dev-20261120-12`, runtime revision `a2fc1bc4f630c0f38c22986150e27de428349d17`. Five complete-source engineering diagnostics, six installed bidirectional route checks, four controlled MING PCM comparisons and disconnect recovery passed. Its first continuousASAP cohort completed11 originals without inference errors before this conversation stopped the load after read-only GPU temperatures rose through93/98/101C. That interrupted cohort is not a20-recording pass. An independent continuous20-original1x realtime cohort is now being prepared on the identical image and parameters; no successful rows are grafted between cohorts. Sustained performance, complete-corpus scores and listening acceptance are not yet asserted. Earlier failed cohorts remain separate evidence.
- Only Chinese-to-English and English-to-Chinese are enabled. Intel is unimplemented. Future Intel support must use a separate OpenVINO backend behind the same session contract, without silently falling back to CPU or XPU.
- These are MINE development images and experimental PTL charts, not formal releases or market submissions. ARM64 has build/import validation only, without GPU hardware acceptance.

Tencent ASAP input-weighted total S2ST RTF is 0.7565, with a worst recording of 0.9285. Its realtime cohort has a maximum FIFO playback gap of 17.20 seconds and a maximum playback tail of 25.53 seconds. Throughput and protocol completion have passed; uninterrupted listening has not. Continuous original-source and arrival-paced context clips are retained to distinguish source pauses from additional waiting. Coarse reference segments alone cannot establish continuous speech.

The image12 environment diagnostic completed atRTF0.804, and health/science realtime controls completed atRTF1.011/1.013. Education retained all4246 original2s ASR characters and the correct meaning of "four years or bust". Its final translation differs from the prior quality control in several synonymous phrases, so byte-identical MT is not claimed. The full text comparison and the rejected exact-string gate are retained.

The continuous-load image12 sport diagnostic reachedRTF1.452; a separate single run after passive idle, with the same loaded image, parameters, recording and identical ASR/MT, reached0.665. These are both raw observations, not replacements for each other. Temperature, SM clocks and NVML event masks show materially different operating states; low idle clocks alone are not thermal evidence. No fan, power, clock, driver or GPU scheduling setting was changed. The new full dataset policy explicitly freezes `performance_acceptance=false`: characterization does not waive the throughput or delivery goals.

## Frozen upstreams and models

- Tencent SimulInfer: `7efcb18c521a7025b37d4bc2230a1df8ade83950`.
- Tencent audio GGUF llama.cpp fork: `d38e66aac7bb18101f417338b795e4546a5a1a59`.
- Tencent GGUF snapshot: `8d4e9381419ac17d17e29359a297109afce02066`.
- Youdao T3PO: `0ac9d0c806f8f139ebaa83d394ffbd5057ff3ae3`.
- Youdao R2T2: `62757213befa282e7fbfe1bfde3c29577781527c`.
- Youdao native llama.cpp: `ad6c66839af3c5646fba8c6c2e2087a1e4e38948`.
- MING source: `028c1eace193e7acdb256453da89b6e9207e052b`; snapshot: `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0`.

The pinned Tencent `serve_llamacpp.py` bridge explicitly does not preserve encoder state. This engine embeds the author's persistent-state native CLI instead of substituting repeated stateless audio completions.

## Shared MING contract

Youdao reuses Tencent's seven modules byte-for-byte: `ming_worker.py`, `speech.py`, `delivery.py`, `clause.py`, `playout.py`, `spoken_numbers.py` and `spoken_scores.py`. Their settings are unchanged. There is no Youdao-specific TTS configuration.

Source-speaker conditioning uses the official CAMPPlus CUDA encoder, CFG1, temperature0, seed0, native text preparation length24 and normal EOS checks. The pinned SimulInfer client names its OpenAI-compatible HTTP transport `vllm_omni`; that legacy enum selects only the request shape to the private MING endpoint. No vLLM-Omni package or engine is installed or executed. Its legacy request/diagnostic seed42 is not the synthesis seed: the fixed native MING worker explicitly calls `torch.manual_seed(0)` per request. The frozen MING implementation and native request events, rather than the SDK transport label, define the actual backend and settings. Generated speech is never fed back as conditioning. English input contains complete words. The shared frontend renders numbers and unambiguous contextual sports scores for pronunciation; raw model translation is retained. An isolated dangling preposition is held upstream until subsequent translated content or EOF, without changing MING.

A causal bounded playout queue performs the accepted pitch-preserving rate adjustment. Synthesis PCM, actual delivered PCM and arrival-paced replay are distinct evidence. Replay reconstructs recorded FIFO arrival waits and is not a physical sound-card recording. No silence is removed to disguise latency. Nonstreaming output contains the complete audio without artificial delivery-wait silence.

All40 Tencent outputs have exact concatenated MING input text matching their corresponding canonical translations, including whitespace, and all native fragments ended normally. They generated528 synthesis requests with ASAP input and1911 with realtime input. Input cadence affects phrase grouping even with the same downstream code; each cohort therefore requires independent generated-audio assessment. Four controlled PCM matches under concurrent ASR/MT load prove those inputs only, not every possible pronunciation.

## Necessary Tencent and shared transport adaptations

1. The native CLI gains bounded PCM16 input, telemetry, private-FIFO ownership and a shared-library lifecycle entry. The model arguments, encoder, decoder, sampler, prompt and audio turn sizes remain unchanged. Two opposite-direction, original unspliced40-second controls match standalone native token IDs, UTF-8 output, consumed samples, audio turns, EOG values and context positions. Timing is outside that equivalence assertion.
2. Native translation and MING share one GPU PID. The first two-process candidate exposed alternating roughly ten-second GPU waits. Native cleanup and lock release complete before terminal success; cleanup failure is explicit. No scheduler or preload setting is changed.
3. WebSocket upload drains validated PCM frames to a private temporary spool bounded by the900-second monoPCM16 limit,28.8MB. Inference still consumes samples in order and speaker conditioning advances with actual consumption. EOF drains residual input; disconnect cancels the session and closes its spool.
4. Complete HTTP responses establish headers before long inference to keep the proxy connection alive. Callers must validate the terminal payload, not infer model completion from HTTP200 or response headers. NDJSON and WebSocket delivery remain incremental.

Worker death, abnormal native EOS, incomplete source consumption, uncommitted text and bounded-buffer overflow fail explicitly. There is no synthesis retry or reference-text substitution.

## Necessary Youdao adaptations

The installed image12 PTL arguments are:

```text
--cache-dir /data/s2st --translation-latency native --asr-chunk-ms 2000 --asr-chunk-ms-zh 4000 --translation-chunk-ms 4000 --overlap-asr --overlap-translation
```

1. **Original recognition and finalization.** The rolling audio/text ledger uses actual sample counts. EOF drains residual PCM before the author's finalization, including the empty-buffer case. The original CUDA neural VAD remains. Explicit500ms EOS padding is logged separately and excluded from the source-duration denominator. Physical VAD markers do not falsely finalize text; the original ASR reset is the finality acknowledgement.
2. **Audited HF dispatch.** Original R2T2 streaming algorithms use the official HF BF16 neural forward. A private Transformers4.57.6/tokenizers0.22.2 namespace isolates ASR from MING's unchanged Transformers4.46.3/tokenizers0.20.3 stack. The original method hashes and audited dispatch changes are recorded. This is a declared backend dispatch adaptation, not a claim that an unmodified vLLM server is used.
3. **Lossless source boundaries.** Partial numbers, decimal points and Chinese scales wait for the actual remaining number/unit before append-only MT; recognized text is not rewritten. English recognition remains2s, Chinese4s. English source increments are losslessly coalesced to4s before MT to reduce repeated dispatch. Actual ASR reset and EOF drain pending text immediately. There is no benchmark-text detection, forced short-latency translation, extra complete-sentence TTS gate or speech-oriented prompt.
4. **Official translation policy.** The pinned prompt, native/balanced policy and incremental T3PO SSE are retained. The minimum-one-token OpenAI setting is mapped to llama.cpp's nonempty grammar, and the repetition-penalty spelling is translated. The age of pending source text, rather than repeated ASR arrivals, controls the bounded translation deadline. Raw foreign-script fragments remain visible with quality warnings, without rewriting or retry. Final speech input is checked against canonical MT using the pinned official join function, with explicit word-boundary whitespace accounted for.
5. **Opt-in scheduling overlap.** `--overlap-asr` bypasses only the global ASR stage exclusion; `--overlap-translation` bypasses only the nativeMT/MING stage exclusion within the same GPU PID. Serialized behavior remains the engine default. Actual ASR and MT request observers verify the selected flags; completion records preserve the settings. Neural steps, precision, prompts and MING settings do not change.
6. **Transport and cancellation.** The Youdao-owned internal MING HTTP server closes each completed connection rather than reusing idle loopback sockets. This is a safeguard following a ReadError whose exact network cause was not established; prior native synthesis had normal EOS. Public diagnostics preserve the actual error without retry. Concurrent cleanup owns and closes each socket once. Two cooperative yields in the author's two ASR processor loops allow event-loop progress; removing only those yields makes the actual original source AST identical. No neural call, parameter, state transition or PCM operation is removed.
7. **Independent bounded bridge reception.** Downstream speech backpressure must not stop reading the upstream WebSocket. The selected receiver drains original messages into a FIFO capped at64MiB and65536records, with explicit overflow, ordered delivery and cancellation. Source watermarks and speaker conditioning advance only when the consumer processes each original ordered event; receiving future messages does not expose future conditioning. Original heartbeat/read deadlines and WebSocket flow-control settings remain unchanged. Peak queue use is recorded per completed request.

The bridge sends `loading` before `init_ok`; the adapter waits on the same connection under one initialization deadline and never feeds PCM early. Real initialization errors fail the request. Loading progress is not an inference retry.

## Placement, packaging and Chart verification

Actual placement is checked after an uncached speaker extraction: CAMPPlus executed281 weighted nodes on CUDA and zero weighted CPU nodes; R2T2 parameters/nonempty buffers and the original VAD are CUDA resident; T3PO reports49/49 GPU layers and a CUDA0 weight buffer. Host NVML and container audit PIDs are linked through the process-namespace PID chain. A cache hit with zero profiled weighted executions is insufficient proof. Owned reference/embedding caches are archived using Olares Files only.

Each CUDA workflow builds AMD64 and ARM64 under one manifest. Registry-compressed layers are approximately3.493/3.646GiB for Tencent image9 and3.502/3.655GiB for Youdao image12. No weights are embedded. Torch2.13.0+cu130, CUDA13.0.3, torchaudio2.11.0+cu130, required Triton/headers/compiler tools and the validated imageio-ffmpeg PCM path are retained. The immutable dependency parent preserves the tested runtime; `vendor.Dockerfile` contains its source bootstrap recipe. The private T3PO WebUI is disabled with upstream CMake options; inference is unchanged.

Experimental PTL versions remain0.1.0/appVersion1.0.0 and use the same llm-init v1.7.26/Common pattern as current TTS charts. Serving only consumes existing llm-init Common weights and never downloads them. The normal Youdao PTL Chart was reinstalled without the temporary LAN endpoint override and passed all six bidirectional routes, disconnect recovery and four controlled PCM comparisons; its actual endpoint resolves to the system-configured HF service. This verifies normal installation with already cached pinned weights, not a fresh empty-Common public download. During earlier download isolation, a read-only LAN endpoint served only SHA/Git-blob-verified public files at the pinned revisions; llm-init performed all Common writes.

The R2T2 Chart downloads its complete repository with one weight file and small configuration/tokenizer files. The two VAD files are separate frozen HF sources because llm-init v1.7.26's repeated include forwarding otherwise leaves only the final pattern in its actual HF CLI. No additional weight version is selected.

POST `/v1/audio/speech-translations` supports complete WAV input with complete JSON/WAV or incremental NDJSON output. WebSocket `/v1/audio/speech-translations/stream` accepts live mono16kHzPCM16 and delivers translated text and24kHz speech. Validation covers both directions, actual observed ASR increments, complete input drain, invalid frames, disconnect cleanup, fresh recovery and zero inference retries.

## Benchmark interpretation and retained failures

The frozen RealSI protocol contains20 complete original recordings,10 per direction,777 reference segments and5747.040125 seconds of source audio. References, transcripts and reference boundaries are never inference input. ASAP throughput and1x realtime-arrival cohorts are distinct. Timing includes per-session initialization on a warm service, with source-reference and speaker-embedding caches emptied before each cohort; a clean-cache service cold-start benchmark is not claimed.

All timed inference finishes and releases the GPU before independent generated-audio ASR or COMET starts. ASR uses uniform16kHz, non-overlapping chunks of at most60s, complete sample coverage and no reference prompt/retry. BLEU, chrF++ and COMET are reported separately by direction for raw MT and generated-speech ASR. Scores are not accuracy percentages or reproductions of vendor GEMBA84.7/77.7. Some recordings were used during development, so this is not an untouched held-out evaluation. RTF below1 does not establish uninterrupted speech delivery; first audio, FIFO gaps, playback tail and continuous original-source context remain necessary.

Read-only temperature, SM-clock and NVML event masks are retained. Different thermal histories prevent simple code-speed attribution; no fan, clock, power, driver or scheduling setting is changed. Performance values are not normalized to an imagined clock rate.

Retained failed attempts are never retried into their old passing denominator:

- Tencent image8:19/20 ASAP, with one prolonged-upload proxy TCP timeout; no Pod restart/OOM. Image9's separate474s WebSocket regression and two fresh complete cohorts passed.
- Youdao image5 ASR2s:10 complete English originals, seven aboveRTF1, worst1.562; stopped before a full campaign. ASR4s improved throughput but misrecognized "or bust" as "are bust", changing meaning. Its full attempt retained17 successes and3 failures after a platform StopByUser operation; originating client unknown, no observed OOM.
- Image6: educationASAP0.658 with original source fidelity; sport1.499 and health HTTP ReadError. Image7: education0.703 and health1x complete, but sport1.423. Image8 only repaired overwritten audit metadata.
- Images9/10: whole sport failed with1011 keepalive timeouts after approximately101/89s. The initial ASR-only attribution was insufficient. A real loopback test with the production websockets15.0.1 version reproduced the outer bridge's16-frame backpressure timeout. Image11's independent bounded reader preserves every message and heartbeats under the same pause and deadlines; all58 S2ST tests passed before whole-source diagnostics. The full benchmark remains separate evidence.


## Adaptive upstream deadline protection and retained failure

The image11 all-ASR-serialized alternative completed the previously failing environment recording atRTF0.817, but sport reached1.452 and the diagnostic stopped before promotion. That fixed exclusion profile is not selected. Image12 retains normal ASR/MT overlap and uses the existing live FIFO ticket age to give an old MING request priority. After20s of pending/active MING work, new ASR and MT admissions queue behind it; already running calls finish normally. The shared60s MING deadline, seven MING modules, all neural operations, precision, translation prompts and source increments stay unchanged. This is upstream resource admission, not a TTS configuration change or retry. Queue identity checks reject dead/reused PID owners, cancellation removes only the caller's ticket, and actual protected calls are counted. Short/young MING requests keep the existing overlap. Real FIFO tests cover both modes, aged-MING protection, stale owners, resumption and cancellation. All63 S2ST tests and the dual-architecture CI passed. The installed candidate passed five complete-source engineering controls. Full sustained performance and generated-speech quality remain separately measured; these controls are not a full Benchmark pass.

## Device thermal interruption

The image12 ASAP campaign retains11 complete successful originals and state`interrupted`; this conversation sentSIGINT and stopped its own application after hostnvidia-smi andNVML showed101C with eventmask32. The GPU cooled to71C after releasing the load and55C at the next idle baseline. This is a disclosed measurement interruption, not an inferredOOM, model exception or discarded slow row. The new standalone1x cohort stops the owned workload at100C once or95C in two consecutive2s samples; it never pauses, cools and resumes inside a timed session. Fan, power, clock, driver and Olares scheduling settings remain unchanged. A complete1x result cannot substitute for the missing sustainedASAP throughput result.

## Optional upstream source admission candidate

`--max-source-speed` is a Youdao-only, opt-in source admission limit. The default0 retains unbounded admission. A value1.25 permits four seconds of initial source audio, then admits additional original samples at no more than1.25 times realtime. It delays before upstream sending; it does not trim audio, change2s/4s recognition windows, alter neural computation or modify any of the seven MING modules/settings. The500ms finalization pad remains separate from original-source admission. Cancellation cannot count an unadmitted frame as complete. Terminal metadata records the selected limit, actual waiting and admitted original samples.

This is a resource scheduling experiment for the observed sustained-load thermal limitation, not a proven acceleration or deployment selection. Its cost is a long-file admission floor of roughlyRTF0.8 at1.25x, with all waiting included in end-to-end measurements. Normally paced1x input adds no admission wait; actual long-recording quality, timing and temperatures still require independent validation. There is no temperature-dependent manipulation, silent cooling pause, dropped input, inference retry or altered hardware setting. The active image12 realtime cohort remains unchanged while this candidate is built.

An additional opt-in Youdao diagnostic uses `--share-upstream-gpu` with the unlocked overlap flags disabled. ASR and MT acquire shared process-file locks; MING retains its exclusive lock. A waiting MING ticket prevents later upstream work from passing it, so completed readers must drain before synthesis starts. This changes application-stage admission only; Olares time slicing, GPU settings, neural inference, source chunks and all seven trusted MING files/settings remain unchanged. Default admission remains exclusive. Actual worker health, ASR/MT observations and completion metadata expose the selected policy. This candidate is not accepted before sustained measurements and independent speech checks.

The legacy `raw_tts_output_samples` completion field counts the pinned SDK output after its fixed pitch-preserving 1.1/1.2 tempo transform, before adaptive playout. It is not the untransformed native MING sample count. Benchmark audits record native, fixed-tempo and adaptive-tempo stages separately; they must not assert equal sample counts across tempo transforms.

The alternative diagnostic `--share-asr-ming-gpu` pairs the two PyTorch stages while native T3PO stays exclusive. The producer's MING ticket explicitly carries its shared mode across the separate service processes; no MING implementation or synthesis setting is changed. FIFO writer fairness remains, and new ASR work stops joining an active MING phrase after its existing20-second protection threshold. The two pair-sharing flags are mutually exclusive and cannot be combined with unlocked overlap flags. Neither experimental profile is selected by default or accepted without measured quality and sustained performance.


## Rejected independent-ASR-stream experiment

The development-only separate CUDA-stream experiment is not retained in this runtime. Its first three complete diagnostic requests were slower than the existing ASR/MING-sharing profile; final raw diagnostics remain associated with development image `s2st-youdao-dev-20261121-2`. Thermal history prevents attributing every difference to that change. This removal restores the prior validated default-stream behavior and all seven trusted MING files/settings remain unchanged. The fifth hot diagnostic failed native MING EOS validation after252 decode steps. That failure is retained, not retried or reported as a valid timing result.

## Native Chinese completion delivery

Chinese frontend translation deltas remain immediate. Only the speech bridge waits until each existing native T3PO response commits, then sends its exact accumulated deltas as one queue item. This prevents a slow token stream from feeding the unchanged650ms MING batcher fragments such as `约翰，你怎么` followed by `看？是啊，`. It does not wait for the whole source or a complete sentence, force translation, rewrite text, increase synthesis budgets or retry failures. English delivery remains unchanged. The observed split is proven by the retained native logs; its causal role in the failed EOS and any quality/performance benefit still require installed validation. All seven trusted MING modules/settings remain identical to Tencent.
