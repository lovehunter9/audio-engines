# CUDA integration boundaries

The pinned author source is QwenAudio/S3-S2ST 1ca52ea064ec2081f12675d853525d394571bf73. The original launcher generates configuration; its complete evaluator, READ/WRITE policy, full-prefix speaker refresh, native vocabulary sampling, hidden alignment, global Talker, ten-step FM solver, vocoder, both CUDA graph optimizations and EOS finalization remain selected.

The unavailable reference compiled vLLM build is replaced by public vLLM 0.11.1 compiled kernels and their PyTorch 2.9 ABI runtime, while the author Python engine/model definitions remain in use. Two absent Gemma RMS bindings call the corresponding FlashInfer fused CUDA kernels. The optional unused grouped INT4 registration checks that its own operator exists. Real FlashAttention is installed; its imports are not removed or deferred.

The native multi-process engines remain separate. On the target HAMI device, CUDA IPC export previously failed. The transport serializes exact CUDA tensor storage bytes through host memory and reconstructs the same dtype and shape on CUDA; no neural computation, sampling, or acoustic operation moves to the host. Worker initialization installs the original author capture and model hooks before the original vLLM Worker initialization. It does not replace EngineCore, its scheduler, or its execution backend.

Checkpoint loading uses CPU mmap before copying unchanged selected tensors to CUDA, avoiding transient whole-checkpoint GPU allocation. The source records before/after hashes for that loader change. Original two-record warmup is retained; the author's referenced warmup audio is absent from the public repository, so the approved unspliced Chinese and English FLEURS sources fill the original --warmup input.

Zstandard encodes filesystem layers without changing their contents. Weights remain outside the image. CUDA inference and direct-author versus wrapper equivalence must be checked on hardware; successful builds and import tests alone do not establish quality or performance.
