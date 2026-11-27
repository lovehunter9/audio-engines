# Tencent and Youdao CUDA speech translation

Both images provide one composite service for Chinese-English speech translation.
Input is mono PCM16 WAV at16kHz or live PCM16; output is translated text and24kHz speech.

- Tencent: Hy-SimulMT Q4_K_M with MING0.5B.
- Youdao: R2T2 BF16, T3PO Q5_K_M and MING0.5B.
- CUDA images contain AMD64 andARM64 slices; Intel is not enabled.
- One active inference session per instance; shared task APIs queue file requests.

## API

- POST `/v1/audio/translations`: complete JSON/WAV or NDJSON with `stream=true`.
- WS `/v1/audio/translations/stream`: live PCM input and translated text/audio events.
- GET `/v1/models`.
- GET/DELETE `/v1/tasks/{id}`, GET `/v1/tasks`, GET `/v1/tasks/{id}/result`.
- The same task operations also exist below `/v1/audio/tasks`.

Register as mode`audio`, capability`audio_s2s`; no separate S2ST capability is assumed.

## Runtime

Model assets, dependency versions and original source revisions are pinned in each payload manifest.
Authored adapters own service startup, container paths and transport; model inference remains in the locked modules.
Youdao uses an image-local official GPU client in its three worker processes to avoid idle handoff delays.
The loader adapter does not change node libraries, allocation limits or scheduling configuration.

## Build

Each model workflow accepts public`v*` tags, its model-specific tags and manual dispatch.
Manual dispatch can supply an immutable verified dependency cache when publishing the same runtime to another namespace.
The default cache lookup/build remains available when no cache is supplied.
