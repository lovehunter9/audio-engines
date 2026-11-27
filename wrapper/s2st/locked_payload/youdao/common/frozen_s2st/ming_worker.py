import argparse
from pathlib import Path


def build(options):
    import base64
    import hashlib
    import json
    import threading
    import time

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import onnxruntime as ort
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    COMPILE_FLOW = options.compile_flow
    OUT = options.cache
    (OUT / "references").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    log = (OUT / "events.jsonl").open("a", buffering=1)

    def record(kind, **fields):
        log.write(
            json.dumps(
                {"event": kind, "monotonic_seconds": time.monotonic(), **fields}, ensure_ascii=False
            )
            + "\n"
        )

    def soundfile_load(path, frame_offset=0, num_frames=-1, channels_first=True, **kwargs):
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        data = data[frame_offset:]
        if num_frames >= 0:
            data = data[:num_frames]
        tensor = torch.from_numpy(np.ascontiguousarray(data))
        return (tensor.T.contiguous() if channels_first else tensor), rate

    torchaudio.load = soundfile_load
    import spkemb_extractor

    def gpu_speaker_init(self, campplus_model, target_sr=16000):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2
        options.enable_profiling = True
        options.profile_file_prefix = str(OUT / "campplus-profile")
        self.campplus_session = ort.InferenceSession(
            campplus_model,
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": 536870912,
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                )
            ],
        )
        if self.campplus_session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CAMPPlus did not load CUDA provider")
        self.target_sr = target_sr

    spkemb_extractor.SpkembExtractor.__init__ = gpu_speaker_init
    from ming_tts import MingTTS
    import modeling_bailingmm
    from frontend.number_en import normalize_numbers
    from .spoken_scores import spoken_scores
    from .spoken_numbers import spoken_zh_numbers

    modeling_bailingmm.MAX_CACHE_LEN = 8192
    torch.set_num_threads(4)
    model = MingTTS(str(options.model), device="cuda:0", text_normalization=False)
    devices = sorted({str(p.device) for p in model.model.parameters()})
    if not devices or any(not d.startswith("cuda") for d in devices):
        raise RuntimeError("MING neural weights are not all CUDA: " + repr(devices))
    rate = model.sample_rate
    original_native_sample = model.model.sample

    def validated_native_sample(*args, **kwargs):
        terminal = False
        steps = 0
        for latent, is_last in original_native_sample(*args, **kwargs):
            steps += 1
            terminal = bool(is_last)
            yield latent, is_last
        record("native_generation_end", steps=steps, eos=terminal)
        if not terminal:
            raise RuntimeError("MING native generation exhausted its budget without stop signal")

    model.model.sample = validated_native_sample

    speaker_directory = OUT / "speaker-embeddings"
    speaker_directory.mkdir(exist_ok=True)

    def speaker_file(reference, digest):
        target = speaker_directory / (digest + ".pt")
        if not target.exists():
            _, embeddings = model._load_prompt([str(reference)])
            torch.save(embeddings[0].detach().cpu().float(), target)
        return target

    if COMPILE_FLOW:
        cfm = model.model.flowloss.cfm
        original_sample = cfm.sample
        compiled_sample = torch.compile(
            original_sample,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )

        def canonical_sample(noise, c, latent_history, sde_noise, **kwargs):
            values = [
                x.clone(memory_format=torch.contiguous_format)
                for x in [noise, c, latent_history, sde_noise]
            ]
            return compiled_sample(
                noise=values[0],
                c=values[1],
                latent_history=values[2],
                sde_noise=values[3],
                **kwargs,
            )

        cfm.sample = canonical_sample
        record("compilation_warmup_start", diffusion_steps=10, cfg=1.0, temperature=0.0)
        started = time.monotonic()
        torch.manual_seed(0)
        warm = model.synthesize(
            task="TTS",
            text="今天下午，我们一起去图书馆，然后沿着河边慢慢走回家。",
            language_id="zh",
            use_zero_spk_emb=True,
            cfg=1.0,
            temperature=0.0,
            raw_text=True,
            max_decode_steps=364,
        )
        sf.write(
            OUT / "compiled-warmup.wav",
            warm.detach().float().cpu().numpy().reshape(-1),
            model.sample_rate,
        )
        warm_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        record(
            "compilation_warmup_complete",
            seconds=time.monotonic() - started,
            output_samples=warm.numel(),
            unique_graphs=warm_graphs,
        )
    audit = {
        "model_devices": devices,
        "speaker_providers": model.spkemb_extractor.campplus_session.get_providers(),
        "flow_compiled": COMPILE_FLOW,
        "diffusion_steps": 10,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "onnxruntime": ort.__version__,
        "source_revision": "028c1eace193e7acdb256453da89b6e9207e052b",
        "conditioning": "official spk_emb_path; source speaker only; CFG1; native max_length24; no cache-dependent VAE RNG",
        "runtime_changes": [
            "soundfile audio loader matching released engine",
            "released engine decode-budget ceiling 8192",
            "CAMPPlus CUDA provider; no CPU neural weight offload",
            "spoken number frontend; contextual score separator rendering",
        ],
    }
    (OUT / "device-audit.json").write_text(json.dumps(audit, indent=2))
    record("ready", **audit)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "model": "Ming-3.0-tts-0.5B",
            "sample_rate": 24000,
            "native_sample_rate": rate,
        }

    @app.get("/v1/simulinfer/tts_capabilities")
    def capabilities():
        return {
            "source_voice_history": "source-voice-history-v1",
            "backend": "MING native",
            "source_conditioning": "original source snapshot; generated acoustic history disabled",
        }

    @app.post("/v1/audio/speech")
    def speech(body: dict):
        text = body["input"]
        reference = body.get("extra_params", {}).get("simulinfer_source_audio") or body.get(
            "ref_audio"
        )
        if not reference or not reference.startswith("data:audio/"):
            raise ValueError("Source reference must be inline audio")
        raw = base64.b64decode(reference.split(",", 1)[1])
        digest = hashlib.sha256(raw).hexdigest()
        path = OUT / "references" / (digest + ".wav")
        if not path.exists():
            path.write_bytes(raw)
        language = "zh" if str(body.get("language", "")).lower() in ["chinese", "zh"] else "en"
        spoken = (
            spoken_zh_numbers(text) if language == "zh" else normalize_numbers(spoken_scores(text))
        )

        def stream():
            with lock:
                started = time.monotonic()
                graphs_before = (
                    int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                    if COMPILE_FLOW
                    else 0
                )
                record(
                    "request_start",
                    text=text,
                    spoken_text=spoken,
                    reference_sha256=digest,
                    language=language,
                )
                torch.manual_seed(0)
                parts = model.prepare_text(spoken, "TTS", language, 24)
                samples = 0
                for fragment in parts:
                    wave = model.synthesize(
                        task="TTS",
                        text=fragment,
                        language_id=language,
                        spk_emb_path=str(speaker_file(path, digest)),
                        use_spk_emb=True,
                        cfg=1.0,
                        temperature=0.0,
                        raw_text=True,
                        max_decode_steps=min(1000, max(64, len(fragment) * 14)),
                    )
                    record(
                        "pcm_serialization",
                        returned_tensor_device=str(wave.device),
                        neural_parameter_devices=devices,
                    )
                    wave = torchaudio.functional.resample(wave.detach().float().cpu(), rate, 24000)
                    pcm = (wave.numpy().reshape(-1).clip(-1, 1) * 32767).astype("<i2").tobytes()
                    samples += len(pcm) // 2
                    for offset in range(0, len(pcm), round(0.2 * 24000) * 2):
                        yield pcm[offset : offset + round(0.2 * 24000) * 2]
                if not audit.get("speaker_profile"):
                    profile = model.spkemb_extractor.campplus_session.end_profiling()
                    events = json.load(open(profile))
                    weighted = [
                        e
                        for e in events
                        if e.get("args", {}).get("op_name")
                        in ["Conv", "FusedConv", "Gemm", "MatMul", "BatchNormalization"]
                    ]
                    cpu_weighted = [
                        e
                        for e in weighted
                        if e.get("args", {}).get("provider") != "CUDAExecutionProvider"
                    ]
                    audit.update(
                        speaker_profile=profile,
                        speaker_weighted_node_executions=len(weighted),
                        speaker_cpu_weighted_node_executions=len(cpu_weighted),
                    )
                    (OUT / "device-audit.json").write_text(json.dumps(audit, indent=2))
                    if cpu_weighted:
                        raise RuntimeError("CAMPPlus weighted neural nodes fell back to CPU")
                record(
                    "request_complete",
                    seconds=time.monotonic() - started,
                    audio_seconds=samples / 24000,
                    new_compile_graphs=(
                        int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) - graphs_before
                    )
                    if COMPILE_FLOW
                    else 0,
                )

        return StreamingResponse(
            stream(),
            media_type="application/octet-stream",
            headers={"X-Audio-Sample-Rate": "24000", "X-Audio-Channels": "1"},
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--compile-flow", action="store_true")
    parser.add_argument("--port", type=int, default=18091)
    options = parser.parse_args()
    if not options.model.is_dir():
        raise ValueError("MING requires an existing local model snapshot")
    import uvicorn

    uvicorn.run(build(options), host="127.0.0.1", port=options.port, log_level="info")


if __name__ == "__main__":
    main()
