import argparse
import json
from pathlib import Path
import sys
import time
from .gpu_queue import configure, stage


def main(args):
    configure(args.gpu_queue)
    root = Path(args.root)
    source = root / "sources/Confucius4-R2T2"
    sys.path.insert(0, str(source))
    from r2t2 import R2T2ASRModel

    original_factory = R2T2ASRModel.LLM
    log = (root / "results/asr-observer.jsonl").open("a", buffering=1)

    def record(kind, **fields):
        log.write(
            json.dumps(
                {
                    "event": kind,
                    "wall_time": time.time(),
                    "monotonic_seconds": time.monotonic(),
                    **fields,
                }
            )
            + "\n"
        )

    def factory(*positional, **keywords):
        original_options = dict(keywords)
        keywords.update(
            gpu_memory_utilization=args.memory_fraction,
            max_model_len=args.context,
            max_num_seqs=1,
            limit_mm_per_prompt={"audio": 1},
        )
        record(
            "model_factory",
            backend="official_vllm_bf16",
            original_options=original_options,
            deployment_options=keywords,
        )
        return original_factory(*positional, **keywords)

    R2T2ASRModel.LLM = staticmethod(factory)
    import inspect
    import textwrap

    module = sys.modules[R2T2ASRModel.__module__]

    def rolling_window(state, incoming_samples):
        ledger = getattr(state, "_actual_chunk_samples", [])
        if len(ledger) != len(state.chunk_text):
            raise RuntimeError("Audio/text chunk ledger mismatch before append")
        ledger.append(incoming_samples)
        if sum(ledger) != len(state.audio_accum):
            raise RuntimeError("Audio window length does not match sample ledger")
        if len(state.audio_accum) > 16 * 16000:
            removed = count = 0
            while removed < 8 * 16000:
                removed += ledger[count]
                count += 1
            if count >= len(ledger):
                raise RuntimeError("Window trim would remove current inference chunk")
            state.audio_accum = state.audio_accum[removed:]
            state.chunk_text = state.chunk_text[count:]
            ledger = ledger[count:]
            state._first_chunk_discarded = True
            record(
                "aligned_window_trim",
                removed_samples=removed,
                removed_text_chunks=count,
                retained_samples=len(state.audio_accum),
                retained_prior_text_chunks=len(state.chunk_text),
                configured_chunk_ms=args.chunk_ms,
            )
        state._actual_chunk_samples = ledger

    for method_name, incoming in [
        ("streaming_transcribe_no_reset", "chunk"),
        ("finish_streaming_transcribe_no_reset", "tail"),
    ]:
        original_method = getattr(R2T2ASRModel, method_name)
        code = textwrap.dedent(inspect.getsource(original_method))
        if method_name == "finish_streaming_transcribe_no_reset":
            empty_guard = "if state.buffer is None or state.buffer.shape[0] == 0:"
            if code.count(empty_guard) != 1:
                raise RuntimeError("Unexpected official final-buffer guard")
            code = code.replace(
                empty_guard,
                "if state.buffer is None or (state.buffer.shape[0] == 0 and state.audio_accum.shape[0] == 0):",
                1,
            )
        lines = code.splitlines(keepends=True)
        start = next(
            i
            for i, line in enumerate(lines)
            if "# Once the accumulated audio reaches 16 seconds" in line
        )
        end = next(
            i
            for i in range(start, len(lines))
            if 'prefix_text = "".join(state.chunk_text)' in lines[i]
        )
        indent = lines[end][: len(lines[end]) - len(lines[end].lstrip())]
        code = (
            "".join(lines[:start])
            + indent
            + "_feasibility_rolling_window(state, len("
            + incoming
            + "))\n\n"
            + "".join(lines[end:])
        )
        namespace = dict(vars(module), _feasibility_rolling_window=rolling_window)
        exec(
            compile(code, inspect.getfile(original_method) + ":aligned-wrapper", "exec"), namespace
        )
        setattr(R2T2ASRModel, method_name, namespace[method_name])
        (root / "results" / (method_name + "-aligned.py")).write_text(code)
        record(
            "aligned_method_installed",
            method=method_name,
            official_method_sha256=__import__("hashlib")
            .sha256(inspect.getsource(original_method).encode())
            .hexdigest(),
        )
    for name in [
        "streaming_transcribe",
        "finish_streaming_transcribe",
        "streaming_transcribe_no_reset",
        "finish_streaming_transcribe_no_reset",
    ]:
        original = getattr(R2T2ASRModel, name)

        def observe(self, *positional, _original=original, _name=name, **keywords):
            started = time.monotonic()
            state = (
                (positional[0] if _name.startswith("finish") else positional[1])
                if positional
                else keywords.get("state")
            )
            incoming = 0 if _name.startswith("finish") else len(positional[0])
            budget_index = 1 if _name.startswith("finish") else 2
            budget = (
                positional[budget_index]
                if len(positional) > budget_index
                else keywords.get("max_new_tokens")
            )
            before_buffer = len(state.buffer) if state is not None else None
            before_tail = state.last_fixed_text[-100:] if state is not None else None
            try:
                with stage("r2t2") as gpu_wait:
                    result = _original(self, *positional, **keywords)
            except BaseException as exc:
                record(
                    _name + "_failed",
                    input_samples=incoming,
                    elapsed_seconds=time.monotonic() - started,
                    detail=repr(exc),
                )
                raise
            record(
                _name,
                input_samples=incoming,
                elapsed_seconds=time.monotonic() - started,
                gpu_queue_wait_seconds=gpu_wait,
                max_new_tokens=budget,
                input_buffer_samples=before_buffer,
                finalized_tail_before=before_tail if _name.startswith("finish") else None,
                finalized_tail_after=state.last_fixed_text[-100:]
                if _name.startswith("finish")
                else None,
                state_buffer_samples=len(state.buffer) if state is not None else None,
                state_audio_window_samples=len(state.audio_accum) if state is not None else None,
            )
            return result

        setattr(R2T2ASRModel, name, observe)
    sys.argv = [
        str(source / "ws_server.py"),
        "--ip",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--asr_model_path",
        str(root / "models/R2T2-HF"),
        "--vad_model_path",
        str(root / "models/Stream-VAD"),
    ]
    record("bootstrap_started", source_revision="62757213befa282e7fbfe1bfde3c29577781527c")
    import hashlib

    original_source = (source / "ws_server.py").read_text()
    before, handler = original_source.split("async def asr_stream_api_v1(", 1)
    needle = "                        first_max_new_tokens = max(1, int((step + lookahead) / 1280))"
    if handler.count(needle) != 1:
        raise RuntimeError("Unexpected official EOS block; refusing deployment patch")
    extra = "                        residual = audio_buf[tmp_audio_pointer:]\n                        if len(residual):\n                            asr_model.streaming_transcribe_no_reset(residual, asr_state, int(max_new_tokens), nmt_fast)\n"
    handler = handler.replace(needle, extra + needle, 1)
    code = before + "async def asr_stream_api_v1(" + handler
    if args.chunk_ms != 160:
        code = code.replace(
            "CHUNK_ASR_SECONDS = 0.16", "CHUNK_ASR_SECONDS = " + str(args.chunk_ms / 1000), 1
        )
    record(
        "deployment_patch",
        original_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
        description="Pass actual residual input to original ASR before original finish; no silence padding",
        chunk_ms=args.chunk_ms,
    )
    (root / "results/deployed-ws-server.py").write_text(code)
    globals_ = {
        "__name__": "__main__",
        "__file__": str(source / "ws_server.py"),
        "__package__": None,
    }
    exec(compile(code, str(source / "ws_server.py"), "exec"), globals_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--gpu-queue", required=True)
    parser.add_argument("--port", type=int, default=18094)
    parser.add_argument("--memory-fraction", type=float, default=0.20)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--chunk-ms", type=int, default=4000)
    main(parser.parse_args())
