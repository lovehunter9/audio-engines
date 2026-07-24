# Entrypoint for the qwen audio base: pick the capability module from
# MODEL_SUPPORTS and run it. Each instance loads exactly one model:
#   stt / stt_stream  -> Qwen3-ASR engine (caps.stt_stream, one load serves both)
#   align             -> Qwen3-ForcedAligner (caps.align; a separate model)
# stt-family and align are mutually exclusive (different models, never co-loaded).
import sys

from .contract import parse_supports


def main():
    supports = parse_supports()
    if not supports:
        print("[app] MODEL_SUPPORTS is empty; nothing to serve", flush=True)
        sys.exit(1)

    has_asr = ("stt" in supports) or ("stt_stream" in supports)
    has_align = "align" in supports

    if has_asr and has_align:
        print("[app] WARN stt-family and align are separate models; serving stt-family, "
              "ignoring align (clone align as its own instance)", flush=True)

    if has_asr:
        from .caps import stt_stream
        stt_stream.run([s for s in supports if s in ("stt", "stt_stream")])
    elif has_align:
        from .caps import align
        align.run(["align"])
    else:
        print("[app] unsupported MODEL_SUPPORTS for qwen base: %s "
              "(expected stt / stt_stream / align)" % supports, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
