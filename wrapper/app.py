# Entrypoint for every base image: run the one wrapper.caps module that serves this clone's caps.
import os
import sys

from .contract import parse_supports

# base -> ordered [(caps served together, module)]; first match wins, since one instance = one model.
ROUTES = {
    "qwen": [
        (("stt", "stt_stream"), "stt_stream"),
        (("align",), "align"),
    ],
    "fasterwhisper": [
        (("stt",), "whisper"),
    ],
    "pyannote": [
        (("vad",), "vad"),
        (("diar",), "diar"),
        (("speaker_embed",), "embed"),
        (("enhance",), "enhance"),
    ],
    "nemo": [
        (("diar_stream",), "diar_stream"),
    ],
}


def main():
    base = (os.environ.get("AUDIO_BASE") or "").strip()
    routes = ROUTES.get(base)
    if routes is None:
        print("[app] AUDIO_BASE=%r is not a known base (%s); the image is built wrong"
              % (base, ", ".join(sorted(ROUTES))), flush=True)
        sys.exit(1)

    supports = parse_supports()
    if not supports:
        print("[app] MODEL_SUPPORTS is empty; nothing to serve", flush=True)
        sys.exit(1)

    matched = [(caps, mod) for caps, mod in routes if set(caps) & set(supports)]
    if not matched:
        print("[app] MODEL_SUPPORTS=%s has nothing the %s base can serve (it serves %s)"
              % (",".join(supports), base,
                 ", ".join(c for caps, _ in routes for c in caps)), flush=True)
        sys.exit(1)

    caps, module = matched[0]
    served = [s for s in supports if s in caps]
    if len(matched) > 1:
        ignored = [c for cs, _ in matched[1:] for c in cs if c in supports]
        print("[app] WARN %s need separate models; serving %s, ignoring %s "
              "(clone each as its own instance)"
              % (",".join(supports), ",".join(served), ",".join(ignored)), flush=True)

    mod = __import__("wrapper.caps.%s" % module, fromlist=["run"])
    mod.run(served)


if __name__ == "__main__":
    main()
