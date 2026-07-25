# Entrypoint shared by every audio base image: pick the capability module from
# MODEL_SUPPORTS and run it. ONE instance loads exactly ONE model, so when the
# requested capabilities span several modules only the first is served — the
# others belong to their own clones.
#
# AUDIO_BASE (baked into each image) selects the routing table: the same cap
# name maps to different engines per base (`stt` is qwen-asr on the qwen base
# and CTranslate2 on the fasterwhisper one).
import os
import sys

from .contract import parse_supports

# base -> ordered [(capability set served together, module in wrapper.caps)].
# Order is the tie-break when a clone asks for caps from several modules.
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
