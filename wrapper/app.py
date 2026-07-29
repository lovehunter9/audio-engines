# Entrypoint for every base image: run the one wrapper.caps module that serves this clone's caps.
import os
import sys

from . import catalog
from .contract import SUPPORTS_PREFIX, parse_supports


def main():
    base = (os.environ.get("AUDIO_BASE") or "").strip()
    routes = catalog.BASES.get(base)
    if routes is None:
        print("[app] AUDIO_BASE=%r is not a known base (%s); the image is built wrong"
              % (base, ", ".join(sorted(catalog.BASES))), flush=True)
        sys.exit(1)

    supports, bad = parse_supports()
    if bad:
        print("[app] MODEL_SUPPORTS entries must be %s-prefixed capability keys: %s"
              % (SUPPORTS_PREFIX, ", ".join(bad)), flush=True)
        sys.exit(1)
    if not supports:
        print("[app] MODEL_SUPPORTS is empty; nothing to serve", flush=True)
        sys.exit(1)

    implements = catalog.implements(base)
    unknown = [c for c in supports if c not in implements]
    if unknown:
        print("[app] MODEL_SUPPORTS asks for %s, which the %s base does not implement (it has %s)"
              % (",".join(unknown), base, ", ".join(implements)), flush=True)
        sys.exit(1)

    # Every declared cap is implemented here, so at least one route matches.
    matched = [(caps, mod) for caps, mod in routes if set(caps) & set(supports)]
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
