# A load that hangs forever is invisible: k8s probes llm-init, not us, so time it out here.
import logging
import os
import threading
import time

log = logging.getLogger("audio-watchdog")

# Loading is cache-local (llm-init already downloaded the weights), so half an hour is generous.
LOAD_TIMEOUT_S = float(os.environ.get("LOAD_TIMEOUT_S", "1800") or 0)
_POLL_S = 5.0
_EXIT_CODE = 70


def arm(is_ready, failed=None, what="engine"):
    # is_ready: () -> bool ; failed: () -> truthy once loading gave up with a reason.
    if LOAD_TIMEOUT_S <= 0:
        return

    def wait():
        t0 = time.time()
        while True:
            if is_ready():
                log.info("%s became ready after %.0fs", what, time.time() - t0)
                return
            if failed is not None and failed():
                # A refused model is diagnosable through /v1/models; a crash loop is not.
                log.error("%s failed to load; staying up so the reason stays visible", what)
                return
            left = LOAD_TIMEOUT_S - (time.time() - t0)
            if left <= 0:
                break
            time.sleep(min(_POLL_S, left))   # never sleep past the deadline we are enforcing
        log.error("%s still not ready after %.0fs and no error was reported: exiting so the "
                  "container is restarted", what, LOAD_TIMEOUT_S)
        os._exit(_EXIT_CODE)

    threading.Thread(target=wait, name="audio-load-watchdog", daemon=True).start()
    log.info("load watchdog armed: %s must be ready within %.0fs", what, LOAD_TIMEOUT_S)
