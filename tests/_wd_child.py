"""Child process for watchdog_smoke.py: arms the watchdog, then just lives for a while."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper import watchdog  # noqa: E402

mode = sys.argv[1]
timeout_s = float(sys.argv[2])
state = {"ready": False, "error": None}
if mode == "ready":
    def flip():
        time.sleep(0.3)
        state["ready"] = True

    threading.Thread(target=flip, daemon=True).start()
if mode == "failed":
    state["error"] = "model refused to load"

watchdog.arm(lambda: state["ready"], lambda: state["error"], "test engine", timeout_s=timeout_s)
time.sleep(4)
print("survived")
