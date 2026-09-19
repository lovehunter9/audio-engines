"""The load watchdog, and the vLLM capture-size probe that has to survive version drift.

    python tests/watchdog_smoke.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FAILED = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ((" -> " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(name)


def child(mode, timeout="1"):
    p = subprocess.run([sys.executable, os.path.join(HERE, "_wd_child.py"), mode, timeout],
                       capture_output=True, timeout=30)
    return p.returncode, (p.stdout + p.stderr).decode()[-300:]


def main():
    print("\n[watchdog]")
    rc, out = child("wedge")
    check("a load that never finishes exits non-zero so k8s rebuilds", rc == 70,
          (rc, out.strip()[-120:]))
    rc, out = child("ready")
    check("a load that finishes is left alone", rc == 0 and "survived" in out, (rc, out[-80:]))
    rc, out = child("failed")
    check("a load that failed loudly stays up to be diagnosed", rc == 0 and "survived" in out,
          (rc, out[-120:]))
    rc, out = child("wedge", timeout="0")
    check("a zero deadline disables it", rc == 0 and "survived" in out, (rc, out[-80:]))

    print("\n" + ("FAILURES: %s" % FAILED if FAILED else "all checks passed"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
