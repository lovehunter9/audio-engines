"""The load watchdog, and the vLLM capture-size probe that has to survive version drift.

    python tests/watchdog_smoke.py
"""
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FAILED = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ((" -> " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(name)


def child(mode, timeout="1"):
    env = dict(os.environ, LOAD_TIMEOUT_S=timeout)
    p = subprocess.run([sys.executable, os.path.join(HERE, "_wd_child.py"), mode],
                       env=env, capture_output=True, timeout=30)
    return p.returncode, (p.stdout + p.stderr).decode()[-300:]


def fake_vllm(fields):
    for name in ("vllm", "vllm.config"):
        sys.modules.pop(name, None)
    if fields is None:
        sys.modules["vllm"] = None      # import raises, like an older build
        return
    cfg = types.ModuleType("vllm.config")
    cfg.CompilationConfig = type("CompilationConfig", (), {"model_fields": dict.fromkeys(fields)})
    pkg = types.ModuleType("vllm")
    pkg.config = cfg
    sys.modules["vllm"] = pkg
    sys.modules["vllm.config"] = cfg


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
    check("LOAD_TIMEOUT_S=0 disables it", rc == 0 and "survived" in out, (rc, out[-80:]))

    print("\n[vLLM capture-size probe]")
    from wrapper.caps import stt_stream as q

    os.environ.pop("VLLM_ENFORCE_EAGER", None)
    os.environ.pop("VLLM_CAPTURE_SIZES", None)
    fake_vllm(["cudagraph_capture_sizes", "level"])
    check("the current field name is used",
          q._capture_kw() == {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]}},
          q._capture_kw())
    fake_vllm(["capture_sizes"])
    check("an older field name is used instead",
          q._capture_kw() == {"compilation_config": {"capture_sizes": [1, 2, 4, 8]}},
          q._capture_kw())
    fake_vllm(["something_else"])
    check("an unknown config is left alone rather than guessed", q._capture_kw() == {})
    fake_vllm(None)
    check("an unimportable vllm.config is survived", q._capture_kw() == {})
    fake_vllm(["cudagraph_capture_sizes"])
    os.environ["VLLM_CAPTURE_SIZES"] = "1, 4"
    check("the capture set is overridable",
          q._capture_kw() == {"compilation_config": {"cudagraph_capture_sizes": [1, 4]}},
          q._capture_kw())
    os.environ["VLLM_ENFORCE_EAGER"] = "1"
    check("the eager escape hatch wins over everything",
          q._capture_kw() == {"enforce_eager": True}, q._capture_kw())

    print("\n" + ("FAILURES: %s" % FAILED if FAILED else "all checks passed"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
