# faster-qwen3-tts 0.3.2 builds its token-suppression mask one index at a time:
#
#     for i in range(suppress_start, vocab_size):
#         if i != eos_id:
#             suppress_mask[i] = True
#
# On a CUDA tensor every assignment is its own kernel, 1023 of them before each prefill. Alone on
# the GPU that is ~40 ms. Sharing it with an LLM (MemorySlice: both contexts decode at once and
# the driver time-slices between them) each tiny kernel waits for a slice, and on pptest03 the
# first audio of every sentence went from 0.23 s to 2.8 s while sglang decoded, which put the
# whole stream behind real time (RTF 0.9). Two slice writes build the same mask: first audio
# 0.52 s, RTF 1.41 under the same load.
#
# The library is pinned and its source is rewritten at load, not forked: a release that changes
# the loop leaves it untouched and says so, and the engine runs as before.
import inspect
import logging
import re

log = logging.getLogger("audio-tts")

_LOOP = re.compile(
    r"( *)for i in range\(suppress_start, vocab_size\):\n"
    r"\1    if i != eos_id:\n"
    r"\1        suppress_mask\[i\] = True\n")


def rewrite(src):
    """(source with the loop replaced by slice writes, how many loops were replaced)."""
    def repl(m):
        pad = m.group(1)
        return ("%ssuppress_mask[suppress_start:] = True\n"
                "%sif suppress_start <= eos_id < vocab_size:\n"
                "%s    suppress_mask[eos_id] = False\n" % (pad, pad, pad))
    return _LOOP.subn(repl, src)


def _patch_function(module, name):
    fn = getattr(module, name, None)
    if fn is None:
        return 0
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return 0
    new, n = rewrite(src)
    if n:
        # Redefined in the module's own globals, so its imports and decorators resolve as before;
        # model.py imports these at call time and picks the new ones up.
        exec(compile(new, inspect.getsourcefile(fn) or "<tts_mask>", "exec"), module.__dict__)
    return n


_applied = []


def apply():
    """Patch the generate loops of an imported faster_qwen3_tts; returns how many were replaced.

    Once per process: getsource reads the file on disk, not the redefinition, so a second pass
    would find the shipped loop again and redefine everything for nothing.
    """
    if _applied:
        return _applied[0]
    try:
        from faster_qwen3_tts import generate, streaming
    except ImportError:
        return 0
    targets = ((streaming, "fast_generate_streaming"), (streaming, "parity_generate_streaming"),
               (generate, "fast_generate"))
    done = sum(_patch_function(mod, name) for mod, name in targets)
    _applied.append(done)
    if done:
        log.info("suppress mask built with slice writes in %d faster-qwen3-tts loops", done)
    else:
        log.warning("faster-qwen3-tts no longer has the per-index suppress mask loop; "
                    "left as shipped")
    return done
