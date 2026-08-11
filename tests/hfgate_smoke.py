"""Gate diagnosis: it must name a gate only when the hub confirms one, and never invent one.

    python tests/hfgate_smoke.py

The hub is faked throughout. A real call would make this depend on someone's HuggingFace account
and on which repos are gated this month, and the branch worth pinning down -- gated, but this app
was let through -- needs an approved token nobody can commit.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wrapper import hfgate                                             # noqa: E402

FAILED = []
ERR = EOFError("Ran out of input")


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ((" -> " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(name)


class Denied(Exception):
    def __init__(self, status):
        super().__init__("denied")
        self.response = types.SimpleNamespace(status_code=status)


def fake_hub(gated=False, files=("pytorch_model.bin",), info_raises=None, head_raises=None):
    """A hub that answers however the case under test needs, and counts what was asked."""
    calls = []

    class HfApi:
        def model_info(self, repo, token=None, timeout=None):
            calls.append("info")
            if info_raises:
                raise info_raises
            return types.SimpleNamespace(
                gated=gated,
                siblings=[types.SimpleNamespace(rfilename=f) for f in files])

    def get_hf_file_metadata(url, token=None, timeout=None):
        calls.append("head")
        if head_raises:
            raise head_raises

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = HfApi
    mod.get_hf_file_metadata = get_hf_file_metadata
    mod.hf_hub_url = lambda repo, filename: "https://hub/%s/%s" % (repo, filename)
    sys.modules["huggingface_hub"] = mod
    return calls


def gate_named(out):
    return "is a gated repo" in out and "huggingface.co" in out


def main():
    os.environ.pop("HF_TOKEN", None)

    print("\n[nothing to ask about]")
    calls = fake_hub()
    check("a bare model name is not a hub id, so the hub is never called",
          hfgate.explain("silero-v5", ERR) == str(ERR) and calls == [], calls)
    check("an empty source is left alone", hfgate.explain("", ERR) == str(ERR))

    print("\n[the gate is real]")
    fake_hub(gated="auto", head_raises=Denied(403))
    out = hfgate.explain("pyannote/embedding", ERR)
    check("a refused download names the gate", gate_named(out), out[:90])
    check("the original error is kept for whoever needs it", str(ERR) in out)
    check("with no token, that is what it says", "no HuggingFace token" in out, out[:120])

    os.environ["HF_TOKEN"] = "hf_pretend"
    fake_hub(gated="auto", head_raises=Denied(403))
    out = hfgate.explain("pyannote/embedding", ERR)
    check("with a token, it blames approval rather than the token",
          "not approved" in out, out[:120])
    os.environ.pop("HF_TOKEN", None)

    fake_hub(gated="manual", head_raises=Denied(401))
    check("401 counts as a refusal too", gate_named(hfgate.explain("a/b", ERR)))

    class GatedRepoError(Exception):
        pass

    fake_hub(gated=True, head_raises=GatedRepoError("nope"))
    check("so does the hub's own exception, which carries no response",
          gate_named(hfgate.explain("a/b", ERR)))

    print("\n[no gate: say nothing]")
    calls = fake_hub(gated=False)
    check("an ungated repo is not accused, and its files are not even asked about",
          hfgate.explain("openai/whisper-tiny", ERR) == str(ERR) and calls == ["info"], calls)
    fake_hub(gated="auto")
    check("gated but let through is not the cause of this failure",
          hfgate.explain("a/b", ERR) == str(ERR))
    fake_hub(gated="auto", files=())
    check("a repo with no files to try is not guessed at",
          hfgate.explain("a/b", ERR) == str(ERR))
    fake_hub(info_raises=Denied(404))
    check("a misspelled repo is not called gated", hfgate.explain("nobody/typo", ERR) == str(ERR))
    fake_hub(info_raises=OSError("Name or service not known"))
    check("an offline node keeps its own error", hfgate.explain("a/b", ERR) == str(ERR))
    sys.modules["huggingface_hub"] = None
    check("a base image without huggingface_hub still reports the load failure",
          hfgate.explain("a/b", ERR) == str(ERR))

    print("\n" + ("FAILURES: %s" % FAILED if FAILED else "all checks passed"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
