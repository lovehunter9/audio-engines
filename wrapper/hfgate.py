# A gated HuggingFace repo does not fail in a way anyone can read. llm-init stops with hf_gated and
# the weights never land, so the engine loads whatever the cache holds -- an empty file -- and torch
# reports "EOFError: Ran out of input". The real reason is on llm-init's dashboard, but the app log
# shows this instead, so name the gate here too.
import logging
import os

log = logging.getLogger("audio-hfgate")

# The caller is the load thread and the watchdog reads its outcome, so this must never hang.
_PROBE_TIMEOUT_S = 10


def explain(repo, err):
    """The load error, or the gate that actually caused it.

    Called only from a failed load: the happy path never reaches the hub, and anything short of a
    confirmed gate gives the original error back untouched -- a wrong accusation would cost more
    than the cryptic message it replaces.
    """
    text = str(err)
    if not repo or "/" not in repo:
        return text          # not a hub id (silero ships inside the image), no gate to speak of
    try:
        with _online():
            if not _refused(repo):
                return text
    except Exception as probe:
        # Offline, throttled, misspelled repo, hub outage: none of these are a gate.
        log.info("could not check whether %s is gated (%s)", repo, probe)
        return text

    why = "the token in use is not approved for it" if _token() else "no HuggingFace token reached this app"
    log.error("%s is gated and %s", repo, why)
    return ("%s is a gated repo and %s. Accept the conditions at https://huggingface.co/%s, connect "
            "that HuggingFace account under Olares Settings, then restart this app. "
            "(load error: %s)" % (repo, why, repo, text))


class _online:
    """Let this one probe reach the hub even though the engine runs offline.

    The wrapper scripts export HF_HUB_OFFLINE so that no inference path can ever touch the
    network, which also means the gate check could never answer: it would fail with "offline mode
    is enabled" and report that instead of the gate. Confirming a gate requires asking, and this
    runs once, after a load has already failed, with a 10s timeout. The flag is a module constant
    read at import time, so setting the environment variable back would have no effect here.
    """

    def __enter__(self):
        from huggingface_hub import constants

        self._constants = constants
        self._was = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = False
        return self

    def __exit__(self, *exc):
        self._constants.HF_HUB_OFFLINE = self._was
        return False


def _refused(repo):
    """True only when the hub confirms this app may read the repo page but not its files.

    Gating is what `gated` announces and what a download is refused over; the two are asked
    separately because a `gated: auto` repo serves its metadata to anyone, so reading that alone
    says nothing about whether the conditions were ever accepted.
    """
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    info = HfApi().model_info(repo, token=_token(), timeout=_PROBE_TIMEOUT_S)
    if not getattr(info, "gated", False):
        return False
    named = [s.rfilename for s in (getattr(info, "siblings", None) or []) if s.rfilename]
    if not named:
        return False
    try:
        get_hf_file_metadata(hf_hub_url(repo, named[0]), token=_token(), timeout=_PROBE_TIMEOUT_S)
    except Exception as e:
        return _denied(e)
    return False             # gated, but this app has been let through


def _token():
    return os.environ.get("HF_TOKEN") or None


def _denied(e):
    if type(e).__name__ in ("GatedRepoError", "LocalTokenNotFoundError"):
        return True
    # The hub dresses a refusal up as 404/RepositoryNotFound, so read the status, not the class.
    return getattr(getattr(e, "response", None), "status_code", None) in (401, 403, 404)
