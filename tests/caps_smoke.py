"""Wire-level check of every capability: the async result must equal the sync one, byte for byte.

No GPU and no model weights: each engine is replaced by a stub, so what is under test is the
wiring (form fields, dispatch, progress, result shape), which is exactly where a refactor slips.

    python tests/caps_smoke.py        # needs fastapi, httpx, numpy, python-multipart
"""
import base64
import copy
import io
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

SR = 16000
FAILED = []


def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL ") + name + ((" -> " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(name)


def wav_of(seconds):
    """Silence of a given length as real RIFF WAV bytes, for caps that measure what they got."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"\x00\x00" * int(round(seconds * SR)))
    return buf.getvalue()


def fake_soundfile():
    # align's batch mode decodes and re-writes slices with soundfile, absent from a bare venv.
    m = types.ModuleType("soundfile")
    m.read = lambda buf, dtype=None, always_2d=False: (
        np.zeros((SR * 4, 1), dtype="float32"), SR)

    def write(target, data, sr, format=None, subtype=None, **k):
        # Deterministic and derived from the samples, so "async bytes == sync bytes" means something.
        if hasattr(target, "write"):
            target.write(b"FAKE" + np.asarray(data, dtype="float32").tobytes())

    m.write = write
    m.check_format = lambda *a, **k: True
    # What probe_seconds reads when a cap hands the file to a model instead of decoding it.
    m.info = lambda *a, **k: types.SimpleNamespace(frames=SR * 4, samplerate=SR)
    sys.modules["soundfile"] = m
    return m


class Wav:
    # Stands in for the (channels, time) torch tensor the decoders return.
    def __init__(self, n=SR * 4):
        self.shape = (1, n)

    def squeeze(self, _dim=0):
        return self

    def contiguous(self):
        return self


class Seg:
    def __init__(self, a, b):
        self.start, self.end = a, b


TASKS = "/v1/tasks"           # llm-init docs/api/openapi.yaml async-tasks contract
LEGACY = "/v1/audio/tasks"    # what this engine shipped first; same runner


def poll(c, tid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = c.get("%s/%s" % (TASKS, tid)).json()
        if d["status"] in ("succeeded", "failed", "canceled"):
            return d
        time.sleep(0.02)
    return {"status": "timeout"}


def contract(c, doc, label, legacy=True):
    """What every engine behind llm-init must answer, not just this one."""
    check("%s doc is object=task kind=audio" % label,
          doc.get("object") == "task" and doc.get("kind") == "audio",
          (doc.get("object"), doc.get("kind")))
    check("%s poll/result_url point at the contract path" % label,
          doc.get("poll") == "%s/%s" % (TASKS, doc["id"])
          and doc.get("result_url") == "%s/%s/result" % (TASKS, doc["id"]),
          (doc.get("poll"), doc.get("result_url")))
    if legacy:
        alias = c.get("%s/%s" % (LEGACY, doc["id"])).json()
        check("%s the legacy alias answers the same task" % label, alias == doc, alias.get("id"))
    else:
        check("%s does not mount the legacy task alias" % label,
              c.get("%s/%s" % (LEGACY, doc["id"])).status_code == 404)
    cap = (c.get(TASKS).json() or {}).get("capacity") or {}
    check("%s the list reports queue capacity" % label,
          set(cap) == {"queued", "running", "limit", "accepting"} and cap["limit"] > 0, cap)
    if legacy:
        check("%s the legacy list alias agrees" % label,
              c.get(LEGACY).json() == c.get(TASKS).json())
    else:
        check("%s the legacy list is gone" % label, c.get(LEGACY).status_code == 404)
    got = c.get(TASKS, params={"status": "succeeded"}).json()["data"]
    check("%s ?status= filters" % label,
          got and all(d["status"] == "succeeded" for d in got),
          [d["status"] for d in got])
    one = c.get(TASKS, params={"limit": "1"}).json()
    finished = [d for d in one["data"] if d["status"] in ("succeeded", "failed", "canceled")]
    check("%s ?limit= caps the finished tasks" % label, len(finished) <= 1, len(finished))
    check("%s a rejected parameter is a 400" % label,
          c.get(TASKS, params={"status": "done"}).status_code == 400
          and c.get(TASKS, params={"limit": "0"}).status_code == 400)


def metered(c, sync, doc, label, want):
    """Duration is the engine's to report: nothing downstream ever holds the decoded audio.

    Per-second pricing reads the response header, and a task's reader reads the document, so the
    two have to carry the same number or one of the two callers bills something else.
    """
    from wrapper.tasks import INPUT_SECONDS_HEADER, OUTPUT_SECONDS_HEADER

    names = {"input": INPUT_SECONDS_HEADER, "output": OUTPUT_SECONDS_HEADER}
    for key in want:
        got = sync.headers.get(names[key])
        field = (doc or {}).get("%s_duration_seconds" % key)
        agrees = (got is not None and field is not None
                  and abs(float(got) - float(field)) < 0.01)
        check("%s reports the %s duration on the response and the task" % (label, key),
              agrees, (got, field))


def both_ways(c, path, files, data, label, json_result=True, meters=()):
    """POST once sync, once with async=1, and insist the answers agree."""
    r1 = c.post(path, files=files, data=data)
    ok1 = r1.status_code == 200
    check("%s sync 200" % label, ok1, (r1.status_code, r1.text[:160]))
    r2 = c.post(path, files=files, data=dict(data, **{"async": "1"}))
    check("%s async 202" % label, r2.status_code == 202, (r2.status_code, r2.text[:160]))
    if r2.status_code != 202:
        return None
    doc = poll(c, r2.json()["task"]["id"])
    check("%s task succeeded" % label, doc["status"] == "succeeded",
          (doc.get("status"), (doc.get("error") or {}).get("message")))
    if not ok1 or doc["status"] != "succeeded":
        return doc
    contract(c, doc, label)
    metered(c, r1, doc, label, meters)
    if json_result:
        check("%s async result == sync result" % label, doc["result"] == r1.json())
        rr = c.get("%s/%s/result" % (TASKS, doc["id"]))
        check("%s .../result agrees too" % label, rr.json() == r1.json(), rr.status_code)
    else:
        rr = c.get("%s/%s/result" % (TASKS, doc["id"]))
        check("%s async bytes == sync bytes" % label, rr.content == r1.content,
              (len(rr.content), len(r1.content)))
        check("%s content-type survives" % label,
              rr.headers["content-type"] == r1.headers["content-type"],
              (rr.headers.get("content-type"), r1.headers.get("content-type")))
    return doc


def mounted(c):
    eps = c.get("/api/engine-spec").json()["endpoints"]
    return {(e["method"], e["path"]) for e in eps if e["available"]}


def advertises_tasks(c, label, legacy=True):
    paths = mounted(c)
    check("%s advertises the task API" % label,
          ("GET", TASKS + "/{id}") in paths, sorted(p[1] for p in paths))
    if legacy:
        check("%s advertises the legacy task alias" % label,
              ("GET", LEGACY + "/{id}") in paths, sorted(p[1] for p in paths))
        deprecated = {e["path"] for e in c.get("/api/engine-spec").json()["endpoints"]
                      if e.get("deprecated")}
        check("%s marks the legacy paths deprecated" % label,
              deprecated == {LEGACY, LEGACY + "/{id}", LEGACY + "/{id}/result"},
              sorted(deprecated))
    else:
        check("%s does not advertise the legacy task alias" % label,
              all(p != LEGACY and not str(p).startswith(LEGACY + "/")
                  for _m, p in paths),
              sorted(p[1] for p in paths))


WAV = {"file": ("a.wav", b"RIFF0000WAVEfake", "audio/wav")}


def t_vad():
    from fastapi.testclient import TestClient
    from wrapper.caps import vad

    vad.decode_mono = lambda *a, **k: Wav()
    vad._state.update(ready=True, model=object(),
                      get_ts=lambda wav, model, **kw: [{"start": 0, "end": SR}])
    with TestClient(vad.build_app(["vad"])) as c:
        advertises_tasks(c, "vad")
        both_ways(c, "/v1/audio/vad", WAV, {"threshold": "0.4"}, "vad", meters=("input",))


def t_diar():
    from fastapi.testclient import TestClient
    from wrapper.caps import diar

    class Ann:
        def __init__(self, *spans):
            self.spans = spans

        def itertracks(self, yield_label=False):
            for start, end, spk in self.spans:
                yield Seg(start, end), None, spk

    CLEAN = Ann((0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01"))
    OVERLAP = Ann((0.0, 2.0, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01"))

    class Out:
        """What pyannote 4 hands back: two diarizations and the clustering's centroids."""

        def __init__(self, exclusive=True, centroids=((0.1, 0.2), (0.3, 0.4))):
            self.speaker_diarization = OVERLAP
            if exclusive:
                self.exclusive_speaker_diarization = CLEAN
            if centroids is not None:
                self.speaker_embeddings = centroids

    seen = {"hook": 0, "at_call": None}

    class Pipe:
        """A pipeline whose hyper-parameters can be read and written, as pyannote's can."""

        def __init__(self):
            self.out = None  # None = the v3 shape, a bare Annotation
            self.params = {"segmentation": {"min_duration_off": 0.0},
                           "clustering": {"threshold": 0.6, "method": "centroid"}}

        def parameters(self, instantiated=False):
            return self.params

        def instantiate(self, params):
            self.params = params

        def __call__(self, _inp, hook=None, **kw):
            seen["at_call"] = copy.deepcopy(self.params)
            for i in range(4):
                if hook:
                    seen["hook"] += 1
                    hook("segmentation", None, total=4, completed=i + 1)
                time.sleep(0.05)
            if hook:
                hook("embeddings", None, total=2, completed=1)
            return CLEAN if self.out is None else self.out

    pipe = Pipe()
    diar.decode = lambda *a, **k: (Wav(), SR)
    diar._state.update(ready=True, pipeline=pipe, device="cpu")
    diar._seed(pipe)
    with TestClient(diar.build_app(["diar"])) as c:
        advertises_tasks(c, "diar")
        doc = both_ways(c, "/v1/audio/diarization", WAV, {"num_speakers": "2"}, "diar",
                         meters=("input",))
        check("diar ends at stage=done, ratio=1",
              (doc or {}).get("progress", {}).get("ratio") == 1.0
              and (doc or {}).get("progress", {}).get("stage") == "done",
              (doc or {}).get("progress"))
        # A stage counts its own units: carrying the previous stage's over reads as "8 of 6".
        check("a new stage does not inherit the old stage's counts",
              "done" not in (doc or {}).get("progress", {})
              or (doc or {})["progress"]["done"] <= (doc or {})["progress"].get("total", 1e9),
              (doc or {}).get("progress"))
        r = c.post("/v1/audio/diarization", files=WAV, data={"async": "1"})
        mid = {}
        for _ in range(40):
            mid = c.get("%s/%s" % (TASKS, r.json()["task"]["id"])).json()
            if (mid.get("progress") or {}).get("stage") == "segmentation":
                break
            time.sleep(0.01)
        check("diar relays pyannote's stage + counts while it runs",
              seen["hook"] > 0 and (mid.get("progress") or {}).get("stage") == "segmentation"
              and (mid["progress"].get("total") or 0) == 4,
              (seen["hook"], mid.get("progress")))
        poll(c, r.json()["task"]["id"])

        pipe.out = Out()
        doc = c.post("/v1/audio/diarization", files=WAV, data={"exclusive": "1"}).json()
        check("exclusive=1 answers with the turns that do not overlap",
              doc.get("exclusive") is True and doc["segments"][0]["end"] == 1.5, doc.get("segments"))
        doc = c.post("/v1/audio/diarization", files=WAV).json()
        check("without exclusive the overlapping turns stand",
              doc.get("exclusive") is False and doc["segments"][0]["end"] == 2.0,
              doc.get("segments"))
        check("centroids come back keyed by the speaker they belong to",
              sorted(doc.get("speaker_centroids") or {}) == ["SPEAKER_00", "SPEAKER_01"],
              doc.get("speaker_centroids"))

        pipe.out = Out(exclusive=False)
        doc = c.post("/v1/audio/diarization", files=WAV, data={"exclusive": "1"}).json()
        check("a build without exclusive diarization says so rather than failing",
              doc.get("exclusive") is False and doc["num_segments"] == 2, doc.get("exclusive"))

        # A cluster that produced no turn shifts every label after it, so the rows can no
        for label, centroids in (("a centroid with no turns", ((0.1,), (0.2,), (0.3,))),
                                 ("a non-finite centroid", ((float("nan"), 0.2), (0.3, 0.4)))):
            pipe.out = Out(centroids=centroids)
            doc = c.post("/v1/audio/diarization", files=WAV).json()
            check("%s drops the centroids rather than mislabelling them" % label,
                  "speaker_centroids" not in doc, doc.get("speaker_centroids"))

        pipe.out = Out()
        base = copy.deepcopy(diar._state["params"])
        doc = c.post("/v1/audio/diarization", files=WAV,
                     data={"min_duration_off": "0.5", "clustering_threshold": "0.8"}).json()
        check("a request's hyper-parameters are in force while it runs",
              seen["at_call"]["segmentation"]["min_duration_off"] == 0.5
              and seen["at_call"]["clustering"]["threshold"] == 0.8, seen["at_call"])
        check("and the response says which values ran",
              doc.get("min_duration_off") == 0.5 and doc.get("clustering_threshold") == 0.8, doc)
        check("the pipeline is left as it was found", pipe.params == base, pipe.params)
        doc = c.post("/v1/audio/diarization", files=WAV).json()
        check("so the next request sees the pipeline's own values",
              doc.get("min_duration_off") == 0.0 and doc.get("clustering_threshold") == 0.6, doc)
        r = c.post("/v1/audio/diarization", files=WAV, data={"clustering_threshold": "high"})
        check("an unparsable hyper-parameter is a 400, not a job that fails later",
              r.status_code == 400, (r.status_code, r.text[:120]))

        pipe.params = {"segmentation": {"min_duration_off": 0.0}}
        diar._seed(pipe)
        r = c.post("/v1/audio/diarization", files=WAV, data={"clustering_threshold": "0.8"})
        check("a knob this pipeline does not have is refused up front",
              r.status_code == 400, (r.status_code, r.text[:120]))


def t_diar_speakrs_models_dir():
    import shutil
    import tempfile
    from wrapper.caps import diar_speakrs as ds

    root = tempfile.mkdtemp(prefix="hfcache-")
    try:
        os.environ["HF_HUB_CACHE"] = root
        repo = "avencera/speakrs-models"
        snaps = os.path.join(root, "models--avencera--speakrs-models", "snapshots")
        # The engine is Rust and has no huggingface_hub to resolve this layout for it.
        for name, when in (("older", 1000000), ("newer", 2000000)):
            d = os.path.join(snaps, name)
            os.makedirs(d)
            os.utime(d, (when, when))
        # llm-init writes the resolved path for engines that share its run directory. Guessing
        told_dir = tempfile.mkdtemp(prefix="rundir-")
        try:
            with open(os.path.join(told_dir, "model_path"), "w") as f:
                f.write("/somewhere/llm-init/decided\n")
            ds.RUN_DIR = told_dir
            check("diar_speakrs prefers the path llm-init wrote over the cache layout",
                  ds._models_dir(repo) == "/somewhere/llm-init/decided", ds._models_dir(repo))
        finally:
            shutil.rmtree(told_dir, ignore_errors=True)
            ds.RUN_DIR = "/nonexistent-run-dir"

        check("diar_speakrs resolves the HF snapshot directory",
              ds._models_dir(repo) == os.path.join(snaps, "newer"), ds._models_dir(repo))

        # A re-download leaves the previous revision in place; the fresh one is the one llm-init
        shutil.rmtree(snaps)
        flat = os.path.join(root, "models--avencera--speakrs-models")
        check("diar_speakrs falls back to the repo directory when there are no snapshots",
              ds._models_dir(repo) == flat, ds._models_dir(repo))
    finally:
        shutil.rmtree(root, ignore_errors=True)
        os.environ.pop("HF_HUB_CACHE", None)


def t_diar_speakrs():
    from fastapi.testclient import TestClient
    from wrapper.caps import diar_speakrs as ds

    sent = []

    def fake_run(payload):
        sent.append(payload)
        return {"ok": True, "device": "cuda",
                "segments": [[0.0, 1.5, "SPEAKER_00"], [1.5, 3.0, "SPEAKER_01"]]}

    ds._child.run = fake_run
    ds._state.update(ready=True, error=None, device="cuda", params={})
    with TestClient(ds.build_app(["diar"])) as c:
        advertises_tasks(c, "diar_speakrs")
        doc = both_ways(c, "/v1/audio/diarization", WAV, {}, "diar_speakrs",
                        meters=("input",))
        # both_ways hands back the task document; the diarization itself is its result.
        body = doc["result"]
        check("diar_speakrs names the speakers it found",
              body["num_speakers"] == 2 and body["speakers"] == ["SPEAKER_00", "SPEAKER_01"],
              (body["num_speakers"], body["speakers"]))
        check("diar_speakrs reports the device the engine ran on",
              body["device"] == "cuda", body["device"])

        # This engine cannot stand in for pyannote: speaker-count knobs must 400, not be ignored.
        for field in ("num_speakers", "min_speakers", "max_speakers"):
            r = c.post("/v1/audio/diarization", files=WAV, data={field: "3"})
            check("diar_speakrs refuses %s rather than ignoring it" % field,
                  r.status_code == 400 and field in r.json()["detail"],
                  (r.status_code, r.json()))

        # Seconds on the wire, frames to the engine.
        sent.clear()
        c.post("/v1/audio/diarization", files=WAV, data={"min_duration_off": "0.5"})
        check("diar_speakrs converts seconds to frames for the engine",
              sent[-1]["min_duration_off_frames"] == 30, sent[-1]["min_duration_off_frames"])

        # Unset and zero must stay different: speakrs' fast modes default this filter to 3.
        sent.clear()
        c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs sends null, not 0, for a knob nobody set",
              sent[-1]["min_duration_off_frames"] is None, sent[-1]["min_duration_off_frames"])
        sent.clear()
        c.post("/v1/audio/diarization", files=WAV, data={"min_duration_off": "0"})
        check("diar_speakrs sends 0 when a caller really asked for 0",
              sent[-1]["min_duration_off_frames"] == 0, sent[-1]["min_duration_off_frames"])

        # The engine holds the whole clip decoded, so an unbounded upload is an OOM kill, which
        from wrapper import limits

        ds.BOUNDS.seconds = 1.0
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs refuses audio longer than it can hold",
              r.status_code == 413 and "memory" in r.json()["detail"],
              (r.status_code, r.json()))
        r = c.post("/v1/audio/diarization", files=WAV, data={"async": "1"})
        check("diar_speakrs refuses it before a task exists, not inside one",
              r.status_code == 413, (r.status_code, r.text[:120]))
        ds.BOUNDS.seconds = 14400.0

        # A container the header readers cannot parse may still be one the engine decodes, and it
        real_probe, real_sub = limits.probe_seconds, limits.subprocess
        limits.probe_seconds = lambda _p: None
        limits.subprocess = types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="7200.0\n", stderr=""),
            SubprocessError=real_sub.SubprocessError)
        ds.BOUNDS.seconds = 60.0
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs falls back to ffprobe when the header readers cannot",
              r.status_code == 413 and "7200s" in r.json()["detail"],
              (r.status_code, r.json()))

        # ffprobe absent or unable: unknown stays a real answer rather than a refusal.
        limits.subprocess = types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="bad"),
            SubprocessError=real_sub.SubprocessError)
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs lets an unmeasurable clip through rather than refusing it",
              r.status_code == 200, (r.status_code, r.text[:120]))
        limits.probe_seconds, limits.subprocess = real_probe, real_sub
        ds.BOUNDS.seconds = 14400.0

        # An upload too big to hold is refused while it is still arriving.
        ds.BOUNDS.megabytes = 0.001  # ~1 KiB
        big = {"file": ("big.wav", b"RIFF" + b"\0" * 4096, "audio/wav")}
        r = c.post("/v1/audio/diarization", files=big)
        check("diar_speakrs refuses an upload past its byte bound",
              r.status_code == 413 and "MiB" in r.json()["detail"],
              (r.status_code, r.json()))
        ds.BOUNDS.megabytes = 1024.0

        sent.clear()
        doc = c.post("/v1/audio/diarization", files=WAV, data={"exclusive": "1"}).json()
        check("diar_speakrs passes exclusive through and echoes what ran",
              sent[-1]["exclusive"] is True and doc["exclusive"] is True,
              (sent[-1]["exclusive"], doc["exclusive"]))

        # A dead child must read as "not ready", never as an empty result.
        def dead(_payload):
            raise RuntimeError("engine process exited with code 1")

        ds._child.run = dead
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs fails the job when the engine is gone",
              r.status_code == 500, (r.status_code, r.text[:120]))

    ds._child.run = fake_run
    ds._state.update(ready=False, error="engine failed to load the model")
    with TestClient(ds.build_app(["diar"])) as c:
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs 503s while the engine is loading",
              r.status_code == 503 and "load" in r.json()["detail"],
              (r.status_code, r.json()))

    # Everything this container serves lives in that child; a pod that outlives it is a leak.
    from wrapper import watchdog

    exits, real_exit, real_grace = [], ds._exit, ds._EXIT_GRACE_S
    ds._exit, ds._EXIT_GRACE_S = exits.append, 0.0
    ds._child._proc = types.SimpleNamespace(wait=lambda: 9)
    try:
        ds._stopping.clear()
        ds._child._reap()
        check("diar_speakrs exits so k8s restarts a container whose engine died",
              exits == [watchdog.EXIT_CODE], exits)

        # The same exit during a rollout would turn every normal shutdown into an error.
        exits.clear()
        ds._stopping.set()
        ds._child._reap()
        check("diar_speakrs does not call a shutdown a crash", exits == [], exits)
    finally:
        ds._exit, ds._EXIT_GRACE_S = real_exit, real_grace
        ds._stopping.clear()
        ds._child._proc = None


def t_diar_stream_offline():
    """llm-init downloads, the engine loads offline.

    Reaching for the hub when the cache is empty fails minutes later with a network error that
    names none of the actual fault, and on the machines where it succeeds the engine serves
    weights nobody downloaded.
    """
    from wrapper.caps import diar_stream

    # The image carries torch and NeMo; a test box does not, and the point here is which branch
    fetched = []
    mods = {
        "torch": {"cuda": types.SimpleNamespace(is_available=lambda: False)},
        "nemo": {},
        "nemo.collections": {},
        "nemo.collections.asr": {},
        "nemo.collections.asr.models": {
            "SortformerEncLabelModel": types.SimpleNamespace(
                restore_from=lambda **k: None,
                from_pretrained=lambda *a, **k: fetched.append(a) or None)},
    }
    saved = {name: sys.modules.get(name) for name in mods}
    for name, attrs in mods.items():
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod

    real_find = diar_stream._find_nemo
    diar_stream._find_nemo = lambda: None
    try:
        diar_stream._state["error"] = None
        diar_stream._load()
    finally:
        diar_stream._find_nemo = real_find
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    check("diar_stream does not reach for the hub when the cache is empty", fetched == [], fetched)
    err = diar_stream._state["error"] or ""
    check("diar_stream fails naming the empty cache rather than reaching for the hub",
          "shared cache" in err and "offline" in err, err[:160])


def t_embed():
    from fastapi.testclient import TestClient
    from wrapper.caps import embed

    embed.decode = lambda *a, **k: (Wav(), SR)
    embed._state.update(ready=True, device="cpu",
                        inference=lambda _inp: np.arange(8, dtype="float32"))
    with TestClient(embed.build_app(["speaker_embed"])) as c:
        advertises_tasks(c, "embed")
        both_ways(c, "/v1/audio/embeddings", WAV, {}, "embed", meters=("input",))


def t_enhance():
    from fastapi.testclient import TestClient
    from wrapper.caps import enhance

    enhance.CHUNK_S, enhance.OVERLAP_S = 1.0, 0.1     # force the windowed path
    enhance.decode_mono = lambda *a, **k: np.zeros((1, SR * 5), dtype="float32")

    def slow(noisy):
        time.sleep(0.05)                              # slow enough to observe mid-flight
        return np.zeros(noisy.shape[-1], dtype="float32")

    enhance._run = slow
    enhance._encode = lambda out, want: (b"OggS" + b"\0" * 64, "audio/ogg", "ogg/opus")
    enhance._state.update(ready=True, kind="waveform", device="cpu", model=object())
    with TestClient(enhance.build_app(["enhance"])) as c:
        advertises_tasks(c, "enhance")
        both_ways(c, "/v1/audio/enhance", WAV, {"format": "ogg"}, "enhance", json_result=False,
                  meters=("input", "output"))
        r = c.post("/v1/audio/enhance", files=WAV, data={"async": "1"})
        mid = {}
        for _ in range(60):
            mid = c.get("%s/%s" % (TASKS, r.json()["task"]["id"])).json()
            if (mid.get("progress") or {}).get("total"):
                break
            time.sleep(0.01)
        check("enhance counts its windows while it runs",
              (mid.get("progress") or {}).get("total", 0) >= 5, mid.get("progress"))
        doc = poll(c, r.json()["task"]["id"])
        check("and lands on stage=done, ratio=1",
              doc.get("progress") == {"stage": "done", "ratio": 1.0}, doc.get("progress"))


def t_align():
    from fastapi.testclient import TestClient
    from wrapper.caps import align

    fake_soundfile()
    unit = {"text": "hi", "start_time": 0.1, "end_time": 0.4}
    model = types.SimpleNamespace(align=lambda audio=None, text=None, language=None: ([unit],))
    align._state.update(ready=True, model=model, device="cpu")
    with TestClient(align.build_app(["align"])) as c:
        advertises_tasks(c, "align")
        single = both_ways(c, "/v1/audio/align", WAV, {"text": "hi there"}, "align single",
                           meters=("input",))
        check("align single bills the whole file it was handed",
              (single or {}).get("input_duration_seconds") == 4.0,
              (single or {}).get("input_duration_seconds"))
        # The upload decodes to four seconds and the segments ask for two of them.
        batch = both_ways(c, "/v1/audio/align", WAV,
                          {"segments":
                           '[{"start":0,"end":1,"text":"hi"},{"start":1,"end":2,"text":"yo"}]'},
                          "align batch", meters=("input",))
        check("align batch bills the slices it aligned, not the file",
              (batch or {}).get("input_duration_seconds") == 2.0,
              (batch or {}).get("input_duration_seconds"))


def t_whisper():
    from fastapi.testclient import TestClient
    from wrapper.caps import whisper

    class S:
        def __init__(self, a, b, t):
            self.start, self.end, self.text = a, b, t
            self.id, self.seek, self.words = 0, 0, None

    def transcribe(path, **kw):
        segs = (S(0.0, 1.0, "hello "), S(1.0, 2.0, "world"))
        info = types.SimpleNamespace(language="en", duration=2.0)
        return iter(segs), info

    whisper._state.update(ready=True, model=types.SimpleNamespace(transcribe=transcribe),
                          pipeline=None, device="cpu", compute="int8")
    # A real WAV of the length ffmpeg would actually have produced.
    whisper._ffmpeg_slice_wav = lambda src, start, dur: wav_of(
        max(0.0, min(float(dur), 2.0 - float(start))))
    with TestClient(whisper.build_app(["stt"])) as c:
        advertises_tasks(c, "whisper")
        both_ways(c, "/v1/audio/transcriptions", WAV, {}, "whisper stt", meters=("input",))
        both_ways(c, "/v1/audio/transcriptions", WAV, {"response_format": "verbose_json"},
                  "whisper verbose", meters=("input",))
        both_ways(c, "/v1/audio/translations", WAV, {}, "whisper translations",
                  meters=("input",))
        batch = both_ways(
            c, "/v1/audio/transcriptions", WAV,
            {"segments": '[{"start":0,"end":1},{"start":1,"end":9}]'}, "whisper batch",
            meters=("input",))
        check("whisper batch bills the audio that existed, not the window asked for",
              (batch or {}).get("input_duration_seconds") == 2.0,
              (batch or {}).get("input_duration_seconds"))
        r = c.post("/v1/audio/transcriptions", files=WAV, data={"response_format": "text"})
        check("whisper text/plain still comes back as text",
              r.status_code == 200 and r.headers["content-type"].startswith("text/plain"),
              (r.status_code, r.headers.get("content-type")))
        r2 = c.post("/v1/audio/transcriptions", files=WAV,
                    data={"response_format": "text", "async": "1"})
        doc = poll(c, r2.json()["task"]["id"])
        rr = c.get("%s/%s/result" % (TASKS, doc["id"]))
        check("and as a task it is served from .../result as text",
              rr.status_code == 200 and rr.content == r.content
              and rr.headers["content-type"].startswith("text/plain"),
              (rr.status_code, rr.headers.get("content-type"), rr.content[:40]))


def t_whisper_ov():
    from fastapi.testclient import TestClient
    from wrapper.caps import whisper_ov as w

    w._decode = lambda data, fn: np.zeros(SR * 2, dtype="float32")
    w._generate = lambda audio, task, language: "hello world"
    w._state.update(ready=True, pipeline=object(), device="GPU", error=None)
    with TestClient(w.build_app(["stt"])) as c:
        advertises_tasks(c, "whisper_ov")
        both_ways(c, "/v1/audio/transcriptions", WAV, {}, "whisper_ov stt", meters=("input",))
        both_ways(c, "/v1/audio/translations", WAV, {}, "whisper_ov translations",
                  meters=("input",))
        both_ways(
            c, "/v1/audio/transcriptions", WAV,
            {"segments": '[{"start":0,"end":1},{"start":1,"end":2}]'}, "whisper_ov batch",
            meters=("input",))
        r = c.post("/v1/audio/transcriptions", files=WAV,
                   data={"segments": '[{"start":0,"end":1},{"start":1,"end":2}]'})
        check("whisper_ov batch returns one result per span",
              r.status_code == 200 and len((r.json() or {}).get("results") or []) == 2,
              r.text[:200])


def t_qwen():
    from fastapi.testclient import TestClient
    from wrapper.caps import stt_stream as q

    q._decode_to_16k_mono = lambda raw, fn: np.zeros(SR * 4, dtype="float32")
    # One result per clip, saying how many samples that clip carried. The real engine
    calls = []

    def fake_transcribe(audio=None, language=None, return_time_stamps=None):
        # Two shapes reach here: the serial path hands one (clip, sr) tuple, the batched
        clips = audio if isinstance(audio, list) else [audio]
        calls.append(len(clips))
        return [types.SimpleNamespace(text=str(len(c))) for c, _sr in clips]

    q._state.update(ready=True, asr=types.SimpleNamespace(
        transcribe=fake_transcribe,
        sampling_params=types.SimpleNamespace(max_tokens=32)))
    with TestClient(q.build_app(["stt", "stt_stream"])) as c:
        advertises_tasks(c, "qwen stt")
        check("the WS stream endpoint is still advertised",
              ("WS", "/v1/audio/stream") in mounted(c))
        spec = c.get("/api/engine-spec").json()
        align_eps = [e for e in spec["endpoints"] if e.get("capability") == "align"]
        check("align is on qwen but not served without MODEL_SUPPORTS",
              bool(align_eps) and all(not e.get("available") for e in align_eps))
        both_ways(c, "/v1/audio/transcriptions", WAV, {}, "qwen stt", meters=("input",))
        both_ways(c, "/v1/audio/transcriptions", WAV,
                  {"segments": '[{"start":0,"end":1},{"start":1,"end":2}]'}, "qwen batch",
                  meters=("input",))
        batch_over_cap(c, q, calls)
        repetition_fallback(q)
        repetition_request_spellings(q)
        the_report_does_not_need_a_request(q)
        malformed_span_is_isolated(c, q, calls)
        one_bad_span_does_not_sink_its_group(c, q, calls)


def one_bad_span_does_not_sink_its_group(c, q, calls):
    """A span the engine refuses must cost only itself, not the group it happened to land in.

    A batched call has no per-span outcome, so the first version marked the whole group --
    measured against a real engine, one 0.5 ms span failed all five of its group. Retrying
    cannot help either: grouping is by index, so the retry rebuilds the same group around
    the same span. Halving a failed group ends the search on the span responsible.
    """
    was = q.MAX_BATCH_SPANS
    q.MAX_BATCH_SPANS = 8
    bad = {"n": 0}
    real = q._offline_transcribe_many

    def refusing(clips):
        # The engine refuses any call that carries the pathological clip, as a real one does.
        bad["n"] += 1
        if any(len(cl) == 8 for cl in clips):
            raise RuntimeError("engine refused this batch")
        return real(clips)

    q._offline_transcribe_many = refusing
    try:
        del calls[:]
        # index 2 is 8 samples long: hi > lo so it is handed to the engine, and a real one
        spans = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.2005), (0.4, 0.5), (0.5, 0.6)]
        body = ",".join('{"start":%s,"end":%s}' % (a, b) for a, b in spans)
        r = c.post("/v1/audio/transcriptions", files=WAV,
                   data={"segments": "[" + body + "]"})
        doc = r.json() if r.status_code == 200 else {}
    finally:
        q._offline_transcribe_many = real
        q.MAX_BATCH_SPANS = was
    got = doc.get("results") or []
    errs = [i for i, e in enumerate(got) if "error" in e]
    check("a refused span still leaves one entry per request entry", len(got) == 5, len(got))
    check("only the span the engine refused carries the error", errs == [2], errs)
    check("its neighbours keep their transcripts",
          len(got) == 5 and all("text" in got[i] for i in (0, 1, 3, 4)), got)


def malformed_span_is_isolated(c, q, calls):
    """One unusable span must not take the rest of the request with it.

    The serial path has always answered per span: a bad one gets {"error": ...} and its
    neighbours still get transcripts. parse_segments only checks the payload is a JSON
    array, so anything at all can arrive as an element -- and the batched path has to keep
    the same promise, that the reply carries one entry per request entry, in order.
    """
    was = q.MAX_BATCH_SPANS
    q.MAX_BATCH_SPANS = 8
    try:
        del calls[:]
        r = c.post("/v1/audio/transcriptions", files=WAV,
                   data={"segments": '[{"start":0,"end":0.2},"oops",'
                                     '{"start":"n/a","end":1},{"start":0.3,"end":0.5}]'})
        doc = r.json() if r.status_code == 200 else {}
    finally:
        q.MAX_BATCH_SPANS = was
    got = doc.get("results") or []
    check("a malformed span does not 500 the whole request", r.status_code == 200,
          r.status_code)
    check("every requested span still gets an entry, in order", len(got) == 4, got)
    check("the good spans still carry text",
          len(got) == 4 and "text" in got[0] and "text" in got[3], got)
    check("only the malformed ones carry an error",
          len(got) == 4 and "error" in got[1] and "error" in got[2], got)


def the_report_does_not_need_a_request(q):
    """Whatever it has to say must be said without waiting for an offline transcription.

    It used to hang off the token budget, whose callers both sit inside
    `if sp is not None:` -- so the branch written for "this build exposes no
    sampling_params" could not run, and a deployment that only streams never reached any
    of it. Those are the two silences the report exists to break, so it is said once when
    the model finishes loading, and the load path is where this pins it.
    """
    import inspect

    src = inspect.getsource(q._load_blocking)
    check("the report is made when the model loads, not from a request path",
          "_say_repetition_once()" in src)
    ov_src = inspect.getsource(q._load_ov)
    check("the OpenVINO load path reports the same way",
          "_say_repetition_once()" in ov_src)
    budget = inspect.getsource(q._token_budget)
    check("the budget no longer has to be the thing that reports",
          "if sp is not None" not in budget)


def repetition_request_spellings(q):
    """How --repetition-detection is spelled must not flip what it means.

    bool("false") is True, so a value that spells the feature OFF used to read as a JSON
    override and switch it ON. The detector then failed to build and logged one line about
    an ignored value -- while everything keyed on "was it asked for" carried on believing
    it had been. A word is a word; only real JSON is an override.
    """
    from wrapper.contract import EngineArgs

    cases = [
        ("",                                        False, ""),
        ("--repetition-detection",                  True,  ""),
        ("--repetition-detection true",             True,  ""),
        ("--repetition-detection false",            False, ""),
        ("--repetition-detection 0",                False, ""),
        ("--repetition-detection off",              False, ""),
        # Words nobody listed: a word list would let these through, and each must still mean off.
        ("--repetition-detection disabled",         False, ""),
        ("--repetition-detection none",             False, ""),
        ("--repetition-detection never",            False, ""),
        ("--repetition-detection nope",             False, ""),
        ('--repetition-detection {"min_count":30}', True,  '{"min_count":30}'),
        ('--repetition-detection [1,2]',            True,  '[1,2]'),
    ]
    for raw, want_on, want_override in cases:
        on, override, note = q.repetition_request(EngineArgs(raw))
        check("spelling %r asks for it: %s" % (raw or "(nothing)", want_on),
              on == want_on, on)
        check("spelling %r overrides: %r" % (raw or "(nothing)", want_override),
              override == want_override, override)
    for raw in ("--repetition-detection disabled", "--repetition-detection nope"):
        check("an unrecognised word %r is reported, not silently obeyed" % raw.split()[-1],
              q.repetition_request(EngineArgs(raw))[2] is not None)
    for raw in ("--repetition-detection", "--repetition-detection false",
                '--repetition-detection {"min_count":30}'):
        check("a spelling this engine knows says nothing extra: %r" % raw,
              q.repetition_request(EngineArgs(raw))[2] is None)


def repetition_fallback(q):
    """The fallback replaces the detector, never joins it, and can be turned off by name.

    Three states have to stay distinguishable, because from outside they all look like an
    engine that transcribes: the detector running, the fallback capping, and neither. The
    default is the number that used to be hard-coded, so this exposes it without moving it.
    """
    stock = q.OFFLINE_MAX_TOKENS
    was_on, was_n, was_cls = q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC, q._repdet_class[:]
    try:
        check("the default is the number that was hard-coded",
              q.REPETITION_FALLBACK_TOKENS_PER_SEC == 12, q.REPETITION_FALLBACK_TOKENS_PER_SEC)

        # A build that HAS the detector: the fallback is redundant there and must not apply.
        del q._repdet_class[:]
        q._repdet_class.append(object)
        q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC = True, 12
        check("a build with the detector keeps the engine's own budget",
              q._token_budget(600.0) == stock, q._token_budget(600.0))

        # A build that has NOT got it: the fallback is what protection means there.
        del q._repdet_class[:]
        q._repdet_class.append(None)
        want = min(stock, int(10.0 * 12) + q.TOKENS_FLOOR)
        check("without the detector the fallback bounds the output",
              q._token_budget(10.0) == want, (q._token_budget(10.0), want))

        q.REPETITION_FALLBACK_TOKENS_PER_SEC = 0
        check("setting it to 0 gives the engine's own budget back",
              q._token_budget(10.0) == stock, q._token_budget(10.0))

        # Never asked for protection: neither mechanism may touch the budget.
        q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC = False, 12
        check("asking for nothing leaves the budget alone",
              q._token_budget(10.0) == stock, q._token_budget(10.0))

        # And it must not go looking for the detector either.
        del q._repdet_class[:]
        del q._repdet_said[:]
        q._say_repetition_once()
        check("asking for nothing does not probe for the detector",
              q._repdet_class == [], q._repdet_class)
    finally:
        q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC = was_on, was_n
        del q._repdet_class[:]
        q._repdet_class.extend(was_cls)


def batch_over_cap(c, q, calls):
    """A request larger than --batch-max-spans is split, and nothing moves or goes missing.

    The cap bounds memory in one generate() call, so a caller sending more spans than it
    gets grouped. What the caller must not be able to tell is that this happened: same
    count, same order, every span answered. Distinct span lengths make each one
    identifiable, since the stub answers with the sample count it was handed.
    """
    spans = [(0.0, 0.1), (0.1, 0.4), (0.4, 0.5), (0.5, 1.1),
             (1.1, 1.2), (1.2, 1.9), (1.9, 2.0)]
    want = [str(int(round(e * SR)) - int(round(s * SR))) for s, e in spans]
    body = ",".join('{"start":%s,"end":%s}' % (s, e) for s, e in spans)

    was = q.MAX_BATCH_SPANS
    q.MAX_BATCH_SPANS = 3
    try:
        del calls[:]
        r = c.post("/v1/audio/transcriptions", files=WAV,
                   data={"segments": "[" + body + "]"})
        doc = r.json()
    finally:
        q.MAX_BATCH_SPANS = was

    check("over-cap batch answers 200", r.status_code == 200, r.status_code)
    got = doc.get("results") or []
    check("every span comes back", len(got) == len(spans), len(got))
    check("no span is left unanswered", all(x is not None for x in got), got)
    check("spans keep the order they were sent in",
          [x.get("text") for x in got] == want, [x.get("text") for x in got])
    check("the request was split at the cap, not sent whole", calls == [3, 3, 1], calls)


def t_ov():
    """OpenVINO base: same HTTP/WS contract, generate() only after the client stops."""
    import json
    from fastapi.testclient import TestClient
    from wrapper.caps import stt_stream as q

    class Pipe:
        def __init__(self):
            self.calls = []

        def generate(self, raw, streamer=None, **kw):
            self.calls.append({"n": len(raw), "kw": kw, "has_streamer": streamer is not None})
            if streamer:
                streamer("hello")
                streamer(" world")
            return types.SimpleNamespace(texts=["hello world"], languages=["English"])

    os.environ["AUDIO_BASE"] = "ov"
    pipe = Pipe()
    q._decode_to_16k_mono = lambda raw, fn: np.zeros(SR * 4, dtype="float32")
    q._state.update(ready=True, asr=pipe, error=None)
    pcm = b"\x00\x00" * (SR // 10)
    with TestClient(q.build_app(["stt", "stt_stream"])) as c:
        spec = c.get("/api/engine-spec").json()
        check("ov align is implemented but not served on an stt-only instance",
              [e for e in spec["endpoints"]
               if e.get("capability") == "align" and not e["available"]])
        check("ov still advertises the WS stream",
              ("WS", "/v1/audio/stream") in mounted(c))
        both_ways(c, "/v1/audio/transcriptions", WAV, {}, "ov stt", meters=("input",))
        pipe.calls.clear()
        with c.websocket_connect("/v1/audio/stream") as ws:
            ready = json.loads(ws.receive_text())
            check("ov stream ready", ready.get("type") == "ready", ready)
            ws.send_bytes(pcm)
            ws.send_bytes(pcm)
            check("ov did not transcribe while audio was still arriving",
                  pipe.calls == [])
            ws.send_text(json.dumps({"type": "stop"}))
            kinds = []
            while True:
                msg = json.loads(ws.receive_text())
                kinds.append(msg["type"])
                if msg["type"] in ("closed", "error"):
                    break
        check("ov generate ran once after stop", len(pipe.calls) == 1, pipe.calls)
        check("ov generate used a streamer",
              pipe.calls and pipe.calls[0]["has_streamer"])
        check("ov stream token-streamed after stop",
              "partial" in kinds and "final" in kinds and "closed" in kinds, kinds)
        check("ov stream did not error", "error" not in kinds, kinds)


class FakeTTS:
    """Stands in for FasterQwen3TTS. The model object is the seam; everything above it is wiring."""

    sample_rate = 24000

    def __init__(self, custom=True, design=False):
        self.calls = []
        self.refuse_over = None
        if custom:
            kind = "custom_voice"
        elif design:
            kind = "voice_design"
        else:
            kind = "base"
        self.model = types.SimpleNamespace(model=types.SimpleNamespace(tts_model_type=kind))

    def get_supported_speakers(self):
        return ["aiden", "nofish"]

    def warmup(self, prefill_len=100):
        self.calls.append(("warmup", {"prefill_len": prefill_len}))

    @staticmethod
    def _audio():
        # A list of arrays, which is what the codec decoder hands back.
        return [np.linspace(-0.5, 0.5, 480, dtype="float32")]

    def _record(self, kind, kw):
        self.calls.append((kind, kw))
            # refuse_over stands in for the engine's own limit so the test watches the message travel.
        if self.refuse_over and len(str(kw.get("text") or "")) > self.refuse_over:
            raise ValueError("Input is too long: prefill has 2148 tokens but max_seq_len=2048. "
                             "Use shorter text or shorter reference audio.")

    def last(self, kind):
        for name, kw in reversed(self.calls):
            if name == kind:
                return kw
        return None

    def kinds(self):
        return [name for name, _kw in self.calls]

    def generate_custom_voice(self, **kw):
        self._record("custom", kw)
        return self._audio(), self.sample_rate

    def generate_voice_clone(self, **kw):
        self._record("clone", kw)
        return self._audio(), self.sample_rate

    def generate_custom_voice_streaming(self, **kw):
        self._record("custom_stream", kw)
        for _ in range(3):
            yield self._audio()[0], self.sample_rate, {}

    def generate_voice_clone_streaming(self, **kw):
        self._record("clone_stream", kw)
        for _ in range(3):
            yield self._audio()[0], self.sample_rate, {}

    def generate_voice_design(self, **kw):
        self._record("design", kw)
        return self._audio(), self.sample_rate

    def generate_voice_design_streaming(self, **kw):
        self._record("design_stream", kw)
        for _ in range(3):
            yield self._audio()[0], self.sample_rate, {}


def install_tts(custom=True, design=False):
    from wrapper.caps import tts

    fake_soundfile()
    m = FakeTTS(custom=custom, design=design)
    tts._state.update(ready=True, error=None, model=m, custom_voice=custom, voice_design=design,
                      speakers=m.get_supported_speakers() if custom else [])
    return tts, m


REF = "data:audio/wav;base64,UklGRiQAAABXQVZF"
PCM_BYTES = 3 * 480 * 2   # three streamed chunks of 480 float samples, as int16


def t_tts():
    from fastapi.testclient import TestClient

    tts, m = install_tts(custom=True)

    with TestClient(tts.build_app(["tts"])) as c:
        advertises_tasks(c, "tts")
        r = c.post("/v1/audio/speech", json={"input": "hello"})
        check("tts sync 200", r.status_code == 200, (r.status_code, r.text[:120]))
        kw = m.last("custom")
        check("tts speaks a preset voice in the configured language",
              kw and kw["speaker"] in m.get_supported_speakers() and kw["language"], kw)
        check("tts an empty input is a 400",
              c.post("/v1/audio/speech", json={"input": "  "}).status_code == 400)
        check("tts an unsupported response_format is a 400",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "response_format": "midi"}).status_code == 400)
        check("tts a non-JSON body is a 400",
              c.post("/v1/audio/speech", content=b"input=hi").status_code == 400)
        # A voice the checkpoint does not have used to reach the model and die as a 500.
        check("tts an unknown voice is a 400",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "voice": "nobody"}).status_code == 400)
        # CustomVoice weights cannot clone: better a 400 here than a ValueError mid-generation.
        check("tts a preset instance refuses ref_audio",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "ref_audio": REF}).status_code == 400)
        check("tts lists its voices",
              c.get("/v1/audio/voices").json()["voices"] == [{"id": s} for s
                                                             in m.get_supported_speakers()])
            # Only the engine knows what fits; a character-count estimate refused text that works.
        m.refuse_over = 8
        rl = c.post("/v1/audio/speech", json={"input": "far too much text for this one"})
        check("tts the engine's own length error reaches the caller intact",
              "max_seq_len=2048" in rl.text and "prefill has 2148 tokens" in rl.text,
              (rl.status_code, rl.text[:160]))
        check("tts a text the engine accepts is not refused by the wrapper",
              c.post("/v1/audio/speech", json={"input": "short"}).status_code == 200)
        m.refuse_over = None
        r2 = c.post("/v1/audio/speech?async=1", json={"input": "hello"})
        check("tts async 202", r2.status_code == 202, (r2.status_code, r2.text[:120]))
        if r2.status_code == 202:
            doc = poll(c, r2.json()["task"]["id"])
            check("tts task succeeded", doc["status"] == "succeeded", doc.get("status"))
            contract(c, doc, "tts")
            rr = c.get("%s/%s/result" % (TASKS, doc["id"]))
            check("tts async bytes == sync bytes", rr.content == r.content,
                  (len(rr.content), len(r.content)))
            metered(c, r, doc, "tts", ("output",))
            check("tts the stored result carries the duration too",
                  rr.headers.get("x-audio-output-duration-seconds") is not None,
                  dict(rr.headers))

        rs = c.post("/v1/audio/speech", json={"input": "hi", "stream": True,
                                              "response_format": "pcm"})
        check("tts stream 200 raw pcm", rs.status_code == 200 and len(rs.content) == PCM_BYTES,
              (rs.status_code, len(rs.content)))
        check("tts stream took the streaming entry point", "custom_stream" in m.kinds(), m.kinds())
        rw = c.post("/v1/audio/speech", json={"input": "hi", "stream": True,
                                              "response_format": "wav"})
        check("tts a streamed wav opens with a RIFF header",
              rw.content[:4] == b"RIFF" and len(rw.content) == 44 + PCM_BYTES,
              (rw.content[:4], len(rw.content)))
        # mp3 cannot be produced incrementally by soundfile, so promising it would be a lie.
        check("tts a container format cannot be streamed",
              c.post("/v1/audio/speech", json={"input": "hi", "stream": True,
                                               "response_format": "mp3"}).status_code == 400)

        rb = c.post("/v1/audio/speech/batch",
                    json={"response_format": "pcm", "voice": "nofish",
                          "items": [{"input": "one"}, {"input": "two", "voice": "aiden"}]})
        check("tts batch 200", rb.status_code == 200, (rb.status_code, rb.text[:160]))
        if rb.status_code == 200:
            items = rb.json()["items"]
            check("tts batch answers one item per input, in order",
                  [i["index"] for i in items] == [0, 1] and all(i["audio"] for i in items),
                  [(i["index"], len(i["audio"])) for i in items])
            # Top-level keys are defaults, not overrides: item 2 asked for a different voice.
            check("tts batch items override the shared body",
                  m.calls[-1][1]["speaker"] == "aiden"
                  and m.calls[-2][1]["speaker"] == "nofish",
                  [m.calls[-2][1].get("speaker"), m.calls[-1][1].get("speaker")])
        check("tts batch an empty items[] is a 400",
              c.post("/v1/audio/speech/batch", json={"items": []}).status_code == 400)
        check("tts batch a bad item is a 400 before anything runs",
              c.post("/v1/audio/speech/batch",
                     json={"items": [{"input": "ok"}, {"input": " "}]}).status_code == 400)

        with c.websocket_connect("/v1/audio/speech/stream") as ws:
            # Audio leaves as it is decoded, one frame per chunk — that is the whole point.
            def spoken(chunks=3):
                return b"".join(ws.receive_bytes() for _ in range(chunks))

            ws.send_json({"type": "session.config", "language": "English"})
            ready = ws.receive_json()
            check("ws announces the sample rate the frames are in",
                  ready["type"] == "session.ready" and ready["sample_rate"] == m.sample_rate,
                  ready)
            # The point of the socket: a half-sentence waits, a finished one is spoken at once.
            ws.send_json({"type": "input.text", "text": "Half a sen"})
            ws.send_json({"type": "input.text", "text": "tence. And the rest"})
            got = spoken()
            check("ws speaks as soon as a sentence closes", len(got) == PCM_BYTES, len(got))
            check("ws holds back the unfinished tail",
                  m.last("custom_stream")["text"].strip() == "Half a sentence.",
                  m.last("custom_stream")["text"])
            ws.send_json({"type": "input.done"})
            tail = spoken()
            check("ws flushes the tail on input.done", len(tail) == PCM_BYTES, len(tail))
            check("ws spoke the tail it was holding",
                  m.last("custom_stream")["text"].strip() == "And the rest",
                  m.last("custom_stream")["text"])
            check("ws then says the turn is over",
                  ws.receive_json()["type"] == "session.done")
                # One refused sentence used to take the socket down and every later sentence with it.
            m.refuse_over = 8
            ws.send_json({"type": "input.text", "text": "far too much text to speak. "})
            err = ws.receive_json()
            check("ws reports a refused sentence with the engine's own numbers",
                  err["type"] == "error" and "max_seq_len=2048" in err["message"], err)
            m.refuse_over = None
            ws.send_json({"type": "input.text", "text": "Still here. "})
            check("ws keeps speaking after one sentence failed", len(spoken()) == PCM_BYTES)


def t_tts_clone():
    from fastapi.testclient import TestClient

    tts, m = install_tts(custom=False)

    with TestClient(tts.build_app(["tts_clone"])) as c:
        advertises_tasks(c, "tts_clone")
        # Base weights have no speakers at all, so a bare speech call cannot be served.
        check("a clone instance refuses speech with no reference",
              c.post("/v1/audio/speech", json={"input": "hi"}).status_code == 400)
        r = c.post("/v1/audio/speech", json={"input": "hi", "ref_audio": REF})
        check("a clone instance speaks from ref_audio", r.status_code == 200,
              (r.status_code, r.text[:160]))
        check("and the reference reaches the model as a file on disk",
              str(m.last("clone")["ref_audio"]).endswith(".wav"), m.last("clone"))
        check("a ref_audio that is not a data: URL is a 400",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "ref_audio": "https://x/a.wav"}).status_code == 400)
            # A phone recording is an mp4; naming it .wav on disk produced "Format not recognised".
        mp4 = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"
        c.post("/v1/audio/speech", json={"input": "hi", "ref_audio": mp4})
        check("a reference clip keeps the extension of what it actually is",
              str(m.last("clone")["ref_audio"]).endswith(".mp4"), m.last("clone"))
        # audioread.NoBackendError carries no message, so reporting str(e) sent the client "".
        class NoBackendError(Exception):
            pass

        check("a decoder failure with no message still explains itself",
              "decode" in tts._explain(NoBackendError()), tts._explain(NoBackendError()))
            # Base weights have no presets, so the route must be absent, not an empty list.
        check("a clone instance does not serve the voice list at all",
              c.get("/v1/audio/voices").status_code == 404
              and ("GET", "/v1/audio/voices") not in mounted(c))

        del m.calls[:]
        both_ways(c, "/v1/audio/speech/clone", WAV, {"input": "hello"}, "tts clone",
                  json_result=False)
            # Without ref_text or the flag the model refuses, which broke the capability's main use.
        check("a bare clip clones from the speaker embedding",
              m.calls and all(kw.get("xvec_only") is True for _k, kw in m.calls), m.calls[:1])
        del m.calls[:]
        c.post("/v1/audio/speech/clone", files=WAV,
               data={"input": "hi", "ref_text": "what the clip says"})
        check("a supplied ref_text clones from audio and text instead",
              m.last("clone")["ref_text"] == "what the clip says"
              and m.last("clone")["xvec_only"] is False, m.last("clone"))
        del m.calls[:]
        c.post("/v1/audio/speech/clone", files=WAV,
               data={"input": "hi", "ref_text": "x", "x_vector_only_mode": "true"})
        check("an explicit x_vector_only_mode still wins",
              m.last("clone")["xvec_only"] is True, m.last("clone"))
        check("an empty upload is a 400",
              c.post("/v1/audio/speech/clone", files={"file": ("r.wav", b"", "audio/wav")},
                     data={"input": "hi"}).status_code == 400)


def t_tts_design():
    from fastapi.testclient import TestClient

    tts, m = install_tts(custom=False, design=True)

    with TestClient(tts.build_app(["tts"])) as c:
        advertises_tasks(c, "tts")
        check("a design instance refuses speech with no instructions",
              c.post("/v1/audio/speech", json={"input": "hi"}).status_code == 400)
        r = c.post("/v1/audio/speech",
                   json={"input": "hi", "instructions": "a calm low female narrator"})
        check("a design instance speaks from instructions", r.status_code == 200,
              (r.status_code, r.text[:160]))
        kw = m.last("design")
        check("and the description reaches the model as instruct",
              kw and kw["instruct"] == "a calm low female narrator" and kw["text"] == "hi", kw)
        check("a design instance refuses a preset voice",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "voice": "vivian",
                           "instructions": "calm"}).status_code == 400)
        check("a design instance refuses ref_audio",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "instructions": "calm",
                           "ref_audio": REF}).status_code == 400)
        check("a design instance has no voice list",
              c.get("/v1/audio/voices").status_code == 404)

        rs = c.post("/v1/audio/speech",
                    json={"input": "hi", "instructions": "calm", "stream": True,
                          "response_format": "pcm"})
        check("a design instance streams", rs.status_code == 200 and "design_stream" in m.kinds(),
              (rs.status_code, m.kinds()))

        rb = c.post("/v1/audio/speech/batch",
                    json={"response_format": "pcm",
                          "instructions": "a calm narrator",
                          "items": [{"input": "one"},
                                    {"input": "two", "instructions": "a bright child"}]})
        check("a design batch 200", rb.status_code == 200, (rb.status_code, rb.text[:160]))
        if rb.status_code == 200:
            check("a design batch items override the shared instruct",
                  m.calls[-1][1]["instruct"] == "a bright child"
                  and m.calls[-2][1]["instruct"] == "a calm narrator",
                  [m.calls[-2][1].get("instruct"), m.calls[-1][1].get("instruct")])


def t_tts_warmup():
    # Ready has to mean warmed, or the first caller waits longer than llm-init holds a proxy open.
    tts, m = install_tts(custom=True)
    tts._state["ready"] = False

    tts._warmup()
    check("a tts warmup captures the graphs and then speaks",
          m.kinds() == ["warmup", "custom"], m.kinds())
    check("warming up does not by itself declare the engine ready", tts._state["ready"] is False)

    tts, m = install_tts(custom=False)
    tts._warmup()
    check("a clone warmup brings its own reference",
          m.kinds() == ["warmup", "clone"]
          and str(m.last("clone")["ref_audio"]).endswith(".wav"), m.kinds())

    def refuse(**kw):
        raise RuntimeError("model said no")

    m.generate_voice_clone = refuse
    tts._warmup()
    check("a refused warmup is survivable", True)

    tts, m = install_tts(custom=False, design=True)
    tts._warmup()
    check("a design warmup speaks from a throwaway instruct",
          m.kinds() == ["warmup", "design"] and m.last("design")["instruct"], m.kinds())

    import wave

    with wave.open(io.BytesIO(tts._tone_wav())) as w:
        check("the warmup reference is a real mono wav",
              w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getnframes() > 0,
              (w.getnchannels(), w.getframerate(), w.getnframes()))


class FakeDasheng:
    """Stands in for Dasheng-AudioGen. generate() takes the whole list in one call, which is the
    property worth protecting: a batch that quietly became a loop would still pass a naive test."""

    TAGS = (("caption", "<|caption|>"), ("speech", "<|speech|>"), ("asr", "<|asr|>"),
            ("sfx", "<|sfx|>"), ("music", "<|music|>"), ("env", "<|env|>"))

    def __init__(self):
        self.calls = []
        self.config = types.SimpleNamespace(sample_rate=16000)

    def to(self, **_kw):
        return self

    def eval(self):
        return self

    @staticmethod
    def compose_prompt(prompt=None, caption=None, **aspects):
        # Upstream's real validation, because the 400s under test are exactly these two rules.
        if prompt is not None:
            prompt = str(prompt).strip()
            if not prompt:
                raise ValueError("The `prompt` string is empty.")
            if not prompt.startswith("<|caption|>"):
                raise ValueError("The `prompt` string must start with the <|caption|> tag.")
            return prompt
        if caption is None or not str(caption).strip():
            raise ValueError("The `caption` field is required and cannot be empty.")
        values = dict(aspects, caption=caption)
        return " ".join("%s %s" % (tag, values[key]) for key, tag in FakeDasheng.TAGS
                        if values.get(key))

    def generate(self, prompts, **kw):
        self.calls.append((list(prompts), kw))
        return _FakeWave(len(prompts))


class _FakeWave:
    """What torch hands back: (batch, samples), reachable only through .float().cpu().numpy()."""

    def __init__(self, n):
        self._a = np.linspace(-0.5, 0.5, 480, dtype="float32")[None, :].repeat(n, axis=0)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def install_sound_fx():
    from wrapper.caps import sound_fx

    fake_soundfile()
    m = FakeDasheng()
    sound_fx._state.update(ready=True, error=None, model=m)
    return sound_fx, m


def t_sound_fx():
    from fastapi.testclient import TestClient

    sound_fx, m = install_sound_fx()

    with TestClient(sound_fx.build_app(["sound_fx"])) as c:
        advertises_tasks(c, "sound_fx")

        r = c.post("/v1/audio/speech", json={"input": "Rain on a metal roof."})
        check("sound_fx sync 200", r.status_code == 200, (r.status_code, r.text[:120]))
        prompts, kw = m.calls[-1]
        check("sound_fx sends the caption as a tagged prompt",
              prompts == ["<|caption|> Rain on a metal roof."], prompts)
        check("sound_fx uses upstream's generation defaults",
              kw == {"num_steps": 25, "guidance_scale": 5.0, "sway_sampling_coef": -1.0}, kw)

        r = c.post("/v1/audio/speech", json={"input": "A storm.", "sfx": "thunder",
                                             "env": "open field at night"})
        prompts, _ = m.calls[-1]
        check("sound_fx carries the optional aspects in the model's own tags",
              prompts == ["<|caption|> A storm. <|sfx|> thunder <|env|> open field at night"],
              prompts)

        r = c.post("/v1/audio/speech", json={"input": "A click.", "num_steps": 8,
                                             "guidance_scale": 2.5})
        _, kw = m.calls[-1]
        check("sound_fx takes per-request sampler settings",
              kw["num_steps"] == 8 and kw["guidance_scale"] == 2.5, kw)

        r = c.post("/v1/audio/speech", json={"input": ""})
        check("sound_fx an empty input is a 400", r.status_code == 400, r.status_code)

        r = c.post("/v1/audio/speech", json={"prompt": "rain, but untagged"})
        check("sound_fx a prompt without <|caption|> is a 400 before any GPU work",
              r.status_code == 400 and "caption" in r.text, (r.status_code, r.text[:160]))

        r = c.post("/v1/audio/speech", json={"input": "hi", "num_steps": 9999})
        check("sound_fx an absurd num_steps is a 400", r.status_code == 400, r.status_code)

        # The whole point of this base's batch: one denoising pass for the lot.
        before = len(m.calls)
        r = c.post("/v1/audio/speech/batch",
                   json={"items": [{"input": "A cat meowing."},
                                   {"input": "Thunder rolling."},
                                   {"input": "A piano."}]})
        check("sound_fx batch 200", r.status_code == 200, (r.status_code, r.text[:120]))
        check("sound_fx batch is one native pass, not a loop", len(m.calls) - before == 1,
              len(m.calls) - before)
        prompts, _ = m.calls[-1]
        check("sound_fx batch denoises every item together", len(prompts) == 3, prompts)
        items = r.json()["items"]
        check("sound_fx batch returns one audio per item, in order",
              [i["index"] for i in items] == [0, 1, 2] and all(i["audio"] for i in items), items)

        r = c.post("/v1/audio/speech/batch",
                   json={"items": [{"input": "x"}] * (sound_fx.MAX_BATCH + 1)})
        check("sound_fx batch past the ceiling is a 400", r.status_code == 400, r.status_code)

        r = c.post("/v1/audio/speech/batch",
                   json={"items": [{"input": "a", "num_steps": 10},
                                   {"input": "b", "num_steps": 20}]})
        check("sound_fx batch refuses items that disagree on the schedule",
              r.status_code == 400 and "one pass has one schedule" in r.text,
              (r.status_code, r.text[:200]))

        paths = mounted(c)
        check("sound_fx mounts no streaming route (diffusion emits the clip whole)",
              not any("stream" in p for _m, p in paths), sorted(p for _m, p in paths))
        check("sound_fx mounts no voices route (it has no speakers)",
              ("GET", "/v1/audio/voices") not in paths, sorted(p for _m, p in paths))

        sync = c.post("/v1/audio/speech", json={"input": "A door creaking."})
        r = c.post("/v1/audio/speech?async=1", json={"input": "A door creaking."})
        check("sound_fx async dispatches a task", r.status_code in (200, 202), r.status_code)
        if r.status_code == 202:
            metered(c, sync, poll(c, r.json()["task"]["id"]), "sound_fx", ("output",))


def t_sound_fx_not_ready():
    from fastapi.testclient import TestClient
    from wrapper.caps import sound_fx

    sound_fx._state.update(ready=False, error="weights are missing", model=None)
    with TestClient(sound_fx.build_app(["sound_fx"])) as c:
        for path, call in (("/v1/audio/speech",
                            lambda: c.post("/v1/audio/speech", json={"input": "hi"})),
                           ("/v1/audio/speech/batch",
                            lambda: c.post("/v1/audio/speech/batch",
                                           json={"items": [{"input": "hi"}]}))):
            r = call()
            check("sound_fx %s 503s while unloaded" % path, r.status_code == 503,
                  (r.status_code, r.text[:120]))
            check("sound_fx %s says why" % path, "missing" in r.text, r.text[:120])


def t_sound_fx_engine_args():
    """The sampler defaults are settable at deploy time but must stay per-request overridable.

    A deploy-time default that a caller cannot move is a different, worse feature, so both halves
    are pinned here. The module reads ENGINE_ARGS at import, hence the reloads.
    """
    import importlib
    from fastapi.testclient import TestClient
    from wrapper.caps import sound_fx

    was = os.environ.get("ENGINE_ARGS", "")
    try:
        os.environ["ENGINE_ARGS"] = "--num-steps 12 --guidance-scale 3.5"
        importlib.reload(sound_fx)
        _, m = install_sound_fx()
        with TestClient(sound_fx.build_app(["sound_fx"])) as c:
            c.post("/v1/audio/speech", json={"input": "A click."})
            _, kw = m.calls[-1]
            check("sound_fx takes its generation defaults from ENGINE_ARGS",
                  kw["num_steps"] == 12 and kw["guidance_scale"] == 3.5, kw)

            c.post("/v1/audio/speech", json={"input": "A click.", "num_steps": 40})
            _, kw = m.calls[-1]
            check("sound_fx still lets one request outvote the deployed default",
                  kw["num_steps"] == 40, kw)

        # Out of range as a flag would otherwise 400 every call for the life of the deployment.
        os.environ["ENGINE_ARGS"] = "--num-steps 0"
        importlib.reload(sound_fx)
        check("sound_fx ignores an out-of-range --num-steps instead of 400ing forever",
              sound_fx.DEFAULT_STEPS == 25, sound_fx.DEFAULT_STEPS)
    finally:
        os.environ["ENGINE_ARGS"] = was
        importlib.reload(sound_fx)


class FakeSoulX:
    """Stands in for SoulXPodcast. The property worth protecting is that a whole script goes
    through forward_longform exactly once: turn N is conditioned on turns 1..N-1, so a version
    that looped per turn would still return the right audio while losing the conversation."""

    def __init__(self):
        self.calls = []

    def forward_longform(self, **kw):
        self.calls.append(kw)
        return {"generated_wavs": [_FakeTurn() for _ in kw["text_tokens_for_llm"]]}


class _FakeTurn:
    """One turn off the vocoder: a [1, T] tensor reachable only via .detach().float().cpu()."""

    def __init__(self, n=240):
        self._a = np.linspace(-0.5, 0.5, n, dtype="float32")[None, :]

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a


def install_tts_dialogue():
    """Stub the SoulX package. The image clones it onto PYTHONPATH; a test box has no such thing."""
    from wrapper.caps import tts_dialogue

    fake_soundfile()
    seen = {"seeds": [], "inputs": []}

    def process_single_input(dataset, target_text_list, prompt_wav_list, prompt_text_list,
                             use_dialect_prompt, dialect_prompt_text_list):
            # The real dataloader opens these paths, so the cap holds the temp files across generation.
        seen["inputs"].append({"lines": list(target_text_list),
                               "refs": [(p, os.path.exists(p)) for p in prompt_wav_list],
                               "ref_texts": list(prompt_text_list),
                               "dialect": use_dialect_prompt})
        return {"text_tokens_for_llm": [[1] for _ in target_text_list], "spk_ids": [0]}

    mods = {
        "soulxpodcast": {},
        "soulxpodcast.utils": {},
        "soulxpodcast.utils.commons": {"set_all_random_seed": seen["seeds"].append},
        "soulxpodcast.utils.infer_utils": {"process_single_input": process_single_input,
                                           "initiate_model": lambda *a, **k: (None, None)},
    }
    for name, attrs in mods.items():
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
    for parent, child in (("soulxpodcast", "utils"), ("soulxpodcast.utils", "commons"),
                          ("soulxpodcast.utils", "infer_utils")):
        setattr(sys.modules[parent], child, sys.modules["%s.%s" % (parent, child)])

    m = FakeSoulX()
    tts_dialogue._state.update(ready=True, error=None, model=m, dataset=object())
    return tts_dialogue, m, seen


def _spk(i):
    return {"ref_audio": "data:audio/wav;base64," + base64.b64encode(b"RIFF%d" % i).decode(),
            "ref_text": "参考文本%d" % i}


def t_tts_dialogue():
    from fastapi.testclient import TestClient

    tts_dialogue, m, seen = install_tts_dialogue()
    script = {"speakers": [_spk(0), _spk(1)],
              "turns": [{"speaker": 0, "text": "今天天气不错。"},
                        {"speaker": 1, "text": "是啊，适合出门。"},
                        {"speaker": 0, "text": "那走吧。"}]}

    with TestClient(tts_dialogue.build_app(["tts_dialogue"])) as c:
        advertises_tasks(c, "tts_dialogue")

        r = c.post("/v1/audio/speech", json=script)
        check("tts_dialogue sync 200", r.status_code == 200, (r.status_code, r.text[:160]))
        check("tts_dialogue synthesizes the script in one pass, not one call per turn",
              len(m.calls) == 1, len(m.calls))
        sent = seen["inputs"][-1]
        check("tts_dialogue tags each line with a 1-based speaker",
              sent["lines"] == ["[S1]今天天气不错。", "[S2]是啊，适合出门。", "[S1]那走吧。"],
              sent["lines"])
        check("tts_dialogue has every reference clip on disk while generating",
              len(sent["refs"]) == 2 and all(ok for _p, ok in sent["refs"]), sent["refs"])
        check("tts_dialogue removes the temp clips afterwards",
              not any(os.path.exists(p) for p, _ok in sent["refs"]), sent["refs"])
        check("tts_dialogue pairs each clip with its own transcript",
              sent["ref_texts"] == ["参考文本0", "参考文本1"], sent["ref_texts"])
        check("tts_dialogue leaves dialect prompting off on the mandarin checkpoint",
              sent["dialect"] is False, sent["dialect"])
        check("tts_dialogue returns the conversation as one file",
              r.headers.get("X-Audio-Turns") == "3", dict(r.headers))

        r = c.post("/v1/audio/speech", json=dict(script, per_turn=True))
        doc = r.json()
        check("tts_dialogue per_turn returns one clip per turn",
              [t["index"] for t in doc.get("turns", [])] == [0, 1, 2], doc)
        check("tts_dialogue per_turn keeps each turn's speaker",
              [t["speaker"] for t in doc.get("turns", [])] == [0, 1, 0], doc)

        c.post("/v1/audio/speech", json=dict(script, seed=42))
        check("tts_dialogue applies a per-request seed", seen["seeds"][-1:] == [42], seen["seeds"])

        for name, body in (
                ("a script with no speakers", {"turns": script["turns"]}),
                ("a script with no turns", {"speakers": script["speakers"]}),
                ("an empty line", dict(script, turns=[{"speaker": 0, "text": "  "}])),
                ("a turn pointing at a speaker that was not supplied",
                 dict(script, turns=[{"speaker": 5, "text": "喂"}])),
                ("a reference clip with no transcript",
                 dict(script, speakers=[{"ref_audio": _spk(0)["ref_audio"]}])),
                ("a reference clip that is not a data: URL",
                 dict(script, speakers=[dict(_spk(0), ref_audio="https://example/a.wav")])),
                ("more speakers than the model takes",
                 dict(script, speakers=[_spk(i) for i in range(5)])),
                ("an unknown response_format", dict(script, response_format="aiff"))):
            r = c.post("/v1/audio/speech", json=body)
            check("tts_dialogue rejects %s with a 400" % name, r.status_code == 400,
                  (r.status_code, r.text[:140]))

        sync = c.post("/v1/audio/speech", json=script)
        r = c.post("/v1/audio/speech?async=1", json=script)
        check("tts_dialogue async dispatches a task", r.status_code in (200, 202), r.status_code)
        if r.status_code == 202:
            metered(c, sync, poll(c, r.json()["task"]["id"]), "tts_dialogue", ("output",))


def t_tts_dialogue_not_ready():
    from fastapi.testclient import TestClient
    from wrapper.caps import tts_dialogue

    tts_dialogue._state.update(ready=False, error="weights are missing", model=None, dataset=None)
    with TestClient(tts_dialogue.build_app(["tts_dialogue"])) as c:
        r = c.post("/v1/audio/speech", json={"speakers": [_spk(0)], "turns": []})
        check("tts_dialogue 503s while unloaded", r.status_code == 503,
              (r.status_code, r.text[:120]))
        check("tts_dialogue says why", "missing" in r.text, r.text[:120])


def t_tts_not_ready():
    # A model that never loaded must say so on every route, not 500 on a None.
    from fastapi.testclient import TestClient
    from wrapper.caps import tts

    tts._state.update(ready=False, error="weights are missing", model=None, custom_voice=True,
                      voice_design=False, speakers=[])
    with TestClient(tts.build_app(["tts"])) as c:
        for path, call in (("/v1/audio/speech", lambda: c.post("/v1/audio/speech",
                                                               json={"input": "hi"})),
                           ("/v1/audio/speech/batch",
                            lambda: c.post("/v1/audio/speech/batch",
                                           json={"items": [{"input": "hi"}]})),
                           ("/v1/audio/voices", lambda: c.get("/v1/audio/voices"))):
            r = call()
            check("%s is a 503 while the model is not loaded" % path, r.status_code == 503,
                  (r.status_code, r.text[:100]))
            check("%s says why" % path, "weights are missing" in r.text, r.text[:100])
        with c.websocket_connect("/v1/audio/speech/stream") as ws:
            frame = ws.receive_json()
            check("the socket refuses instead of accepting text it cannot speak",
                  frame["type"] == "error" and "weights are missing" in frame["message"], frame)


class FakeCrispSession:
    # Same shape the cap calls; 1 s of silence so duration metering has a known number.
    def __init__(self):
        self.backend = "voxtral-tts"
        self.texts = []
        self.voice = None
        self.n = 24000

    def speakers(self):
        return ["de_female", "casual_male"]

    def output_sample_rate(self):
        return 24000

    def set_speaker_name(self, voice):
        self.voice = voice

    def set_voice(self, voice):
        self.voice = voice

    def synthesize(self, text):
        self.texts.append(text)
        return np.zeros((self.n,), dtype="float32")


def install_crispasr():
    from wrapper.caps import crispasr_tts

    fake_soundfile()
    sess = FakeCrispSession()
    crispasr_tts._state.update(ready=True, error=None, session=sess,
                               speakers=sess.speakers(), out_sr=24000, quant="Q8_0",
                               voice_set="")
    return crispasr_tts, sess


def t_crispasr():
    from fastapi.testclient import TestClient

    cap, sess = install_crispasr()
    with TestClient(cap.build_app(["tts"])) as c:
        advertises_tasks(c, "crispasr")
        r = c.post("/v1/audio/speech", json={"input": "hello"})
        check("crispasr sync 200", r.status_code == 200, (r.status_code, r.text[:120]))
        check("crispasr an empty input is a 400",
              c.post("/v1/audio/speech", json={"input": "  "}).status_code == 400)
        check("crispasr an unknown voice is a 400",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "voice": "nobody"}).status_code == 400)
        check("crispasr refuses ref_audio rather than answering in a preset",
              c.post("/v1/audio/speech",
                     json={"input": "hi", "ref_audio": "data:audio/wav;base64,AA"}).status_code == 400)
        check("crispasr lists its voices",
              c.get("/v1/audio/voices").json()["voices"] == [{"id": "de_female"},
                                                            {"id": "casual_male"}])
        r2 = c.post("/v1/audio/speech?async=1", json={"input": "hello"})
        check("crispasr async 202", r2.status_code == 202, (r2.status_code, r2.text[:120]))
        if r2.status_code == 202:
            doc = poll(c, r2.json()["task"]["id"])
            check("crispasr task succeeded", doc["status"] == "succeeded", doc.get("status"))
            contract(c, doc, "crispasr")
            metered(c, r, doc, "crispasr", ("output",))
        paths = mounted(c)
        check("crispasr mounts no websocket (ggml speaks whole utterances)",
              not any(m == "WS" for m, _p in paths), sorted(paths))
        rb = c.post("/v1/audio/speech/batch",
                    json={"items": [{"input": "one"}, {"input": "two", "voice": "casual_male"}]})
        check("crispasr batch 200", rb.status_code == 200, (rb.status_code, rb.text[:160]))
        if rb.status_code == 200:
            items = rb.json()["items"]
            check("crispasr batch answers one item per input",
                  [i["index"] for i in items] == [0, 1] and all(i["audio"] for i in items),
                  [(i["index"], len(i.get("audio") or "")) for i in items])
            check("crispasr batch items override the shared voice",
                  sess.voice == "casual_male", sess.voice)


def t_crispasr_not_ready():
    from fastapi.testclient import TestClient
    from wrapper.caps import crispasr_tts

    crispasr_tts._state.update(ready=False, error="weights are missing", session=None,
                               speakers=[])
    with TestClient(crispasr_tts.build_app(["tts"])) as c:
        for path, call in (("/v1/audio/speech",
                            lambda: c.post("/v1/audio/speech", json={"input": "hi"})),
                           ("/v1/audio/speech/batch",
                            lambda: c.post("/v1/audio/speech/batch",
                                           json={"items": [{"input": "hi"}]})),
                           ("/v1/audio/voices", lambda: c.get("/v1/audio/voices"))):
            r = call()
            check("crispasr %s 503s while unloaded" % path, r.status_code == 503,
                  (r.status_code, r.text[:120]))
            check("crispasr %s says why" % path, "missing" in r.text, r.text[:120])


class FakeELBackend:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def presets(self):
        return [{"voice_id": "p1", "name": "Preset One", "category": "premade",
                 "instruction": "A clear young female voice.",
                 "sample_text": "Hello there friend.",
                 "description": "test preset"}]

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        from wrapper.caps import tts_el

        self.calls.append(("clone", text, prompt_text, kw.get("instruction"), kw.get("settings")))
        tts_el.job_tick(kw.get("ctx"), 1, 1)
        n = max(2400, len(text) * 120)
        return np.zeros(n, dtype="float32"), 24000

    def design(self, instruction, text, **kw):
        from wrapper.caps import tts_el

        self.calls.append(("design", instruction, text, kw.get("settings")))
        tts_el.job_tick(kw.get("ctx"), 1, 1, stage="design")
        n = max(2400, len(text) * 120)
        return np.zeros(n, dtype="float32"), 24000, {"plan": "ok"}


class TickCtx:
    """Stand-in for tasks.Context: count checkpoints, record progress, cancel on the Nth tick."""

    def __init__(self, cancel_after=None):
        self.n = 0
        self.ticks = []
        self.cancel_after = cancel_after

    def checkpoint(self):
        from wrapper.tasks import Cancelled

        self.n += 1
        if self.cancel_after is not None and self.n >= self.cancel_after:
            raise Cancelled("canceled by client")

    def progress(self, **kw):
        self.ticks.append(kw)


def t_voice_cards():
    import json
    import tempfile

    from wrapper.caps import tts_el

    fr = tts_el.builtin_presets("firered")
    br = tts_el.builtin_presets("breeze")
    ids = [c.get("voice_id") for c in fr]
    langs = {(c.get("labels") or {}).get("language") for c in fr}
    check("shared builtin pack is zh+en only",
          len(fr) == 4 and len(set(ids)) == 4
          and set(ids) == {"zh-f", "zh-m", "en-f", "en-m"}, ids)
    check("no language outside zh/en", langs <= {"zh", "en"}, langs)
    check("firered and breeze read the same cards",
          [c.get("voice_id") for c in br] == ids, ids)
    check("voice_id is a role slug, not an engine prefix",
          ids and all(v and not v.startswith(("fr3-", "br2-")) for v in ids), ids)
    check("cards are premade with an instruction",
          all(c.get("category") == "premade" and c.get("instruction") for c in fr + br))
    check("list name and description are bilingual",
          all(" / " in (c.get("name") or "")
              and any("\u4e00" <= ch <= "\u9fff" for ch in (c.get("description") or ""))
              and any("a" <= ch.lower() <= "z" for ch in (c.get("description") or ""))
              for c in fr),
          [(c.get("voice_id"), c.get("name")) for c in fr])
    dest = tempfile.mkdtemp(prefix="voices-seed-")
    seeded = tts_el.seed_builtins("firered", dest)
    check("seed copies shared cards onto the runtime dir", len(seeded) == len(fr))
    pack_card = next(c for c in fr if c.get("voice_id") == "zh-f")
    check("clone pack plants factory wav and transcript",
          os.path.isfile(dest + "/zh-f/prompt.wav")
          and os.path.isfile(dest + "/zh-f/prompt.txt"))
    factory_wav = open(dest + "/zh-f/prompt.wav", "rb").read()
    meta_path = dest + "/zh-f/meta.json"
    old = json.loads(open(meta_path).read())
    old["name"] = "旧名"
    old["description"] = "旧介绍"
    old["instruction"] = "这是一条已废弃的设计词"
    open(meta_path, "w").write(json.dumps(old, ensure_ascii=False))
    os.makedirs(dest + "/ja-f", exist_ok=True)
    open(dest + "/ja-f/meta.json", "w").write(
        '{"voice_id": "ja-f", "name": "old", "category": "premade"}')
    tts_el.seed_builtins("firered", dest)
    check("clone instruction change keeps the factory wav",
          os.path.isfile(dest + "/zh-f/prompt.wav")
          and open(dest + "/zh-f/prompt.wav", "rb").read() == factory_wav)
    patched = json.loads(open(meta_path).read())
    check("display fields follow the pack when the ref matches",
          patched.get("name") == pack_card.get("name")
          and patched.get("description") == pack_card.get("description")
          and patched.get("instruction") == pack_card.get("instruction"))
    check("seed drops a premade that left the pack", not os.path.isdir(dest + "/ja-f"))
    for vid in ("fr3-warm-zh-f", "br2-clear-en-f"):
        os.makedirs(dest + "/" + vid, exist_ok=True)
        open(dest + "/" + vid + "/meta.json", "w").write(
            '{"voice_id": "%s", "name": "old", "category": "premade"}' % vid)
    leftover = dest + "/clone-keep/meta.json"
    os.makedirs(dest + "/clone-keep", exist_ok=True)
    open(leftover, "w").write('{"voice_id": "clone-keep", "name": "mine", "category": "cloned"}')
    reused = dest + "/fr3-calm-zh-m/meta.json"
    os.makedirs(dest + "/fr3-calm-zh-m", exist_ok=True)
    open(reused, "w").write('{"voice_id": "fr3-calm-zh-m", "name": "mine", "category": "cloned"}')
    tts_el.seed_builtins("firered", dest)
    check("seed drops leftover premade ids",
          not os.path.isdir(dest + "/fr3-warm-zh-f")
          and not os.path.isdir(dest + "/br2-clear-en-f"))
    check("seed does not drop a cloned card", open(leftover).read().find("clone-keep") >= 0)
    check("seed does not drop a user card that reused an old id",
          open(reused).read().find("cloned") >= 0)
    check("shared cards are factory clone",
          all(c.get("source") == "clone" for c in fr)
          and all(os.path.isfile(os.path.join(str(tts_el.voices_root()), c["voice_id"], "prompt.wav"))
                  for c in fr),
          [c.get("source") for c in fr])
    from wrapper.caps.firered import FireRedBackend
    backend_ids = [c["voice_id"] for c in FireRedBackend(None).presets()]
    check("firered backend presets read the cards", backend_ids == ids, backend_ids)

    class OnlyNative:
        uses_shared_pack = False

        def native_presets(self):
            return [{"voice_id": "model-a", "name": "Model A", "category": "premade",
                     "instruction": "native"}]

    native_only = tempfile.mkdtemp(prefix="voices-native-")
    planted = tts_el.seed_builtins("firered", native_only, backend=OnlyNative())
    planted_ids = [c["voice_id"] for c in planted]
    check("native-only backend does not take our pack",
          planted_ids == ["model-a"], planted_ids)

    class NativePlusOurs:
        uses_shared_pack = True

        def native_presets(self):
            return [{"voice_id": "model-a", "name": "Model A", "category": "premade",
                     "instruction": "native"}]

    both = tempfile.mkdtemp(prefix="voices-both-")
    planted = tts_el.seed_builtins("firered", both, backend=NativePlusOurs())
    planted_ids = [c["voice_id"] for c in planted]
    check("native plus our pack lists both",
          planted_ids[0] == "model-a" and set(planted_ids) == set(["model-a"] + ids),
          planted_ids)


def t_voice_cards_clone_seed():
    """Clone premade identity is wav+transcript; instruction is Voice Direction only."""
    import json
    import tempfile

    from wrapper.caps import tts_el

    pack = tempfile.mkdtemp(prefix="pack-clone-")
    dest = tempfile.mkdtemp(prefix="dest-clone-")
    card = {"voice_id": "zh-f", "name": "中文女 / Chinese Female", "category": "premade",
            "source": "clone", "instruction": "keep direction",
            "description": "普通话女声。Mandarin female."}
    os.makedirs(os.path.join(pack, "zh-f"))
    open(os.path.join(pack, "zh-f", "meta.json"), "w").write(json.dumps(card, ensure_ascii=False))
    open(os.path.join(pack, "zh-f", "prompt.wav"), "wb").write(b"RIFF" + b"A" * 80)
    open(os.path.join(pack, "zh-f", "prompt.txt"), "w").write("今天天气很好")
    tts_el.seed_builtins("breeze", dest, cards=[card], pack=pack)
    check("clone seed plants wav and transcript",
          open(os.path.join(dest, "zh-f", "prompt.txt")).read() == "今天天气很好"
          and open(os.path.join(dest, "zh-f", "prompt.wav"), "rb").read().startswith(b"RIFF"))
    card["instruction"] = "new direction"
    card["name"] = "新名 / New"
    open(os.path.join(pack, "zh-f", "meta.json"), "w").write(json.dumps(card, ensure_ascii=False))
    tts_el.seed_builtins("breeze", dest, cards=[card], pack=pack)
    check("clone instruction change keeps factory wav",
          open(os.path.join(dest, "zh-f", "prompt.wav"), "rb").read() == b"RIFF" + b"A" * 80)
    patched = json.loads(open(os.path.join(dest, "zh-f", "meta.json")).read())
    check("clone instruction change patches display",
          patched.get("name") == "新名 / New" and patched.get("instruction") == "new direction",
          patched)
    open(os.path.join(pack, "zh-f", "prompt.wav"), "wb").write(b"RIFF" + b"B" * 80)
    tts_el.seed_builtins("breeze", dest, cards=[card], pack=pack)
    check("clone wav change replants",
          open(os.path.join(dest, "zh-f", "prompt.wav"), "rb").read() == b"RIFF" + b"B" * 80)
    open(os.path.join(pack, "zh-f", "prompt.txt"), "w").write("换一句")
    tts_el.seed_builtins("breeze", dest, cards=[card], pack=pack)
    check("clone transcript change replants txt",
          open(os.path.join(dest, "zh-f", "prompt.txt")).read() == "换一句")


def t_breeze_split_speak():
    from wrapper.caps import breeze

    check("short chinese stays one piece",
          breeze.split_speak("第一句。第二句！第三句？") == ["第一句。第二句！第三句？"])
    check("short english stays one piece",
          breeze.split_speak("Hello there. Next one!") == ["Hello there. Next one!"])
    check("newlines always split",
          breeze.split_speak("一行。\n二行。") == ["一行。", "二行。"])
    tagged = breeze.split_speak("[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。下一句。")
    check("inline laugh stays in the line",
          tagged == ["[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。下一句。"], tagged)
    en = breeze.split_speak("(laugh) Hello there. Next one.")
    check("english laugh stays in the line",
          en == ["(laugh) Hello there. Next one."], en)
    check("short ellipsis stays one piece",
          breeze.split_speak("Wait... Then this.") == ["Wait... Then this."])
    limit = breeze.speak_limit()
    zh = ("啊" * (limit - 2)) + "。下一句！"
    zh_parts = breeze.split_speak(zh)
    check("over-budget chinese cuts on 。！？",
          len(zh_parts) == 2 and zh_parts[1] == "下一句！", zh_parts)
    pad = ("alpha " * (limit // 6 + 4)).strip()
    en_over = pad + " sees 3.14 exactly. Next one."
    en_parts = breeze.split_speak(en_over)
    check("english decimal is not a sentence cut",
          any("3.14 exactly." in p for p in en_parts), en_parts)
    check("english period plus space cuts when over budget",
          any(p.strip() == "Next one." for p in en_parts), en_parts)
    check("english title period is not a sentence cut",
          breeze.split_speak(("word " * (limit // 5 + 3)).strip() + " saw Mr. Smith today. Done.")
          and any("Mr. Smith today." in p for p in breeze.split_speak(
              ("word " * (limit // 5 + 3)).strip() + " saw Mr. Smith today. Done.")),
          breeze.split_speak(("word " * (limit // 5 + 3)).strip() + " saw Mr. Smith today. Done."))
    long = "一二，" * ((limit // 2) + 2)
    parts = breeze.split_speak(long)
    check("over-budget sentence splits on clauses",
          len(parts) > 1 and all(len(p) <= limit for p in parts),
          (limit, [len(p) for p in parts]))
    thousands = ("word " * (limit // 5 + 3)).strip() + " costs 1,000 now, then more."
    th_parts = breeze.split_speak(thousands)
    check("english thousands comma is not a clause cut",
          any("1,000 now," in p or "1,000 now, then more." in p for p in th_parts), th_parts)
    blob = "啊" * (limit + 40)
    pieces = breeze.split_speak(blob)
    check("unpunctuated over-budget is hard cut",
          len(pieces) > 1 and all(len(p) <= limit for p in pieces)
          and "".join(pieces) == blob, [len(p) for p in pieces])


def t_breeze_seeds_every_generation():
    """Breeze seeds by calling out to the runtime, so a skipped call is not a default —
    it is the previous request's leftover RNG, and no stability setting can pin it."""
    import types

    from wrapper.caps import breeze, tts_el

    seen = []
    runtime = types.ModuleType("breeze_infer.runtime")
    runtime.set_all_seeds = lambda s: seen.append(int(s))
    templates = types.ModuleType("breeze_infer.templates")
    templates.get_template = lambda name: name
    templates.prepare_inputs = lambda *a, **k: {}
    parent = types.ModuleType("breeze_infer")
    parent.runtime, parent.templates = runtime, templates
    for name, mod in (("breeze_infer", parent), ("breeze_infer.runtime", runtime),
                      ("breeze_infer.templates", templates)):
        sys.modules[name] = mod

    class OneChunk:
        sample_rate = 24000

        def iter_audio_chunks(self, inputs, request_id=None):
            yield [0.0, 0.1, 0.0]

    back = breeze.BreezeBackend(OneChunk(), None, None, None)
    saved = tts_el.SEED
    tts_el.SEED = None
    try:
        list(back._iter_generate("你好", "", seed=None))
        check("an unasked-for seed still seeds, at upstream's own default",
              seen == [breeze._UPSTREAM_SEED], seen)
        seen.clear()
        list(back._iter_generate("你好", "", seed=7))
        check("an asked-for seed is the one used", seen == [7], seen)
    finally:
        tts_el.SEED = saved
        for name in ("breeze_infer.templates", "breeze_infer.runtime", "breeze_infer"):
            sys.modules.pop(name, None)


def t_breeze_clone_slices():
    fake_soundfile()
    from wrapper.caps import tts_el
    from wrapper.caps.breeze import BreezeBackend, speak_limit

    be = BreezeBackend(None, None, None, None)
    seen = []
    cfgs = []

    def gen(text, instruction, ref_path=None, ref_text=None, cfg=None, **kw):
        seen.append(text)
        cfgs.append(cfg)
        return np.ones(2400, dtype="float32"), 24000

    be._generate = gen
    audio, sr = be.clone("第一句。第二句。", np.zeros(2400, dtype="float32"), 24000, "ref")
    check("short clone stays one generate", seen == ["第一句。第二句。"], seen)
    check("bare clone uses cfg scale 1", cfgs[-1] == tts_el.CFG_SCALE, cfgs)
    seen.clear()
    cfgs.clear()
    audio, sr = be.clone("第一句。\n第二句。", np.zeros(2400, dtype="float32"), 24000, "ref")
    check("newline clone speaks each line once", seen == ["第一句。", "第二句。"], seen)
    from wrapper.caps import tts_long
    zh_gap = int(tts_long.PAUSE_MS_ZH / 1000.0 * 24000)
    check("clone inserts a chinese pause between slices",
          len(audio) == 2400 + 2400 + zh_gap and sr == 24000, (len(audio), zh_gap))
    quiet = np.concatenate([
        np.ones(1200, dtype="float32"),
        np.full(400, 0.01, dtype="float32"),
        np.zeros(800, dtype="float32"),
    ])
    mid = np.concatenate([np.ones(1600, dtype="float32"), np.zeros(4000, dtype="float32")])
    joined, _ = tts_long.join([mid, quiet], 24000, gap_ms=0)
    mid_kept = len(joined) - len(quiet)
    check("middle slice drops trailing silence only",
          1600 <= mid_kept <= 1600 + int(0.08 * 24000) + 2, mid_kept)
    check("last slice keeps a quiet last syllable",
          float(np.max(np.abs(joined[-1200:]))) >= 0.009, float(np.max(np.abs(joined[-1200:]))))
    seen.clear()
    cfgs.clear()
    audio, sr = be.clone("Hello.\nNext.", np.zeros(2400, dtype="float32"), 24000, "ref")
    en_gap = int(tts_long.PAUSE_MS_EN / 1000.0 * 24000)
    check("clone inserts a shorter english pause",
          len(audio) == 2400 + 2400 + en_gap, (len(audio), en_gap))
    seen.clear()
    cfgs.clear()
    be.clone("短句。", np.zeros(2400, dtype="float32"), 24000, "ref",
             instruction="说得慢一点")
    check("request direction uses design cfg", cfgs[-1] == tts_el.DESIGN_CFG, cfgs)
    seen.clear()
    be.design("A warm voice.", "第一句。第二句。第三句。")
    check("design stays one shot", seen == ["第一句。第二句。第三句。"], seen)
    seen.clear()
    blob = "啊" * (speak_limit() + 40)
    be.clone(blob, np.zeros(2400, dtype="float32"), 24000, "ref")
    check("clone hard-cuts an unsplittable piece",
          len(seen) > 1 and all(len(s) <= speak_limit() for s in seen)
          and "".join(seen) == blob, [len(s) for s in seen])


def t_breeze_stream_chunks():
    fake_soundfile()
    from wrapper.caps.breeze import BreezeBackend

    be = BreezeBackend(None, None, None, None)
    seen = []

    def it(text, instruction, ref_path=None, ref_text=None, cfg=None, **kw):
        seen.append(text)
        yield np.ones(80, dtype="float32"), 24000
        yield np.ones(80, dtype="float32"), 24000

    be._iter_generate = it
    waves = list(be.iter_clone(
        "第一句。\n第二句。", np.zeros(2400, dtype="float32"), 24000, "ref",
        pace_each=True))
    check("pace_each yields codec chunks and a pause",
          len(waves) == 5 and seen == ["第一句。", "第二句。"], (len(waves), seen))
    check("first two yields are audio",
          len(waves[0][0]) == 80 and len(waves[1][0]) == 80, (len(waves[0][0]), len(waves[1][0])))
    gap = waves[2][0]
    check("pause sits between parts",
          len(gap) > 0 and float(np.max(np.abs(gap))) == 0.0, len(gap))
    check("stream_gap stays off", be.stream_gap("中文。", 24000) is None)
    raw = np.ones(16, dtype="float32")
    check("stream_next does not fade every chunk",
          np.array_equal(be.stream_next_slice(raw, 24000), raw))


def _tempo_all(wave, sr, factor):
    """One whole buffer through the shared streaming tempo, for the offline assertions."""
    from wrapper.caps.tts_long import TempoStream

    stream = TempoStream(sr, factor)
    try:
        parts = stream.write(wave, sr) + stream.finish()
    finally:
        stream.abort()
    return np.concatenate(parts)


def t_breeze_pace():
    import shutil

    if shutil.which("ffmpeg") is None:
        print("  skip tempo checks: ffmpeg is not installed in the lint runner")
        return

    from wrapper.caps.tts_long import TempoStream, _atempo_chain, pace
    from wrapper.caps import tts_el

    # Both engines share one tempo filter; lengths and pitch are the contract.
    note = 440.0
    t = np.arange(24000, dtype="float32") / 24000.0
    tone = (0.4 * np.sin(2.0 * np.pi * note * t)).astype("float32")

    def heard(wave, sr):
        crossings = int(np.count_nonzero(np.diff(np.signbit(wave))))
        return crossings / 2.0 / (len(wave) / float(sr))

    out, sr = pace(tone, 24000, 2.0)
    check("2x halves samples", 11000 <= len(out) <= 13000 and sr == 24000, len(out))
    check("2x is still the same note", abs(heard(out, 24000) - note) < 25.0, heard(out, 24000))
    keep, _ = pace(tone, 24000, 1.0)
    check("1x keeps length", len(keep) == 24000, len(keep))
    fast, _ = pace(tone, 24000, 4.0)
    check("4x is about a quarter", 5500 <= len(fast) <= 7000, len(fast))
    slow, _ = pace(tone, 24000, 0.5)
    check("0.5x doubles samples", 45000 <= len(slow) <= 51000, len(slow))
    check("0.5x is still the same note", abs(heard(slow, 24000) - note) < 25.0, heard(slow, 24000))

    # atempo is FireRed's, and it is the one that has to keep pitch across chunks.
    t = np.arange(24000, dtype="float32") / 24000.0
    tone = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    paced = _tempo_all(tone, 24000, 2.0)
    check("atempo halves samples", 11500 <= len(paced) <= 13000, len(paced))
    check("atempo quarters at 4x", 5500 <= len(_tempo_all(tone, 24000, 4.0)) <= 7000)
    check("atempo doubles at 0.5x", 45000 <= len(_tempo_all(tone, 24000, 0.5)) <= 51000)
    check("atempo quadruples at 0.25x", 88000 <= len(_tempo_all(tone, 24000, 0.25)) <= 102000)

    spectrum = np.abs(np.fft.rfft(paced * np.hanning(len(paced))))
    hz = np.fft.rfftfreq(len(paced), 1.0 / 24000.0)[int(np.argmax(spectrum))]
    check("tempo keeps pitch", abs(hz - 440.0) < 8.0, hz)

    high = _tempo_all((0.25 * np.sin(2.0 * np.pi * 7000.0 * t)).astype("float32"), 24000, 2.0)
    high_spectrum = np.abs(np.fft.rfft(high * np.hanning(len(high))))
    high_hz = np.fft.rfftfreq(len(high), 1.0 / 24000.0)[int(np.argmax(high_spectrum))]
    check("tempo does not alias high frequencies", abs(high_hz - 7000.0) < 20.0, high_hz)

    stream = TempoStream(24000, 2.0)
    streamed = []
    try:
        for chunk in np.array_split(tone, 7):
            streamed.extend(stream.write(chunk, 24000))
        streamed.extend(stream.finish())
    finally:
        stream.abort()
    streamed = np.concatenate(streamed)
    check("tempo state crosses input chunk boundaries",
          len(streamed) == len(paced) and np.allclose(streamed, paced),
          (len(streamed), len(paced)))

    # A separate stream, because this one is about latency rather than samples:
    live = TempoStream(24000, 2.0)
    live_chunks = 0
    deadline = time.time() + 10.0
    try:
        while live_chunks == 0 and time.time() < deadline:
            for chunk in np.array_split(tone, 7):
                live_chunks += len(live.write(chunk, 24000))
    finally:
        live.abort()
    check("tempo emits before the request finishes", live_chunks > 0, live_chunks)
    check("atempo extremes use portable chains",
          _atempo_chain(0.25) == "atempo=0.5,atempo=0.5"
          and _atempo_chain(4.0) == "atempo=2,atempo=2",
          (_atempo_chain(0.25), _atempo_chain(4.0)))

    knobs = tts_el.resolve_settings(
        {**tts_el._DEFAULT_SETTINGS, "speed": 2.0}, 1.0,
        "沉稳男声", text="你好", include_speed_direction=False)
    check("breeze does not ask the model to apply speed twice",
          "快" not in knobs.instruction and "slow" not in knobs.instruction.lower(),
          knobs.instruction)


def t_tts_job_tick():
    from wrapper.caps import tts_el
    from wrapper.tasks import Cancelled

    ctx = TickCtx()
    tts_el.job_tick(None, 1, 2)
    tts_el.job_tick(ctx, 1, 4)
    check("tick publishes done/total",
          ctx.ticks and ctx.ticks[-1].get("done") == 1 and ctx.ticks[-1].get("total") == 4,
          ctx.ticks)
    ctx2 = TickCtx(cancel_after=1)
    try:
        tts_el.job_tick(ctx2, 0, 2)
        check("tick raises Cancelled", False)
    except Cancelled:
        check("tick raises Cancelled", True)


def t_stream_ctx_cancel():
    import threading
    import time

    from wrapper.caps import tts_el
    from wrapper import tasks
    from wrapper.tasks import Cancelled

    stop = threading.Event()
    ctx = tasks.stream_ctx(stop)
    seen = []

    def work():
        try:
            for i, part in enumerate(["a", "b", "c"]):
                tts_el.job_tick(ctx, i, 3)
                seen.append(part)
                time.sleep(0.25)
        except Cancelled:
            return

    th = threading.Thread(target=work)
    th.start()
    time.sleep(0.05)
    stop.set()
    th.join(2)
    check("stream_ctx cancel stops before later pieces", seen == ["a"], seen)


def t_live_stream_stops_iter():
    import asyncio
    import tempfile
    import threading
    import time

    from wrapper.caps import tts_el

    fake_soundfile()
    seen = []
    started = threading.Event()

    class SlowEL(FakeELBackend):
        def iter_clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
            ctx = kw.get("ctx")
            for i, part in enumerate(["one", "two", "three"]):
                tts_el.job_tick(ctx, i, 3)
                seen.append(part)
                started.set()
                time.sleep(0.35)
                yield np.ones(80, dtype="float32"), 24000

    class FlipReq:
        def __init__(self):
            self.gone = False

        async def is_disconnected(self):
            return self.gone

    backend = SlowEL()
    root = tempfile.mkdtemp(prefix="el-stream-stop-")
    tts_el.install(backend, store=tts_el.VoiceStore(root, backend.presets()))
    req = FlipReq()

    hid_box = [None]

    async def run():
        resp = await tts_el._live_stream("p1", "hello stream", "", "mp3", None, req)
        check("stream opened", getattr(resp, "status_code", 200) == 200, getattr(resp, "status_code", None))
        hid_box[0] = resp.headers.get("history-item-id")
        started.wait(2)
        req.gone = True
        async for _ in resp.body_iterator:
            pass

    asyncio.run(run())
    # Wait for the reading thread to finish rather than guessing at it.
    doc = {}
    deadline = time.time() + 20
    while time.time() < deadline:
        doc = tts_el._history().get(hid_box[0]) or {}
        if doc.get("state") in ("created", "canceled"):
            break
        time.sleep(0.05)
    # A response ending is not the reading ending; unread slices must still be spoken for Range.
    check("disconnect keeps speaking for a resume", "three" in seen, seen)
    check("a reading the client left is finished, not canceled",
          doc.get("state") == "created", doc.get("state"))


def t_live_stream_named_id():
    """A client that names its reading can stop it before the request has landed."""
    import asyncio
    import tempfile

    from wrapper.caps import tts_el

    fake_soundfile()
    backend = FakeELBackend()
    root = tempfile.mkdtemp(prefix="el-named-")
    tts_el.install(backend, store=tts_el.VoiceStore(root, backend.presets()))

    named = "hi" + "0" * 20
    check("a name that is not an id is refused",
          _raises_status(lambda: tts_el._named_id("../escape", "hi")) == 400, None)
    check("a well-formed name is accepted", tts_el._named_id(named, "hi") == named, None)

    async def speak(hid):
        return await tts_el._live_stream("p1", "hello", "", "mp3", None, None, hid=hid)

    resp = asyncio.run(speak(named))
    check("the reading answers to the name it was given",
          resp.headers.get("history-item-id") == named, resp.headers.get("history-item-id"))

    # Stopped while the request is still in flight; the later reading is never spoken.
    refused = "hi" + "1" * 20
    tts_el._remember_refusal(refused)
    stopped = asyncio.run(speak(refused))
    check("a reading stopped before it started does not speak",
          getattr(stopped, "status_code", None) == 499, getattr(stopped, "status_code", None))
    check("the refusal is spent once", tts_el._was_refused(refused) is False, None)


def t_live_grace_widens_for_followers():
    """One bad minute must not kill a reading whose client has been coming back."""
    import threading

    from wrapper.caps import tts_el

    live = tts_el._LiveReading("hi" + "2" * 20, threading.Event())
    check("a reading nobody resumed keeps the short grace",
          live.grace() == tts_el._RESUME_GRACE_SECONDS, live.grace())
    live.followed = True
    check("a followed reading is given the long grace",
          live.grace() == tts_el._RESUME_GRACE_FOLLOWED_SECONDS, live.grace())
    check("the long grace is the wider of the two",
          tts_el._RESUME_GRACE_FOLLOWED_SECONDS > tts_el._RESUME_GRACE_SECONDS, None)


def _raises_status(fn):
    from fastapi import HTTPException

    try:
        fn()
    except HTTPException as e:
        return e.status_code
    return None


def t_breeze_clone_cancel():
    fake_soundfile()
    from wrapper.caps.breeze import BreezeBackend
    from wrapper.tasks import Cancelled

    be = BreezeBackend(None, None, None, None)
    seen = []

    def gen(text, instruction, ref_path=None, ref_text=None, cfg=None, **kw):
        seen.append(text)
        return np.ones(80, dtype="float32"), 24000

    be._generate = gen
    ctx = TickCtx(cancel_after=3)
    try:
        be.clone("第一句。\n第二句。", np.zeros(2400, dtype="float32"), 24000, "ref", ctx=ctx)
        check("canceled breeze clone did not finish", False)
    except Cancelled:
        check("breeze clone stops after the current piece", seen == ["第一句。"], seen)
    check("breeze clone published 1/2 before the next piece",
          any(t.get("done") == 1 and t.get("total") == 2 for t in ctx.ticks), ctx.ticks)


def t_firered_clone_cancel():
    from wrapper.caps import firered
    from wrapper.tasks import Cancelled

    seen = []

    class FakeModel:
        def _apply_frontend(self, text):
            parts, buf = [], ""
            for ch in text:
                buf += ch
                if ch in "。！？!?":
                    if buf.strip():
                        parts.append(buf)
                    buf = ""
            if buf.strip():
                parts.append(buf)
            return text, "zh", parts or [text]

        def generate_tts(self, **kw):
            seen.append(("tts", kw.get("text")))
            return np.ones(80, dtype="float32"), 24000

        def generate_voice_design(self, **kw):
            seen.append(("design", kw.get("text")))
            return np.ones(80, dtype="float32"), 24000, "plan"

    orig = firered._as_torch
    firered._as_torch = lambda audio: np.asarray(audio, dtype="float32")
    try:
        be = firered.FireRedBackend(FakeModel())
        be._join = lambda waves, sr, fade_ms=50.0, **kw: (
            np.concatenate([np.asarray(w, dtype="float32").reshape(-1) for w in waves]), sr)
        be._pace = lambda audio, sr, instruction="", speed=None: (audio, sr)
        ctx = TickCtx(cancel_after=3)
        try:
            be.clone("第一句。第二句。", np.zeros(80, dtype="float32"), 24000, "ref",
                     ctx=ctx, speed=1.0)
            check("canceled firered clone did not finish", False)
        except Cancelled:
            check("firered clone stops after the current sentence",
                  seen == [("tts", "第一句。")], seen)
        check("firered clone published 1/2 before the next sentence",
              any(t.get("done") == 1 and t.get("total") == 2 for t in ctx.ticks), ctx.ticks)
        seen.clear()
        ctx2 = TickCtx(cancel_after=3)
        try:
            be.design("沉稳男声", "第一句。第二句。", ctx=ctx2, speed=1.0)
            check("canceled firered design did not finish", False)
        except Cancelled:
            check("firered design stops after the current sentence",
                  seen == [("design", "第一句。")], seen)
        check("firered design published 1/2 before the next sentence",
              any(t.get("done") == 1 and t.get("total") == 2 and t.get("stage") == "design"
                  for t in ctx2.ticks), ctx2.ticks)
    finally:
        firered._as_torch = orig


def t_firered_triplet_pad():
    from wrapper.caps.firered import as_tts_triplet

    check("2-tuple becomes a triplet", as_tts_triplet(("a", 24000)) == ("a", 24000, None))
    check("triplet is left alone", as_tts_triplet(("a", 24000, "x")) == ("a", 24000, "x"))


def t_firered_design_instruction():
    from wrapper.caps.tts_el import _design_identity, _design_instruction

    check("plan beats the short blurb",
          _design_instruction({"instruction": "年轻女声", "plan": "口音：四川话"})
          == "口音：四川话")
    check("extra.plan is enough",
          _design_instruction({"instruction": "年轻女声", "extra": {"plan": "语速稍慢"}})
          == "语速稍慢")
    check("blurb is the fallback",
          _design_instruction({"instruction": "年轻女声"}) == "年轻女声")
    check("factory clone is not a design identity",
          not _design_identity({"source": "clone", "category": "premade"}))
    check("user clone is not a design identity",
          not _design_identity({"category": "cloned"}))
    check("generated card is a design identity",
          _design_identity({"category": "generated", "source": "design"}))
    check("legacy premade without source is a design identity",
          _design_identity({"category": "premade", "instruction": "年轻女声"}))


def t_output_format():
    from fastapi import HTTPException
    from wrapper.caps import tts_el

    d = tts_el._output_format(None)
    check("default token is EL mp3 44.1/128",
          d.kind == "mp3" and d.sr == 44100 and d.bitrate == 128, (d.kind, d.sr, d.bitrate))
    m = tts_el._output_format("mp3_44100_192")
    check("mp3 192 token", m.kind == "mp3" and m.sr == 44100 and m.bitrate == 192, m.token)
    w = tts_el._output_format("wav_48000")
    check("wav 48k token", w.kind == "wav" and w.sr == 48000 and w.bitrate is None, w.token)
    eight = tts_el._output_format("wav_8000")
    check("wav 8k token", eight.kind == "wav" and eight.sr == 8000, eight.token)
    bare = tts_el._output_format("mp3")
    check("bare mp3 is EL default", bare.sr == 44100 and bare.bitrate == 128, bare.token)
    pcm = tts_el._output_format("pcm_24000")
    check("pcm 24k token", pcm.kind == "pcm" and pcm.sr == 24000 and pcm.bitrate is None,
          (pcm.kind, pcm.sr, pcm.bitrate))
    try:
        tts_el._output_format("aac_44100")
        check("unknown codec is 400", False)
    except HTTPException as e:
        check("unknown codec is 400", e.status_code == 400, e.detail)


def t_voice_settings_map():
    from wrapper.caps import tts_el

    base = 2.0
    d = tts_el.resolve_settings(tts_el._DEFAULT_SETTINGS, base, "")
    check("default knobs keep chart CFG", abs(d.cfg - base) < 1e-6, d.cfg)
    check("default knobs pass no seed of ours", d.seed is None, d.seed)
    check("default knobs add no instruction", d.instruction == "", d.instruction)
    high = tts_el.resolve_settings({**tts_el._DEFAULT_SETTINGS, "similarity_boost": 1.0}, base, "")
    check("high similarity raises CFG", high.cfg > base, high.cfg)
    off = tts_el.resolve_settings({**tts_el._DEFAULT_SETTINGS, "use_speaker_boost": False}, base, "")
    check("speaker_boost off lowers CFG", off.cfg < base, off.cfg)
    steady = tts_el.resolve_settings({**tts_el._DEFAULT_SETTINGS}, base, "")
    check("a steady voice leaves the model the seed its authors chose",
          steady.seed is None, steady.seed)
    low = tts_el.resolve_settings({**tts_el._DEFAULT_SETTINGS, "stability": 0.1}, base, "")
    check("low stability varies the seed", low.seed is not None, low.seed)
    styled = tts_el.resolve_settings({**tts_el._DEFAULT_SETTINGS, "style": 0.8}, base, "")
    check("style wraps an English direction",
          "expressive" in styled.instruction.lower(), styled.instruction)
    check("style raises CFG", styled.cfg > base, styled.cfg)
    slow = tts_el.resolve_settings({**tts_el._DEFAULT_SETTINGS, "speed": 0.6}, base, "沉稳男声")
    check("slow speed wraps a Chinese direction", "慢" in slow.instruction, slow.instruction)


def t_firered_settings_apply():
    from wrapper.caps import firered, tts_el

    seen = []

    class FakeModel:
        def generate_tts(self, **kw):
            seen.append(kw)
            return np.ones(80, dtype="float32"), 24000

        def generate_voice_design(self, **kw):
            seen.append(kw)
            return np.ones(80, dtype="float32"), 24000, "plan"

    orig = firered._as_torch
    firered._as_torch = lambda audio: np.asarray(audio, dtype="float32")
    try:
        be = firered.FireRedBackend(FakeModel())
        be._pace = lambda audio, sr, speed=None: (audio, sr)
        be._join = lambda waves, sr, text="", **kw: (waves[0], sr)
        be.clone("你好", np.zeros(80, dtype="float32"), 24000, "ref",
                 settings={**tts_el._DEFAULT_SETTINGS, "similarity_boost": 1.0})
        check("firered clone uses mapped CFG",
              seen and seen[-1].get("inference_cfg") > tts_el.INFERENCE_CFG,
              seen[-1] if seen else None)
        seen.clear()
        be.design("沉稳男声", "你好",
                  settings={**tts_el._DEFAULT_SETTINGS, "style": 0.8, "speed": 0.6})
        check("firered design uses mapped CFG",
              seen and seen[-1].get("inference_cfg") > tts_el.DESIGN_CFG,
              seen[-1] if seen else None)
        check("firered design instruction carries style wrap",
              seen and "夸张" in str(seen[-1].get("instruction") or ""),
              seen[-1] if seen else None)
    finally:
        firered._as_torch = orig


def t_firered_speaker_boost():
    """Boost has no flag on this model: the tail of the slice before goes into the
    prompt, which is in-context, so the next slice keeps the voice it just heard."""
    from wrapper.caps import firered, tts_el

    def run(boost):
        seen = []

        class FakeModel:
            def _apply_frontend(self, text, **kw):
                parts = [p + "。" for p in (text or "").split("。") if p]
                return "".join(parts), "Chinese", parts

            def generate_tts(self, **kw):
                seen.append(kw)
                return np.ones(24000, dtype="float32"), 24000

        orig = firered._as_torch
        firered._as_torch = lambda audio: np.asarray(audio, dtype="float32")
        try:
            be = firered.FireRedBackend(FakeModel())
            be._pace = lambda audio, sr, speed=None: (audio, sr)
            be._join = lambda waves, sr, text="", **kw: (waves[0], sr)
            be.clone("第一句。第二句。", np.zeros(240, dtype="float32"), 24000, "参考文本",
                     settings={**tts_el._DEFAULT_SETTINGS, "use_speaker_boost": boost})
        finally:
            firered._as_torch = orig
        return seen

    on = run(True)
    check("boost speaks each sentence once", len(on) == 2, [k.get("text") for k in on])
    check("boost feeds the first sentence into the second prompt",
          "第一句。" in str(on[-1].get("prompt_text")), on[-1].get("prompt_text"))
    check("boost ends the prompt on the clean reference",
          str(on[-1].get("prompt_text")).endswith("参考文本"),
          on[-1].get("prompt_text"))
    check("overlap trim never wipes a whole slice",
          firered._strip_ctx_overlap("作为全球服务贸易领域极具影响力的国际性开放合作平台，服贸会持续以展为桥、",
                                     "作为全球服务贸易领域极具影响力的国际性开放合作平台，服贸会持续以展为桥、")
          == "作为全球服务贸易领域极具影响力的国际性开放合作平台，服贸会持续以展为桥、")
    check("a short clip does not carry the unsaid rest of the sentence",
          firered._spoken_text("作为全球服务贸易领域极具影响力的国际性开放合作平台，服贸会持续以展为桥、",
                               np.zeros(4800, dtype="float32"), 24000) !=
          "作为全球服务贸易领域极具影响力的国际性开放合作平台，服贸会持续以展为桥、")
    check("boost feeds its audio in too",
          len(on[-1]["prompt_audio"]) > len(on[0]["prompt_audio"]),
          (len(on[0]["prompt_audio"]), len(on[-1]["prompt_audio"])))
    off = run(False)
    check("without boost every slice hears only the reference",
          {str(k.get("prompt_text")) for k in off} == {"参考文本"},
          [k.get("prompt_text") for k in off])


def t_firered_split_paragraphs():
    from wrapper.caps import firered

    text = (
        "京华九月，秋启新程。今天，2026年中国国际服务贸易交易会如约启幕，"
        "以90个国家（地区）和国际组织设展办会、1830余家企业线下参展、"
        "200余项新产品新成果集中发布的扩容升级之势，为全球服务贸易开放合作注入新动能。\n\n"
        "作为全球服务贸易领域极具影响力的国际性开放合作平台，服贸会持续以展为桥、"
        "以会聚力，清晰勾勒出中国服务贸易从规模扩容向质效跃升、从深耕内功向赋能全球的进阶轨迹。"
    )
    be = firered.FireRedBackend(type("M", (), {"redae": None})())
    parts = be._sentences(text)
    check("a new paragraph is not glued to the sentence before it",
          any(p.startswith("作为全球") for p in parts), parts)
    check("the 作为 paragraph starts a new slice",
          any(p.startswith("作为全球") for p in parts), parts)
    check("今天， is not a slice of its own",
          not any(p.strip() == "今天，" for p in parts), parts)
    check("today and 启幕 stay in one sentence slice",
          any("今天，" in p and "启幕，" in p and "注入新动能" in p for p in parts), parts)
    check("a long sentence is not recut on commas",
          not any(p.strip() in ("今天，", "如约启幕，") for p in parts), parts)
    check("FireRed join is not the Breeze 520 ms window",
          firered._pause_ms("秋启新程。", text) == firered._SLICE_GAP_MS
          and firered._SLICE_GAP_MS < 200,
          firered._pause_ms("秋启新程。", text))

    class Official:
        redae = None

        def _apply_frontend(self, text, **kw):
            parts = []
            buf = ""
            for ch in text.replace("\n", ""):
                buf += ch
                if ch in "。！？" and buf.strip():
                    parts.append(buf.strip())
                    buf = ""
            if buf.strip():
                parts.append(buf.strip())
            return "".join(parts), "Chinese", parts

    glued = firered.FireRedBackend(Official())._sentences(text)
    check("official long sentence keeps 今天 with 启幕",
          any("今天，" in p and "启幕，" in p for p in glued), glued)
    check("official glue still leaves 作为 as its own slice",
          any(p.startswith("作为全球") for p in glued), glued)


def t_firered_unsaid_requeues():
    """generate_tts often stops at a comma. The leftover clause must get its own slice."""
    from wrapper.caps import firered, tts_el

    text = "但基础硬件出海，并不等于整套服务能力落地可用。"
    short = np.ones(24000, dtype="float32")
    rest = firered._unsaid(text, short, 24000)
    check("a short clip requeues the clause after the comma",
          rest.startswith("并不等于"), rest)
    long = np.ones(96000, dtype="float32")
    check("a finished sentence is not requeued",
          firered._unsaid(text, long, 24000) == "")

    seen = []

    class FakeModel:
        def generate_tts(self, **kw):
            seen.append(kw.get("text"))
            return np.ones(24000, dtype="float32"), 24000

    orig = firered._as_torch
    firered._as_torch = lambda audio: np.asarray(audio, dtype="float32")
    try:
        be = firered.FireRedBackend(FakeModel())
        be._pace = lambda audio, sr, speed=None: (audio, sr)
        be._join = lambda waves, sr, text="", **kw: (waves[0], sr)
        be.clone(text + "今年走了。", np.zeros(240, dtype="float32"), 24000, "参考文本",
                 settings={**tts_el._DEFAULT_SETTINGS, "use_speaker_boost": True})
    finally:
        firered._as_torch = orig
    check("clone asks for 并不等于 after a truncated generate",
          any(isinstance(t, str) and t.startswith("并不等于") for t in seen), seen)


def t_speed_is_applied_once():
    """Speed is a filter here, not a request to the model. Doing both speaks it twice:
    the model hurries, then the tempo filter hurries the hurried audio again."""
    import inspect

    from wrapper.caps import breeze, firered

    for name, src in (("breeze", inspect.getsource(breeze)),
                      ("firered", inspect.getsource(firered))):
        calls = src.count("resolve_settings(")
        waived = src.count("include_speed_direction=False")
        check("%s never asks the model to hurry as well" % name, calls == waived,
              "%d resolve_settings, %d waived" % (calls, waived))


def t_design_alternates_are_opt_in():
    """Offering a choice must not change what a caller who never asks for one gets."""
    import tempfile

    from fastapi.testclient import TestClient
    from wrapper.caps import tts_el

    base = tts_el.resolve_settings(None, tts_el.DESIGN_CFG, "")
    check("no alternate asked for is the seed callers already had",
          tts_el.resolve_settings(None, tts_el.DESIGN_CFG, "", seed_jitter=0).seed == base.seed,
          base.seed)
    later = tts_el.resolve_settings(None, tts_el.DESIGN_CFG, "", seed_jitter=1).seed
    check("an alternate is a different voice", later != base.seed, (base.seed, later))

    fake_soundfile()
    backend = FakeELBackend()
    root = tempfile.mkdtemp(prefix="el-takes-")
    tts_el.install(backend, store=tts_el.VoiceStore(root, backend.presets()))
    from wrapper.caps.breeze import build_app

    with TestClient(build_app(["tts", "tts_clone", "tts_design"])) as c:
        one = c.post("/v1/text-to-voice/design", json={"voice_description": "沉稳青年女声"})
        check("a request that says nothing still gets exactly one voice",
              one.status_code == 200 and len(one.json().get("previews") or []) == 1,
              one.text[:160])
        three = c.post("/v1/text-to-voice/design",
                       json={"voice_description": "沉稳青年女声", "takes": 3})
        got = three.json().get("previews") or []
        check("asking for three offers three", len(got) == 3, len(got))
        check("each one is savable on its own",
              len({p.get("generated_voice_id") for p in got}) == 3,
              [p.get("generated_voice_id") for p in got])
        bad = c.post("/v1/text-to-voice/design",
                     json={"voice_description": "沉稳青年女声", "takes": 9})
        check("more than the engine offers is refused, not silently trimmed",
              bad.status_code == 400, bad.status_code)


def t_firered_designs_speak_their_sample():
    """Voice design here is conditioned on the text it is handed, so designing again
    at read time hands back a different speaker — often a different sex. The approved
    sample is the only stable answer to who this voice is."""
    from wrapper.caps.firered import FireRedBackend

    check("a saved design speaks its frozen sample, not a fresh design",
          FireRedBackend.prefer_design_speak is False,
          FireRedBackend.prefer_design_speak)
    check("a description of the speaker is not a direction for the sentence",
          FireRedBackend.card_instruction_is_direction is False,
          FireRedBackend.card_instruction_is_direction)


def t_firered_design_anchors():
    """A design has no wav to hold it steady, so every slice re-imagines the speaker.
    Boost makes the first slice that missing reference for the ones after it."""
    from wrapper.caps import firered, tts_el

    def run(boost):
        seen = []

        class FakeModel:
            def _apply_frontend(self, text, **kw):
                parts = [p + "。" for p in (text or "").split("。") if p]
                return "".join(parts), "Chinese", parts

            def generate_voice_design(self, **kw):
                seen.append(("design", kw))
                return np.ones(240, dtype="float32"), 24000, "plan"

            def generate_tts(self, **kw):
                seen.append(("tts", kw))
                return np.ones(240, dtype="float32"), 24000

        orig = firered._as_torch
        firered._as_torch = lambda audio: np.asarray(audio, dtype="float32")
        try:
            be = firered.FireRedBackend(FakeModel())
            be._pace = lambda audio, sr, speed=None: (audio, sr)
            list(be.iter_design("沉稳男声", "第一句。第二句。第三句。",
                                settings={**tts_el._DEFAULT_SETTINGS,
                                          "use_speaker_boost": boost}))
        finally:
            firered._as_torch = orig
        return seen

    on = run(True)
    check("boost speaks each sentence once", len(on) == 3, [k for k, _ in on])
    check("the first slice still invents the voice", on[0][0] == "design", on[0][0])
    check("later slices speak against it instead of re-imagining it",
          [k for k, _ in on[1:]] == ["tts", "tts"], [k for k, _ in on])
    check("the anchor they hear is the first slice",
          "第一句。" in str(on[1][1].get("prompt_text")), on[1][1].get("prompt_text"))
    check("and it stays the anchor as the reading goes on",
          "第一句。" in str(on[-1][1].get("prompt_text")), on[-1][1].get("prompt_text"))
    off = run(False)
    check("without boost every slice invents the voice again",
          {k for k, _ in off} == {"design"}, [k for k, _ in off])


def t_firered_speed_for():
    """--speak-speed is the baseline this model needs; the EL knob multiplies it."""
    from wrapper.caps import tts_el
    from wrapper.caps.firered import clamp_speed, tempo_for

    check("clamp floors at 0.25", clamp_speed(0.01) == 0.25)
    check("clamp caps at 4.0", clamp_speed(9) == 4.0)
    check("a default request slows the machine-gun pace down",
          tempo_for(1.0) == tts_el.SPEAK_SPEED and tempo_for(1.0) < 1.0, tempo_for(1.0))
    check("asking for faster multiplies the baseline rather than replacing it",
          abs(tempo_for(2.0) - 2.0 * tts_el.SPEAK_SPEED) < 1e-9, tempo_for(2.0))
    check("asking for slower stays inside what atempo can chain",
          0.25 <= tempo_for(0.25) < tts_el.SPEAK_SPEED, tempo_for(0.25))


def t_firered_design_speak():
    import tempfile

    from fastapi.testclient import TestClient
    from wrapper.caps import tts_el

    class FakeFireRed(FakeELBackend):
        prefer_design_speak = True

    fake_soundfile()
    backend = FakeFireRed()
    root = tempfile.mkdtemp(prefix="el-voices-")
    store = tts_el.VoiceStore(root, backend.presets())
    tts_el.install(backend, store=store)
    from wrapper.caps.firered import build_app

    with TestClient(build_app(["tts", "tts_clone", "tts_design"])) as c:
        r = c.post("/v1/text-to-speech/p1", json={"text": "hello from design speak"})
        check("design-speak premade returns 200", r.status_code == 200, r.text[:120])
        clones = [call for call in backend.calls if call[0] == "clone"]
        designs = [call for call in backend.calls if call[0] == "design"]
        check("design-speak never falls back to clone", clones == [], backend.calls)
        check("freeze still designs a sample",
              any(call[2] != "hello from design speak" for call in designs), designs)
        check("speak redesigns with the user text",
              any(call[2] == "hello from design speak" for call in designs), designs)
        check("premade speak uses the frozen plan not the blurb",
              any(call[0] == "design" and call[1] == "ok"
                  and call[2] == "hello from design speak" for call in backend.calls),
              backend.calls)
        preview = c.post("/v1/text-to-voice/design",
                         json={"voice_description": "A raspy pirate", "text": "Ahoy"})
        check("design preview keeps a plan",
              preview.status_code == 200
              and (preview.json().get("previews") or [{}])[0].get("plan") == "ok",
              preview.text[:160])
        gid = (preview.json().get("previews") or [{}])[0].get("generated_voice_id")
        created = c.post("/v1/text-to-voice",
                         json={"generated_voice_id": gid, "voice_name": "Pirate"})
        vid = created.json().get("voice_id")
        backend.calls.clear()
        spoken = c.post("/v1/text-to-speech/%s" % vid, json={"text": "more pirate"})
        check("saved design speak returns 200", spoken.status_code == 200, spoken.text[:120])
        check("saved design speak uses the frozen plan",
              any(call[0] == "design" and call[1] == "ok" and call[2] == "more pirate"
                  for call in backend.calls), backend.calls)

    class FakeClonePack(FakeELBackend):
        prefer_design_speak = True

        def presets(self):
            return [{"voice_id": "zh-f", "name": "中文女", "category": "premade",
                     "source": "clone", "instruction": "语速稍慢",
                     "description": "nav"}]

    clone_be = FakeClonePack()
    clone_root = tempfile.mkdtemp(prefix="el-voices-clone-")
    os.makedirs(os.path.join(clone_root, "zh-f"), exist_ok=True)
    open(os.path.join(clone_root, "zh-f", "meta.json"), "w").write(
        '{"voice_id":"zh-f","name":"中文女","category":"premade","source":"clone",'
        '"instruction":"语速稍慢"}')
    open(os.path.join(clone_root, "zh-f", "prompt.wav"), "wb").write(wav_of(0.4))
    open(os.path.join(clone_root, "zh-f", "prompt.txt"), "w").write("准备出发")
    tts_el.install(clone_be, store=tts_el.VoiceStore(clone_root, clone_be.presets()))
    with TestClient(build_app(["tts", "tts_clone", "tts_design"])) as c:
        spoken = c.post("/v1/text-to-speech/zh-f", json={"text": "请直行"})
        check("factory clone speak returns 200", spoken.status_code == 200, spoken.text[:120])
        check("factory clone speak uses the wav, not design",
              any(call[0] == "clone" and call[1] == "请直行" for call in clone_be.calls)
              and not any(call[0] == "design" and call[2] == "请直行" for call in clone_be.calls),
              clone_be.calls)
        check("factory clone passes Voice Direction to clone",
              any(call[0] == "clone" and call[3] == "语速稍慢" for call in clone_be.calls),
              clone_be.calls)


def t_tts_el(module_name="firered"):
    import tempfile

    from fastapi.testclient import TestClient
    from wrapper.caps import tts_el

    fake_soundfile()
    backend = FakeELBackend()
    if module_name == "breeze":
        backend.card_instruction_is_direction = False
    root = tempfile.mkdtemp(prefix="el-voices-")
    store = tts_el.VoiceStore(root, backend.presets())
    tts_el.install(backend, store=store)
    build = __import__("wrapper.caps.%s" % module_name, fromlist=["build_app"]).build_app
    wav = wav_of(0.4)

    with TestClient(build(["tts", "tts_clone", "tts_design"])) as c:
        advertises_tasks(c, module_name, legacy=False)
        spec_paths = mounted(c)
        check("%s advertises ElevenLabs list/design/speak" % module_name,
              ("GET", "/v1/voices") in spec_paths
              and ("GET", "/v1/voices/settings/default") in spec_paths
              and ("POST", "/v1/voices/{voice_id}/settings/edit") in spec_paths
              and ("POST", "/v1/text-to-voice/design") in spec_paths
              and ("POST", "/v1/text-to-speech/{voice_id}") in spec_paths
              and ("POST", "/v1/text-to-speech/{voice_id}/stream") in spec_paths
              and ("POST", "/v1/voices/add") in spec_paths,
              sorted(p[1] for p in spec_paths if p[1].startswith("/v1/")))
        check("%s does not advertise OpenAI speech aliases" % module_name,
              ("GET", "/v1/audio/voices") not in spec_paths
              and ("POST", "/v1/audio/speech") not in spec_paths
              and ("POST", "/v1/audio/speech/clone") not in spec_paths,
              sorted(p[1] for p in spec_paths if p[1].startswith("/v1/")))
        listed = c.get("/v1/voices").json()["voices"]
        check("%s lists premade voices before they are frozen" % module_name,
              listed and listed[0]["voice_id"] == "p1" and listed[0]["category"] == "premade",
              listed)
        check("%s OpenAI voices alias is gone" % module_name,
              c.get("/v1/audio/voices").status_code == 404)
        r = c.post("/v1/text-to-speech/p1", json={"text": "hello from eleven"})
        check("%s speaks a premade voice_id" % module_name, r.status_code == 200,
              (r.status_code, r.text[:120]))
        check("%s first premade speak froze a design sample" % module_name,
              any(call[0] == "design" for call in backend.calls)
              and any(call[0] == "clone" for call in backend.calls),
              backend.calls)
        if module_name == "breeze":
            check("breeze premade speak does not send the card blurb as Voice Direction",
                  any(call[0] == "clone" and (len(call) < 4 or not call[3])
                      for call in backend.calls), backend.calls)
        r2 = c.post("/v1/audio/speech", json={"input": "hello again", "voice": "p1"})
        check("%s OpenAI speech alias is gone" % module_name, r2.status_code == 404,
              r2.status_code)
        preview = c.post("/v1/text-to-voice/design",
                         json={"voice_description": "A raspy pirate", "text": "Ahoy there"})
        check("%s design returns a generated_voice_id" % module_name, preview.status_code == 200,
              (preview.status_code, preview.text[:160]))
        gid = (preview.json().get("previews") or [{}])[0].get("generated_voice_id")
        created = c.post("/v1/text-to-voice",
                         json={"generated_voice_id": gid, "voice_name": "Pirate"})
        check("%s persist design yields a voice_id" % module_name,
              created.status_code == 200 and created.json().get("voice_id"),
              created.text[:160])
        add = c.post("/v1/voices/add",
                     data={"name": "Me", "description": "this is the transcript"},
                     files={"file": ("r.wav", wav, "audio/wav")})
        check("%s clone add returns voice_id" % module_name,
              add.status_code == 200 and add.json().get("voice_id"),
              (add.status_code, add.text[:160]))
        vid = add.json().get("voice_id")
        check("%s advertises delete and edit" % module_name,
              ("DELETE", "/v1/voices/{voice_id}") in spec_paths
              and ("POST", "/v1/voices/{voice_id}/edit") in spec_paths,
              sorted(p[1] for p in spec_paths if "voices" in p[1]))
        edited = c.post("/v1/voices/%s/edit" % vid, data={"name": "Renamed"})
        check("%s edit returns ok" % module_name,
              edited.status_code == 200 and edited.json().get("status") == "ok",
              (edited.status_code, edited.text[:160]))
        check("%s edit updates GET" % module_name,
              c.get("/v1/voices/%s" % vid).json().get("name") == "Renamed")
        check("%s edit unknown is 404" % module_name,
              c.post("/v1/voices/nobody/edit", data={"name": "X"}).status_code == 404)
        check("%s edit premade is 400" % module_name,
              c.post("/v1/voices/p1/edit", data={"name": "Nope"}).status_code == 400)
        deleted = c.delete("/v1/voices/%s" % vid)
        check("%s delete clone returns ok" % module_name,
              deleted.status_code == 200 and deleted.json().get("status") == "ok",
              (deleted.status_code, deleted.text[:160]))
        check("%s deleted voice is gone" % module_name,
              c.get("/v1/voices/%s" % vid).status_code == 404)
        check("%s delete premade is 400" % module_name,
              c.delete("/v1/voices/p1").status_code == 400)
        check("%s delete unknown is 404" % module_name,
              c.delete("/v1/voices/nobody").status_code == 404)
        check("%s clone without a transcript is a 400" % module_name,
              c.post("/v1/voices/add", data={"name": "Nope"},
                     files={"file": ("r.wav", wav, "audio/wav")}).status_code == 400)
        check("%s empty text is a 400" % module_name,
              c.post("/v1/text-to-speech/p1", json={"text": "  "}).status_code == 400)
        check("%s unknown voice_id is a 404" % module_name,
              c.post("/v1/text-to-speech/nobody", json={"text": "hi"}).status_code == 404)
        check("%s stream rejects async=1" % module_name,
              c.post("/v1/text-to-speech/p1/stream?async=1",
                     json={"text": "later"}).status_code == 400)
        streamed = c.post("/v1/text-to-speech/p1/stream",
                          json={"text": "hello stream", "output_format": "mp3"})
        check("%s stream returns audio as it is produced" % module_name,
              streamed.status_code == 200
              and "mpeg" in (streamed.headers.get("content-type") or "")
              and len(streamed.content) > 0,
              (streamed.status_code, streamed.headers.get("content-type"),
               len(streamed.content)))
        stream_hid = streamed.headers.get("history-item-id")
        stream_history = c.get("/v1/history/%s" % stream_hid).json() if stream_hid else {}
        check("%s completed stream publishes output duration" % module_name,
              stream_history.get("state") == "created"
              and float(stream_history.get("output_duration_seconds") or 0) > 0,
              stream_history)
        default_s = c.get("/v1/voices/settings/default")
        check("%s default settings look like ElevenLabs" % module_name,
              default_s.status_code == 200
              and default_s.json().get("stability") == 0.5
              and default_s.json().get("similarity_boost") == 0.75
              and "speed" in default_s.json(),
              default_s.json())
        got_s = c.get("/v1/voices/p1/settings")
        check("%s premade settings GET works" % module_name,
              got_s.status_code == 200 and got_s.json().get("speed") == 1.0,
              got_s.json())
        edited_s = c.post("/v1/voices/p1/settings/edit",
                          json={"speed": 0.8, "stability": 0.2, "unknown_knob": 9})
        check("%s premade settings edit is allowed" % module_name,
              edited_s.status_code == 200
              and edited_s.json().get("speed") == 0.8
              and edited_s.json().get("stability") == 0.2
              and "unknown_knob" not in edited_s.json(),
              edited_s.json())
        check("%s settings persist on GET" % module_name,
              c.get("/v1/voices/p1/settings").json().get("speed") == 0.8)
        backend.calls.clear()
        knobs_r = c.post("/v1/text-to-speech/p1",
                         json={"text": "knobs please",
                               "voice_settings": {"style": 0.9, "similarity_boost": 1.0}})
        check("%s speak accepts voice_settings" % module_name, knobs_r.status_code == 200,
              knobs_r.text[:120])
        check("%s speak forwards voice_settings to the backend" % module_name,
              any((call[0] == "clone" and (call[4] or {}).get("style") == 0.9)
                  or (call[0] == "design" and len(call) > 3
                      and (call[3] or {}).get("style") == 0.9)
                  for call in backend.calls),
              backend.calls)
        async_r = c.post("/v1/text-to-speech/p1?async=1", json={"text": "later"})
        check("%s async 202" % module_name, async_r.status_code == 202, async_r.text[:120])
        if async_r.status_code == 202:
            doc = poll(c, async_r.json()["task"]["id"])
            check("%s async task succeeded" % module_name, doc["status"] == "succeeded",
                  doc.get("status"))
            contract(c, doc, module_name, legacy=False)
        design_async = c.post("/v1/text-to-voice/design?async=1",
                              json={"voice_description": "A raspy pirate", "text": "Ahoy again"})
        check("%s design async 202" % module_name, design_async.status_code == 202,
              design_async.text[:160])
        if design_async.status_code == 202:
            doc = poll(c, design_async.json()["task"]["id"])
            check("%s design async succeeded" % module_name, doc["status"] == "succeeded",
                  doc.get("status"))
            preview_async = c.get("%s/%s/result" % (TASKS, doc["id"])).json()
            gid_async = (preview_async.get("previews") or [{}])[0].get("generated_voice_id")
            check("%s design async result has preview" % module_name, bool(gid_async),
                  preview_async)
            persist_async = c.post("/v1/text-to-voice?async=1",
                                   json={"generated_voice_id": gid_async,
                                         "voice_name": "PirateAsync"})
            check("%s persist async 202" % module_name, persist_async.status_code == 202,
                  persist_async.text[:160])
            if persist_async.status_code == 202:
                doc = poll(c, persist_async.json()["task"]["id"])
                check("%s persist async succeeded" % module_name, doc["status"] == "succeeded",
                      doc.get("status"))
        add_async = c.post("/v1/voices/add?async=1",
                           data={"name": "MeAsync", "description": "this is the transcript"},
                           files={"file": ("r.wav", wav, "audio/wav")})
        check("%s add async 202" % module_name, add_async.status_code == 202,
              (add_async.status_code, add_async.text[:160]))
        if add_async.status_code == 202:
            doc = poll(c, add_async.json()["task"]["id"])
            check("%s add async succeeded" % module_name, doc["status"] == "succeeded",
                  doc.get("status"))
            added = c.get("%s/%s/result" % (TASKS, doc["id"])).json()
            check("%s add async result has voice_id" % module_name, bool(added.get("voice_id")),
                  added)

    tts_el._state.update(ready=False, error="weights are missing", backend=None, store=None)
    with TestClient(build(["tts", "tts_clone", "tts_design"])) as c:
        r = c.get("/v1/voices")
        check("%s list is 503 while loading" % module_name, r.status_code == 503
              and "weights are missing" in r.text, r.text[:100])


def t_tts_el_edge_safe():
    """A body over the public-hop budget is stored as mp3; a small wav is left alone."""
    from wrapper.caps import tts_el

    old_lim = tts_el._EDGE_SAFE_BYTES
    old_enc = tts_el._encode
    old_fit = tts_el._encode_mp3_fit
    try:
        tts_el._encode = lambda audio, sr, fmt: b"RIFF" + b"W" * 1000
        tts_el._encode_mp3_fit = lambda audio, sr: b"ID3mp3"
        tts_el._EDGE_SAFE_BYTES = 200
        silent = np.zeros(8, dtype="float32")
        body, fmt = tts_el._encode_edge(silent, 24000, "wav_44100")
        check("oversize wav becomes mp3", fmt == "mp3" and body == b"ID3mp3", (fmt, body[:8]))
        tts_el._EDGE_SAFE_BYTES = 8000
        body, fmt = tts_el._encode_edge(silent, 24000, "wav_44100")
        check("under-limit wav is unchanged", fmt == "wav" and body.startswith(b"RIFF"),
              (fmt, len(body)))
    finally:
        tts_el._EDGE_SAFE_BYTES = old_lim
        tts_el._encode = old_enc
        tts_el._encode_mp3_fit = old_fit


def t_tts_el_live_mux_duration():
    """Logical duration is identical across containers and includes every write (including gaps)."""
    from wrapper.caps import tts_el

    class Sink:
        def write(self, _data):
            return None

        def flush(self):
            return None

    class Proc:
        stdin = Sink()

    durations = {}
    old_encode = tts_el._encode_out
    try:
        tts_el._encode_out = lambda *_args: b"encoded"
        for kind in ("pcm", "wav", "mp3", "flac", "opus"):
            mux = tts_el._LiveMux(tts_el.OutSpec(kind, 24000, 64 if kind in ("mp3", "opus") else None), 24000)
            if kind == "wav":
                mux._oneshot = True
            elif kind not in ("pcm", "wav"):
                mux.proc = Proc()
            mux.write(np.zeros(2400, dtype="float32"), 24000)
            mux.write(np.zeros(4800, dtype="float32"), 24000)
            durations[kind] = mux.output_duration_seconds
    finally:
        tts_el._encode_out = old_encode
    check("live mux duration is container-independent",
          all(abs(value - 0.3) < 1e-9 for value in durations.values()), durations)


def t_tts_el_history_duration():
    """Only a normally completed history item publishes a final duration."""
    import tempfile
    from wrapper.caps import tts_el

    store = tts_el.HistoryStore(tempfile.mkdtemp(prefix="el-history-duration-"))
    args = dict(voice_id="p1", voice_name="Voice", voice_category="premade",
                text="hello", settings={}, content_type="audio/mpeg", output_format="mp3")
    created = store.begin(**args)
    doc = store.finish(created["history_item_id"], "created", 1.25)
    check("created history carries duration", doc.get("output_duration_seconds") == 1.25, doc)
    canceled = store.begin(**args)
    doc = store.finish(canceled["history_item_id"], "canceled", 9.0)
    check("canceled history omits duration", "output_duration_seconds" not in doc, doc)


def t_tts_el_limits():
    """Product gates live in ENGINE_ARGS. 0 stays off for a bare process."""
    import importlib
    import tempfile

    from fastapi.testclient import TestClient
    from wrapper.caps import tts_el

    was = os.environ.get("ENGINE_ARGS", "")
    try:
        os.environ["ENGINE_ARGS"] = (
            "--cfg-scale 1 --design-cfg 4 --ref-min-seconds 3 --ref-max-seconds 30 "
            "--ref-max-mb 5 --design-min-chars 3 --design-max-chars 500 "
            "--max-new-tokens 1500 --max-seq-len 2048"
        )
        importlib.reload(tts_el)
        check("tts_el reads Breeze product gates from ENGINE_ARGS",
              tts_el.REF_MIN_SECONDS == 3 and tts_el.REF_MAX_SECONDS == 30
              and tts_el.REF_MAX_MB == 5
              and tts_el.DESIGN_MIN_CHARS == 3 and tts_el.DESIGN_MAX_CHARS == 500
              and tts_el.MAX_NEW_TOKENS == 1500 and tts_el.MAX_SEQ_LEN == 2048,
              (tts_el.REF_MIN_SECONDS, tts_el.REF_MAX_SECONDS, tts_el.MAX_SEQ_LEN))
        fake_soundfile()
        backend = FakeELBackend()
        root = tempfile.mkdtemp(prefix="el-limits-")
        tts_el.install(backend, store=tts_el.VoiceStore(root, backend.presets()))
        with TestClient(tts_el.build_app(["tts", "tts_clone", "tts_design"],
                                         module="breeze")) as c:
            def add(wav):
                return c.post(
                    "/v1/voices/add",
                    data={"name": "Me", "description": "this is the transcript"},
                    files={"file": ("r.wav", wav, "audio/wav")})
            short = add(wav_of(1))
            check("clone shorter than --ref-min-seconds is 400",
                  short.status_code == 400 and "ref-min-seconds" in short.text,
                  (short.status_code, short.text[:160]))
            mid = add(wav_of(10))
            check("clone inside 3–30s is 200",
                  mid.status_code == 200 and mid.json().get("voice_id"),
                  (mid.status_code, mid.text[:160]))
            long = add(wav_of(35))
            check("clone longer than --ref-max-seconds is 400",
                  long.status_code == 400 and "ref-max-seconds" in long.text,
                  (long.status_code, long.text[:160]))
            fat = add(b"RIFF" + b"\x00" * (6 * 1024 * 1024))
            check("clone bigger than --ref-max-mb is 400",
                  fat.status_code == 400 and "ref-max-mb" in fat.text,
                  (fat.status_code, fat.text[:160]))
            tiny = c.post("/v1/text-to-voice/design",
                          json={"voice_description": "ab", "text": "Ahoy"})
            check("design shorter than --design-min-chars is 400",
                  tiny.status_code == 400 and "design-min-chars" in tiny.text,
                  (tiny.status_code, tiny.text[:160]))
            huge = c.post("/v1/text-to-voice/design",
                          json={"voice_description": "x" * 501, "text": "Ahoy"})
            check("design longer than --design-max-chars is 400",
                  huge.status_code == 400 and "design-max-chars" in huge.text,
                  (huge.status_code, huge.text[:160]))
            ok = c.post("/v1/text-to-voice/design",
                        json={"voice_description": "A raspy pirate", "text": "Ahoy"})
            check("design inside 3–500 characters is 200",
                  ok.status_code == 200, (ok.status_code, ok.text[:160]))
    finally:
        if was:
            os.environ["ENGINE_ARGS"] = was
        else:
            os.environ.pop("ENGINE_ARGS", None)
        importlib.reload(tts_el)


def t_engine_args():
    # Every knob is a flag in ENGINE_ARGS now, so this parser is the one gate they all pass through.
    from wrapper.contract import EngineArgs

    a = EngineArgs("--beam-size 8 --compute-type=int8 --enforce-eager --stage-init-timeout 1200")
    check("value after a space", a.count("--beam-size", 5) == 8)
    check("value after an equals", a.text("--compute-type", "float16") == "int8")
    check("bare flag reads as on", a.switch("--enforce-eager") is True)
    check("absent flag falls back", a.count("--max-model-len", 8192) == 8192)
    check("unclaimed flags pass through", a.passthrough() == ["--stage-init-timeout", "1200"])

    b = EngineArgs("--gpu-memory-utilization -0.5 --amp")
    check("a negative value is not read as a flag", b.number("--gpu-memory-utilization", 0.3) == -0.5)
    check("a trailing bare flag survives", b.switch("--amp") is True)
    check("nothing set means nothing claimed", EngineArgs("").passthrough() == [])
    check("garbage does not raise", EngineArgs("--beam-size abc").count("--beam-size", 5) == 5)


def t_every_engine_args_user_reports_leftovers():
        # A parser that works proves nothing until a cap calls warn_unclaimed; tts.py once did not.
    caps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wrapper", "caps")
    for name in sorted(os.listdir(caps)):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        with open(os.path.join(caps, name), encoding="utf-8") as fh:
            src = fh.read()
        if "EngineArgs(" not in src:
            continue
        check(f"{name} warns about unclaimed engine args",
              "warn_unclaimed" in src or ".passthrough()" in src)


def main():
    print("\n[engine args]")
    t_engine_args()
    t_every_engine_args_user_reports_leftovers()
    # AUDIO_BASE is baked into each image, so the spec of a base is only right under its own base.
    for name, fn, base in (("vad", t_vad, "pyannote"), ("diar", t_diar, "pyannote"),
                           ("embed", t_embed, "pyannote"), ("enhance", t_enhance, "pyannote"),
                           ("align", t_align, "qwen"),
                           ("whisper (fasterwhisper)", t_whisper, "fasterwhisper"),
                           ("whisper_ov", t_whisper_ov, "whisperov"),
                           ("qwen", t_qwen, "qwen"),
                           ("ov", t_ov, "ov"),
                           ("tts", t_tts, "qwen3tts"),
                           ("tts clone", t_tts_clone, "qwen3tts"),
                           ("tts design", t_tts_design, "qwen3tts"),
                           ("tts warmup", t_tts_warmup, "qwen3tts"),
                           ("tts not ready", t_tts_not_ready, "qwen3tts"),
                           ("crispasr tts", t_crispasr, "crispasr"),
                           ("crispasr not ready", t_crispasr_not_ready, "crispasr"),
                           ("sound_fx", t_sound_fx, "dasheng"),
                           ("sound_fx not ready", t_sound_fx_not_ready, "dasheng"),
                           ("sound_fx engine args", t_sound_fx_engine_args, "dasheng"),
                           ("tts_dialogue", t_tts_dialogue, "soulx"),
                           ("tts_dialogue not ready", t_tts_dialogue_not_ready, "soulx"),
                           ("firered elevenlabs", lambda: t_tts_el("firered"), "firered"),
                           ("breeze elevenlabs", lambda: t_tts_el("breeze"), "breeze"),
                           ("breeze split speak", t_breeze_split_speak, "breeze"),
                           ("breeze clone slices", t_breeze_clone_slices, "breeze"),
                           ("breeze stream chunks", t_breeze_stream_chunks, "breeze"),
                           ("breeze pace", t_breeze_pace, "breeze"),
                           ("tts job tick", t_tts_job_tick, "firered"),
                           ("stream ctx cancel", t_stream_ctx_cancel, "breeze"),
                           ("live stream stops iter", t_live_stream_stops_iter, "breeze"),
                           ("live stream named id", t_live_stream_named_id, "breeze"),
                           ("live grace widens for followers", t_live_grace_widens_for_followers, "breeze"),
                           ("breeze clone cancel", t_breeze_clone_cancel, "breeze"),
                           ("firered clone cancel", t_firered_clone_cancel, "firered"),
                           ("tts_el limits", t_tts_el_limits, "breeze"),
                           ("tts_el edge-safe", t_tts_el_edge_safe, "breeze"),
                           ("tts_el live mux duration", t_tts_el_live_mux_duration, "breeze"),
                           ("tts_el history duration", t_tts_el_history_duration, "breeze"),
                           ("voice cards", t_voice_cards, "firered"),
                           ("voice cards clone seed", t_voice_cards_clone_seed, "breeze"),
                           ("firered triplet pad", t_firered_triplet_pad, "firered"),
                           ("firered speed_for", t_firered_speed_for, "firered"),
                           ("firered speaker boost", t_firered_speaker_boost, "firered"),
                           ("firered split paragraphs", t_firered_split_paragraphs, "firered"),
                           ("firered unsaid requeues", t_firered_unsaid_requeues, "firered"),
                           ("speed is applied once", t_speed_is_applied_once, "breeze"),
                           ("design alternates are opt-in", t_design_alternates_are_opt_in, "breeze"),
                           ("firered designs speak their sample", t_firered_designs_speak_their_sample, "firered"),
                           ("firered design anchors", t_firered_design_anchors, "firered"),
                           ("output format tokens", t_output_format, "firered"),
                           ("voice settings map", t_voice_settings_map, "firered"),
                           ("firered settings apply", t_firered_settings_apply, "firered"),
                           ("firered design instruction", t_firered_design_instruction, "firered"),
                           ("firered design speak", t_firered_design_speak, "firered"),
                           ("diar_speakrs", t_diar_speakrs, "speakrs"),
                           ("diar_speakrs models dir", t_diar_speakrs_models_dir, "speakrs"),
                           ("diar_stream offline", t_diar_stream_offline, "nemo")):
        print("\n[%s]" % name)
        os.environ["AUDIO_BASE"] = base
        try:
            fn()
        except Exception as e:
            import traceback

            traceback.print_exc()
            check("%s raised" % name, False, e)
    print("\n" + ("FAILURES: %s" % FAILED if FAILED else "all capability wiring checks passed"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
