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


def contract(c, doc, label):
    """What every engine behind llm-init must answer, not just this one."""
    check("%s doc is object=task kind=audio" % label,
          doc.get("object") == "task" and doc.get("kind") == "audio",
          (doc.get("object"), doc.get("kind")))
    check("%s poll/result_url point at the contract path" % label,
          doc.get("poll") == "%s/%s" % (TASKS, doc["id"])
          and doc.get("result_url") == "%s/%s/result" % (TASKS, doc["id"]),
          (doc.get("poll"), doc.get("result_url")))
    alias = c.get("%s/%s" % (LEGACY, doc["id"])).json()
    check("%s the legacy alias answers the same task" % label, alias == doc, alias.get("id"))
    cap = (c.get(TASKS).json() or {}).get("capacity") or {}
    check("%s the list reports queue capacity" % label,
          set(cap) == {"queued", "running", "limit", "accepting"} and cap["limit"] > 0, cap)
    check("%s the legacy list alias agrees" % label,
          c.get(LEGACY).json() == c.get(TASKS).json())
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


def advertises_tasks(c, label):
    paths = mounted(c)
    check("%s advertises the task API" % label,
          ("GET", TASKS + "/{id}") in paths and ("GET", LEGACY + "/{id}") in paths,
          sorted(p[1] for p in paths))
    deprecated = {e["path"] for e in c.get("/api/engine-spec").json()["endpoints"]
                  if e.get("deprecated")}
    check("%s marks the legacy paths deprecated" % label,
          deprecated == {LEGACY, LEGACY + "/{id}", LEGACY + "/{id}/result"},
          sorted(deprecated))


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
        # longer be named; and a NaN would leave the response unparseable to a strict client.
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
        # when we have been told is how the two answers drift apart.
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
        # signalled on, so picking the older would load weights nobody asked for.
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

        # The whole reason this engine cannot stand in for pyannote everywhere. Accepting the
        # constraint and quietly ignoring it would leave a caller unable to tell that it was
        # dropped, so the only honest answer is a 400 naming the parameters it refused.
        for field in ("num_speakers", "min_speakers", "max_speakers"):
            r = c.post("/v1/audio/diarization", files=WAV, data={field: "3"})
            check("diar_speakrs refuses %s rather than ignoring it" % field,
                  r.status_code == 400 and field in r.json()["detail"],
                  (r.status_code, r.json()))

        # Seconds on the wire, frames to the engine: every other engine in this repo states
        # durations in seconds, and speakrs states them in frames.
        sent.clear()
        c.post("/v1/audio/diarization", files=WAV, data={"min_duration_off": "0.5"})
        check("diar_speakrs converts seconds to frames for the engine",
              sent[-1]["min_duration_off_frames"] == 30, sent[-1]["min_duration_off_frames"])

        # 🔴 Unset and zero must stay different: speakrs' fast modes default this filter to 3
        # frames, so sending 0 for "the caller said nothing" would silently switch it off.
        sent.clear()
        c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs sends null, not 0, for a knob nobody set",
              sent[-1]["min_duration_off_frames"] is None, sent[-1]["min_duration_off_frames"])
        sent.clear()
        c.post("/v1/audio/diarization", files=WAV, data={"min_duration_off": "0"})
        check("diar_speakrs sends 0 when a caller really asked for 0",
              sent[-1]["min_duration_off_frames"] == 0, sent[-1]["min_duration_off_frames"])

        # The engine holds the whole clip decoded, so an unbounded upload is an OOM kill, which
        # reaches the caller as a dropped connection rather than as something it can act on.
        ds.MAX_AUDIO_SECONDS = 1.0
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs refuses audio longer than it can hold",
              r.status_code == 413 and "memory" in r.json()["detail"],
              (r.status_code, r.json()))
        r = c.post("/v1/audio/diarization", files=WAV, data={"async": "1"})
        check("diar_speakrs refuses it before a task exists, not inside one",
              r.status_code == 413, (r.status_code, r.text[:120]))
        ds.MAX_AUDIO_SECONDS = 14400.0

        # A container the header readers cannot parse may still be one the engine decodes, and it
        # is exactly the file that would slip past the length check. ffprobe shares ffmpeg's
        # demuxers, so it measures what the engine can read.
        real_probe, real_sub = ds.probe_seconds, ds.subprocess
        ds.probe_seconds = lambda _p: None
        ds.subprocess = types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="7200.0\n", stderr=""),
            SubprocessError=real_sub.SubprocessError)
        ds.MAX_AUDIO_SECONDS = 60.0
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs falls back to ffprobe when the header readers cannot",
              r.status_code == 413 and "7200s" in r.json()["detail"],
              (r.status_code, r.json()))

        # ffprobe absent or unable: unknown stays a real answer rather than a refusal, because a
        # file nothing can measure is one the engine will fail on cheaply, before spending memory.
        ds.subprocess = types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="bad"),
            SubprocessError=real_sub.SubprocessError)
        r = c.post("/v1/audio/diarization", files=WAV)
        check("diar_speakrs lets an unmeasurable clip through rather than refusing it",
              r.status_code == 200, (r.status_code, r.text[:120]))
        ds.probe_seconds, ds.subprocess = real_probe, real_sub
        ds.MAX_AUDIO_SECONDS = 14400.0

        sent.clear()
        doc = c.post("/v1/audio/diarization", files=WAV, data={"exclusive": "1"}).json()
        check("diar_speakrs passes exclusive through and echoes what ran",
              sent[-1]["exclusive"] is True and doc["exclusive"] is True,
              (sent[-1]["exclusive"], doc["exclusive"]))

        # A dead child must read as "not ready", never as an empty result: zero segments is a
        # legitimate answer for silent audio, so it can never double as an error.
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
        # The upload decodes to four seconds and the segments ask for two of
        # them. Billing the file here would charge for audio nothing aligned,
        # and a real caller sends an hour with a handful of seconds in it.
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
    # A real WAV of the length ffmpeg would actually have produced: the
    # recording is two seconds long, so a segment asking for more gets what
    # exists, and billing has to follow the slice rather than the request.
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


def t_qwen():
    from fastapi.testclient import TestClient
    from wrapper.caps import stt_stream as q

    q._decode_to_16k_mono = lambda raw, fn: np.zeros(SR * 4, dtype="float32")
    q._state.update(ready=True, asr=types.SimpleNamespace(
        transcribe=lambda audio=None, language=None, return_time_stamps=None:
            [types.SimpleNamespace(text="ni hao")],
        sampling_params=types.SimpleNamespace(max_tokens=32)))
    with TestClient(q.build_app(["stt", "stt_stream"])) as c:
        advertises_tasks(c, "qwen stt")
        check("the WS stream endpoint is still advertised",
              ("WS", "/v1/audio/stream") in mounted(c))
        spec = c.get("/api/engine-spec").json()
        check("align shows up as implemented but not served",
              [e for e in spec["endpoints"]
               if e.get("capability") == "align" and not e["available"]])
        both_ways(c, "/v1/audio/transcriptions", WAV, {}, "qwen stt", meters=("input",))
        both_ways(c, "/v1/audio/transcriptions", WAV,
                  {"segments": '[{"start":0,"end":1},{"start":1,"end":2}]'}, "qwen batch",
                  meters=("input",))


class FakeTTS:
    """Stands in for FasterQwen3TTS. The model object is the seam; everything above it is wiring."""

    sample_rate = 24000

    def __init__(self, custom=True):
        self.calls = []
        self.refuse_over = None
        self.model = types.SimpleNamespace(model=types.SimpleNamespace(
            tts_model_type="custom_voice" if custom else "base"))

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


def install_tts(custom):
    from wrapper.caps import tts

    fake_soundfile()
    m = FakeTTS(custom=custom)
    tts._state.update(ready=True, error=None, model=m, custom_voice=custom,
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
                      speakers=[])
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
                           ("qwen", t_qwen, "qwen"),
                           ("tts", t_tts, "qwen3tts"),
                           ("tts clone", t_tts_clone, "qwen3tts"),
                           ("tts warmup", t_tts_warmup, "qwen3tts"),
                           ("tts not ready", t_tts_not_ready, "qwen3tts"),
                           ("crispasr tts", t_crispasr, "crispasr"),
                           ("crispasr not ready", t_crispasr_not_ready, "crispasr"),
                           ("sound_fx", t_sound_fx, "dasheng"),
                           ("sound_fx not ready", t_sound_fx_not_ready, "dasheng"),
                           ("sound_fx engine args", t_sound_fx_engine_args, "dasheng"),
                           ("tts_dialogue", t_tts_dialogue, "soulx"),
                           ("tts_dialogue not ready", t_tts_dialogue_not_ready, "soulx"),
                           ("diar_speakrs", t_diar_speakrs, "speakrs"),
                           ("diar_speakrs models dir", t_diar_speakrs_models_dir, "speakrs")):
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
