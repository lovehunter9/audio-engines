"""Wire-level check of every capability: the async result must equal the sync one, byte for byte.

No GPU and no model weights: each engine is replaced by a stub, so what is under test is the
wiring (form fields, dispatch, progress, result shape), which is exactly where a refactor slips.

    python tests/caps_smoke.py        # needs fastapi, httpx, numpy, python-multipart
"""
import base64
import copy
import io
import logging
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


class said:
    """Capture what a logger says while the block runs, so a fix whose whole content is a
    warning can be tested at all.

    🔴 This did not exist, and everything it guards was written without it. Four changes in
    one review round had "it now says so once" as their entire content -- an unreadable GPU
    limit, a checkpoint whose dimensions cannot be read, a tokenizer that refused a span,
    an ordering forced on by the padding rule -- and none was testable, because the harness
    never looked at a log record. One of the four was then reverted by a mutation and all
    886 checks still passed.
    """

    def __init__(self, name="audio-align"):
        self.log = logging.getLogger(name)
        self.records = []

    def __enter__(self):
        class _Sink(logging.Handler):
            def emit(_, record):
                self.records.append(record.getMessage())
        self.handler = _Sink()
        self.kept = self.log.disabled
        self.log.disabled = False
        self.log.addHandler(self.handler)
        return self

    def __exit__(self, *exc):
        self.log.removeHandler(self.handler)
        self.log.disabled = self.kept
        return False

    def mentions(self, *needles):
        return any(all(n in m for n in needles) for m in self.records)


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



def kappa(align, pair):
    """Pin one of the two fitted factor pairs for a check that is about that build.

    🔴 The shipped default is the pair fitted WITH a KV cache, because --no-kv-cache is
    opt-in. A check about the cache-off calibration has to say so, or it reads the shipping
    constants and asserts things that are true of the other build.
    """
    keep = (align.KAPPA_LM, align.KAPPA_ENCODER, align.ENCODER_FIXED_BYTES,
            align._encoder_rate)
    align.KAPPA_LM, align.KAPPA_ENCODER, align.ENCODER_FIXED_BYTES = pair
    align._encoder_rate = None
    return keep


def kappa_restore(align, keep):
    (align.KAPPA_LM, align.KAPPA_ENCODER, align.ENCODER_FIXED_BYTES,
     align._encoder_rate) = keep


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


def t_diar_speakrs_openvino_models_dir():
    """Preparing the batched segmentation model, and every way it is allowed to give up.

    The giving-up cases matter more than the happy one: this runs at startup, and anything
    that stops the engine from launching costs more than the batching it was buying.
    """
    import shutil
    import tempfile
    from wrapper.caps import diar_speakrs as ds
    from wrapper.contract import EngineArgs

    root = tempfile.mkdtemp(prefix="ovmodels-")
    farm = ds._farm_for(root)
    shutil.rmtree(farm, ignore_errors=True)
    original = ds.EXECUTION_MODE
    batching = ds.OPENVINO_BATCHING
    try:
        # Off for every other backend: they compile the stock export and batch with it.
        open(os.path.join(root, "segmentation-3.0-b32.onnx"), "wb").close()
        ds.EXECUTION_MODE = "cuda"
        check("non-openvino modes are handed the cache directory untouched",
              ds._openvino_models_dir(root) == root, ds._openvino_models_dir(root))

        ds.EXECUTION_MODE = "openvino"
        # 🔴 A directory that is not there. The wrapper script starts this process offline when
        # llm-init never names the model, and _models_dir then returns a path that does not
        # exist so the engine can report the missing weights by name. Listing it unguarded
        # raised at import, on the Intel image only, and the engine never got to say anything.
        absent = os.path.join(root, "not-downloaded-yet")
        check("a models directory that does not exist is handed over untouched",
              ds._openvino_models_dir(absent) == absent, ds._openvino_models_dir(absent))

        # No stock export to derive from -- the cpu file list does not fetch one.
        empty = tempfile.mkdtemp(prefix="ovempty-")
        try:
            check("a cache with no batched export is handed over untouched",
                  ds._openvino_models_dir(empty) == empty, ds._openvino_models_dir(empty))
        finally:
            shutil.rmtree(empty, ignore_errors=True)

        # 🔴 The one that has to hold: a file that is not a model at all. Anything thrown while
        # deriving must come back as the original directory, so the engine starts unbatched
        # rather than not at all.
        check("an unreadable batched export falls back to the cache directory",
              ds._openvino_models_dir(root) == root, ds._openvino_models_dir(root))
        # The farm is built before the export is read, so a failure used to leave a directory
        # of symlinks that looked exactly like a working one -- to anyone asking why batching
        # was off, evidence pointing the wrong way.
        check("and takes its half-built farm with it", not os.path.exists(farm))

        try:
            import onnx
            from onnx import helper, TensorProto
        except ImportError:
            # 🔴 A failure on CI rather than a print. Returning here skips every assertion
            # below and the suite still reports that everything passed -- which is the shape
            # this repository already paid for once, with the tempo assertions that shelled
            # out to a missing ffmpeg on every run. CI installs onnx; if it ever stops, this
            # says so instead of going quiet.
            if os.environ.get("CI"):
                check("onnx is installed so the derivation itself is exercised", False)
                return
            print("  (skipped the derivation itself: onnx is not installed here)")
            return

        # 🔴 An op from a domain onnx knows nothing about, deliberately. This stood in as an
        # Identity, and Identity ties the output's shape to the input's: shape inference
        # propagated [32, 1, 160000] over the declared output and the derivation's edit to the
        # frame dimension could not be observed at all. The real export's output is not a copy
        # of its input -- [32, 589, 7], windows by frames by speakers -- and an op with no
        # inference rule is what lets a two-node fake say so.
        node = helper.make_node("Segment", ["input"], ["output"], domain="test.fake")
        graph = helper.make_graph(
            [node], "seg",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 1, 160000])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 589, 7])])
        model = helper.make_model(graph)
        model.opset_import.append(helper.make_opsetid("test.fake", 1))
        onnx.save(model, os.path.join(root, "segmentation-3.0-b32.onnx"))

        got = ds._openvino_models_dir(root)
        check("a derivable export produces a directory of its own", got == farm, got)

        # The switch, asserted where deriving succeeds -- against a case that already returns
        # the cache directory it would prove nothing. Off means the cache directory; a value
        # that is neither on nor off must not quietly become off, because that is a 4x loss
        # with nothing in the output saying why.
        ds.OPENVINO_BATCHING = "off"
        check("--openvino-batching off hands over the cache directory",
              ds._openvino_models_dir(root) == root, ds._openvino_models_dir(root))
        ds.OPENVINO_BATCHING = "OFF"
        check("and the value is not case-sensitive", ds._openvino_models_dir(root) == root)
        # 🔴 The whole vocabulary, walked rather than spelled out here. Every other boolean in
        # this wrapper answers to 0/false/no/off -- --exclusive is read four lines from where
        # this flag is -- so a person turning batching off for a measurement has four ways to
        # write what reads to them as one word, and any spelling that falls through to the
        # unrecognised branch derives: the measurement then reports "batching off" for a run
        # with batching on. Walking EngineArgs' own tuples is the point rather than a tidiness
        # -- a list copied into this file goes green on the day a fifth spelling is added to
        # them, which is the same second copy that put the defect here in the first place.
        for word in EngineArgs.OFF_WORDS:
            ds.OPENVINO_BATCHING = word
            check("--openvino-batching %s hands over the cache directory" % word,
                  ds._openvino_models_dir(root) == root, ds._openvino_models_dir(root))
        for word in EngineArgs.ON_WORDS:
            ds.OPENVINO_BATCHING = word
            check("--openvino-batching %s derives" % word,
                  ds._openvino_models_dir(root) == farm, ds._openvino_models_dir(root))
        ds.OPENVINO_BATCHING = "maybe"
        check("an unrecognised value still derives", ds._openvino_models_dir(root) == farm)
        ds.OPENVINO_BATCHING = batching

        # 🔴 A device may be named, and on a two-card machine it will be: the engine takes
        # openvino:<device> and the chart could start sending one. So the guard has to be a
        # prefix test, and this has to be asserted where deriving SUCCEEDS -- asserted against
        # an unreadable export instead, both answers are the cache directory and the case
        # proves nothing, which is how it was written the first time.
        ds.EXECUTION_MODE = "openvino:GPU.1"
        check("a mode naming a device still derives", ds._openvino_models_dir(root) == farm,
              ds._openvino_models_dir(root))
        ds.EXECUTION_MODE = "openvino"
        prepared = os.path.join(farm, "segmentation-3.0-b32-dynseq.onnx")
        check("the prepared model is there under the name speakrs looks for",
              os.path.isfile(prepared) and not os.path.islink(prepared), prepared)
        # 🔴 Followed, not just identified as a link. os.path.islink is True for a dangling
        # one, so a farm of links that all point at themselves satisfied the old wording of
        # this check -- "reachable through it" was the one thing it did not test. The engine
        # joins the embedding weights and the PLDA directory off this path; dead links there
        # turn "slower but running" into a pipeline that cannot build, which is the failure
        # this whole function is written to avoid.
        linked = os.path.join(farm, "segmentation-3.0-b32.onnx")
        check("the rest of the cache is reachable through it",
              os.path.islink(linked) and os.path.isfile(linked)
              and os.path.realpath(linked).startswith(os.path.realpath(root)),
              os.path.realpath(linked))
        for name in os.listdir(farm):
            entry = os.path.join(farm, name)
            check("every entry in the farm resolves: %s" % name, os.path.exists(entry),
                  os.path.realpath(entry))

        dims = onnx.load(prepared).graph.input[0].type.tensor_type.shape.dim
        # Batch static, samples dynamic. The other way round was measured and still fails.
        check("the batch dimension stays fixed", dims[0].dim_value == 32, dims[0].dim_value)
        check("the sample dimension becomes dynamic",
              dims[2].dim_param == "samples" and dims[2].dim_value == 0, str(dims[2]))
        out = onnx.load(prepared).graph.output[0].type.tensor_type.shape.dim
        # Both ends, or the graph contradicts itself: a dynamic sample count feeding a fixed
        # frame count. onnx's own checker accepts that and ORT runs it on CPU, so nothing
        # downstream of here would notice the output edit going missing.
        check("the frame dimension becomes dynamic too",
              out[1].dim_param == "frames" and out[1].dim_value == 0, str(out[1]))
        check("the speaker dimension stays fixed", out[2].dim_value == 7, out[2].dim_value)
        check("no half-written model is left behind",
              not os.path.exists(prepared + ".partial"))

        # 🔴 The name has to track the export, not a literal. speakrs builds what it looks
        # for out of its own PRIMARY_BATCH_SIZE, so a cache shipping a different batch size is
        # the case where a hardcoded 32 here and the constant there stop agreeing -- and the
        # failure is batching silently off, which nothing errors on.
        other = os.path.join(root, "segmentation-3.0-b64.onnx")
        shutil.copyfile(os.path.join(root, "segmentation-3.0-b32.onnx"), other)
        ds._openvino_models_dir(root)
        check("a different batch size is derived under its own name",
              os.path.isfile(os.path.join(farm, "segmentation-3.0-b64-dynseq.onnx")),
              sorted(os.listdir(farm)))
        check("and the original is still derived alongside it",
              os.path.isfile(os.path.join(farm, "segmentation-3.0-b32-dynseq.onnx")))
        # 🔴 The pattern must not match what this function writes. Asserting that no
        # `-dynseq-dynseq` file appears could not fail: the exports are read from the cache
        # and the derivatives are written to the farm, two directories that are never the
        # same one, so the scenario it described cannot arise. Ask the pattern directly
        # instead -- that is the property, and it is the one a loosened regex breaks.
        check("the pattern does not match a derived model",
              not ds._BATCHED_SEGMENTATION.fullmatch("segmentation-3.0-b32-dynseq.onnx"))
        check("and still matches a real export",
              bool(ds._BATCHED_SEGMENTATION.fullmatch("segmentation-3.0-b32.onnx")))
        os.remove(other)

        # 🔴 /tmp is a mounted volume and survives a restart. A farm left from a previous
        # revision would aim at weights the cache has moved past, and nothing would say so,
        # so a second call must rebuild rather than find its own work and keep it.
        stale = os.path.join(farm, "left-from-a-previous-revision.onnx")
        open(stale, "wb").close()
        os.utime(prepared, (1000000, 1000000))
        ds._openvino_models_dir(root)
        check("a farm left over from an earlier run is rebuilt, not reused",
              not os.path.exists(stale))
        check("and the model is derived again rather than found",
              os.stat(prepared).st_mtime > 1000000, os.stat(prepared).st_mtime)
        # 🔴 Two caches, two farms. /tmp is a mounted volume shared by every replica of this
        # application, and under one fixed name the second replica tears down and rebuilds the
        # directory the first is serving out of -- while that one holds open handles to paths
        # that have just been replaced. Asserted as the property rather than by restating the
        # formula, which would be this module asked twice.
        check("a second cache directory gets a farm of its own",
              ds._farm_for(root) != ds._farm_for(root + "-other"),
              ds._farm_for(root))

        # 🔴 One unconvertible export must not take the others with it. This loop used to sit
        # inside a single try, so a cache carrying both a b32 and a b64 export lost BOTH when
        # either would not convert: the farm was torn down, the engine ran a window at a time,
        # and the log named a file that had nothing wrong with it.
        broken = os.path.join(root, "segmentation-3.0-b64.onnx")
        open(broken, "wb").write(b"not an onnx graph")
        got = ds._openvino_models_dir(root)
        check("a broken export does not cost the ones that convert", got == farm, got)
        check("and the good one is still derived",
              os.path.isfile(os.path.join(farm, "segmentation-3.0-b32-dynseq.onnx")),
              sorted(os.listdir(farm)))
        check("while the broken one leaves no half-written file behind",
              not os.path.exists(os.path.join(farm, "segmentation-3.0-b64-dynseq.onnx.partial")))
        os.remove(broken)

        # 🔴 That the function is reached at all. Every check above calls it directly, so
        # deleting the call from the engine's argv left all of them green and turned
        # batching silently off -- which is the entire reason this code exists.
        source = open(ds.__file__, encoding="utf-8").read()
        wired = [line for line in source.splitlines()
                 if "--models-dir" in line
                 or ("_openvino_models_dir(" in line and "def " not in line)]
        check("the engine is handed the prepared directory, not the raw cache",
              any("_openvino_models_dir(" in line for line in wired), wired)
    finally:
        ds.EXECUTION_MODE = original
        ds.OPENVINO_BATCHING = batching
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(farm, ignore_errors=True)


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

        # The cap is on a span, not on the upload: a caller legitimately sends half an
        # hour of audio with seconds of speech marked in it. Lowered here because the
        # stub decodes to four seconds; 300 is what ships.
        was, align.MAX_SPAN_SEC = align.MAX_SPAN_SEC, 1.5
        try:
            over = c.post("/v1/audio/align", files=WAV,
                          data={"segments": '[{"start":0,"end":1,"text":"hi"},'
                                            '{"start":1,"end":4,"text":"yo"}]'}).json()
            got = over.get("results") or []
            check("a span over the limit fails alone, and says the limit",
                  len(got) == 2 and not got[0].get("error")
                  # The knob name is part of the message on purpose: the number alone reads
                  # as a property of the model rather than a setting a deployment can move.
                  and "--max-span-seconds" in (got[1].get("error") or ""),
                  got)
            check("the spans under it still align",
                  bool((got[0] or {}).get("units")), got[:1])
            single_over = c.post("/v1/audio/align", files=WAV, data={"text": "hi there"})
            check("single mode refuses an over-long clip with 413",
                  single_over.status_code == 413
                  and "aligns at most" in single_over.text, single_over.status_code)
        finally:
            align.MAX_SPAN_SEC = was

        # 🔴 Outside the lowered-ceiling block: in there the stub clip is
        # already over the limit, the request is refused with 413 and the work
        # never runs, so the assertion would pass without measuring anything.
        # The single-file path measures too. It is the largest shape this engine
        # runs -- the whole upload -- so it sets the process's all-time high, and
        # `_used` answers zero for anything below that. Unmeasured, one ordinary
        # successful request here froze the correction for the life of the process:
        # every grouped call afterwards measured as nothing. ⚠️ Nothing unlucky is
        # needed; every other way the peak could stick needed a failure, this one
        # needs a success.
        # ⚠️ The counters only move where there is a card to read, and there is none
        # here, so the pair is produced: a reading before, and a used figure after.
        keep_mem = (align._memory_now, align._used_bytes)
        try:
            align._memory_now = lambda: (0, 0)
            align._used_bytes = lambda *_: 10 ** 8
            seen_before = align._calls_seen
            c.post("/v1/audio/align", files=WAV, data={"text": "hi"})
            check("a single-file request is measured, not just served",
                  align._calls_seen > seen_before,
                  (seen_before, align._calls_seen))

            # 🔴 And measured in the SAME unit the other two call sites use. `_observe`'s
            # first argument is a cost -- the larger of the encoder and language-model
            # terms -- and a single span is exactly the shape where the encoder wins, so
            # the raw position count is 2.25x low with the cache on and 12.2x low with
            # `--no-kv-cache`. `_scale` is global and shared with the grouped path, so one
            # such request divides every later request's budget by that factor: grouping
            # collapses to one span a call, and the log blames the library. Both arguments
            # type as a number and both are plausible sizes, so nothing else can catch it.
            # ⚠️ The two are told apart by which FUNCTION answered, not by a number: both
            # are position counts and both are plausible. So each is stubbed to its own
            # marker value and the assertion is on which marker arrived.
            keep = (align._observe, align._cost, align._span_positions)
            seen_cost = []
            align._observe = lambda cost, used: seen_cost.append(cost)
            align._span_positions = lambda *a, **k: 111.0     # the raw count
            align._cost = lambda members, extra=None: 777.0   # what a call really costs
            try:
                c.post("/v1/audio/align", files=WAV, data={"text": "hi there"})
            finally:
                align._observe, align._cost, align._span_positions = keep
            check("and what it reports is the call's cost, not the span's position count",
                  seen_cost == [777.0], {"reported": seen_cost,
                                         "cost": 777.0, "raw positions": 111.0})

            # 🔴 And the failing half, which is the likelier one: 300 s is the largest
            # shape this engine runs and the one most likely to hit the wall on a 4 GiB
            # grant -- this file's own header prices it at 2.2 GiB, 3.3 with dense text.
            # The commit that added the measurement above covered only the success.
            # 🔴 `_align_raw`, not `_align`. The reset now lives in the `_align` wrapper,
            # so a stub installed on `_align` REPLACES the thing under test and this check
            # goes green on a build with no reset anywhere. Stub the layer below whatever
            # owns the behaviour, or the test measures the stub.
            resets = []
            keep_rp, keep_al = align._reset_peak, align._align_raw

            def _raises_oom(*a, **k):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

            align._reset_peak = lambda: resets.append(1)
            align._align_raw = _raises_oom
            try:
                c.post("/v1/audio/align", files=WAV, data={"text": "hi"})
            except Exception:
                pass
            finally:
                align._reset_peak, align._align_raw = keep_rp, keep_al
            check("and a single-file request that fails puts the peak back",
                  bool(resets), resets)
        finally:
            align._memory_now, align._used_bytes = keep_mem

        # 🔴 Outside the lowered-limit block on purpose. Inside it the stub clip is already
        # over the ceiling, so a reader that never consults the stub still answers 413 and
        # the assertion passes for the wrong reason -- which is what the first version of
        # this check did. Here the ceiling is the shipped one and the clip is well under it,
        # so only a path that reads the WIDE measurement can refuse.
        #
        # What it is guarding: `audioio.probe_seconds` tries soundfile and the stdlib `wave`
        # module and nothing else -- measured, libsndfile 1.2.2 answers "Format not
        # recognised" for m4a and lists no M4A/MP4/AAC format at all -- while the engine
        # decodes through ffmpeg and reads it happily. A clip that runs and cannot be
        # measured walks past a limit keyed to the measurement, and 200 with saturated
        # timestamps is the outcome the limit exists to prevent.
        keep_dur = align.duration
        try:
            align.duration = lambda _p: 3600.0      # only the wide reader can answer this
            unmeasurable = c.post("/v1/audio/align", files=WAV, data={"text": "hi there"})
            check("a clip only ffprobe can measure is still held to the span limit",
                  unmeasurable.status_code == 413 and "aligns at most" in unmeasurable.text,
                  (unmeasurable.status_code, unmeasurable.text[:90]))
            # ⚠️ None still passes through, deliberately: ffprobe may be absent from an
            # image, and a clip three readers cannot measure is one the engine fails on by
            # itself. Refusing here would reject working files on a thin image.
            align.duration = lambda _p: None
            unknown = c.post("/v1/audio/align", files=WAV, data={"text": "hi there"})
            check("but a duration nothing can read is passed through, not refused",
                  unknown.status_code == 200, unknown.status_code)
        finally:
            align.duration = keep_dur

        # A container whose header parses and whose body does not. The probe in front of the
        # decode reads the format block and stops, so this gets past it -- measured, a FLAC
        # truncated to half its length: `info` returns, `read` raises. The decode then runs on
        # the worker, where an unguarded exception becomes a 500, and a 5xx is what tells a
        # retrying caller to come back for a file that will never decode. WAV hides this:
        # libsndfile returns the frames it has rather than raising.
        sf = sys.modules["soundfile"]
        kept_read = sf.read

        def _body_will_not_decode(*a, **k):
            raise RuntimeError("Error opening: File contains data in an unknown format.")

        sf.read = _body_will_not_decode
        try:
            before = len(align._telemetry)
            bad = c.post("/v1/audio/align", files=WAV,
                         data={"segments": '[{"start":0,"end":1,"text":"hi"}]'})
            check("a body that will not decode is a 400, not a 500",
                  bad.status_code == 400 and "could not decode audio" in bad.text,
                  (bad.status_code, bad.text[:160]))
            # 🔴 And it is recorded on the way out. This exact defect has already come back
            # once: the serial path below was fixed with a `finally` and the batched path's
            # early exit, which leaves ABOVE both `finally` blocks, repeated it. A request
            # that exits here unrecorded leaves `_last_request_ended` at the request before
            # it, so the NEXT request's `idle_before` spans two requests -- and that gap is
            # the half of the measurement that says whether a neighbour moved the wall while
            # this engine was not running. A wrong gap is worse than a missing one.
            check("and the request is still recorded, so the next idle gap starts here",
                  len(align._telemetry) == before + 1,
                  (before, len(align._telemetry)))
        finally:
            sf.read = kept_read



def t_align_batching():
    """The grouping rule, the budget it finds, and what happens when a call fails.

    🔴 The batched path was never exercised before this: the stub returned one result
    however many spans were sent, which pairs only the first member and looks like a pass.
    This stub returns one result a span, so a group that silently dropped members would
    fail here.
    """
    from fastapi.testclient import TestClient
    from wrapper.caps import align

    fake_soundfile()
    unit = {"text": "hi", "start_time": 0.1, "end_time": 0.4}
    calls = []
    fail_on = {}

    def fake_align(audio=None, text=None, language=None):
        texts = text if isinstance(text, list) else [text]
        calls.append(list(texts))
        for t in texts:
            if t in fail_on:
                raise fail_on[t]
        return [[unit] for _ in texts]

    model = types.SimpleNamespace(align=fake_align)
    align._state.update(ready=True, model=model, device="cpu")
    # 🔴 A loaded engine has been weighed: `_calibrate` runs one throwaway call before
    # `ready`, and grouping is held off until something has. Without this the stub is a
    # machine where nothing ever measured -- a real state, tested on its own below -- and
    # every grouping case here would quietly become a test of the gate instead of the rule.
    align._scale_seen = 1

    # 🔴 What a position costs follows the CHECKPOINT, not the flag. The two part company
    # on one that ships `use_cache: false`: nothing overrides it any more, so no cache is
    # built -- and a cost model reading the flag prices 138 kB for a cache that does not
    # exist, over-predicting by five, which this file's own note calls the slow failure that
    # raises nothing. The log line and telemetry read the same value for the same reason.
    keep_pricing = (align.CACHE_ON, align.KAPPA_LM, align.KAPPA_ENCODER,
                    align.ENCODER_FIXED_BYTES)
    try:
        align._rebind_cache_pricing([None, None, True])      # what the shipped one holds
        with_cache = (align.CACHE_ON, align.KAPPA_LM)
        align._rebind_cache_pricing([False, False, False])   # a checkpoint that ships it off
        without = (align.CACHE_ON, align.KAPPA_LM)
        check("a checkpoint that ships the cache off is priced without one",
              with_cache[0] is True and without[0] is False and with_cache[1] != without[1],
              (with_cache, without))
        # 🔴 The third row, wrong here for one commit: nothing readable at all. An empty
        # list means `_kv_configs` matched no object, so the write loop was a no-op and the
        # checkpoint builds whatever it ships -- a cache, on this family. Pricing it as none
        # under-predicts by five, the direction that walks into the card; pricing it as a
        # cache costs speed and nothing else.
        for nothing in ([], [None, None, None]):
            align._rebind_cache_pricing(nothing)
            check("nothing readable prices as a cache, not as none (%r)" % (nothing,),
                  align.CACHE_ON is True, (nothing, align.CACHE_ON))
        align._rebind_cache_pricing([True, False])
        check("configs that disagree are priced as a cache",
              align.CACHE_ON is True, align.CACHE_ON)
    finally:
        (align.CACHE_ON, align.KAPPA_LM, align.KAPPA_ENCODER,
         align.ENCODER_FIXED_BYTES) = keep_pricing

    # ---- the pieces, before the path that uses them ----------------------------------
    check("units split CJK singly and latin on whitespace",
          align._units_of("hi there \u4f60\u597d") == ["hi", "there", "\u4f60", "\u597d"],
          align._units_of("hi there \u4f60\u597d"))
    check("a latin run inside CJK stays one unit",
          align._units_of("\u4f60abc\u597d") == ["\u4f60", "abc", "\u597d"],
          align._units_of("\u4f60abc\u597d"))
    # 2 + 13*seconds + per unit (its tokens + two timestamp slots); no tokenizer on the
    # stub, so the fallback counts one token a CJK character.
    check("positions = 2 + 13 a second + 3 a CJK unit",
          align._span_positions(10, "\u4f60\u597d") == 2 + 130 + 6,
          align._span_positions(10, "\u4f60\u597d"))

    long_span = align._Span(0, None, "x", "auto", 30.0, 800)
    short_span = align._Span(1, None, "x", "auto", 2.0, 100)
    mixed = align._cost([long_span, short_span, short_span, short_span])
    uniform = align._cost([long_span, long_span, long_span, long_span])
    check("a batch costs count x longest, so mixed and uniform are equal",
          mixed == uniform == 4 * 800, (mixed, uniform))
    # 🔴 The larger of two phases, not their sum. Measured on the batch axis: one span and
    # two spans of the same length cost the same, because the encoder sets the peak in both,
    # and from four up the cost tracks the batch because the language model has taken over.
    rate = align._encoder_positions_a_second()
    # Little text against long audio, so the encoder is the taller phase for the first few
    # spans -- which is the regime the batch axis exposed: one span and two of the same
    # length measured the same, 227.0 against 230.0 MiB.
    sparse = align._Span(2, None, "x", "auto", 30.0, 300)
    check("while the encoder is the taller phase, adding a span costs nothing",
          align._cost([sparse]) == align._cost([sparse, sparse]) == rate * 30,
          (align._cost([sparse]), align._cost([sparse, sparse]), rate * 30))
    check("and the same spans cost more once enough of them are in the call",
          align._cost([sparse] * 8) == 8 * 300 > rate * 30,
          (align._cost([sparse] * 8), rate * 30))
    check("and once the batch is the taller phase the cost tracks the batch",
          align._cost([long_span] * 4) == 4 * 800 > rate * 30,
          (align._cost([long_span] * 4), rate * 30))
    check("a long span alone is not cheap for being alone",
          align._cost([long_span]) > 800, align._cost([long_span]))

    # ---- the encoder's fixed part -----------------------------------------------------
    # 🔴 Only measurable once the KV cache came off: at 5 MB it is 3% of what the cache
    # alone cost, so while the cache was on it had nowhere to show. Everything above this
    # block runs on a stub with no dimensions, where the term is deliberately zero -- so
    # without this block the term would be in the shipped arithmetic and in none of the
    # checks.
    keep_model = align._state.get("model")
    keep_rate = align._encoder_rate
    keep_k = kappa(align, align.KAPPA_NO_CACHE)
    try:
        align._encoder_rate = None
        align._state["model"] = types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    text_config=types.SimpleNamespace(hidden_size=1024,
                                                      intermediate_size=3072),
                    audio_config=types.SimpleNamespace(downsample_hidden_size=480,
                                                       n_window=50),
                    classify_num=5000),
                parameters=lambda: iter(())))
        fixed = align._encoder_fixed_positions()
        check("the encoder costs something before its per-second term starts",
              abs(fixed - align.ENCODER_FIXED_BYTES
                  / align._bytes_a_position(align._dims())) < 1e-9, fixed)
        tiny = align._Span(3, None, "x", "auto", 0.5, 20)
        check("and on four short spans it is most of what the call costs",
              align._cost([tiny] * 4) > 2 * (4 * 20), align._cost([tiny] * 4))
        # 🔴 With real dimensions and no KV cache, a second of audio is worth about 300
        # positions, so four 30-second spans are priced by the ENCODER (9016) and not by the
        # batch (3200). While the cache was on the same four priced by the batch. This is
        # the behaviour change that commit makes and it is not a small one: long spans got
        # about five times dearer against short ones, which is what moves the division.
        rate4 = align._encoder_positions_a_second()
        check("with the cache off a long span is priced by the encoder, not by the batch",
              align._cost([long_span] * 4) == rate4 * 30 + fixed > 4 * 800,
              (align._cost([long_span] * 4), rate4 * 30 + fixed, 4 * 800))
        check("and the batch takes over once enough spans ride in the call",
              align._cost([long_span] * 16) == 16 * 800,
              (align._cost([long_span] * 16), 16 * 800))
        # 🔴 The fixed part belongs to the encoder phase, so where the language model is the
        # taller phase it does not appear at all. Charged to the call instead, it would sit
        # on top of both -- and the six shapes rule that out, see the constant's own note.
        check("and where the batch decides, the encoder's fixed part is not charged",
              align._cost([long_span] * 16) % 800 == 0, align._cost([long_span] * 16))
    finally:
        align._state["model"] = keep_model
        align._encoder_rate = keep_rate
        kappa_restore(align, keep_k)
    check("with no dimensions to read there is no fixed term either, not a guessed floor",
          align._encoder_fixed_positions() == 0.0, align._encoder_fixed_positions())

    # ---- the host's account, which is the one that kills --------------------------------
    # 🔴 Everything above prices the GPU. Crossing the GPU's limit raises an exception this
    # module catches, halves and retries; crossing the container's memory limit is an OOMKill
    # that takes the process and every in-flight request with it. So the guard refuses rather
    # than trying, and these read the source because what they check cannot be seen at
    # runtime until it has already gone wrong.
    src2 = open(os.path.join(os.path.dirname(__file__), "..", "wrapper", "caps",
                             "align.py")).read()
    # 🔴 Sliced on CODE landmarks, never on a comment. `work` used to end at the text of a
    # comment, so deleting that comment silently changed what this check reads -- a comment a
    # test greps for is not a comment, it is an interface, and nothing marks it as one.
    # `_slice` takes the source of one function by name, off the parse tree.
    def _slice(src, fname):
        import ast as _ast
        lines = src.split("\n")
        for node in _ast.walk(_ast.parse(src)):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == fname:
                return "\n".join(lines[node.lineno - 1:node.end_lineno])
        raise AssertionError("%s is gone from align.py, so this check reads nothing" % fname)

    handler = src2[src2.index("if MAX_UPLOAD_BYTES > 0"):src2.index("def _work_batch(ctx):")]
    work = _slice(src2, "_work_batch")
    check("the decode happens on the task worker, not in the request handler",
          "_decode_all()" in work and "await asyncio.to_thread(_decode_all)" not in handler,
          "_decode_all()" in work)
    # 🔴 That IS the admission control: one worker means one decoded request at a time, and
    # the queue behind it holds undecoded bytes. A semaphore beside it was written first and
    # removed -- it was a second scheduler next to the one `tasks` already runs, and the only
    # thing it could add was a permit to leak.
    check("and there is no second admission counter beside the queue",
          "_inflight" not in src2 and "_Slot" not in src2,
          [w for w in ("_inflight", "_Slot") if w in src2])
    # ---- what a group the card refuses costs ------------------------------------------
    # 🔴 The recovery that used to live here worked and was removed on purpose: 4000 spans
    # came back with zero errors (V76), and the price was a ceiling that never rose, so one
    # oversized call left the process at one span a call for its whole life at about a tenth
    # of the throughput, with a complete response and a zero error count. What is left has to
    # stay gone, so the check is on the source.
    src3 = open(os.path.join(os.path.dirname(__file__), "..", "wrapper", "caps",
                             "align.py")).read()
    # 🔴 The split is fine and stays. What was toxic is the MEMORY: a ceiling that only fell,
    # so one unlucky call left the process at one span a call for its whole life.
    for gone, what in (("_ceiling", "a ceiling that outlives the call"),
                       ("DEGRADED", "a degraded state to report")):
        check("no %s" % what, gone not in src3, gone)
    check("a group that will not fit is split and retried",
          '"split"' in src3 and "group[:half]" in src3, "absent")
    check("and a group that fails for some other reason is isolated, one member at a time",
          '"isolate"' in src3, "absent")


    check("the header is read before anything is decoded",
          "_sf.info(" in handler and handler.index("_sf.info(") < handler.index("status_code=400"),
          "sf.info before the refusal it answers")
    # 🔴 This branch refuses nothing on how much audio an upload holds. The guard that
    # would, and the measured rate it rests on, are deferred: that rate came from one machine
    # and from a probe that could not separate "cost follows the file" from "cost follows the
    # sum of the spans". What holds here is structural, and the test for it is above -- the
    # decode is on the single worker, so the multiplication an OOMKill is made of is gone.
    check("and no limit on how much audio a request may hold ships in this branch",
          "MAX_AUDIO_SECONDS" not in src2 and "_host_audio_seconds" not in src2,
          [w for w in ("MAX_AUDIO_SECONDS", "_host_audio_seconds") if w in src2])
    check("unreadable audio is still a 400 rather than a server error",
          "status_code=400" in handler, handler.count("status_code=400"))
    # 🔴 A 503 without Retry-After says "no" rather than "not yet", and what a client does
    # with the difference is the client's decision rather than this engine's assumption. One
    # caller read in 2026-09 retries 5xx and abandons the rest; the next will differ.
    tsrc = open(os.path.join(os.path.dirname(__file__), "..", "wrapper", "tasks.py")).read()
    check("a busy engine says how long to wait, not just no",
          "Retry-After" in tsrc and "status_code=503" in tsrc, "Retry-After present")
    check("and so does an engine that is not loaded yet",
          "Retry-After" in src2[:src2.index("async def _work_batch") if
                                "async def _work_batch" in src2
                                else src2.index("def _work_batch")], "Retry-After present")

    # 🔴 The six shapes the constants were fitted from, with what each actually peaked at
    # on hzydemo01 with the cache off. **This is a consistency check, not a validation** --
    # these are the points the fit saw, so passing it says the constants in the module are
    # the ones that were fitted and no digit was lost on the way in. It is worth having for
    # exactly that: a typo in KAPPA_LM changes no test above, because everything above runs
    # on a stub with no dimensions.
    keep_model, keep_rate = align._state.get("model"), align._encoder_rate
    try:
        align._encoder_rate = None
        align._state["model"] = types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    text_config=types.SimpleNamespace(hidden_size=1024,
                                                      intermediate_size=3072),
                    audio_config=types.SimpleNamespace(downsample_hidden_size=480,
                                                       n_window=50),
                    classify_num=5000),
                parameters=lambda: iter(())))
        keep_k3 = kappa(align, align.KAPPA_NO_CACHE)
        perpos = align._bytes_a_position(align._dims())
        # 🔴 The six were measured on a build with no KV cache, so they check that pair. The
        # shipping pair is the other one, and a check that read it would be asserting the
        # wrong build's numbers while still passing something.
        measured = ((4, 100, 2.0, 21.8), (8, 126, 4.0, 38.6), (32, 99, 1.0, 88.7),
                    (64, 99, 1.0, 178.2), (16, 112, 2.0, 51.0), (8, 178, 8.0, 72.3))
        worst, where = 0.0, None
        for n, pos, sec, mb in measured:
            group = [align._Span(i, None, "x", "auto", sec, pos) for i in range(n)]
            off = abs(align._cost(group) * perpos / 1e6 - mb) / mb
            if off > worst:
                worst, where = off, (n, sec, align._cost(group) * perpos / 1e6, mb)
        check("the cache-off pair reproduces the six shapes it was fitted from",
              worst < 0.05, (round(worst * 100, 1), where))
        kappa_restore(align, keep_k3)
        # And the shipping default is the OTHER pair, which is what makes --no-kv-cache
        # opt-in rather than a thing that happened.
        check("what ships is the pair fitted with a cache, not this one",
              (align.KAPPA_LM, align.KAPPA_ENCODER,
               align.ENCODER_FIXED_BYTES) == align.KAPPA_WITH_CACHE,
              (align.KAPPA_LM, align.KAPPA_ENCODER, align.ENCODER_FIXED_BYTES))
    finally:
        align._state["model"], align._encoder_rate = keep_model, keep_rate

    # ---- the cache that is built and never read ---------------------------------------
    # 🔴 `_kv_configs` walks a nesting rather than naming it, because the decorator reads
    # whichever `self.config` the module it decorates has. A walk that stopped at the first
    # level would set the flag on an object nothing reads and report success.
    leaf = types.SimpleNamespace(use_cache=True)
    mid = types.SimpleNamespace(text_config=leaf, use_cache=True)
    top = types.SimpleNamespace(thinker_config=mid, use_cache=True)
    found = align._kv_configs(types.SimpleNamespace(model=types.SimpleNamespace(config=top)))
    check("every config on the path the decoder reads use_cache from is found",
          found == [top, mid, leaf], found)
    check("a model that is not loaded yields no configs rather than raising",
          align._kv_configs(types.SimpleNamespace()) == [],
          align._kv_configs(types.SimpleNamespace()))
    # A config that points at itself would otherwise loop forever inside model loading.
    loop = types.SimpleNamespace()
    loop.thinker_config = loop
    check("a config that points at itself terminates",
          align._kv_configs(types.SimpleNamespace(
              model=types.SimpleNamespace(config=loop))) == [loop], "no hang")

    # ---- the whole path --------------------------------------------------------------
    was = (align.BATCH, align._budget, align.MAX_SPAN_SEC)
    # 🔴 The promise the switches make is that the *grouping* is off until asked for. Read
    # from the module rather than from the table in the comment, which is where the two
    # would drift.
    #
    # ⚠️ There used to be three switches here and the other two answered nothing: the
    # padding rule brings its own ascending walk, and handing the library arrays rather
    # than temp wavs changes no output byte. They are gone; what is left is the one switch
    # that decides whether a request is grouped at all.
    # 🔴 An ordering bug, so the check is on the order. `warn_unclaimed` reports what is
    # left over and drains the bad values it has seen SO FAR: a flag claimed after it is
    # announced as one this engine does not take, and a typo in that flag's value falls
    # back to the default in silence. It sat above two of the four claims with a comment
    # over it explaining why it must sit below all of them.
    import re as _re
    _src = io.open(os.path.join(os.path.dirname(align.__file__), "align.py"),
                   encoding="utf-8").read()
    _claims = [m.start() for m in
               _re.finditer(r"_args\.(number|count|text|flag|switch)\(", _src)]
    _warn = _src.index("_args.warn_unclaimed(")
    check("every ENGINE_ARGS flag is claimed before the unclaimed ones are reported",
          bool(_claims) and _warn > max(_claims),
          (len(_claims), _warn, max(_claims) if _claims else None))

    # 🔴 On by default as of this branch, and that is the whole point of the branch: the
    # recommended configuration is this flag with `--no-kv-cache`, and a default of off left
    # every deployment one edit away from the behaviour the work was for. Asserted rather
    # than left to the flag line because the default IS the product decision -- flipping it
    # back is a thing someone may do to quiet a failing acceptance check, and it has to cost
    # a deliberate edit here.
    # ⚠️ What a deployment that does nothing gets: grouped calls, and output that is NOT
    # byte-identical. 226 of 17550 timestamps moved on the corpus this was measured against.
    check("--align-batch defaults on, so the shipped image groups without being asked",
          align.BATCH is True, align.BATCH)
    # 🔴 And "off" still has to mean the machinery does not RUN, not that it runs and decides
    # to do nothing. An operator who types `--align-batch off` is asking for main's behaviour
    # back -- most likely because something went wrong -- so the fallback has to be a path
    # these 1300 lines are not on at all, and the way to hold a promise like that is to make
    # the code that would break it explode. Reading `if BATCH:` proves only that somebody
    # wrote it.
    # ⚠️ Set on the module rather than reloaded with a flag: `_run_group` and the rest read
    # the global per request (there is a second site below that flips it the same way to
    # compare billing across the two paths), and a reload here would rebuild every stub this
    # function has installed.
    # 🔴 And the memory counter is stubbed to ANSWER, which is the whole point. This check
    # used to run on a host with no card, where `_memory_now()` returns None and the
    # single-file path's measurement block is skipped for that reason alone -- so it went
    # green about the test host rather than about the code, while a real deployment with
    # grouping off called `_observe` on every single-file request. A check that asserts a
    # property of the machine it runs on cannot fail on the machine that matters.
    _kept_on = align.BATCH
    align.BATCH = False
    _kept_mem = align._memory_now
    align._memory_now = lambda: (0, 0)
    _boom = types.SimpleNamespace()
    _names = ("_batched", "_next_group", "_run_group", "_call", "_solve_budget",
              "_effective_budget", "_observe", "_record_group", "_cost", "_span_positions")
    _kept = {n: getattr(align, n) for n in _names}
    try:
        for _n in _names:
            setattr(align, _n, lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("the batching path ran with --align-batch off")))
        with TestClient(align.build_app(["align"])) as c_off:
            _b = c_off.post("/v1/audio/align", files=WAV,
                            data={"segments": '[{"start":0,"end":1,"text":"hi"},'
                                              '{"start":1,"end":2,"text":"yo"}]'})
            _s = c_off.post("/v1/audio/align", files=WAV, data={"text": "hi there"})
        check("and with it off neither path touches any of the sizing machinery",
              _b.status_code == 200 and _s.status_code == 200
              and len(_b.json().get("results") or []) == 2,
              (_b.status_code, _s.status_code, _b.text[:120]))
        # 🔴 `batch` is the multi-segment form and is true on main too; `grouped` is the one
        # that answers what --align-batch did. A caller reading `batch` to find out whether
        # grouping is on gets true from the day the image ships, since the chart passes no
        # flags -- which is the reason for the second field, so it has to be asserted apart.
        check("with grouping off the segment form still reports batch, and grouped is off",
              _b.json().get("batch") is True and _b.json().get("grouped") is False,
              {k: _b.json().get(k) for k in ("batch", "grouped")})
    finally:
        for _n, _v in _kept.items():
            setattr(align, _n, _v)
        align.BATCH, align._memory_now = _kept_on, _kept_mem
    # 🔴 Every tunable an operator may set is an ENGINE_ARGS flag, which is this engine's
    # own convention and what the other five capabilities do. Three of these used to be
    # separate environment variable names; the module must not grow them back.
    # ⚠️ The check is on how a value is READ, not on the names: ALIGN_FIXED_BUDGET and
    # ALIGN_GROUP_SLACK are module constants and keep their names.
    check("no tunable hides in an environment variable of its own",
          'os.environ.get("ALIGN' not in _src,
          [ln for ln in _src.splitlines() if 'os.environ.get("ALIGN' in ln])
    check("the four flags an operator may set are all claimed",
          all(f in _src for f in ("--align-batch", "--gpu-budget-fraction", "--max-span-seconds")),
          [f for f in ("--align-batch", "--gpu-budget-fraction", "--max-span-seconds")
           if f not in _src])
    # 🔴 The line exists because reading the source does not answer "what is this process
    # doing". Two measurements were taken on an engine left in a previous arm's state and
    # both looked like results.
    # 🔴 The cache and its calibration are ONE change, and the direction that hurts is
    # silent: the cache-off pair with a cache on prices a position at a fifth of what it
    # costs, which is a budget 5.1x too large and a batch that walks into the card. Nothing
    # raises. The invariant is that the three move as a unit and follow the switch.
    check("the factors are whichever pair the cache switch selects, all three together",
          (align.KAPPA_LM, align.KAPPA_ENCODER, align.ENCODER_FIXED_BYTES)
          == (align.KAPPA_NO_CACHE if align.NO_KV_CACHE else align.KAPPA_WITH_CACHE),
          (align.KAPPA_LM, align.KAPPA_ENCODER, align.ENCODER_FIXED_BYTES))
    check("and the default is the with-cache pair, so a fresh deployment is what ships",
          align.NO_KV_CACHE is False
          and (align.KAPPA_LM, align.KAPPA_ENCODER,
               align.ENCODER_FIXED_BYTES) == align.KAPPA_WITH_CACHE,
          (align.NO_KV_CACHE, align.KAPPA_LM))
    _d = {"hidden": 1024, "ffn": 3072, "classes": 5000, "width": 2,
          "downsample": 480, "window": 50}
    _wrong = align.KAPPA_NO_CACHE[0] * (6 * 1024 + 2 * 3072 + 5000) * 2
    check("and mixing them the wrong way under-prices a position about five-fold",
          4.5 < align._bytes_a_position(_d) / _wrong < 5.5,
          round(align._bytes_a_position(_d) / _wrong, 2))

    # ---- what the engine records about the machine it is on ---------------------------
    # 🔴 Two questions decide the next design and neither has an answer today: what makes the
    # wall move and how often, and whether the transcription engine runs beside us or finishes
    # before us. The pipeline reaches alignment after transcription, so the second one is
    # genuinely open, and `outside` -- free plus our own reserved pool, which cancels our own
    # allocations out -- is what would show it.
    with TestClient(align.build_app(["align"])) as c_t:
        before = len(align._telemetry)
        t0 = c_t.get("/v1/audio/align/telemetry")
        check("the engine serves its own telemetry, which the bench copy used to hold alone",
              t0.status_code == 200 and set(t0.json()) == {"now", "budget", "model",
                                                           "requests"},
              (t0.status_code, sorted(t0.json()) if t0.status_code == 200 else t0.text[:80]))
        check("and reports the sample size beside the correction, never the correction alone",
              {"scale", "scale_seen", "calls_seen"} <= set(t0.json()["model"]),
              sorted(t0.json()["model"]))
        # ⚠️ Batching ships off, so a sample taken only inside it would be empty on every
        # deployment that has not turned it on -- which is all of them.
        was_batch = align.BATCH
        try:
            align.BATCH = False
            c_t.post("/v1/audio/align", files=WAV,
                     data={"segments": '[{"start":0,"end":1,"text":"hi"}]'})
            check("a serial request is recorded too, not only a batched one",
                  len(align._telemetry) == before + 1, len(align._telemetry))
            rec = align._telemetry[-1]
            check("and it carries both ends plus the gap nobody was looking",
                  {"started", "ended", "idle_before", "spans", "calls"} <= set(rec),
                  sorted(rec))
            c_t.post("/v1/audio/align", files=WAV, data={"text": "hi there"})
            check("a single align request is recorded too, not only segmented requests",
                  len(align._telemetry) == before + 2, len(align._telemetry))
            check("and it records one span and one model call",
                  align._telemetry[-1]["spans"] == 1 and align._telemetry[-1]["calls"] == 1,
                  align._telemetry[-1])

            # 🔴 `calls` is the denominator a sweep divides by, so it has to be the calls
            # that happened, not a guess read back off the results. Counting results with
            # units was wrong in BOTH directions at once: an empty-text span is answered
            # without calling the model and was counted, and a span whose call raised is
            # answered with an error and was not. Four spans -- one empty, one that fails,
            # The mix is deliberately lopsided so the two ways of counting disagree: four
            # spans, TWO empty and one that fails, is two calls -- while counting answers
            # that carry units gives three, the two empties plus the one that worked.
            calls_made = []
            real_align = model.align

            def counting_align(audio=None, text=None, language=None, **kw):
                calls_made.append(1)
                if "boom" in (text if isinstance(text, str) else ""):
                    raise RuntimeError("this span is broken")
                return real_align(audio=audio, text=text, language=language, **kw)

            model.align = counting_align
            try:
                mixed = ('[{"start":0,"end":1,"text":"hi"},'
                         '{"start":1,"end":2,"text":""},'
                         '{"start":2,"end":3,"text":""},'
                         '{"start":3,"end":4,"text":"boom"}]')
                c_t.post("/v1/audio/align", files=WAV, data={"segments": mixed})
                rec2 = align._telemetry[-1]
                check("serial calls counts the calls, not the answers that carry units",
                      rec2["calls"] == len(calls_made),
                      (rec2["calls"], len(calls_made)))
                check("and the two empty spans made no call at all",
                      len(calls_made) == 2, len(calls_made))
            finally:
                model.align = real_align
        finally:
            align.BATCH = was_batch
    # ⚠️ A telemetry field must not be able to break a request: the body parser admits
    # shapes the engine cannot read, and a sum written the obvious way raised out of the
    # loop that exists so one bad span does not cost the others theirs.
    check("the covered-seconds sum cannot raise, whatever the body holds",
          align._covered_seconds([{"start": "x", "end": 1}, "not a dict", None,
                                  {"start": 0, "end": 2}]) == 2.0,
          align._covered_seconds([{"start": "x", "end": 1}, "not a dict", None,
                                  {"start": 0, "end": 2}]))
    check("and the ring is bounded, because a history in a process holding a model is a leak",
          align._telemetry.maxlen == align.TELEMETRY_KEEP, align._telemetry.maxlen)
    # 🔴 `outside` is the only number here that answers the question, and it answers it by
    # cancelling US out: an allocation of ours lowers free and raises reserved by the same
    # amount, so what is left moves only when another container moves. Written as free alone
    # it would read our own batch as a neighbour arriving, every reading would be wrong, and
    # nothing would say so -- the whole sample would be a plausible wrong answer.
    _torch = types.ModuleType("torch")
    _torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (7 * 2 ** 30, 15 * 2 ** 30),
        memory_reserved=lambda: 2 * 2 ** 30,
        memory_allocated=lambda: 1 * 2 ** 30)
    _keep_torch = sys.modules.get("torch")
    sys.modules["torch"] = _torch
    try:
        card = align._card()
        check("outside is free plus what we hold, so our own allocations cancel out",
              card["outside"] == card["free"] + card["reserved"],
              (card["outside"], card["free"], card["reserved"]))
        # ⚠️ And the sample carries the test for what its own numbers MEAN: under HAMi's
        # meminfo hook `free` is quota arithmetic that nets out nothing another container
        # holds, so a neighbour arriving and leaving both read as no change at all.
        _keep_env = os.environ.get("CUDA_DEVICE_MEMORY_LIMIT_0")
        try:
            os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = "4096m"
            check("and it says whether that free is the card or a quota",
                  align._card()["quota_view"] is False, align._card())
            _torch.cuda.mem_get_info = lambda: (1 * 2 ** 30, 4 * 2 ** 30)
            check("a total at or under the published limit can only be the rewrite",
                  align._card()["quota_view"] is True, align._card())
        finally:
            if _keep_env is None:
                os.environ.pop("CUDA_DEVICE_MEMORY_LIMIT_0", None)
            else:
                os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = _keep_env
    finally:
        if _keep_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = _keep_torch

    # ---- the machine calibrates itself before it serves anything ----------------------
    # 🔴 The two factors are what differs between machines, and `_observe` only learns from
    # calls that have already run -- so without this the FIRST request on an unmeasured
    # machine is sized by a calibration taken somewhere else. Where a quota is enforced the
    # only margin is the fraction itself.
    _src_cal = open(os.path.join(os.path.dirname(align.__file__), "align.py"),
                    encoding="utf-8").read()
    check("the calibration runs before ready, so nothing races it on the card",
          _src_cal.index("_calibrate()") < _src_cal.index('_state["ready"] = True'),
          "calibrate then ready")
    keep_sc = (align._scale, align._scale_seen, align._calls_seen, align._state.get("model"),
               align._encoder_rate)
    try:
        align._encoder_rate = None
        align._state["model"] = types.SimpleNamespace(
            model=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    text_config=types.SimpleNamespace(hidden_size=1024,
                                                      intermediate_size=3072),
                    audio_config=types.SimpleNamespace(downsample_hidden_size=480,
                                                       n_window=50),
                    classify_num=5000),
                parameters=lambda: iter(())))
        align._scale, align._scale_seen, align._calls_seen = 1.0, 0, 0
        seen_text = []
        keep_align = align._align
        keep_mem, keep_used = align._memory_now, align._used_bytes
        try:
            align._align = lambda path, text, language: seen_text.append(text) or [[]]
            # A call that takes twice what the model predicts: the correction lands near 2.
            _warm = align._Span(0, None, "\u5b57" * 120, "auto", 30.0,
                                align._span_positions(30.0, "\u5b57" * 120, "auto"))
            align._memory_now = lambda: (0, 0)
            align._used_bytes = lambda before, peak: int(
                2 * align._cost([_warm]) * align._bytes_a_position(align._dims()))
            with said() as heard_cal:
                align._calibrate()
            check("it sends one span of the production shape, not silence",
                  len(seen_text) == 1 and len(seen_text[0]) == 120, seen_text)
            check("and folds what it measured into the correction before any traffic",
                  1.5 < align._scale < 2.5 and align._scale_seen == 1,
                  (align._scale, align._scale_seen))
            check("and says how long it took, because nobody has measured that yet",
                  heard_cal.mentions("calibrated on this machine in"), heard_cal.records)
            # 🔴 A calibration that observed nothing must not read as a calibration that ran.
            # Loading the model leaves a historical peak; a warm-up that stays under it makes
            # `_used` answer zero, `_observe` discard the sample, and the success line print
            # "it took 0 MB, so the correction is 1.00" -- a machine reported as calibrated
            # against a correction nothing on it ever checked. Reported by review 2026-09-14.
            align._scale, align._scale_seen = 1.0, 0
            align._used_bytes = lambda before, peak: 0
            with said() as heard_zero:
                align._calibrate()
            check("a calibration that observed no new peak says so, and does not claim to "
                  "have calibrated",
                  heard_zero.mentions("took no measurement")
                  and not heard_zero.mentions("calibrated on this machine")
                  and align._scale_seen == 0,
                  heard_zero.records)
        finally:
            align._align = keep_align
            align._memory_now, align._used_bytes = keep_mem, keep_used
        # ⚠️ A card too full to warm on is a card that will refuse the first real call too.
        # Saying so beats refusing to start.
        align._scale, align._scale_seen = 1.0, 0
        try:
            align._align = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("CUDA out of memory."))
            with said() as heard_bad:
                align._calibrate()
            check("a calibration that cannot run is a warning, not a failed startup",
                  heard_bad.mentions("could not calibrate") and align._scale == 1.0,
                  heard_bad.records)
        finally:
            align._align = keep_align
    finally:
        (align._scale, align._scale_seen, align._calls_seen, align._state["model"],
         align._encoder_rate) = keep_sc

    with said() as heard:
        align._say_config()
    check("and the engine says its whole effective configuration in one line",
          heard.mentions("batching") and heard.mentions("groups sized at run time")
          # ⚠️ Not "as shipped" any more. This line runs before the checkpoint is open, so
          # the only thing it knows is the flag -- and since an explicit `use_cache: false`
          # stopped being overridden, claiming the checkpoint's own state here would be a
          # guess. What is really built is reported by the `use_cache was ...` line at load.
          and heard.mentions("KV cache not turned off here")
          and heard.mentions("% of what the grant leaves"), heard.records)
    check("and it is the only switch there is; the other two collapsed into it",
          not hasattr(align, "SORT_BY_LEN") and not hasattr(align, "NO_TEMPFILE"),
          [n for n in ("SORT_BY_LEN", "NO_TEMPFILE") if hasattr(align, n)])
    # ⚠️ And the promise stops there. Admission is on in both paths by design: a span
    # longer than the model aligns used to come back with units and HTTP 200, and now it
    # comes back refused. That is a deliberate change in what ships and it is not behind a
    # switch, so the assertion above must not be read as "nothing changed".
    check("admission is on without any switch, which is a change in what ships",
          align.MAX_SPAN_SEC > 0 and align.MAX_UNITS_A_SECOND > 0,
          (align.MAX_SPAN_SEC, align.MAX_UNITS_A_SECOND))
    align.BATCH = True
    align._budget = 0.0
    try:
        with TestClient(align.build_app(["align"])) as c:
            four = ('[{"start":0,"end":1,"text":"aa"},{"start":1,"end":2,"text":"bb"},'
                    '{"start":2,"end":3,"text":"cc"},{"start":3,"end":4,"text":"dd"}]')

            calls[:] = []
            got = c.post("/v1/audio/align", files=WAV, data={"segments": four}).json()
            res = got.get("results") or []
            check("every span comes back, in the order it was sent",
                  [r.get("units") and r["units"][0]["text"] for r in res] == ["hi"] * 4, res)
            # 🔴 With no dimensions to read -- the stub is a SimpleNamespace -- the opening
            # budget is zero and the ramp starts at one span, which is the fallback the real
            # engine takes only when it cannot read its own config.
            check("with nothing readable the first call carries one span",
                  calls and len(calls[0]) == 1, calls)
            # 🔴 `grouped` answers what HAPPENED, so it is pinned against the calls rather
            # than against a constant: --align-batch is on for this whole block, and the
            # answer still has to follow whether any call carried more than one span. It
            # does not always -- on the shipped 4 GiB aligner one 290 s span alone exceeds
            # the budget, so every group comes out single and a caller told "grouped" there
            # would believe its spans were combined when each still had its own call.
            check("grouped follows the calls, not the flag",
                  got.get("grouped") == any(len(c) > 1 for c in calls),
                  (align.BATCH, got.get("grouped"), [len(c) for c in calls]))
            # 🔴 And the case that gives the field its reason to exist: --align-batch still
            # on, two spans, and the ramp never gets to carry both -- so every call is a
            # single span and the honest answer is false. This is the shape the shipped
            # 4 GiB aligner is in for meeting-length spans, where one span alone exceeds the
            # budget. Without this case the assertion above passes on a hardcoded true.
            calls[:] = []
            # The ramp grew on the request above, so it is put back where the block set it.
            align._budget = 0.0
            two = '[{"start":0,"end":1,"text":"aa"},{"start":1,"end":2,"text":"bb"}]'
            solo = c.post("/v1/audio/align", files=WAV, data={"segments": two}).json()
            check("two spans that never share a call report grouped false",
                  all(len(x) == 1 for x in calls) and solo.get("grouped") is False,
                  (solo.get("grouped"), [len(x) for x in calls]))
            # 🔴 The third state, and the one both cases above miss: a call that DID carry
            # two spans, was refused by the card, and was split into singles that worked.
            # `_groups` keeps the refused group's record -- that is what `failed` is for --
            # so counting attempts reports `grouped` for a request in which every span ended
            # up in a call of its own. Both assertions above pass either way, because in
            # neither of them does a multi-span call get as far as being attempted.
            # ⚠️ It is the field the 4 GiB round read as evidence that nothing grouped.
            calls[:] = []
            keep_budget = align._budget
            align._budget = 10.0 ** 9          # big enough that both spans travel together
            oom = MemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            fail_on["aa"] = oom
            try:
                split = c.post("/v1/audio/align", files=WAV,
                               data={"segments": two}).json()
            finally:
                fail_on.clear()
            check("a group the card refused and split does not report itself grouped",
                  max(len(x) for x in calls) > 1            # it really was attempted as one
                  and split.get("grouped") is False,        # and no call carried two
                  (split.get("grouped"), [len(x) for x in calls]))
            align._budget = keep_budget        # this block's later cases read it

            check("a checkpoint whose dimensions are unreadable opens at zero",
                  align._opening_budget() == 0.0, align._opening_budget())
            # And with dimensions, the opening is a real number of positions rather than one
            # span: 6h + 2f + classes per position, against half of what the grant leaves.
            d = {"hidden": 1024, "ffn": 3072, "classes": 5000, "width": 2,
                 "downsample": 480, "window": 50}
            check("a position is priced from the model's own dimensions",
                  align._bytes_a_position(d)
                  == align.KAPPA_LM * (6 * 1024 + 2 * 3072 + 5000) * 2,
                  align._bytes_a_position(d))
            # 🔴 The band depends on which build is loaded, and that is the point: a
            # position goes from 143 KB to 28 KB when the cache comes off while a second of
            # audio barely moves, so the same second is worth about 56 positions with a cache
            # and about 300 without. A long span is five times dearer against a batch of short
            # ones in the second build, which is what moves the division. Checking both is
            # what stops one pair being read while the other is loaded.
            band = (20, 100) if not align.NO_KV_CACHE else (100, 600)
            check("a second of audio is worth what this build's pair says it is",
                  band[0] < align._bytes_a_second(d) / align._bytes_a_position(d) < band[1],
                  (align._bytes_a_second(d) / align._bytes_a_position(d), band))
            keep_k4 = kappa(align, align.KAPPA_NO_CACHE)
            try:
                check("and with the cache off the same second is worth hundreds",
                      100 < align._bytes_a_second(d) / align._bytes_a_position(d) < 600,
                      align._bytes_a_second(d) / align._bytes_a_position(d))
            finally:
                kappa_restore(align, keep_k4)
            check("the budget grew while the calls kept working", align._budget > 0,
                  align._budget)

            # Grown budget, same request: now they travel together.
            calls[:] = []
            c.post("/v1/audio/align", files=WAV, data={"segments": four})
            check("a grown budget groups the spans instead of sending singles",
                  max(len(x) for x in calls) > 1, calls)

            # 🔴 And the budget is not the only thing that has to be true. `_calibrate`
            # swallows its own failures and `ready` goes up either way, so an engine can
            # serve with a correction fitted on another machine -- the state this line's
            # own note records a 5x under-prediction from. Grouping against it is the
            # direction the card answers by refusing the call, so it waits.
            # ⚠️ Produced, not asserted: the gate reads `_scale_seen`, so the state has to
            # be the one an unweighed engine is actually in.
            keep_seen = align._scale_seen
            try:
                # ⚠️ The gate needs a cost model to exist before it has anything to
                # distrust. With `_dims()` unreadable there is none -- the ramp takes over,
                # which is safe without a measurement by construction -- so the gate
                # deliberately does not apply, and the stub reads None. Produce the state
                # the gate is actually about.
                keep_dims = align._dims
                align._dims = lambda: {"hidden": 1024, "ffn": 3072, "classes": 5000,
                                       "width": 2, "downsample": 480, "window": 50}
                align._scale_seen = 0
                calls[:] = []
                c.post("/v1/audio/align", files=WAV, data={"segments": four})
                check("an engine nothing has weighed sends one span a call",
                      max(len(x) for x in calls) == 1, calls)
                check("and it says which condition closed the group, not memory",
                      [g["closed_by"] for g in align._groups] == ["unmeasured"] * 4,
                      [g["closed_by"] for g in align._groups])
                # 🔴 A gate, not a switch: the span that went alone is also the
                # measurement, so grouping is back on the next call without a restart.
                # Without this the check above passes just as well for a permanent
                # disable, which is a different and much worse change.
                align._scale_seen = 1
                calls[:] = []
                c.post("/v1/audio/align", files=WAV, data={"segments": four})
                check("and grouping resumes once something has been weighed",
                      max(len(x) for x in calls) > 1, calls)
                # 🔴 The transition, inside ONE request. The two cases above set the
                # counter between requests, so neither of them steps over the moment a
                # measurement lands mid-request -- and that is where the gate was read one
                # call too early: from before the span that produces the measurement, so the
                # second span went alone as well and only the third grouped.
                align._scale_seen = 0
                calls[:] = []
                real_align = align._align

                def _measures_on_first_call(*a, **k):
                    align._scale_seen = 1          # what `_observe` does after a real call
                    align._align = real_align
                    return real_align(*a, **k)

                align._align = _measures_on_first_call
                try:
                    c.post("/v1/audio/align", files=WAV, data={"segments": four})
                finally:
                    align._align = real_align
                check("the span that measures is the only one that goes alone",
                      len(calls) >= 2 and len(calls[0]) == 1 and len(calls[1]) > 1,
                      [len(x) for x in calls])
                # 🔴 The row the first version of this gate shut off. `_observe` returns
                # early when the dimensions do not read, so `_scale_seen` can never leave
                # zero there -- gating on it alone disabled a documented path for the life
                # of the process, while `_effective_budget` still promised it in a warning.
                align._dims = keep_dims          # back to the stub's unreadable answer
                align._scale_seen = 0
                calls[:] = []
                c.post("/v1/audio/align", files=WAV, data={"segments": four})
                check("with no cost model at all the ramp still runs, gate or no gate",
                      max(len(x) for x in calls) > 1, calls)
            finally:
                align._dims = keep_dims
                align._scale_seen = keep_seen

            # 🔴 One bad span must not cost the others theirs.
            calls[:] = []
            fail_on.clear()
            fail_on["cc"] = RuntimeError("this span is broken")
            got = c.post("/v1/audio/align", files=WAV, data={"segments": four}).json()
            res = got.get("results") or []
            check("a failing span fails alone and the rest still align",
                  [bool(r.get("error")) for r in res] == [False, False, True, False], res)
            # Halving beats singles only above four spans, so the claim is checked where
            # it is actually made: eight spans, one bad, against the 1 + n singles cost.
            eight = "[" + ",".join('{"start":%.1f,"end":%.1f,"text":"s%d"}' % (i * 0.5, i * 0.5 + 0.5, i)
                                   for i in range(8)) + "]"
            calls[:] = []
            fail_on.clear()
            fail_on["s5"] = RuntimeError("this span is broken")
            align._budget = 100000.0
            got8 = c.post("/v1/audio/align", files=WAV, data={"segments": eight}).json()
            res8 = got8.get("results") or []
            check("in a group of eight only the bad span fails",
                  [bool(r.get("error")) for r in res8]
                  == [i == 5 for i in range(8)], res8)
            # ⚠️ It costs the failed call plus one a member. Halving would cost about
            # 2*log2(n); the isolation is the same and this has no recursion. The trade is
            # stated here so a later reader does not read the count as an accident.
            check("and it costs the failed call plus one a span, no recursion",
                  len(calls) == 1 + 8, len(calls))

            # 🔴 The two paths must bill the same request the same amount. A failing
            # span still cost the audio it read, and the batched path bills it; the serial
            # path metered after the call and outside the `finally`, so the same four-span
            # request with one failure cost four seconds batched and three serial --
            # the charge depending on a performance switch. The file already states this
            # rule for admission ("both paths refuse through the same function, or the two
            # would drift"); it was missing on the meter.
            four_one_bad = ('[{"start":0,"end":1,"text":"aa"},{"start":1,"end":2,"text":"bb"},'
                            '{"start":2,"end":3,"text":"boom"},{"start":3,"end":4,"text":"dd"}]')
            fail_on.clear()
            fail_on["boom"] = RuntimeError("this span is broken")
            billed = {}
            keep_batch = align.BATCH
            try:
                for mode in (True, False):
                    align.BATCH = mode
                    r = c.post("/v1/audio/align", files=WAV,
                               data={"segments": four_one_bad})
                    billed[mode] = r.headers.get("X-Audio-Input-Duration-Seconds")
            finally:
                align.BATCH = keep_batch
                fail_on.clear()
            check("both paths bill the same four seconds when one span of four fails",
                  billed[True] == billed[False]
                  and float(billed[True] or 0) == 4.0, billed)

            # 🔴 The same rule on the only failure a client can reach directly. The body
            # parser checks that segments is a JSON array and nothing about its elements,
            # so a start that is not a number, a bare string, or a null used to raise out
            # of the preparation loop and answer 500 for the whole request -- while the
            # identical request with batching off returned 200 and one error object. Four
            # shapes, because they fail at four different lines.
            for label, body in (
                    ("a start that is not a number",
                     '[{"start":0,"end":1,"text":"aa"},{"start":"x","end":2,"text":"bb"}]'),
                    ("an element that is a bare string",
                     '[{"start":0,"end":1,"text":"aa"},"hello"]'),
                    ("an element that is null",
                     '[{"start":0,"end":1,"text":"aa"},null]'),
                    ("a start that is not finite",
                     '[{"start":0,"end":1,"text":"aa"},{"start":1e400,"end":2,"text":"bb"}]')):
                r = c.post("/v1/audio/align", files=WAV, data={"segments": body})
                rows = (r.json() or {}).get("results") or []
                check("%s fails alone rather than failing the request" % label,
                      r.status_code == 200 and len(rows) == 2
                      and not (rows[0] or {}).get("error")
                      and bool((rows[1] or {}).get("error")),
                      (r.status_code, rows))

            # 🔴 A group that runs out of memory is split and retried, and the caller is
            # not told: every member is innocent, the problem is how many rode together, and
            # a prediction that was wrong about this machine is ours to find rather than
            # theirs to handle. What must NOT happen is the process remembering it.
            fail_on.clear()
            align._budget = 10000.0
            sizes = []

            def oom_group(audio=None, text=None, language=None):
                texts = text if isinstance(text, list) else [text]
                sizes.append(len(texts))
                if len(texts) > 2:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return [[unit] for _ in texts]

            model.align = oom_group
            keep_budget_oom = align._budget
            with said() as heard_oom:
                r_oom = c.post("/v1/audio/align", files=WAV, data={"segments": four})
            res_oom = (r_oom.json().get("results") or []) if r_oom.status_code == 200 else []
            check("a group the card refuses is split, and every span still comes back",
                  r_oom.status_code == 200 and len(res_oom) == 4
                  and all(not (x or {}).get("error") for x in res_oom),
                  (r_oom.status_code, res_oom))
            check("and it halved rather than going one at a time, so the call count is log-ish",
                  sizes == [4, 2, 2], sizes)
            # 🔴 The budget is untouched. This is the whole difference from the version that
            # recovered and then ran at a tenth of the throughput for the life of the process.
            check("and nothing is remembered: the budget stands and no ceiling appears",
                  align._budget == keep_budget_oom and not hasattr(align, "_ceiling"),
                  (align._budget, keep_budget_oom, getattr(align, "_ceiling", "absent")))
            # ⚠️ Not told to the caller, but not silent either: we need to know when our own
            # prediction is wrong about a machine, and the log and the telemetry are where.
            check("the caller is not told, but the log says it and names the knob",
                  heard_oom.mentions("out of memory on a call carrying")
                  and heard_oom.mentions("--gpu-budget-fraction"), heard_oom.records)
            check("and the telemetry counts it, so how often can be read rather than guessed",
                  align._telemetry[-1]["ooms"] == 1, align._telemetry[-1].get("ooms"))
            # 🔴 A span that fails ALONE is that span not fitting, which retrying cannot
            # change. It comes back as that span's error and the other three still align.
            # A budget this small makes every group a single span, which is the state this
            # is about; with a large budget the four ride together and the path never runs
            # -- the first version of this check passed for that reason.
            align._budget = 1.0

            def oom_alone(audio=None, text=None, language=None):
                texts = text if isinstance(text, list) else [text]
                if len(texts) == 1:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return [[unit] for _ in texts]

            model.align = oom_alone
            alone = c.post("/v1/audio/align", files=WAV, data={"segments": four})
            check("a span that fails on its own is that span's error, inside a 200",
                  alone.status_code == 200
                  and all((r or {}).get("error")
                          for r in (alone.json().get("results") or [{}])),
                  (alone.status_code, alone.json().get("results")))
            model.align = fake_align

            # 🔴 The rule that decides a group, and the reason it gives for closing one.
            # Memory and padding produce the same wall clock and call for opposite changes,
            # so a run that cannot tell them apart cannot be read at all.
            keep_slack = align.ALIGN_GROUP_SLACK
            check("the padding rule ships on, at the figure the sweep measured",
                  keep_slack == 50, keep_slack)
            align._budget = 10 ** 9
            try:
                # 🔴 The default is 50 and it is measured, so the off case has to be set
                # explicitly here: a test that inherited the default would stop testing
                # "off" the day the default moved, and would pass either way.
                align.ALIGN_GROUP_SLACK = 0
                c.post("/v1/audio/align", files=WAV, data={"segments": four})
                check("with the padding rule off a generous budget sends one call",
                      len(align._groups) == 1 and align._groups[0]["closed_by"] == "no more spans",
                      align._groups)
                # Four spans of equal length whose TEXT climbs, so their positions climb
                # and each addition re-pads what is already there. Equal durations on
                # purpose: it is the text half of a span's length that a seconds-based rule
                # would have missed, which is why this rule counts positions.
                climbing = ('[{"start":0,"end":1,"text":"aa"},'
                            '{"start":1,"end":2,"text":"aa bb"},'
                            '{"start":2,"end":3,"text":"aa bb cc"},'
                            '{"start":3,"end":4,"text":"aa bb cc dd"}]')
                align.ALIGN_GROUP_SLACK = 1
                c.post("/v1/audio/align", files=WAV, data={"segments": climbing})
                # The last group has no successor to refuse, so it closes on running out
                # rather than on padding -- distinguishing the two is the whole point of
                # recording a reason, so the check states it instead of accepting either.
                check("a slack of one position closes each group on padding, the last on "
                      "running out of spans",
                      [g["closed_by"] for g in align._groups]
                      == ["padding", "padding", "padding", "no more spans"],
                      align._groups)
                check("and each of those groups still carries exactly one span",
                      all(g["spans"] == 1 for g in align._groups), align._groups)
                # 🔴 The count is the half that makes the rule self-limiting: the same
                # small step costs more the more spans are already paying it. These four
                # climb by three positions each, so at a slack of seven the walk takes two
                # steps (3, then 6) and refuses the third (9) -- a rule that priced only
                # the step would take all four and never close.
                align.ALIGN_GROUP_SLACK = 7
                c.post("/v1/audio/align", files=WAV, data={"segments": climbing})
                check("the re-padding is charged to every span already in the group",
                      [g["spans"] for g in align._groups] == [3, 1], align._groups)

                # 🔴 A fixed budget stops the engine re-deciding between rounds, which is
                # what a sweep needs. It is now the whole answer: nothing lowers it any
                # more, because nothing remembers a wall.
                keep_pin = align.ALIGN_FIXED_BUDGET
                try:
                    align.ALIGN_FIXED_BUDGET, align._budget = 4242.0, 0.0
                    check("a fixed budget is the budget, without solving for one",
                          align._effective_budget() == 4242.0, align._effective_budget())
                    check("and it stays it, because no wall is remembered",
                          align._effective_budget() == 4242.0, align._effective_budget())
                finally:
                    align.ALIGN_FIXED_BUDGET = keep_pin
                    align._budget = 10 ** 9

                align.ALIGN_GROUP_SLACK = 10 ** 9
                c.post("/v1/audio/align", files=WAV, data={"segments": climbing})
                check("a slack larger than any re-padding leaves the grouping to memory",
                      len(align._groups) == 1, align._groups)

                # 🔴 The rule prices an ascending walk, so it brings the ordering with it.
                # This was once a second switch an operator could set the wrong way, and
                # that combination degraded silently in whichever direction the rule
                # happened to be wrong: one enormous over-padded group before both
                # directions were priced, one span a call after. The switch is gone; the
                # behaviour it used to be able to break is what this checks. Sorting is
                # internal -- results are written back by original index.
                try:
                    align.ALIGN_GROUP_SLACK = 50
                    align._budget = 10 ** 9
                    falling_body = ("[" + ",".join(
                        '{"start":%d,"end":%d,"text":"%s"}' % (i, i + 1, "aa " * n)
                        for i, n in enumerate([9, 1, 1, 1])) + "]")
                    r = c.post("/v1/audio/align", files=WAV,
                               data={"segments": falling_body}).json()
                    check("the rule brings its own ordering, so the longest span does not "
                          "drag three short ones into its padding",
                          len(align._groups) == 2
                          and [g["spans"] for g in align._groups] == [3, 1],
                          [(g["spans"], g["closed_by"]) for g in align._groups])
                    check("and every span still comes back against its own index",
                          len(r.get("results") or []) == 4
                          and all(not (x or {}).get("error")
                                  for x in (r.get("results") or [])),
                          r.get("results"))
                    align.ALIGN_GROUP_SLACK = 0
                    c.post("/v1/audio/align", files=WAV, data={"segments": falling_body})
                    check("with the rule off the order is left alone and memory decides",
                          len(align._groups) == 1, align._groups)

                    # 🔴 A budget of zero is the ramp's opening state on a machine with no
                    # memory authority, not the card refusing. Every process's first group
                    # hits it, and calling that `memory` puts a reading in the record that
                    # claims the card declined a call nobody offered it.
                    align._budget = 0.0
                    align.ALIGN_GROUP_SLACK = 50
                    spans4 = [align._Span(i, None, "x", "auto", 1.0, 100) for i in range(4)]
                    keep_priced = align._budget_priced
                    try:
                        align._budget_priced = False
                        g, _, why = align._next_group(spans4, list(range(4)), 0)
                        check("a group closed with no budget yet says ramping, not memory",
                              (len(g), why) == (1, "ramping"), (len(g), why))
                        # 🔴 The other zero. A grant that was read and is spoken for prices
                        # at 0.0, and that IS the card refusing -- keying the label off
                        # `budget > 0` merged the two states one layer below the place
                        # `_solve_budget` exists to separate them, and merged them for as
                        # long as the quota stayed full rather than for one group.
                        # `_effective_budget` re-derives the flag, which is right -- so
                        # the state has to be produced, not asserted: a grant that reads
                        # and has nothing left.
                        keep_solve = align._solve_budget
                        try:
                            align._solve_budget = lambda: 0.0
                            align._budget = 0.0
                            g2, _, why2 = align._next_group(spans4, list(range(4)), 0)
                        finally:
                            align._solve_budget = keep_solve
                        # 🔴 And the label must survive the ramp. Once doubling has
                        # produced a positive budget, a test on the budget's value passes
                        # and the flag is never read -- so the first group said `ramping`
                        # and every group after it went back to blaming a card nobody had
                        # asked, which is most of the groups in a request.
                        align._budget_priced = False
                        align._budget = 10.0 ** 9
                        g3, _, why3 = align._next_group(
                            spans4 + [align._Span(9, None, "x", "auto", 1.0, 10 ** 9)],
                            list(range(5)), 0)
                        check("a group closed on a ramped budget still says ramping",
                              why3 == "ramping", (len(g3), why3))
                        # ⚠️ There used to be a second half to this: a wall that had been
                        # hit also forced `memory`, because the out-of-memory path returned
                        # from inside `except` and never reached the line that prices the
                        # budget. Nothing remembers a wall now, so the flag is the whole
                        # answer and the two labels mean exactly what they say.
                        align._budget_priced = True
                        g4, _, why4 = align._next_group(
                            spans4 + [align._Span(9, None, "x", "auto", 1.0, 10 ** 9)],
                            list(range(5)), 0)
                        check("and a priced budget that binds says memory",
                              why4 == "memory", (len(g4), why4))
                        # 🔴 The invariant the three-writers-one-reader arrangement rests
                        # on, which was holding by accident until it was written down: the
                        # reader refreshes the flag before reading it. Pinned by making the
                        # flag stale on purpose and checking the answer is still right --
                        # move the `_effective_budget()` call out of `_next_group`, or read
                        # the flag ahead of it, and this goes red instead of going quiet.
                        keep_solve2 = align._solve_budget
                        try:
                            align._budget_priced = True      # stale, and wrong
                            align._budget = 0.0
                            align._solve_budget = lambda: None
                            g5, _, why5 = align._next_group(spans4, list(range(4)), 0)
                            check("a stale flag is refreshed by the read path, not trusted",
                                  why5 == "ramping", (why5, align._budget_priced))
                        finally:
                            align._solve_budget = keep_solve2
                        check("a zero budget that was actually priced still says memory",
                              (len(g2), why2) == (1, "memory"), (len(g2), why2))
                    finally:
                        align._budget_priced = keep_priced
                    align._budget = 10 ** 9
                finally:
                    align.ALIGN_GROUP_SLACK = 50

                # 🔴 The rule has to work on the order it is given. Charging only "everyone
                # already here gets re-padded" reads zero for every candidate shorter than
                # the group's longest -- so on unsorted input the rule closed nothing and
                # reported "no more spans" while the call was padded to several times its
                # real work. It is reached only by calling _next_group directly now that the
                # walk is always sorted, and it is kept because that is one edit away.
                # Descending is the worst case and the cheapest to state.
                Span = align._Span
                falling = [Span(i, None, "x", "auto", 1.0, p)
                           for i, p in enumerate([600] + [20] * 8)]
                align.ALIGN_GROUP_SLACK = 50
                keep_budget = align._budget
                align._budget = 10 ** 9
                try:
                    g, at, why = align._next_group(falling, list(range(len(falling))), 0)
                    check("a candidate shorter than the group's longest is charged the "
                          "padding it will be given",
                          (len(g), why) == (1, "padding"), (len(g), why))
                    # And the term is zero whenever the order is ascending, which is what
                    # keeps every measured figure in the record valid.
                    rising = [Span(i, None, "x", "auto", 1.0, p)
                              for i, p in enumerate([20] * 8 + [600])]
                    g2, _, why2 = align._next_group(rising, list(range(len(rising))), 0)
                    check("and ascending order is unaffected, so the sweep still holds",
                          (len(g2), why2) == (8, "padding"), (len(g2), why2))
                finally:
                    align._budget = keep_budget
            finally:
                align.ALIGN_GROUP_SLACK = keep_slack

            # 🔴 An answer that does not match the question must not be believed. zip
            # stops at the shorter side, so a library returning fewer results leaves spans
            # null while the group reports itself finished, and one returning a flat list
            # of items gives every span the same wrong answer with HTTP 200.
            calls[:] = []
            fail_on.clear()
            align._budget = 100000.0
            def one_only(audio=None, text=None, language=None):
                calls.append(list(text if isinstance(text, list) else [text]))
                return [[unit]]

            model.align = one_only
            short = c.post("/v1/audio/align", files=WAV, data={"segments": four}).json()
            rs = short.get("results") or []
            check("a short answer never reaches the caller as a null or a wrong result",
                  len(rs) == 4 and all(r and r.get("units") and not r.get("error")
                                       for r in rs), rs)
            # 🔴 And the recovery is the useful part: a library that only ever answers one
            # span degrades into one call a span instead of handing back four copies of the
            # first answer, because the mismatch fails the group and halving reaches singles.
            check("a library that answers one at a time is driven one at a time",
                  calls and max(len(k) for k in calls) > 1
                  and [len(k) for k in calls][-1] == 1, calls)
            # With nothing returned at all even a single span fails, and says what it got.
            def none_at_all(audio=None, text=None, language=None):
                return []

            model.align = none_at_all
            none_back = c.post("/v1/audio/align", files=WAV,
                               data={"segments": '[{"start":0,"end":1,"text":"aa"}]'}).json()
            check("an empty answer is reported against what was asked",
                  "results for" in ((none_back.get("results") or [{}])[0].get("error") or ""),
                  none_back.get("results"))
            model.align = fake_align

            # Metering is additive, so a halved retry must not bill a span twice.
            billed = []
            align._budget = 100000.0
            fail_on.clear()
            fail_on["cc"] = RuntimeError("this span is broken")
            # 🔴 The double bill needs _call to fail *partway through writing results*, so
            # that spans already billed are billed again by the retry. A span failing before
            # the call never reaches the meter, which is why the obvious test misses it.
            def breaks_third(audio=None, text=None, language=None):
                texts = text if isinstance(text, list) else [text]
                calls.append(list(texts))
                # An item _units cannot iterate: the spans before it are written and billed,
                # then this raises out of the middle of the loop.
                return [[unit] if t != "cc" else 12345 for t in texts]

            real_meter = align._bill

            def watched(ctx, m):
                charged = real_meter(ctx, m)
                if charged:
                    billed.append(m.index)
                return charged

            align._bill = watched
            model.align = breaks_third
            try:
                got = c.post("/v1/audio/align", files=WAV,
                             data={"segments": four}).json()
                check("a result the reader cannot walk fails its own span, not the batch",
                      [bool(r.get("error")) for r in (got.get("results") or [])]
                      == [False, False, True, False], got.get("results"))
                check("no span is billed twice when the group is halved and retried",
                      sorted(billed) == [0, 1, 2, 3], billed)
            finally:
                align._bill = real_meter
                model.align = fake_align
            fail_on.clear()

            # A script written without spaces that the unit split does not break up would
            # otherwise be one unit, and the guard would read 1 against 48.
            check("a long unspaced run is counted by its characters, not as one unit",
                  align._dense(2.0, "\u3042" * 3000) > 0,
                  align._dense(2.0, "\u3042" * 3000))
            check("an ordinary long word is still one unit",
                  align._dense(2.0, "internationalisation") == 0,
                  align._dense(2.0, "internationalisation"))

            # 🔴 The library deletes punctuation before it splits, so punctuation never
            # becomes a unit and never separates its neighbours. Counting it doubled this
            # sentence -- 8 units against the library's 4 -- which priced its memory at
            # twice the cost and made the density guard twice as strict as it reads. The
            # case is Chinese on purpose: in English the punctuation sits inside a
            # whitespace-delimited word and the count comes out right by accident.
            check("punctuation is deleted before the split, not counted as a unit",
                  align._units_of("\u554a\u2026\u2026\u5bf9\uff0c\u5bf9\uff0c\u5bf9\u3002")
                  == ["\u554a", "\u5bf9", "\u5bf9", "\u5bf9"],
                  align._units_of("\u554a\u2026\u2026\u5bf9\uff0c\u5bf9\uff0c\u5bf9\u3002"))
            check("and a run of punctuation does not keep its neighbours apart",
                  align._units_of("a.b.c") == ["abc"], align._units_of("a.b.c"))
            check("text that is only punctuation has no units at all",
                  align._units_of("\u2026\u2026!!") == [], align._units_of("\u2026\u2026!!"))
            # The apostrophe is the library's one exception to "letters and numbers only",
            # and without it every contraction splits in two.
            check("an apostrophe is kept, so a contraction stays one unit",
                  align._units_of("don't stop") == ["don't", "stop"],
                  align._units_of("don't stop"))

            # 🔴 The library picks its tokeniser by language name and the name comes
            # from the caller -- every segment carries its own, and nothing validates it.
            # Japanese and Korean run nagisa and soynlp, which cut running kana and hangul
            # into words, while the split above leaves a kana sentence as ONE unit: the
            # span would be priced at a fraction of its cost and the density guard would
            # wave it through. Counting a character a unit is an upper bound, wrong in the
            # direction that costs a call rather than an out-of-memory.
            kana = "\u3053\u3093\u306b\u3061\u306f\u307f\u306a\u3055\u3093"
            check("a kana sentence is one unit on the space path, which is the trap",
                  len(align._units_of(kana)) == 1, align._units_of(kana))
            check("and is counted a character a unit when the caller says japanese",
                  len(align._units_of(kana, "Japanese")) == len(kana),
                  align._units_of(kana, "Japanese"))
            check("the density note therefore sees a kana span the space path misses",
                  align._dense(0.2, kana, "japanese") > 0 and align._dense(0.2, kana) == 0,
                  (align._dense(0.2, kana, "japanese"), align._dense(0.2, kana)))
            # \U0001f534 And it is a note, not a refusal: admission turns on duration alone,
            # because that is the one the model cannot answer past. A span nothing can fit
            # fails on its own and comes back as an error anyway; refusing it early would buy
            # one call and risk losing the words of a span that would have aligned.
            check("and density alone never refuses a span",
                  align._admit(0.2, kana, "japanese") is None,
                  align._admit(0.2, kana, "japanese"))
            check("and the language name is matched however it is capitalised",
                  align._units_of(kana, "KOREAN") == align._units_of(kana, "korean"),
                  align._units_of(kana, "KOREAN"))

            # ---- the span ceiling, re-derived from whatever checkpoint loaded -----------
            # 🔴 The figure this build was written against is the shipped model's, worked
            # out by hand. A model upgrade invalidates it without failing anything: a head
            # with fewer classes or a coarser grid still answers 200 and saturates its
            # timestamps, which is the one outcome the ceiling exists to prevent.
            span_was = (align.EXPRESSIBLE_SPAN_SEC, align.MAX_SPAN_SEC)
            try:
                same = {"classes": 5000, "grid_ms": 80.0}
                check("the shipped checkpoint re-derives the figure it was written against",
                      align._expressible_span_sec(same) == span_was[0],
                      (align._expressible_span_sec(same), span_was[0]))
                with said() as heard_same:
                    align._rebind_span_limit(same)
                check("so nothing is rebound and nothing is said",
                      (align.EXPRESSIBLE_SPAN_SEC, align.MAX_SPAN_SEC) == span_was
                      and not heard_same.mentions("output head expresses"),
                      (align.EXPRESSIBLE_SPAN_SEC, align.MAX_SPAN_SEC, heard_same.records))

                # 🔴 This is the case that makes the pair worth having. With only the one
                # above, deleting the whole rebinding leaves both of them green: a function
                # that does nothing passes "it changed nothing" perfectly.
                with said() as heard_diff:
                    align._rebind_span_limit({"classes": 2500, "grid_ms": 80.0})
                check("half the classes halves what the head can express",
                      align.EXPRESSIBLE_SPAN_SEC == 200.0, align.EXPRESSIBLE_SPAN_SEC)
                check("and the limit follows it down",
                      align.MAX_SPAN_SEC == min(align._asked_span_sec, 200.0),
                      align.MAX_SPAN_SEC)
                check("and it says so, because that is the model having changed",
                      heard_diff.mentions("output head expresses", "200.0"),
                      heard_diff.records)
                # The shipped default asks for 300, which a 200 s head cannot give. The
                # import-time warning compared against the old figure, so without this the
                # clamp happens with nothing having mentioned it.
                check("and warns when the deployment asked for more than that",
                      align._asked_span_sec <= 200.0
                      or heard_diff.mentions("--max-span-seconds", "can express"),
                      heard_diff.records)

                # 🔴 The visible consequence, pinned on purpose. The configuration line is
                # printed before the model loads, so after a rebind the startup log says one
                # number and the refusals say another. On the shipped model they agree; a
                # model that moved is exactly when they should not, and somebody reading the
                # log needs that disagreement to be real rather than tidied away.
                with said() as heard_cfg:
                    align._say_config()
                check("the configuration line still quotes the value it was called with",
                      heard_cfg.mentions("longest span 200s"), heard_cfg.records)

                # Read from a checkpoint that does not say, the ceiling stays where it is
                # rather than disappearing: no grid means no derivation, not an unlimited
                # span. 🔴 Restored first on purpose -- the fallback returns whatever the
                # global currently holds, and the check above just moved it, so asking
                # without restoring tests the wrong baseline.
                # 🔴 Everything above calls the rebinding directly, which leaves the one
                # thing that makes it matter untested: that loading a checkpoint actually
                # calls it. `_load` needs torch and real weights, so it cannot be run
                # here, and deleting its one call would leave every check above green
                # while the ceiling never moved on any real machine. Read from the source
                # instead -- a coarse check, and coarse beats absent.
                import inspect as _inspect
                _src = _inspect.getsource(align._load)
                check("loading a checkpoint rebinds the ceiling, after the dims are read",
                      "_rebind_span_limit(" in _src
                      and _src.index("_dims()") < _src.index("_rebind_span_limit("),
                      _src[:0])

                align.EXPRESSIBLE_SPAN_SEC, align.MAX_SPAN_SEC = span_was
                for blind in ({"classes": 5000, "grid_ms": None}, {"classes": 0}, None, {}):
                    check("a checkpoint that does not say keeps the shipped figure",
                          align._expressible_span_sec(blind) == span_was[0], blind)
            finally:
                align.EXPRESSIBLE_SPAN_SEC, align.MAX_SPAN_SEC = span_was

            # ---- the correction that outlives the library ------------------------------
            # 🔴 What makes the model survive the algorithm changing is that it checks
            # itself on every call. These guard the checking, not the model.
            keep = (align._scale, align._scale_seen)
            try:
                align._scale, align._scale_seen = 1.0, 0
                # grid_ms is None because these stubs carry no timestamp_segment_time,
                # which is the case the span ceiling falls back on rather than deriving.
                # 🔴 The key is here because `_dims` now returns it, not because the check
                # needed loosening: an exact-equality test has to name every field, and a
                # field was added. Anyone reading this as a test bent to fit the code has
                # it backwards.
                d2 = {"hidden": 1024, "ffn": 3072, "classes": 5000, "width": 2,
                      "grid_ms": None, "downsample": 480, "window": 50}
                perpos = align._bytes_a_position(d2)
                # A stub has no dimensions, so _observe has nothing to divide by and must
                # leave the correction alone rather than invent one.
                align._observe(100, int(100 * perpos * 3))
                check("with no dimensions to read the correction does not move",
                      (align._scale, align._scale_seen) == (1.0, 0),
                      (align._scale, align._scale_seen))

                align._state["model"] = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(
                            text_config=types.SimpleNamespace(hidden_size=1024,
                                                              intermediate_size=3072),
                            audio_config=types.SimpleNamespace(downsample_hidden_size=480,
                                                               n_window=50),
                            classify_num=5000),
                        parameters=lambda: iter(())))
                check("dimensions come from the checkpoint, not from the class defaults",
                      align._dims() == d2, align._dims())
                # 🔴 The shape the real checkpoint has: AutoModel loads the top-level
                # config and the three fields hang off its thinker. The stub above was
                # the flat shape, so the reader and its test agreed with each other and
                # not with the model -- on the machine, _dims() answered None and the
                # budget quietly fell back to doubling.
                flat = align._state["model"].model.config
                align._state["model"] = types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(thinker_config=flat),
                        parameters=lambda: iter(())))
                check("dimensions are found when the checkpoint nests them under thinker",
                      align._dims() == d2, align._dims())

                # 🔴 The grid is not where the other three are. Upstream reads
                # `model.config.timestamp_segment_time` off the TOP-level config while
                # these hang off the thinker, so a reader that only looks where
                # `classify_num` is finds nothing on the real checkpoint -- and finding
                # nothing is silent, because the ceiling then keeps its shipped value and
                # every span under it behaves exactly as before.
                nested = align._state["model"].model.config
                nested.timestamp_segment_time = 80.0
                check("the timestamp grid is read from the top-level config",
                      (align._dims() or {}).get("grid_ms") == 80.0, align._dims())
                del nested.timestamp_segment_time
                flat.timestamp_segment_time = 40.0
                check("and from the thinker when that is where it sits",
                      (align._dims() or {}).get("grid_ms") == 40.0, align._dims())
                del flat.timestamp_segment_time
                check("and a checkpoint with neither reads None rather than failing",
                      align._dims() == d2, align._dims())
                # 🔴 Asked before the model is up, the encoder rate answers with a fallback
                # 1.9x under this checkpoint's real figure. Caching that answer sets the
                # price of every long span for the life of the process from one early call.
                rate, align._encoder_rate = align._encoder_rate, None
                model_stub, align._state["model"] = align._state["model"], None
                try:
                    fallback = align._encoder_positions_a_second()
                    align._state["model"] = model_stub
                    check("the encoder rate's fallback is not kept once the model can be read",
                          align._encoder_positions_a_second() != fallback,
                          (fallback, align._encoder_positions_a_second()))
                finally:
                    align._state["model"] = model_stub
                    align._encoder_rate = rate
                align._scale, align._scale_seen = 1.0, 0
                for _ in range(align.SCALE_MEMORY * 3):
                    align._observe(100, int(100 * perpos * 3))
                check("a call costing three times its prediction pulls the correction to 3",
                      2.5 < align._scale < 3.2, align._scale)

                # 🔴 What the headroom is measured against, on a machine with no card:
                # the three cases differ by which authority exists, and the wrong one is
                # not a smaller number but somebody else's memory counted as ours.
                fake = types.ModuleType("torch")
                fake.cuda = types.SimpleNamespace(
                    memory_reserved=lambda: 3 * 1024 ** 3,
                    memory_allocated=lambda: 2 * 1024 ** 3,
                    mem_get_info=lambda: (5 * 1024 ** 3, 13 * 1024 ** 3))
                keep_torch = sys.modules.get("torch")
                keep_env = (os.environ.get("CUDA_DEVICE_MEMORY_LIMIT_0"),
                            os.environ.get("REQUIRED_GPU_MEMORY"))
                sys.modules["torch"] = fake
                try:
                    os.environ.pop("REQUIRED_GPU_MEMORY", None)
                    os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = "13312m"
                    check("under an enforced limit the headroom is what HAMi will still "
                          "give plus what the allocator already owns",
                          align._headroom_bytes() == 6 * 1024 ** 3, align._headroom_bytes())
                    # 🔴 A published limit whose counters were never rewritten: the device
                    # still reports the whole card, so its free figure is somebody else's
                    # memory. Told apart by the total, which the rewrite clamps to the
                    # limit. The first version of this block asserted the rewritten case
                    # against code that never read the total -- the same shape as the bug
                    # these three commits started from.
                    fake.cuda.mem_get_info = lambda: (30 * 1024 ** 3, 40 * 1024 ** 3)
                    align._no_grant_logged = True
                    check("a published limit with the card's own counters falls back to the "
                          "limit, not to what the card calls free",
                          align._headroom_bytes() == 11 * 1024 ** 3, align._headroom_bytes())
                    fake.cuda.mem_get_info = lambda: (5 * 1024 ** 3, 13 * 1024 ** 3)
                    os.environ["REQUIRED_GPU_MEMORY"] = "4Gi"
                    check("an enforced limit outranks a declared one rather than being "
                          "taken with it",
                          align._headroom_bytes() == 6 * 1024 ** 3, align._headroom_bytes())
                    # 🔴 HAMi's own parser is the authority for this variable, so these
                    # cases are read off `get_limit_from_env`: last single character for
                    # G/g, M/m, K/k, then a leading integer. Anything else is not a unit.
                    # Reading `4Gi` as 4 GiB the way a Kubernetes table would is wrong by
                    # 2^30 in the direction that over-sizes batches, and silently.
                    for raw, want in (("4096m", 4 * 1024 ** 3), ("4g", 4 * 1024 ** 3),
                                      ("500k", 500 * 1024), ("13312m", 13312 * 1024 ** 2),
                                      ("4Gi", 4), ("4GiB", 4), ("4096MiB", 4096),
                                      # 🔴 HAMi indexes the last character of the raw
                                      # variable and its own file loader trims only a
                                      # newline, so a trailing space survives and is not
                                      # a unit. Stripping first reads this as 4 GiB: a
                                      # factor of 2^20 the over-sizing way, silently.
                                      ("4096m ", 4096), (" 4096m", 4 * 1024 ** 3),
                                      ("-5g", 0), ("nonsense", 0), ("inf", 0)):
                        os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = raw
                        check("HAMi reads %r as %d bytes, and so does this" % (raw, want),
                              align._hami_limit_bytes() == want,
                              align._hami_limit_bytes())
                    # 🔴 Zero is a value, not a failure: hami-core stores it and skips every
                    # check against it, and one of this line's machines is set to `0m`. A
                    # version that warned here sent whoever saw it hunting a typo that did
                    # not exist.
                    # 🔴 Tested through the log, not the return value. Zero and an
                    # unreadable string both return zero -- that is the whole point of the
                    # distinction -- so a check on the number passes either way, and did:
                    # reverting this fix left all 886 checks green.
                    for zero in ("0", "0m"):
                        align._hami_unreadable_logged = False
                        os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = zero
                        with said() as heard:
                            got = align._hami_limit_bytes()
                        check("a limit of %r is HAMi's way of saying no limit, quietly"
                              % zero,
                              got == 0 and not heard.mentions("CUDA_DEVICE_MEMORY_LIMIT_0"),
                              (got, heard.records))
                    # 🔴 `str.isdigit()` is true for superscripts and for every other
                    # script's digits, and `int()` refuses half of them: `"²m"` raised out
                    # of a function every caller expects to return a number, which is an
                    # invariant an earlier version wrote down and the rewrite dropped.
                    for odd in ("\u00b2m", "\u0664\u0660\u0669\u0666m", "\u00bdg"):
                        align._hami_unreadable_logged = False
                        os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = odd
                        check("a digit that is not 0-9 reads as no limit, and does not "
                              "raise (%r)" % odd,
                              align._hami_limit_bytes() == 0, odd)
                    align._hami_unreadable_logged = False
                    os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] = "nonsense"
                    with said() as heard:
                        got = align._hami_limit_bytes()
                    check("and a string with no number in it is zero AND says so",
                          got == 0 and heard.mentions("CUDA_DEVICE_MEMORY_LIMIT_0",
                                                      "no number at all"),
                          (got, heard.records))
                    align._hami_unreadable_logged = False
                    os.environ.pop("CUDA_DEVICE_MEMORY_LIMIT_0")
                    check("with only a declared figure the headroom is that minus what is held",
                          align._headroom_bytes() == 2 * 1024 ** 3, align._headroom_bytes())
                    os.environ.pop("REQUIRED_GPU_MEMORY")
                    # 🔴 None, not zero, and the difference decides whether batching
                    # happens at all. Zero is an answer -- "a grant was read and it has
                    # nothing left" -- so the solver returns 0.0, the caller takes it, and
                    # the ramp that exists for exactly this machine never runs: every group
                    # comes out one span closed by `memory` on a box where no memory
                    # authority was ever found.
                    check("with no authority at all the headroom is None, which is not the "
                          "same answer as zero",
                          align._headroom_bytes() is None, align._headroom_bytes())
                    check("and the budget then solves to None, so the ramp takes over "
                          "instead of the budget standing at zero",
                          align._solve_budget() is None, align._solve_budget())
                finally:
                    if keep_torch is None:
                        sys.modules.pop("torch", None)
                    else:
                        sys.modules["torch"] = keep_torch
                    for name, value in zip(("CUDA_DEVICE_MEMORY_LIMIT_0",
                                            "REQUIRED_GPU_MEMORY"), keep_env):
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value

                # 🔴 Two different zeros. Unreadable is None and hands back to the ramp;
                # a readable grant with nothing left is 0.0 and must shrink the batch, not
                # grow it. One value for both grew the batch when memory was tightest.
                check("an unreadable grant is None, not zero",
                      align._solve_budget() is None, align._solve_budget())
                # 🔴 "There is no cost model" has to be audible wherever the budget came
                # from. It used to be announced from the place that computes an opening
                # budget, and ALIGN_FIXED_BUDGET returns before that -- so the one
                # configuration a sweep runs in was the one where the model going dark
                # printed nothing.
                keep_dims, keep_flag = align._state.get("model"), align._dims_unreadable
                keep_fixed2 = align.ALIGN_FIXED_BUDGET
                try:
                    # 🔴 A model that IS loaded and whose dimensions will not read, which is
                    # the condition this warning is for. It used to be simulated with
                    # `model = None`, and that state cannot occur after a load -- `_state`
                    # takes the model once and nothing ever puts it back -- so None means
                    # "not loaded yet", which is not a fault and is now not warned about.
                    # The route at /v1/audio/align/telemetry has no readiness gate, so a
                    # monitor scraping during startup would otherwise spend this one-shot
                    # flag and the real condition would arrive to silence.
                    align._state["model"] = object()
                    align.ALIGN_FIXED_BUDGET, align._dims_unreadable = 8800.0, False
                    with said() as heard:
                        align._dims()
                    check("an unreadable checkpoint says there is no cost model, even "
                          "with the budget pinned",
                          heard.mentions("no cost model"), heard.records)
                    align._dims_unreadable = False
                    with said() as heard2:
                        align._dims()
                        align._dims()
                    check("and says it once, not once a call",
                          len([m for m in heard2.records if "no cost model" in m]) <= 1,
                          heard2.records)
                finally:
                    align._state["model"] = keep_dims
                    align.ALIGN_FIXED_BUDGET = keep_fixed2
                    align._dims_unreadable = keep_flag
                spent, align._headroom_bytes = align._headroom_bytes, lambda: 0
                try:
                    check("a grant with nothing left solves to zero, which is not None",
                          align._solve_budget() == 0.0, align._solve_budget())
                finally:
                    align._headroom_bytes = spent
                grant, align._headroom_bytes = align._headroom_bytes, lambda: 4 * 1024 ** 3
                try:
                    align._scale, align._scale_seen = 1.0, 0
                    one = align._solve_budget()
                    align._scale = 2.0
                    two = align._solve_budget()
                    check("doubling the correction halves the budget it solves for",
                          one > 0 and abs(two * 2 - one) < one * 0.02, (one, two))
                    check("the solved budget is half the grant divided by a position",
                          abs(one - align.BUDGET_HEADROOM * 4 * 1024 ** 3
                              / align._bytes_a_position(d2)) < 1, one)
                finally:
                    align._headroom_bytes = grant
                # 🔴 Tested against the rule, not against the counter: with no CUDA here
                # the counter path returns zero whatever the rule says, so a version that
                # had the comparison backwards passed this until it was split out.
                check("a call that never rose above an older peak reports nothing",
                      align._used(0, 500, 400) == 0, align._used(0, 500, 400))
                check("a call that did rise reports what it added",
                      align._used(100, 500, 900) == 800, align._used(100, 500, 900))
            finally:
                align._scale, align._scale_seen = keep
                align._state["model"] = model

            # ---- admission: two refusals that must not read alike ----------------------
            align.MAX_SPAN_SEC = 1.5
            over = c.post("/v1/audio/align", files=WAV,
                          data={"segments": '[{"start":0,"end":1,"text":"hi"},'
                                            '{"start":1,"end":4,"text":"yo"}]'}).json()
            check("a span longer than the model aligns says so",
                  "aligns at most" in ((over.get("results") or [{}, {}])[1].get("error") or ""),
                  over.get("results"))
            align.MAX_SPAN_SEC = was[2]
            dense = ("[{\"start\":0,\"end\":1,\"text\":\"hi\"},"
                     "{\"start\":1,\"end\":2,\"text\":\"%s\"}]"
                     % ("\u4f60" * (int(align.MAX_UNITS_A_SECOND) + 5)))
            got = c.post("/v1/audio/align", files=WAV, data={"segments": dense}).json()
            res = got.get("results") or []
            check("a transcript too long for its audio is ALIGNED, not refused",
                  len(res) == 2 and not res[0].get("error") and not res[1].get("error"), res)
            check("and both spans in that request come back with units",
                  bool((res[0] or {}).get("units")) and bool((res[1] or {}).get("units")),
                  res)
    finally:
        (align.BATCH, align._budget, align.MAX_SPAN_SEC) = was


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


def t_qwen():
    from fastapi.testclient import TestClient
    from wrapper.caps import stt_stream as q

    q._decode_to_16k_mono = lambda raw, fn: np.zeros(SR * 4, dtype="float32")
    # One result per clip, saying how many samples that clip carried. The real engine
    calls, requests = [], []

    def fake_transcribe(audio=None, context=None, language=None, return_time_stamps=None):
        # Two shapes reach here: the serial path hands one (clip, sr) tuple, the batched
        clips = audio if isinstance(audio, list) else [audio]
        calls.append(len(clips))
        requests.append({"clips": len(clips), "context": context, "language": language})
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
        the_prompt_reaches_every_qwen_path(c, q, requests)
        batch_over_cap(c, q, calls)
        repetition_fallback(q)
        repetition_request_spellings(q)
        the_report_does_not_need_a_request(q)
        malformed_span_is_isolated(c, q, calls)
        one_bad_span_does_not_sink_its_group(c, q, calls)
        the_sizing_says_what_it_did(c, q, calls)
        the_card_refusing_lowers_the_budget(c, q, calls)
        a_refusal_reshapes_the_rest_of_the_request(c, q, calls)
        the_language_field_reaches_the_model(c, q)
        every_span_says_what_language_it_is(c, q)


class batch_mode:
    """Turn the batch path on the way a deployment does, for one check.

    🔴 Three attributes, not one. Grouping, the ceiling and the measurement are separate,
    and setting the ceiling alone leaves the engine serial -- which every assertion below
    would still pass, having tested nothing.
    """

    def __init__(self, q, ceiling=None, budget=None):
        self.q, self.ceiling, self.budget = q, ceiling, budget

    def __enter__(self):
        q = self.q
        self.was = (q.GROUPING, q.MAX_SPANS, q.grouping.SPAN_FLOOR_SECONDS,
                    q._headroom_bytes)
        # 🔴 The per-span floor is stood down for these checks, and only for these. It is one
        # second, while this file's fixtures are tenths of a second because the test audio is --
        # so with it in force every fixture is one span a call and nothing about the wiring gets
        # exercised. What the floor itself does is pinned in tests/test_grouping.py against
        # numbers that mean something; these checks are about what reaches the handler.
        q.grouping.SPAN_FLOOR_SECONDS = 0.0
        q.GROUPING = True
        q.MAX_SPANS = self.ceiling
        # 🔴 A count no longer forms groups on its own, it only bounds them, and there is no
        # flag that sets a size by hand any more. Forcing a shape therefore means stubbing the
        # measurement, which is the only thing that builds a group.
        want = self.budget if self.budget is not None else 1e9
        q._headroom_bytes = lambda: want * q._bytes_a_padded_second() / q.BUDGET_FRACTION
        return q

    def __exit__(self, *exc):
        q = self.q
        (q.GROUPING, q.MAX_SPANS, q.grouping.SPAN_FLOOR_SECONDS,
         q._headroom_bytes) = self.was
        return False


def span_mode(q, spans):
    return batch_mode(q, ceiling=spans)


def the_prompt_reaches_every_qwen_path(c, q, requests):
    """The OpenAI-compatible prompt is Qwen's context on every execution path."""
    prompt = "Use these canonical spellings for vocabulary and names: \u5c0f\u73fa"

    del requests[:]
    both_ways(c, "/v1/audio/transcriptions", WAV, {"prompt": prompt},
              "qwen prompt single", meters=("input",))
    check("single sync and async calls pass prompt as context",
          len(requests) == 2 and all(x["context"] == prompt for x in requests), requests)

    body = '[{"start":0,"end":0.1},{"start":0.1,"end":0.2},' \
           '{"start":0.2,"end":0.3},{"start":0.3,"end":0.4}]'
    del requests[:]
    with batch_mode(q, budget=0.2):
        both_ways(c, "/v1/audio/transcriptions", WAV,
                  {"segments": body, "prompt": prompt}, "qwen prompt grouped",
                  meters=("input",))
    check("multi-group sync and async calls keep one context for every group",
          len(requests) == 4 and all(x["context"] == prompt for x in requests)
          and all(x["clips"] == 2 for x in requests), requests)

    was_grouping = q.GROUPING
    q.GROUPING = False
    del requests[:]
    try:
        both_ways(c, "/v1/audio/transcriptions", WAV,
                  {"segments": body, "prompt": prompt}, "qwen prompt serial",
                  meters=("input",))
    finally:
        q.GROUPING = was_grouping
    check("serial batch sync and async calls keep context on every span",
          len(requests) == 8 and all(x["context"] == prompt for x in requests)
          and all(x["clips"] == 1 for x in requests), requests)

    real = q._offline_transcribe_many
    attempts = []

    def refusing(clips, language=None, context=""):
        attempts.append((len(clips), context))
        if len(clips) > 1:
            raise RuntimeError("engine refused this batch")
        return real(clips, language=language, context=context)

    q._offline_transcribe_many = refusing
    try:
        with span_mode(q, 8):
            r = c.post("/v1/audio/transcriptions", files=WAV,
                       data={"segments": body, "prompt": prompt})
    finally:
        q._offline_transcribe_many = real
    check("binary retries keep the same context while splitting groups",
          r.status_code == 200 and len(attempts) > 1
          and all(context == prompt for _size, context in attempts), attempts)


def the_card_refusing_lowers_the_budget(c, q, calls):
    """An out-of-memory is the one reading the cost model cannot take for itself.

    🔴 A call the card refuses allocated nothing to measure, so the size it was refused at
    is the only evidence the prediction was too high. Everything else that fails a call --
    a malformed span, a short answer -- says nothing about size, and the check above this
    one holds that those do NOT move the budget.
    """
    spans = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    body = ",".join('{"start":%s,"end":%s}' % (a, b) for a, b in spans)
    real, said, real_p = q._offline_transcribe_many, [], q._p

    contexts = []

    def refusing(clips, language=None, context=""):
        contexts.append(context)
        if len(clips) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(clips, language=language, context=context)

    was = q._refused_above
    # 🔴 This scenario CAUSES a refusal, so it is the one that leaves `_refusal_streak` behind.
    # Fixing the scenario downstream was fixing the victim: a polluter that does not clean up
    # just moves the problem to whoever runs next. Reset on the way in and restore on the way
    # out, the same as everything else this stubs.
    was_streak, was_target = q._refusal_streak, q._streak_target
    q._refusal_streak, q._streak_target = 0, None
    q._offline_transcribe_many, q._p = refusing, said.append
    try:
        with batch_mode(q, budget=1.0):
            del calls[:]
            r = c.post("/v1/audio/transcriptions", files=WAV,
                       data={"segments": "[" + body + "]", "prompt": "\u5c0f\u73fa"})
            doc = r.json() if r.status_code == 200 else {}
    finally:
        q._offline_transcribe_many, q._p = real, real_p
        refused, q._refused_above = q._refused_above, was
        q._refusal_streak, q._streak_target = was_streak, was_target
    got = doc.get("results") or []
    check("every span still comes back after the card refused",
          len(got) == 4 and all("text" in x for x in got), got)
    check("OOM re-planning keeps context on every attempted call",
          bool(contexts) and all(x == "\u5c0f\u73fa" for x in contexts), contexts)
    # 0.4, not the 0.8 the first attempt was refused at: the group was halved and the half
    # was refused too, so the smallest size the card has said no to is the honest ceiling.
    check("the SMALLEST size the card refused becomes the ceiling", refused == 0.4, refused)
    line = [x for x in said if x.startswith("batching:")]
    check("the report says the card refused, not just that a group was split",
          bool(line) and "the card refused 2 calls" in line[0], line)
    # 🔴 A configured number is a ceiling now, re-solved like auto, so a refusal lowers it
    # and the rest of the request is re-planned under it rather than bisected.
    check("a configured number is re-planned under the refusal, not bisected",
          bool(line) and "re-planned 2 times" in line[0]
          and "groups split and retried" not in line[0], line)
    check("and the report says the size the next call will be, unrounded",
          bool(line) and "the next call is sized at 0.2 padded seconds" in line[0], line)
    check("and says that the ceiling lifts, because it does",
          bool(line) and "lifts 60s after the refusal" in line[0]
          and "fixed by configuration" not in line[0], line)


def a_refusal_reshapes_the_rest_of_the_request(c, q, calls):
    """Under `auto`, the card refusing one call makes the WHOLE request smaller.

    🔴 The refusal is about size, so it applies to every group still outstanding. Halving
    only the group that failed leaves the rest of the request queued at the size that just
    failed -- each of them then finds the same wall on its own, and a request with ten
    groups pays ten refused calls to learn one thing.
    """
    spans = [(i * 0.1, i * 0.1 + 0.1) for i in range(8)]
    body = ",".join('{"start":%s,"end":%s}' % (a, b) for a, b in spans)
    real, said, real_p = q._offline_transcribe_many, [], q._p

    def refusing(clips, language=None, context=""):
        # A card that will not take more than two clips at a time, whatever it is told.
        if len(clips) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(clips, language=language, context=context)

    was_refused, was_headroom = q._refused_above, q._headroom_bytes
    was_drop, dropped = q._drop_cache, []
    # 🔴 The streak and its target are module state too, and a scenario that leaves one
    # behind changes what the NEXT scenario's report line says -- the wait doubles, and the
    # sentence about when the ceiling lifts is asserted as a literal below. Saving only
    # `_refused_above` meant the order these run in decided whether that literal held.
    was_streak, was_target = q._refusal_streak, q._streak_target
    q._offline_transcribe_many, q._p = refusing, said.append
    q._headroom_bytes = lambda: 8 * (2 ** 30)
    q._refused_above, q._drop_cache = None, lambda: dropped.append(True)
    q._refusal_streak, q._streak_target = 0, None
    try:
        with batch_mode(q):
            del calls[:]
            r = c.post("/v1/audio/transcriptions", files=WAV,
                       data={"segments": "[" + body + "]"})
            doc = r.json() if r.status_code == 200 else {}
    finally:
        q._offline_transcribe_many, q._p = real, real_p
        q._headroom_bytes, q._refused_above = was_headroom, was_refused
        q._refusal_streak, q._streak_target = was_streak, was_target
        q._drop_cache = was_drop
    got = doc.get("results") or []
    check("every span comes back after the request was reshaped",
          len(got) == 8 and all("text" in x for x in got), got)
    line = [x for x in said if x.startswith("batching:")]
    # 🔴 The exact count. One per refusal is the design: the ceiling halves, so eight spans
    # against a card that takes two go 8 -> 4 -> 2, not one group at a time.
    check("the rest of the request was re-planned once per refusal, not halved",
          bool(line) and "re-planned 2 times" in line[0], line)
    # 🔴 Dropped once per refusal, from the handler: on a shared card the blocks torch
    # caches are blocks the neighbours cannot have.
    check("the allocator's cache goes back to the card on every refusal",
          len(dropped) == 2, dropped)
    # 🔴 The size prints as the number it is: 0.2 rounded to "0" reads as a process with
    # nothing left, which is a different situation with a different cause.
    check("auto says the size of the next call, unrounded",
          bool(line) and "the next call is sized at 0.2 padded seconds" in line[0], line)
    # 🔴 And says it is temporary. A ceiling that never lifted meant one refusal caused by
    # a neighbour's minute left this process smaller for the rest of its life.
    check("the ceiling says when it lifts, because it does",
          bool(line) and "lifts 60s after the refusal" in line[0], line)
    check("refusals track how wrong the guess was, not how many groups there are",
          bool(line) and "the card refused 2 calls" in line[0]
          and "6 calls made" in line[0], line)


def the_sizing_says_what_it_did(c, q, calls):
    """A seconds budget groups by what a call costs, and one line says what happened.

    🔴 The report is the only thing that distinguishes "grouped as asked" from "grouped
    and then split three times because the card refused" -- both answer 200 with every
    span present, and the second is the one worth knowing about.
    """
    # Four 0.1 s spans and one 0.6 s one. A 0.6 s budget takes six of the short ones or one
    # long one, so the long span must come back alone and the short ones together.
    spans = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.8), (0.8, 0.9), (0.9, 1.0)]
    body = ",".join('{"start":%s,"end":%s}' % (a, b) for a, b in spans)
    said, real_p = [], q._p
    q._p = said.append
    try:
        with batch_mode(q, budget=0.6):
            del calls[:]
            r = c.post("/v1/audio/transcriptions", files=WAV,
                       data={"segments": "[" + body + "]"})
            doc = r.json() if r.status_code == 200 else {}
    finally:
        q._p = real_p
    got = doc.get("results") or []
    check("a seconds budget answers 200 with every span", len(got) == len(spans), len(got))
    check("the long span was not padded onto the short ones",
          sorted(calls) == [1, 4], calls)
    line = [x for x in said if x.startswith("batching:")]
    check("the request says what the sizing did", len(line) == 1, said)
    if line:
    # 🔴 The numbers, not the words. A report whose counters never move still carries
    # every phrase, and every assertion about the phrases passes.
        check("it counts the spans it was given", "5 spans planned" in line[0], line[0])
        check("it says the budget it used, not a rounded one",
              "0.6 padded seconds" in line[0], line[0])
        check("it counts the calls it planned", "into 2 calls" in line[0], line[0])
        check("it counts the calls it made", "2 calls made" in line[0], line[0])
        check("nothing was split, and it does not say one was",
              "split" not in line[0], line[0])
        check("it says what the correction is", "cost model 1.00x" in line[0], line[0])


def one_bad_span_does_not_sink_its_group(c, q, calls):
    """A span the engine refuses must cost only itself, not the group it happened to land in.

    A batched call has no per-span outcome, so the first version marked the whole group --
    measured against a real engine, one 0.5 ms span failed all five of its group. Retrying
    cannot help either: grouping is by index, so the retry rebuilds the same group around
    the same span. Halving a failed group ends the search on the span responsible.
    """
    bad = {"n": 0}
    real = q._offline_transcribe_many

    def refusing(clips, language=None, context=""):
        # The engine refuses any call that carries the pathological clip, as a real one does.
        bad["n"] += 1
        if any(len(cl) == 8 for cl in clips):
            raise RuntimeError("engine refused this batch")
        return real(clips, language=language, context=context)

    q._offline_transcribe_many = refusing
    said, real_p = [], q._p
    q._p = said.append
    try:
        with span_mode(q, 8):
            del calls[:]
            # index 2 is 8 samples long: hi > lo so it is handed to the engine, and a real one
            spans = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.2005), (0.4, 0.5), (0.5, 0.6)]
            body = ",".join('{"start":%s,"end":%s}' % (a, b) for a, b in spans)
            r = c.post("/v1/audio/transcriptions", files=WAV,
                       data={"segments": "[" + body + "]"})
            doc = r.json() if r.status_code == 200 else {}
    finally:
        q._offline_transcribe_many = real
        q._p = real_p
    got = doc.get("results") or []
    errs = [i for i, e in enumerate(got) if "error" in e]
    check("a refused span still leaves one entry per request entry", len(got) == 5, len(got))
    check("only the span the engine refused carries the error", errs == [2], errs)
    # 🔴 Planned and made differ here and nowhere else: one group planned, seven calls run.
    # Quoting only the plan would say "5 spans in 1 call" for a request that split three times.
    #
    # ⚠️ Five calls and two splits while groups kept the caller's order. Packing longest first
    # moves the 0.5 ms span from the middle of the group to its end, and halving finds a span at
    # the end in more steps. Same outcome, more steps: nothing orders a group by where a bad
    # span is likely to be, and there is nothing to order it by.
    line = [x for x in said if x.startswith("batching:")]
    check("the report separates the plan from what ran",
          bool(line) and "into 1 calls" in line[0] and "7 calls made" in line[0], line)
    check("and counts the splits that got it there",
          bool(line) and "3 groups split and retried" in line[0], line)
    check("a refusal that was not the card does not move the budget",
          q._refused_above is None, q._refused_above)
    check("its neighbours keep their transcripts",
          len(got) == 5 and all("text" in got[i] for i in (0, 1, 3, 4)), got)


def malformed_span_is_isolated(c, q, calls):
    """One unusable span must not take the rest of the request with it.

    The serial path has always answered per span: a bad one gets {"error": ...} and its
    neighbours still get transcripts. parse_segments only checks the payload is a JSON
    array, so anything at all can arrive as an element -- and the batched path has to keep
    the same promise, that the reply carries one entry per request entry, in order.
    """
    # 🔴 span_mode, not a bare count. The gate is `GROUPING`, so setting the count alone
    # leaves this on the SERIAL path, where every assertion below still passes.
    with span_mode(q, 8):
        del calls[:]
        r = c.post("/v1/audio/transcriptions", files=WAV,
                   data={"segments": '[{"start":0,"end":0.2},"oops",'
                                     '{"start":"n/a","end":1},{"start":0.3,"end":0.5}]'})
        doc = r.json() if r.status_code == 200 else {}
    got = doc.get("results") or []
    check("the malformed-span check ran on the batched path, not the serial one",
          len(calls) > 0 and max(calls) > 1, calls)
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

    with span_mode(q, 3):
        del calls[:]
        r = c.post("/v1/audio/transcriptions", files=WAV,
                   data={"segments": "[" + body + "]"})
        doc = r.json()

    check("over-cap batch answers 200", r.status_code == 200, r.status_code)
    got = doc.get("results") or []
    check("every span comes back", len(got) == len(spans), len(got))
    check("no span is left unanswered", all(x is not None for x in got), got)
    check("spans keep the order they were sent in",
          [x.get("text") for x in got] == want, [x.get("text") for x in got])
    # 🔴 Three calls, and the sizes are what costs least at three: [2, 2, 3] pads to 2.3 s
    # where filling to the cap first ([3, 3, 1]) pads to 2.5 s for the same three calls.
    check("the request was split at the cap, not sent whole", calls == [2, 2, 3], calls)


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
                           ("diar_speakrs openvino models dir",
                            t_diar_speakrs_openvino_models_dir, "speakrs"),
                           ("diar_stream offline", t_diar_stream_offline, "nemo"),
                           ("align batching", t_align_batching, "qwen")):
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


def the_language_field_reaches_the_model(c, q):
    """`language` was declared on the endpoint, parsed by FastAPI -- and then dropped.

    Both offline paths passed a literal None to transcribe(), so forcing a language and
    not forcing one produced byte-identical output. That is the one kind of wrong answer
    a caller cannot see: the transcript looks fine, it is simply in whatever language the
    model guessed. The reply now also carries the language, which is what makes the
    difference observable from outside at all.
    """
    seen = []
    asr = q._state["asr"]
    real = asr.transcribe

    def watching(audio=None, context=None, language=None, return_time_stamps=None):
        seen.append(language)
        clips = audio if isinstance(audio, list) else [audio]
        return [types.SimpleNamespace(text="hi", language=language or "English")
                for _c, _sr in clips]

    asr.transcribe = watching
    try:
        r = c.post("/v1/audio/transcriptions", files=WAV, data={"language": "zh"})
        forced = r.json() if r.status_code == 200 else {}
        auto = c.post("/v1/audio/transcriptions", files=WAV, data={})
        detected = auto.json() if auto.status_code == 200 else {}
        bad = c.post("/v1/audio/transcriptions", files=WAV, data={"language": "klingon"})
        region = c.post("/v1/audio/transcriptions", files=WAV, data={"language": "zh-CN"})
    finally:
        asr.transcribe = real
    # Four requests, four reached the model. The unservable one arrives as None -- the
    # same thing sending no language at all produces -- and is told apart from a silent
    # no-op only by the language the reply carries back.
    check("the model saw exactly what each request asked for",
          seen == ["Chinese", None, None, "Chinese"], seen)
    check("the reply says which language the transcript is in",
          forced.get("language") == "Chinese", forced)
    check("and says what the model detected when nothing was forced",
          detected.get("language") == "English", detected)
    check("a language this model cannot serve falls back instead of failing the request",
          bad.status_code == 200, bad.status_code)
    check("and the reply is what tells the caller the hint was not honoured",
          bad.json().get("language") == "English", bad.json())
    check("a region subtag is accepted and answered like the bare code",
          region.status_code == 200 and region.json().get("language") == "Chinese",
          (region.status_code, region.json()))


def every_span_says_what_language_it_is(c, q):
    """Batch mode answers per span, so the language has to be per span too.

    A caller that batches spans is the one most likely to need this -- it is holding a
    list of transcripts with nothing on them saying which language each is in, which is
    exactly the gap that made the caller go vote on it downstream.
    """
    asr = q._state["asr"]
    real = asr.transcribe
    seen = []

    def answering(audio=None, context=None, language=None, return_time_stamps=None):
        clips = audio if isinstance(audio, list) else [audio]
        seen.append(len(clips))
        return [types.SimpleNamespace(text=str(len(cl)), language=language or "English")
                for cl, _sr in clips]

    asr.transcribe = answering
    try:
        # 🔴 span_mode, as above: the serial path also answers a per-span language, so a
        # bare count leaves this passing while testing the one path it is not about.
        with span_mode(q, 8):
            r = c.post("/v1/audio/transcriptions", files=WAV,
                       data={"segments": '[{"start":0,"end":1},{"start":1,"end":2}]',
                             "language": "ja"})
            doc = r.json() if r.status_code == 200 else {}
    finally:
        asr.transcribe = real
    got = doc.get("results") or []
    check("the language check ran on the batched path, not the serial one",
          seen == [2], seen)
    check("every span in a batch carries its language",
          len(got) == 2 and all(e.get("language") == "Japanese" for e in got), got)


if __name__ == "__main__":
    sys.exit(main())
