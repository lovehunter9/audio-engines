import io
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import wave

from wrapper.caps import diar_sgl


def _wav(seconds, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


class PoolMathTest(unittest.TestCase):
    def test_the_pool_for_a_length_holds_that_length(self):
        for seconds in (60, 860.19, 1800, 3600):
            self.assertGreaterEqual(diar_sgl.seconds_for(diar_sgl.pool_tokens(seconds)), seconds)

    def test_the_pool_survives_sglang_rounding_it_down_to_a_page(self):
        # Asked for 46108 tokens, SGLang on the B70 allocated 46080 and the limit fell to 1799s.
        pool = diar_sgl.pool_tokens(1800)
        self.assertEqual(pool % diar_sgl.POOL_ALIGN, 0)
        self.assertGreaterEqual(diar_sgl.seconds_for(pool // 64 * 64), 1800)

    def test_the_budget_covers_the_densest_measured_meeting(self):
        # AMI R8001: 860 s, 11335 input tokens, 9047 transcript tokens, KV pool 49780.
        budget = diar_sgl.output_budget(49780, 860.19)
        self.assertGreaterEqual(budget, 9047)
        self.assertLessEqual(11335 + budget + 16, 49780)

    def test_the_budget_never_exceeds_the_cap(self):
        self.assertEqual(diar_sgl.output_budget(10 ** 7, 60), diar_sgl.MAX_NEW_TOKENS)

    def test_a_hotword_prompt_comes_out_of_the_budget(self):
        self.assertEqual(diar_sgl.output_budget(25000, 860) - diar_sgl.output_budget(25000, 860, 200),
                         200)


class ModelPathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = os.path.join(self.tmp.name, "run")
        self.cache = os.path.join(self.tmp.name, "hub")
        os.makedirs(self.run_dir)
        self.env = {"HF_HUB_CACHE": self.cache, "MODEL_SOURCE":
                    "hf://OpenMOSS-Team/MOSS-Transcribe-Diarize,hf://w/MOSS-dflash2@abc"}
        self._old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self._run_dir = diar_sgl.RUN_DIR
        diar_sgl.RUN_DIR = self.run_dir

    def tearDown(self):
        diar_sgl.RUN_DIR = self._run_dir
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def snapshot(self, repo):
        path = os.path.join(self.cache, "models--" + repo.replace("/", "--"), "snapshots", "r1")
        os.makedirs(path)
        return path

    def test_llm_init_run_files_win(self):
        with open(os.path.join(self.run_dir, "model_path"), "w") as f:
            f.write("/models/main\n")
        with open(os.path.join(self.run_dir, "extra_model_path"), "w") as f:
            f.write("/models/draft\n")
        self.snapshot("OpenMOSS-Team/MOSS-Transcribe-Diarize")
        self.assertEqual(diar_sgl.model_paths(), ("/models/main", "/models/draft"))

    def test_the_cache_answers_without_run_files_and_drops_the_revision(self):
        main = self.snapshot("OpenMOSS-Team/MOSS-Transcribe-Diarize")
        draft = self.snapshot("w/MOSS-dflash2")
        self.assertEqual(diar_sgl.model_paths(), (main, draft))

    def test_the_engine_waits_for_llm_init_to_finish_the_draft(self):
        import threading

        path = os.path.join(self.run_dir, "extra_model_path")
        threading.Timer(0.3, lambda: open(path, "w").write("/models/draft\n")).start()
        started = time.monotonic()
        diar_sgl.wait_for_draft(timeout_s=10, interval_s=0.05)
        self.assertGreaterEqual(time.monotonic() - started, 0.25)
        self.assertTrue(os.path.exists(path))

    def test_without_a_run_directory_nothing_waits(self):
        diar_sgl.RUN_DIR = os.path.join(self.tmp.name, "absent")
        started = time.monotonic()
        diar_sgl.wait_for_draft(timeout_s=10, interval_s=0.05)
        self.assertLess(time.monotonic() - started, 0.2)

    def test_no_second_source_means_no_draft(self):
        os.environ["MODEL_SOURCE"] = "hf://OpenMOSS-Team/MOSS-Transcribe-Diarize"
        main = self.snapshot("OpenMOSS-Team/MOSS-Transcribe-Diarize")
        self.assertEqual(diar_sgl.model_paths(), (main, None))


class EngineArgvTest(unittest.TestCase):
    def setUp(self):
        self._saved = (diar_sgl.XPU, diar_sgl.USE_DRAFT, diar_sgl.MEM_FRACTION,
                       diar_sgl._passthrough)

    def tearDown(self):
        (diar_sgl.XPU, diar_sgl.USE_DRAFT, diar_sgl.MEM_FRACTION,
         diar_sgl._passthrough) = self._saved

    def test_cuda_runs_the_draft_with_async_decode(self):
        diar_sgl.XPU, diar_sgl.USE_DRAFT = False, True
        argv = diar_sgl.engine_argv("/m", "/d")
        self.assertIn("DFLASH", argv)
        self.assertEqual(argv[argv.index("--asr.engine.speculative_draft_model_path") + 1], "/d")
        self.assertIn("--asr.factory.enable_async_decode", argv)
        self.assertEqual(argv[argv.index("--asr.engine.max_total_tokens") + 1],
                         str(diar_sgl.pool_tokens(diar_sgl.BOUNDS.seconds)))

    def test_cuda_without_draft_weights_runs_plain(self):
        diar_sgl.XPU, diar_sgl.USE_DRAFT = False, True
        self.assertNotIn("DFLASH", diar_sgl.engine_argv("/m", None))

    def test_xpu_uses_intel_attention_and_never_the_draft(self):
        diar_sgl.XPU, diar_sgl.USE_DRAFT = True, True
        argv = diar_sgl.engine_argv("/m", "/d")
        self.assertEqual(argv[argv.index("--asr.engine.attention_backend") + 1], "intel_xpu")
        self.assertEqual(argv[argv.index("--asr.engine.enable_torch_compile") + 1], "false")
        self.assertNotIn("DFLASH", argv)

    def test_unclaimed_flags_reach_the_engine_and_the_fraction_is_optional(self):
        diar_sgl.XPU, diar_sgl.MEM_FRACTION = False, None
        diar_sgl._passthrough = ["--asr.engine.chunked_prefill_size", "2048"]
        argv = diar_sgl.engine_argv("/m", None)
        self.assertNotIn("--mem-fraction-static", argv)
        self.assertEqual(argv[-2:], ["--asr.engine.chunked_prefill_size", "2048"])
        diar_sgl.MEM_FRACTION = "0.4"
        self.assertIn("--mem-fraction-static", diar_sgl.engine_argv("/m", None))


# A stand-in for `sgl-omni serve`: prints the KV pool line, serves /health and a streaming
# /v1/audio/transcriptions, and forks a grandchild the way sgl-omni forks its workers.
FAKE_ENGINE = textwrap.dedent(r'''
    import json, os, re, subprocess, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(sys.argv[sys.argv.index("--port") + 1])
    log = sys.argv[sys.argv.index("--log") + 1]
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(log, "a") as f:
        f.write("worker %d\n" % worker.pid)
    print("KV Cache is allocated. dtype: torch.bfloat16, #tokens: 50000, K size: 2 GB", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode("latin-1")
            budget = re.search(r'name="max_new_tokens"\r\n\r\n(\d+)', body).group(1)
            slow = "slow" in body
            with open(log, "a") as f:
                f.write("request budget=%s\n" % budget)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            parts = ["[0.5][S01]hello[1.5]", "[2.0][S02]world[3.0]"]
            if slow:
                parts = ["[%d.0][S01]tick[%d.5]" % (i, i) for i in range(60)]
            try:
                for p in parts:
                    ev = {"type": "transcript.text.delta", "delta": p}
                    self.wfile.write(("data: %s\n\n" % json.dumps(ev)).encode())
                    self.wfile.flush()
                    time.sleep(0.25 if slow else 0)
                done = {"type": "transcript.text.done", "text": "".join(parts)}
                self.wfile.write(("data: %s\n\ndata: [DONE]\n\n" % json.dumps(done)).encode())
                with open(log, "a") as f:
                    f.write("finished\n")
            except (BrokenPipeError, ConnectionResetError):
                with open(log, "a") as f:
                    f.write("disconnected\n")

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


class FakeEngineTest(unittest.TestCase):
    """The child process and the request path against a stand-in engine."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log = os.path.join(cls.tmp.name, "engine.log")
        script = os.path.join(cls.tmp.name, "fake_engine.py")
        with open(script, "w") as f:
            f.write(FAKE_ENGINE)
        cls._saved = (diar_sgl.ENGINE_PORT, diar_sgl._exit)
        diar_sgl.ENGINE_PORT = 18000 + os.getpid() % 2000
        diar_sgl._exit = lambda code: None
        cls.child = diar_sgl._Child()
        cls.child.start([sys.executable, script, "--port", str(diar_sgl.ENGINE_PORT),
                         "--log", cls.log], dict(os.environ))
        import httpx

        with httpx.Client() as client:
            deadline = time.time() + 20
            while not diar_sgl._healthy(client):
                if time.time() > deadline:
                    raise RuntimeError("fake engine never came up")
                time.sleep(0.1)
        cls._child = diar_sgl._child
        # Each TestClient's shutdown stops _child; the engine has to outlive them.
        diar_sgl._child = diar_sgl._Child()
        diar_sgl._state.update(ready=True, pool=cls.child.pool, error=None,
                               max_seconds=diar_sgl.seconds_for(cls.child.pool))

    @classmethod
    def tearDownClass(cls):
        diar_sgl._stopping.set()
        cls.child.stop(grace_s=5)
        diar_sgl._child = cls._child
        diar_sgl.ENGINE_PORT, diar_sgl._exit = cls._saved
        cls.tmp.cleanup()

    def read_log(self):
        with open(self.log) as f:
            return f.read()

    def client(self):
        from fastapi.testclient import TestClient

        return TestClient(diar_sgl.build_app(["stt", "diar"]))

    def test_the_pool_is_read_off_the_engine_log(self):
        self.assertEqual(self.child.pool, 50000)

    def test_transcription_streams_segments_with_an_explicit_budget(self):
        with self.client() as c:
            r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav(4))},
                       data={"response_format": "diarized_json"})
        self.assertEqual(r.status_code, 200, r.text)
        doc = r.json()
        self.assertEqual([s["speaker"] for s in doc["segments"]], ["S01", "S02"])
        self.assertEqual(doc["text"], "hello world")
        self.assertFalse(doc["truncated"])
        self.assertIn("request budget=%d" % diar_sgl.output_budget(50000, 4.0), self.read_log())

    def test_diarization_answers_turns(self):
        with self.client() as c:
            r = c.post("/v1/audio/diarization", files={"file": ("a.wav", _wav(4))})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["speakers"], ["S01", "S02"])

    def test_a_recording_longer_than_the_pool_is_refused_before_a_task_exists(self):
        saved = diar_sgl._state["max_seconds"]
        diar_sgl._state["max_seconds"] = 2.0
        try:
            with self.client() as c:
                r = c.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav(4))},
                           data={"async": "1"})
        finally:
            diar_sgl._state["max_seconds"] = saved
        self.assertEqual(r.status_code, 413, r.text)
        self.assertIn("KV cache", r.text)

    def test_cancelling_disconnects_the_engine_request_and_reports_progress(self):
        from wrapper import tasks

        class Ctx:
            ratios = []

            def progress(self, ratio, stage, **_):
                self.ratios.append(ratio)

            def checkpoint(self):
                if len(self.ratios) > 1:
                    raise tasks.Cancelled()

        before = self.read_log().count("disconnected")
        path = os.path.join(self.tmp.name, "a.wav")
        with open(path, "wb") as f:
            f.write(_wav(60))
        with self.assertRaises(tasks.Cancelled):
            diar_sgl._transcribe(path, 60.0, "slow", Ctx())
        self.assertGreater(Ctx.ratios[-1], 0.0)
        deadline = time.time() + 10
        while time.time() < deadline and self.read_log().count("disconnected") <= before:
            time.sleep(0.2)
        self.assertGreater(self.read_log().count("disconnected"), before)


class StopTest(unittest.TestCase):
    def test_stop_takes_the_whole_process_group(self):
        self._stop("import time; time.sleep(600)", grace_s=5)

    def test_a_worker_left_after_the_leader_exits_does_not_hold_the_stop(self):
        started = time.monotonic()
        self._stop("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                   "time.sleep(600)", grace_s=20)
        self.assertLess(time.monotonic() - started, 10)

    def _stop(self, worker_code, grace_s):
        child = diar_sgl._Child()
        child.start([sys.executable, "-c", textwrap.dedent('''
            import subprocess, sys, time
            w = subprocess.Popen([sys.executable, "-c", %r])
            print(w.pid, flush=True)
            time.sleep(600)
        ''' % worker_code)], dict(os.environ))
        worker = None
        deadline = time.time() + 10
        while worker is None and time.time() < deadline:
            out = subprocess.run(["pgrep", "-P", str(child.proc.pid)], capture_output=True,
                                 text=True).stdout.split()
            worker = int(out[0]) if out else None
            time.sleep(0.1)
        self.assertIsNotNone(worker)
        time.sleep(0.5)
        diar_sgl._stopping.set()
        try:
            child.stop(grace_s=grace_s)
            child.proc.wait(timeout=5)
            for _ in range(50):
                try:
                    os.kill(worker, 0)
                except OSError:
                    break
                time.sleep(0.1)
            with self.assertRaises(OSError):
                os.kill(worker, 0)
        finally:
            diar_sgl._stopping.clear()
            with __import__("contextlib").suppress(OSError):
                os.kill(worker, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
