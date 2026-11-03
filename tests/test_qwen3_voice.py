# Official Qwen speakers stay in that backend's directory. The shared pack stays the four cards.
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wrapper.caps import qwen3_voice, tts_el
from wrapper.caps.firered import FireRedBackend


def _tone(text, speaker, language):
    return np.zeros(1600, dtype="float32"), 16000


class Qwen3VoiceCardsTest(unittest.TestCase):
    def test_official_speakers_are_not_in_the_shared_pack(self):
        shared = {c["voice_id"] for c in tts_el.builtin_presets("firered")}
        official = {row[0] for row in qwen3_voice.SPEAKERS}
        self.assertEqual(shared, {"zh-f", "zh-m", "en-f", "en-m"})
        self.assertTrue(official.isdisjoint(shared))
        self.assertEqual(len(official), 9)

    def test_seed_keeps_official_speakers_off_firered(self):
        root = tempfile.mkdtemp(prefix="qwen3voice-")
        native = os.path.join(root, ".qwen-native")
        qwen3_voice.bake_native(native, generate=_tone)
        self.assertTrue(os.path.isfile(os.path.join(native, "Vivian", "prompt.wav")))
        self.assertTrue(os.path.isfile(os.path.join(native, "Vivian", "prompt.txt")))

        class Backend:
            uses_shared_pack = True

            def native_voices_dir(self):
                return native

            def native_presets(self):
                return tts_el.load_voice_cards(native, category="premade")

        planted = tts_el.seed_builtins("qwen3_voice", os.path.join(root, "voices"), backend=Backend())
        ids = {c["voice_id"] for c in planted}
        self.assertTrue({row[0] for row in qwen3_voice.SPEAKERS} <= ids)
        self.assertTrue({"zh-f", "en-m"} <= ids)
        with open(os.path.join(root, "voices", "Vivian", "prompt.txt"), encoding="utf-8") as handle:
            transcript = handle.read()
        self.assertEqual(transcript, qwen3_voice.SPEAKERS[0][2])

        firered = tts_el.seed_builtins("firered", os.path.join(root, "firered"), backend=FireRedBackend(None))
        firered_ids = {c["voice_id"] for c in firered}
        self.assertEqual(firered_ids, {"zh-f", "zh-m", "en-f", "en-m"})
        self.assertFalse(os.path.isdir(os.path.join(root, "firered", "Vivian")))


class QwenKnobTest(unittest.TestCase):
    def test_default_knobs_leave_sampling_alone(self):
        gen, speed, seed, language = qwen3_voice._sampling(None, "你好", "")
        self.assertEqual(gen["temperature"], 0.9)
        self.assertNotIn("do_sample", gen)
        self.assertNotIn("instruct", gen)
        self.assertNotIn("top_p", gen)
        self.assertEqual(speed, 1.0)
        self.assertIsNone(seed)
        self.assertEqual(language, "Auto")

    def test_each_knob_moves_the_generator(self):
        slow, speed, _seed, _language = qwen3_voice._sampling({"speed": 0.7}, "你好", "")
        self.assertAlmostEqual(speed, 0.7)
        self.assertNotIn("instruct", slow)
        styled, _, _, _ = qwen3_voice._sampling({"style": 0.8}, "你好", "温柔")
        self.assertEqual(styled["instruct"], "温柔")
        self.assertNotIn("语气夸张", styled["instruct"])
        self.assertLess(styled["temperature"], 0.9)
        self.assertIn("top_p", styled)
        tight, _, _, _ = qwen3_voice._sampling({"similarity_boost": 1.0}, "你好", "")
        self.assertLess(tight["temperature"], 0.9)
        self.assertLess(tight["top_p"], 1.0)
        wild, _, seed, _ = qwen3_voice._sampling({"stability": 0.1}, "你好", "")
        self.assertLess(wild["temperature"], 0.9)
        self.assertIsNotNone(seed)
        loose_stab, _, locked_seed, _ = qwen3_voice._sampling({"stability": 1.0}, "你好", "")
        self.assertGreater(loose_stab["temperature"], 0.9)
        self.assertNotIn("do_sample", loose_stab)
        self.assertIsNone(locked_seed)
        loose, _, _, _ = qwen3_voice._sampling({"use_speaker_boost": False}, "你好", "")
        self.assertGreater(loose["temperature"], 0.9)
        self.assertNotIn("top_p", loose)

    def test_request_sampling_replaces_the_mapped_value(self):
        gen, _speed, _seed, language = qwen3_voice._sampling(
            {"style": 0.8}, "你好", "", sampling={"temperature": 0.2, "language": "Chinese"})
        self.assertEqual(gen["temperature"], 0.2)
        self.assertEqual(language, "Chinese")

    def test_design_base_is_still_a_noop_at_the_defaults(self):
        gen, _speed, seed, language = qwen3_voice._sampling(
            None, "你好", "一个沉稳的男声", base_cfg=tts_el.DESIGN_CFG)
        self.assertEqual(gen["temperature"], 0.9)
        self.assertNotIn("top_p", gen)
        self.assertEqual(gen["instruct"], "一个沉稳的男声")
        self.assertIsNone(seed)
        self.assertEqual(language, "Auto")

    def test_openai_format_names_are_accepted(self):
        self.assertEqual(tts_el._output_format("wav").kind, "wav")
        self.assertEqual(tts_el._output_format("pcm").kind, "pcm")
        self.assertEqual(tts_el._output_format("flac").kind, "flac")
        self.assertEqual(tts_el._output_format("ogg").kind, "opus")
        self.assertEqual(tts_el._output_format("mp3").kind, "mp3")

    def test_run_asks_for_the_openai_routes(self):
        seen = {}

        def fake_run(*args, **kwargs):
            seen["kwargs"] = kwargs

        original = qwen3_voice.tts_el.run
        qwen3_voice.tts_el.run = fake_run
        try:
            qwen3_voice.run(["tts", "tts_clone", "tts_design"])
        finally:
            qwen3_voice.tts_el.run = original
        self.assertTrue(seen["kwargs"].get("openai_alias"))
        self.assertTrue(seen["kwargs"].get("task_legacy"))

    def test_legacy_task_alias_is_mounted(self):
        from fastapi.testclient import TestClient

        with TestClient(qwen3_voice.build_app(["tts", "tts_clone", "tts_design"])) as client:
            spec = client.get("/api/engine-spec").json()["endpoints"]
            legacy = {(row["method"], row["path"]) for row in spec if row["path"].startswith("/v1/audio/tasks")}
            self.assertEqual(legacy, {
                ("GET", "/v1/audio/tasks"),
                ("GET", "/v1/audio/tasks/{id}"),
                ("GET", "/v1/audio/tasks/{id}/result"),
                ("DELETE", "/v1/audio/tasks/{id}"),
            })
            self.assertTrue(all(row.get("deprecated") for row in spec
                                if row["path"].startswith("/v1/audio/tasks")))
            listed = client.get("/v1/audio/tasks")
            self.assertEqual(listed.status_code, 200)
            same = client.get("/v1/tasks")
            self.assertEqual(same.status_code, 200)
            self.assertEqual(listed.json(), same.json())


class _FakeModel:
    def __init__(self):
        self.calls = []

    def _audio(self, kind, kw):
        self.calls.append((kind, kw))
        return np.zeros(1600, dtype="float32"), 24000

    def generate_custom_voice(self, **kw):
        return self._audio("custom", kw)

    def generate_voice_clone(self, **kw):
        return self._audio("clone", kw)

    def generate_voice_design(self, **kw):
        return self._audio("design", kw)

    def _stream(self, kind, kw):
        self.calls.append((kind, kw))
        yield np.full(800, 0.25, dtype="float32"), 24000, None
        yield np.full(400, -0.25, dtype="float32"), 24000, None

    def generate_custom_voice_streaming(self, **kw):
        return self._stream("custom_stream", kw)

    def generate_voice_clone_streaming(self, **kw):
        return self._stream("clone_stream", kw)

    def generate_voice_design_streaming(self, **kw):
        return self._stream("design_stream", kw)


def _backend(fake):
    backend = qwen3_voice.QwenVoiceBackend(fake, "/tmp/qwen-native-test")
    backend._prompt_path = lambda audio, sr: "/tmp/qwen-ref-test.wav"
    backend._custom_model = lambda: fake
    backend._design_model = lambda: fake
    return backend


class QwenRouteTest(unittest.TestCase):
    def test_official_speaker_uses_custom_voice_once(self):
        fake = _FakeModel()
        backend = _backend(fake)
        spoken = "第一句。第二句还很长，不要切开。"
        backend.clone(spoken, np.zeros(8, dtype="float32"), 16000, qwen3_voice.SPEAKERS[0][2],
                      instruction="慢一点", sampling={"language": "Chinese"})
        self.assertEqual([kind for kind, _kw in fake.calls], ["custom"])
        kw = fake.calls[0][1]
        self.assertEqual(kw["speaker"], "Vivian")
        self.assertEqual(kw["text"], spoken)
        self.assertEqual(kw["language"], "Chinese")
        self.assertEqual(kw["instruct"], "慢一点")

    def test_empty_transcript_is_xvec_and_a_user_clone_is_not(self):
        fake = _FakeModel()
        backend = _backend(fake)
        backend.clone("你好", np.zeros(8, dtype="float32"), 16000, "")
        backend.clone("你好", np.zeros(8, dtype="float32"), 16000, "这是用户自己的一句",
                      sampling={"x_vector_only": True, "language": "English"})
        backend.clone("第一句。第二句。", np.zeros(8, dtype="float32"), 16000, "这是用户自己的一句")
        self.assertEqual(fake.calls[0][0], "clone")
        self.assertTrue(fake.calls[0][1]["xvec_only"])
        self.assertEqual(fake.calls[0][1]["ref_text"], "")
        self.assertEqual(fake.calls[0][1]["language"], "Auto")
        self.assertTrue(fake.calls[1][1]["xvec_only"])
        self.assertEqual(fake.calls[1][1]["language"], "English")
        self.assertEqual(fake.calls[2][0], "clone")
        self.assertFalse(fake.calls[2][1]["xvec_only"])
        self.assertEqual(fake.calls[2][1]["text"], "第一句。第二句。")
        self.assertEqual(fake.calls[2][1]["ref_text"], "这是用户自己的一句")

    def test_openvino_speaks_in_a_child_process(self):
        backend = qwen3_voice.QwenVoiceBackend(object(), "/tmp/qwen-native-test")
        backend._prompt_path = lambda audio, sr: "/tmp/qwen-ref-test.wav"
        seen = []
        previous = os.environ.get("AUDIO_BASE")
        os.environ["AUDIO_BASE"] = "qwen3voiceov"

        def hook(repo, method, kwargs):
            seen.append((repo, method, kwargs))
            return np.zeros(1600, dtype="float32"), 24000

        qwen3_voice._OV_GENERATE_HOOK = hook
        try:
            backend.clone("第一句。第二句。", np.zeros(8, dtype="float32"), 16000,
                          qwen3_voice.SPEAKERS[0][2], sampling={"language": "Chinese"})
            backend.clone("你好", np.zeros(8, dtype="float32"), 16000, "")
            backend.design("一个沉稳的男声", "今天下午见。")
        finally:
            qwen3_voice._OV_GENERATE_HOOK = None
            if previous is None:
                os.environ.pop("AUDIO_BASE", None)
            else:
                os.environ["AUDIO_BASE"] = previous
        self.assertEqual(seen[0][0], qwen3_voice.CUSTOM_REPO)
        self.assertEqual(seen[0][1], "generate_custom_voice")
        self.assertEqual(seen[0][2]["speaker"], "Vivian")
        self.assertEqual(seen[0][2]["text"], "第一句。第二句。")
        self.assertEqual(seen[1][0], qwen3_voice.BASE_REPO)
        self.assertEqual(seen[1][1], "generate_voice_clone")
        self.assertTrue(seen[1][2]["xvec_only"])
        self.assertEqual(seen[2][0], qwen3_voice.DESIGN_REPO)
        self.assertEqual(seen[2][1], "generate_voice_design")
        self.assertEqual(seen[2][2]["instruct"], "一个沉稳的男声")

    def test_design_keeps_the_written_instruction(self):
        fake = _FakeModel()
        backend = _backend(fake)
        backend.design("一个沉稳的男声", "今天下午见。", settings={"style": 0.9})
        self.assertEqual(fake.calls[0][0], "design")
        self.assertEqual(fake.calls[0][1]["instruct"], "一个沉稳的男声")
        self.assertNotIn("语气夸张", fake.calls[0][1]["instruct"])
        self.assertEqual(fake.calls[0][1]["language"], "Auto")
        self.assertLess(fake.calls[0][1]["temperature"], 0.9)

    def test_live_iter_uses_streaming_and_one_shot_does_not(self):
        fake = _FakeModel()
        backend = _backend(fake)
        text = qwen3_voice.SPEAKERS[0][2]
        pieces = list(backend.iter_clone(
            "门口见。", np.zeros(8, dtype="float32"), 16000, text, chunk_size=3))
        self.assertEqual([kind for kind, _kw in fake.calls], ["custom_stream"])
        self.assertEqual(fake.calls[0][1]["chunk_size"], 3)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(len(pieces[0][0]) + len(pieces[1][0]), 1200)
        backend.clone("门口见。", np.zeros(8, dtype="float32"), 16000, text)
        self.assertEqual(fake.calls[-1][0], "custom")
        fake.calls.clear()
        designed = list(backend.iter_design("一个沉稳的男声", "今天下午见。"))
        self.assertEqual(fake.calls[0][0], "design_stream")
        self.assertEqual(fake.calls[0][1]["chunk_size"], 8)
        self.assertEqual(len(designed), 2)
        from wrapper.caps.ming_backend import MingBackend
        self.assertFalse(getattr(MingBackend, "streams_while_generating", False))
        self.assertTrue(MingBackend.socket_pcm_only)

    def test_openvino_stream_yields_each_child_chunk(self):
        backend = qwen3_voice.QwenVoiceBackend(object(), "/tmp/qwen-native-test")
        backend._prompt_path = lambda audio, sr: "/tmp/qwen-ref-test.wav"
        seen = []
        previous = os.environ.get("AUDIO_BASE")
        os.environ["AUDIO_BASE"] = "qwen3voiceov"

        def hook(repo, method, kwargs):
            seen.append((repo, method, kwargs.get("chunk_size")))
            return [
                (np.full(4, 0.2, dtype="float32"), 24000),
                (np.full(6, -0.2, dtype="float32"), 24000),
            ]

        qwen3_voice._OV_STREAM_HOOK = hook
        try:
            pieces = list(backend.iter_clone(
                "门口见。", np.zeros(8, dtype="float32"), 16000, qwen3_voice.SPEAKERS[0][2],
                chunk_size=5))
        finally:
            qwen3_voice._OV_STREAM_HOOK = None
            if previous is None:
                os.environ.pop("AUDIO_BASE", None)
            else:
                os.environ["AUDIO_BASE"] = previous
        self.assertEqual(seen, [(qwen3_voice.CUSTOM_REPO, "generate_custom_voice_streaming", 5)])
        self.assertEqual([len(wave) for wave, _sr in pieces], [4, 6])

    def test_ov_frame_file_is_readable_while_the_child_is_still_writing(self):
        import struct
        import threading
        import time

        from wrapper.caps.qwen3_ov_job import _write_frame

        tmp = tempfile.mkdtemp(prefix="qwen-frames-")
        path = os.path.join(tmp, "frames.bin")
        open(path, "wb").close()

        class Proc:
            def __init__(self):
                self.rc = None

            def poll(self):
                return self.rc

        proc = Proc()

        def write():
            time.sleep(0.05)
            with open(path, "r+b", buffering=0) as fh:
                _write_frame(fh, np.full(4, 0.25, dtype="float32"), 24000)
                fh.write(struct.pack("<II", 0, 0))
                fh.flush()
            proc.rc = 0

        threading.Thread(target=write).start()
        fh = open(path, "rb", buffering=0)
        try:
            hdr = qwen3_voice._read_follow(fh, proc, 8)
            sr, n = struct.unpack("<II", hdr)
            raw = qwen3_voice._read_follow(fh, proc, n * 4)
        finally:
            fh.close()
        self.assertEqual((sr, n, len(raw)), (24000, 4, 16))

    def test_openai_stream_sends_codec_chunks_and_the_socket_rejects_wav(self):
        from fastapi.testclient import TestClient

        root = tempfile.mkdtemp(prefix="qwen3-live-")
        native = os.path.join(root, "native")
        qwen3_voice.bake_native(native, generate=_tone)
        fake = _FakeModel()
        backend = qwen3_voice.QwenVoiceBackend(fake, native)
        backend._prompt_path = lambda audio, sr: os.path.join(root, "ref.wav")
        backend._custom_model = lambda: fake
        backend._design_model = lambda: fake
        previous_dir = tts_el.VOICE_DIR
        previous_name = tts_el._MODULE["name"]
        tts_el.VOICE_DIR = Path(root) / "voices"
        tts_el._MODULE["name"] = "qwen3_voice"
        try:
            tts_el.install(backend)
            app = qwen3_voice.build_app(["tts", "tts_clone", "tts_design"])
            with TestClient(app) as client:
                resp = client.post("/v1/audio/speech", json={
                    "input": "门口见。", "voice": "Vivian", "response_format": "pcm",
                    "stream": True, "chunk_size": 3,
                })
                self.assertEqual(resp.status_code, 200, resp.text[:300])
                self.assertEqual(fake.calls[0][0], "custom_stream")
                self.assertEqual(fake.calls[0][1]["chunk_size"], 3)
                self.assertEqual(len(resp.content), 1200 * 2)
                self.assertEqual(resp.headers["x-audio-format"], "pcm")
                bad = client.post("/v1/audio/speech", json={
                    "input": "门口见。", "voice": "Vivian", "response_format": "mp3",
                    "stream": True,
                })
                self.assertEqual(bad.status_code, 400)
                self.assertIn("pcm or wav", bad.text)
                both = client.post("/v1/audio/speech?async=1", json={
                    "input": "门口见。", "voice": "Vivian", "response_format": "pcm",
                    "stream": True,
                })
                self.assertEqual(both.status_code, 400)
                self.assertIn("async", both.text)
                with client.websocket_connect("/v1/audio/speech/stream") as ws:
                    self.assertEqual(ws.receive_json()["type"], "session.ready")
                    ws.send_json({
                        "type": "session.config", "voice": "Vivian", "response_format": "wav",
                        "chunk_size": 6,
                    })
                    err = ws.receive_json()
                    self.assertEqual(err["type"], "error")
                    self.assertIn("raw pcm", err["message"])
                    self.assertEqual(ws.receive_json()["type"], "session.ready")
                    ws.send_json({"type": "session.update", "voice": "Vivian"})
                    ws.send_json({"type": "input.text", "text": "门口见。"})
                    ws.send_json({"type": "input.done"})
                    blobs = []
                    done = None
                    for _ in range(6):
                        msg = ws.receive()
                        if msg.get("bytes"):
                            blobs.append(msg["bytes"])
                            continue
                        done = msg.get("text")
                        if done and "session.done" in done:
                            break
                    self.assertEqual(b"".join(blobs), resp.content)
                    self.assertIn("session.done", done or "")
                    self.assertEqual(fake.calls[-1][0], "custom_stream")
                    self.assertEqual(fake.calls[-1][1]["chunk_size"], 6)
        finally:
            tts_el.VOICE_DIR = previous_dir
            tts_el._MODULE["name"] = previous_name
            tts_el._state["ready"] = False
            tts_el._state["backend"] = None


class QwenSpecTest(unittest.TestCase):
    def test_reported_endpoints_match_ming(self):
        from fastapi.testclient import TestClient

        supports = ["tts", "tts_clone", "tts_design"]
        previous = os.environ.get("AUDIO_BASE"), os.environ.get("MODEL_SUPPORTS")
        os.environ["MODEL_SUPPORTS"] = "supports_tts,supports_tts_clone,supports_tts_design"
        try:
            os.environ["AUDIO_BASE"] = "qwen3voice"
            with TestClient(qwen3_voice.build_app(supports)) as client:
                qwen = client.get("/api/engine-spec").json()["endpoints"]
            from wrapper.caps import ming
            os.environ["AUDIO_BASE"] = "ming"
            with TestClient(ming.build_app(supports)) as client:
                ming_rows = client.get("/api/engine-spec").json()["endpoints"]
        finally:
            for key, value in zip(("AUDIO_BASE", "MODEL_SUPPORTS"), previous):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        qwen_keys = [(row["method"], row["path"]) for row in qwen]
        ming_keys = [(row["method"], row["path"]) for row in ming_rows]
        self.assertEqual(qwen_keys, ming_keys)
        self.assertEqual(len(qwen_keys), 33)
        self.assertIn(("POST", "/v1/audio/speech/batch"), qwen_keys)


if __name__ == "__main__":
    unittest.main()
