"""Ming voice cards and the OpenAI body, with no weights and no GPU."""
import base64
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi.testclient import TestClient

from wrapper.caps import ming, ming_backend, ming_catalog, tts_voice


ROOT = Path(__file__).resolve().parents[1]


def _wav():
    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    buf.close()
    with wave.open(buf.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    data = Path(buf.name).read_bytes()
    os.unlink(buf.name)
    return data


class CatalogTests(unittest.TestCase):
    def test_official_ip_list(self):
        cards = ming_catalog.load_ip_catalog()
        ids = [c["voice_id"] for c in cards]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 100)
        for vid in ("US_James", "US_Stella", "UK_George", "UK_Charlotte", "US_Emma",
                    "灵小柔", "灵楷"):
            self.assertIn(vid, ids)
        self.assertNotIn("灵信远", ids)
        self.assertIn("灵信远_中性", ids)
        james = next(c for c in cards if c["voice_id"] == "US_James")
        self.assertEqual(james["origin"], "model")
        self.assertEqual(james["source"], "ip")
        self.assertEqual(james["labels"]["origin"], "model")
        self.assertEqual(james["category"], "premade")


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = tts_voice.VoiceLibrary(self.tmp.name)
        self.cards = [
            c for c in ming_catalog.load_ip_catalog()
            if c["voice_id"] in ("US_James", "灵小柔")
        ]
        self.lib.reconcile(self.cards, ROOT / "voices")

    def tearDown(self):
        self.tmp.cleanup()

    def test_origins_stay_distinct_and_premade(self):
        model = self.lib.get("灵小柔")
        pack = self.lib.get("zh-f")
        self.assertEqual(model["origin"], "model")
        self.assertEqual(model["source"], "ip")
        self.assertEqual(pack["origin"], "pack")
        self.assertEqual(pack["source"], "clone")
        self.assertEqual(pack["labels"]["origin"], "pack")
        self.assertTrue((Path(self.tmp.name) / "zh-f" / "prompt.wav").is_file())
        self.assertTrue((Path(self.tmp.name) / "zh-f" / "prompt.txt").is_file())
        with self.assertRaises(tts_voice.VoiceError) as cm:
            self.lib.delete("灵小柔")
        self.assertEqual(cm.exception.status_code, 400)
        with self.assertRaises(tts_voice.VoiceError) as cm:
            self.lib.delete("zh-f")
        self.assertEqual(cm.exception.status_code, 400)
        with self.assertRaises(tts_voice.VoiceError) as cm:
            self.lib.edit("US_James", name="nope")
        self.assertEqual(cm.exception.status_code, 400)

    def test_user_clone_survives_reconcile_and_can_be_deleted(self):
        self.lib.add_clone("user-1", "Mine", _wav(), "你好。")
        self.lib.add_design("山东话女声", "山东话女声", "山东话，可爱温柔的青年女声。")
        self.lib.reconcile(self.cards, ROOT / "voices")
        self.assertEqual(self.lib.get("user-1")["origin"], "user")
        designed = self.lib.turn_from_card(self.lib.get("山东话女声"), "你好。")
        self.assertEqual(designed.task, "design")
        self.assertEqual(designed.style, "山东话，可爱温柔的青年女声。")
        self.assertEqual(self.lib.delete("user-1")["status"], "ok")
        self.assertIsNone(self.lib.get("user-1"))
        with self.assertRaises(tts_voice.VoiceError) as cm:
            self.lib.add_clone("灵小柔", "灵小柔", _wav(), "你好。")
        self.assertEqual(cm.exception.status_code, 400)

    def test_turn_follows_the_card(self):
        ip = self.lib.turn_from_card(self.lib.get("US_James"), "Hello.")
        clone = self.lib.turn_from_card(self.lib.get("en-f"), "Hello.")
        self.assertEqual(ip.task, "ip")
        self.assertEqual(ip.ip, "US_James")
        self.assertEqual(clone.task, "clone")
        self.assertTrue(clone.prompt_wav.endswith("prompt.wav"))
        self.assertTrue(clone.prompt_text)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = tts_voice.VoiceLibrary(self.tmp.name)
        cards = [c for c in ming_catalog.load_ip_catalog() if c["voice_id"] == "US_James"]
        self.lib.reconcile(cards, ROOT / "voices")

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_task(self):
        turn = tts_voice.resolve_openai({"input": "Hello.", "voice": "US_James"}, self.lib)
        self.assertEqual(turn.task, "ip")
        turn = tts_voice.resolve_openai(
            {"input": "你好。", "instructions": "成熟男声"}, self.lib)
        self.assertEqual(turn.task, "design")
        self.assertEqual(turn.style, "成熟男声")
        raw = base64.b64encode(_wav()).decode("ascii")
        turn = tts_voice.resolve_openai(
            {"input": "你好。", "ref_audio": "data:audio/wav;base64," + raw, "ref_text": "参考。"},
            self.lib)
        self.assertEqual(turn.task, "clone")
        self.assertEqual(turn.prompt_text, "参考。")

    def test_rejects_mixed_and_unwired_controls(self):
        with self.assertRaises(tts_voice.VoiceError):
            tts_voice.resolve_openai(
                {"input": "x", "voice": "US_James", "instructions": "warm"}, self.lib)
        with self.assertRaises(tts_voice.VoiceError):
            tts_voice.resolve_openai(
                {"input": "x", "ref_audio": "data:audio/wav;base64,YQ=="}, self.lib)
        with self.assertRaises(tts_voice.VoiceError) as cm:
            tts_voice.resolve_openai({"input": "x", "voice": "zh-f", "speed": 1.5}, self.lib)
        self.assertIn("speed", cm.exception.detail)
        turn = tts_voice.resolve_openai({"input": "x", "voice": "zh-f", "speed": 1}, self.lib)
        self.assertEqual(turn.task, "clone")
        self.assertEqual(turn.knobs, {})


class BackendTests(unittest.TestCase):
    def test_tasks_stay_separate_and_knobs_are_refused(self):
        seen = []

        class Model:
            sample_rate = 24000

            def synthesize(self, **kwargs):
                seen.append(kwargs)
                return np.zeros((1, 16), dtype="float32")

        backend = ming_backend.MingBackend(Model())
        audio, sr = backend.synthesize(tts_voice.Turn(
            task="ip", text="Hi", ip="US_James", language_id="en"))
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape, (16,))
        self.assertEqual(seen[0]["task"], "IP")
        self.assertEqual(seen[0]["controls"], {"ip": "US_James"})
        self.assertTrue(seen[0]["use_zero_spk_emb"])
        backend.synthesize(tts_voice.Turn(
            task="clone", text="Hi", prompt_wav="/tmp/a.wav", prompt_text="ref"))
        self.assertEqual(seen[1]["task"], "TTS")
        self.assertEqual(seen[1]["prompt_text"], "ref")
        backend.synthesize(tts_voice.Turn(task="design", text="Hi", style="calm"))
        self.assertEqual(seen[2]["task"], "VOICE_DESIGN")
        with self.assertRaises(ming_backend.UnsupportedControls):
            backend.synthesize(tts_voice.Turn(task="ip", text="Hi", ip="US_James",
                                              knobs={"speed": 1.5}))


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = tts_voice.VoiceLibrary(self.tmp.name)
        cards = [c for c in ming_catalog.load_ip_catalog() if c["voice_id"] == "灵小柔"]
        self.lib.reconcile(cards, ROOT / "voices")
        self.backend = _FakeBackend()
        self._env = {key: os.environ.get(key) for key in ("AUDIO_BASE", "MODEL_SUPPORTS")}
        os.environ["AUDIO_BASE"] = "ming"
        os.environ["MODEL_SUPPORTS"] = "supports_tts,supports_tts_clone,supports_tts_design"
        ming._state["ready"] = True
        ming._state["error"] = None
        self.client = TestClient(ming.build_app(
            ["tts", "tts_clone", "tts_design"], library=self.lib, backend=self.backend))

    def tearDown(self):
        ming._state["ready"] = False
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_speak_list_and_protect(self):
        listed = self.client.get("/v1/audio/voices").json()["voices"]
        origins = {row["voice"]: row["origin"] for row in listed}
        self.assertEqual(origins["灵小柔"], "model")
        self.assertEqual(origins["zh-f"], "pack")
        got = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "response_format": "pcm"})
        self.assertEqual(got.status_code, 200, got.text)
        self.assertEqual(got.headers["x-audio-task"], "ip")
        self.assertEqual(got.headers["x-audio-origin"], "model")
        self.assertEqual(self.backend.turns[-1].ip, "灵小柔")
        denied = self.client.delete("/v1/audio/voices/zh-f")
        self.assertEqual(denied.status_code, 400)
        designed = self.client.post("/v1/audio/voices", json={
            "name": "山东话女声", "instructions": "山东话，可爱温柔的青年女声。"})
        self.assertEqual(designed.status_code, 200, designed.text)
        self.assertEqual(designed.json()["origin"], "user")
        self.assertEqual(designed.json()["source"], "design")
        spoken = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "山东话女声", "response_format": "pcm"})
        self.assertEqual(spoken.status_code, 200, spoken.text)
        self.assertEqual(spoken.headers["x-audio-task"], "design")
        self.assertEqual(self.backend.turns[-1].style, "山东话，可爱温柔的青年女声。")
        raw = base64.b64encode(_wav()).decode("ascii")
        cloned = self.client.post("/v1/audio/voices", json={
            "name": "我的克隆",
            "ref_audio": "data:audio/wav;base64," + raw,
            "ref_text": "参考这句。",
        })
        self.assertEqual(cloned.status_code, 200, cloned.text)
        self.assertEqual(cloned.json()["source"], "clone")
        again = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "我的克隆", "response_format": "pcm"})
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(again.headers["x-audio-task"], "clone")
        self.assertEqual(self.backend.turns[-1].prompt_text, "参考这句。")
        clash = self.client.post("/v1/audio/voices", json={
            "name": "灵小柔", "instructions": "别的声"})
        self.assertEqual(clash.status_code, 400)
        mixed = self.client.post("/v1/audio/voices", json={
            "name": "两个都要", "instructions": "描述",
            "ref_audio": "data:audio/wav;base64," + raw, "ref_text": "参考。"})
        self.assertEqual(mixed.status_code, 400)
        mixed = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "instructions": "warm"})
        self.assertEqual(mixed.status_code, 400)

    def test_models_report_the_loaded_rate(self):
        ming._state["sample_rate"] = 44100
        got = self.client.get("/v1/models")
        self.assertEqual(got.status_code, 200, got.text)
        self.assertEqual(got.json()["data"][0]["meta"]["sample_rate"], 44100)


class DecodeCacheTests(unittest.TestCase):
    def test_widen_replaces_the_512_cap(self):
        import types
        mod = types.ModuleType("modeling_bailingmm")
        mod.MAX_CACHE_LEN = 512
        sys.modules["modeling_bailingmm"] = mod
        try:
            ming_backend.widen_decode_cache()
            self.assertEqual(mod.MAX_CACHE_LEN, ming_backend._DECODE_CACHE_LEN)
        finally:
            sys.modules.pop("modeling_bailingmm", None)


class _FakeBackend:
    def __init__(self):
        self.turns = []

    def synthesize(self, turn):
        self.turns.append(turn)
        return np.zeros(240, dtype="float32"), 24000


if __name__ == "__main__":
    unittest.main()
