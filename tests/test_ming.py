"""Ming voice cards and the shared speech surface, with no weights and no GPU."""
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

from wrapper.caps import ming_backend, ming_catalog, tts_el


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
        self.assertEqual(james["category"], "premade")
        self.assertNotIn("origin", james)
        self.assertNotIn("source", james)
        self.assertNotIn("origin", james["labels"])


class BackendTests(unittest.TestCase):
    def test_tasks_stay_separate_and_knobs_are_applied(self):
        seen = []

        class Model:
            sample_rate = 24000

            def synthesize(self, **kwargs):
                seen.append(kwargs)
                return np.zeros((1, 16), dtype="float32")

        backend = ming_backend.MingBackend(Model())
        audio, sr = backend.speak_native(
            {"voice_id": "US_James", "ip": "US_James", "labels": {"language": "en"}},
            "Hi", instruction="warm", settings={"speed": 1.0})
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.shape, (16,))
        self.assertEqual(seen[0]["task"], "IP")
        self.assertEqual(seen[0]["controls"], {"ip": "US_James"})
        self.assertEqual(seen[0]["cfg"], 2.0)
        self.assertEqual(seen[0]["language_id"], "en")
        self.assertNotIn("instruction", seen[0])
        self.assertNotIn("temperature", seen[0])
        self.assertNotIn("max_decode_steps", seen[0])
        backend.clone("Hi", np.zeros(16, dtype="float32"), 24000, "ref",
                      instruction="warmly",
                      settings={"stability": 0.2, "speed": 1.0})
        self.assertEqual(seen[1]["task"], "TTS")
        self.assertEqual(seen[1]["prompt_text"], "ref")
        self.assertEqual(seen[1]["instruction"], "warmly")
        self.assertEqual(seen[1]["cfg"], 2.18)
        self.assertTrue(seen[1]["use_spk_emb"])
        audio, sr, extra = backend.design(
            "calm", "Hi", settings={"style": 0.9, "speed": 1.0})
        self.assertEqual(seen[2]["task"], "VOICE_DESIGN")
        self.assertEqual(seen[2]["controls"], {"style": "calm"})
        self.assertNotIn("instruction", seen[2])
        self.assertEqual(seen[2]["cfg"], 2.72)
        self.assertEqual(extra, {})
        self.assertEqual(audio.shape, (16,))
        backend.clone("Hi", np.zeros(16, dtype="float32"), 24000, "")
        self.assertIsNone(seen[3]["prompt_text"])
        self.assertTrue(seen[3]["use_spk_emb"])
        self.assertNotIn("instruction", seen[3])
        backend.clone(
            "Hi", np.zeros(16, dtype="float32"), 24000, "ref",
            sampling={"x_vector_only": True, "temperature": 0.4, "max_new_tokens": 30,
                      "top_k": 10, "language": "Auto"})
        self.assertIsNone(seen[4]["prompt_text"])
        self.assertEqual(seen[4]["temperature"], 0.4)
        self.assertEqual(seen[4]["max_decode_steps"], 30)
        self.assertNotIn("top_k", seen[4])
        self.assertNotIn("language_id", seen[4])
        backend.speak_native(
            {"voice_id": "US_James", "ip": "US_James", "labels": {"language": "en"}},
            "Hi", sampling={"language": "ja", "temperature": 0.2})
        self.assertEqual(seen[5]["language_id"], "ja")
        self.assertEqual(seen[5]["temperature"], 0.2)
        self.assertNotIn("instruction", seen[5])

    def test_quiet_minute_in_a_clone_is_replaced(self):
        calls = []

        class Model:
            sample_rate = 24000

            def synthesize(self, **kwargs):
                calls.append(kwargs)
                n = len(kwargs["text"])
                if n > 30:
                    quiet = np.zeros(24000 * 60, dtype="float32")
                    voiced = np.full(24000 * 2, 0.2, dtype="float32")
                    return np.concatenate([quiet, voiced])
                return np.full(24000, 0.2, dtype="float32")

        backend = ming_backend.MingBackend(Model())
        text = "先生您好请问您是二十二号楼二单元二楼二零二户的住户吗您认识陈无欲。短句在后。"
        audio, sr = backend.clone(
            text, np.zeros(1600, dtype="float32"), 24000, "参考这句。")
        self.assertEqual(sr, 24000)
        self.assertGreater(len(calls), 1)
        self.assertLess(len(audio) / sr, 20.0)
        self.assertGreater(float(np.max(audio)), 0.1)
        paths = {c["prompt_wav_path"] for c in calls}
        self.assertEqual(len(paths), 1)
        self.assertTrue(all(c["prompt_text"] == "参考这句。" for c in calls))
        long = next(c for c in calls if len(c["text"]) > 30)
        self.assertEqual(long["max_decode_steps"], ming_backend._steps_for(long["text"]))
        spoken = [c["text"] for c in calls if len(c["text"]) <= 30]
        self.assertTrue(spoken)
        self.assertIn("短句在后。", spoken)

    def test_address_twos_are_rewritten_before_the_model(self):
        seen = []

        class Model:
            sample_rate = 24000

            def synthesize(self, **kwargs):
                seen.append(kwargs["text"])
                return np.full(24000, 0.2, dtype="float32")

        text = "先生，您好！请问您是22号楼2单元2楼202户的住户吗？您认识陈无欲先生吗？"
        backend = ming_backend.MingBackend(Model())
        backend.clone(text, np.zeros(1600, dtype="float32"), 24000, "参考这句。")
        spoken = "".join(seen)
        self.assertIn("二十二号楼", spoken)
        self.assertIn("二单元", spoken)
        self.assertIn("二楼", spoken)
        self.assertIn("二零二户", spoken)
        self.assertNotIn("2", spoken)
        self.assertIn("陈无欲", spoken)

    def test_higher_speed_makes_shorter_audio(self):
        class Model:
            sample_rate = 24000

            def synthesize(self, **kwargs):
                return np.full(24000, 0.2, dtype="float32")

        backend = ming_backend.MingBackend(Model())
        card = {"voice_id": "US_James", "ip": "US_James", "labels": {"language": "en"}}
        normal, _sr = backend.speak_native(card, "Hi", settings={"speed": 1.0})
        faster, _sr = backend.speak_native(card, "Hi", settings={"speed": 2.0})
        self.assertLess(len(faster), len(normal) * 0.7)
        slower, _sr = backend.speak_native(card, "Hi", settings={"speed": 0.5})
        self.assertGreater(len(slower), len(normal) * 1.6)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cards = [c for c in ming_catalog.load_ip_catalog() if c["voice_id"] == "灵小柔"]
        cards.append({
            "voice_id": "zh-f", "name": "zh-f", "category": "premade",
            "description": "", "labels": {"language": "zh"},
        })
        self.backend = _FakeBackend()
        tts_el._MODULE["name"] = "ming"
        tts_el.VOICE_DIR = Path(self.tmp.name)
        tts_el.install(self.backend, store=tts_el.VoiceStore(self.tmp.name, cards),
                       sample_rate=24000)
        self.client = TestClient(tts_el.build_app(
            ["tts", "tts_clone", "tts_design"], module="ming",
            openai_alias=True, task_legacy=True))

    def tearDown(self):
        tts_el._state["ready"] = False
        tts_el._state["backend"] = None
        tts_el._state["store"] = None
        self.tmp.cleanup()

    def test_speak_list_save_and_protect(self):
        listed = self.client.get("/v1/audio/voices").json()["voices"]
        row = next(item for item in listed if item["voice_id"] == "灵小柔")
        self.assertEqual(row["id"], "灵小柔")
        self.assertEqual(row["voice"], "灵小柔")
        self.assertNotIn("origin", row)
        self.assertNotIn("source", row)
        self.assertNotIn("origin", row.get("labels") or {})
        got = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "instructions": "warm",
            "response_format": "pcm"})
        self.assertEqual(got.status_code, 200, got.text)
        self.assertNotIn("x-audio-origin", got.headers)
        self.assertNotIn("x-audio-task", got.headers)
        self.assertEqual(self.backend.native[-1][0]["voice_id"], "灵小柔")
        self.assertEqual(self.backend.native[-1][2], "warm")
        denied = self.client.delete("/v1/audio/voices/zh-f")
        self.assertEqual(denied.status_code, 400)
        designed = self.client.post("/v1/audio/voices", json={
            "name": "山东话女声",
            "instructions": "山东话，可爱温柔的青年女声。",
            "input": "今天天气不错。",
        })
        self.assertEqual(designed.status_code, 200, designed.text)
        body = designed.json()
        self.assertTrue(body["voice_id"].startswith("des"))
        self.assertEqual(body["name"], "山东话女声")
        self.assertNotIn("origin", body)
        slot = Path(self.tmp.name) / body["voice_id"]
        self.assertTrue((slot / "prompt.wav").is_file())
        self.assertEqual((slot / "prompt.txt").read_text(), "今天天气不错。")
        spoken = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": body["voice_id"], "response_format": "pcm"})
        self.assertEqual(spoken.status_code, 200, spoken.text)
        self.assertEqual(self.backend.clones[-1][3], "今天天气不错。")
        raw = base64.b64encode(_wav()).decode("ascii")
        cloned = self.client.post("/v1/audio/voices", json={
            "name": "我的克隆",
            "ref_audio": "data:audio/wav;base64," + raw,
            "ref_text": "参考这句。",
        })
        self.assertEqual(cloned.status_code, 200, cloned.text)
        self.assertTrue(cloned.json()["voice_id"].startswith("clone"))
        self.assertEqual(cloned.json()["name"], "我的克隆")
        self.assertNotEqual(cloned.json()["voice_id"], "我的克隆")
        defaults = self.client.get("/v1/voices/settings/default").json()
        self.assertEqual(defaults["stability"], 0.5)
        self.assertEqual(defaults["similarity_boost"], 0.75)
        self.assertEqual(defaults["style"], 0.0)
        self.assertIs(defaults["use_speaker_boost"], True)
        self.assertEqual(defaults["speed"], 1.0)

    def test_openai_call_matches_the_separate_models(self):
        missing = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "no-such", "response_format": "pcm"})
        self.assertEqual(missing.status_code, 400, missing.text)
        self.assertIn("unknown voice", missing.json()["detail"])
        self.assertIn("GET /v1/audio/voices", missing.json()["detail"])
        el_missing = self.client.post("/v1/text-to-speech/no-such", json={"text": "你好。"})
        self.assertEqual(el_missing.status_code, 404, el_missing.text)
        el_opus = self.client.post("/v1/text-to-speech/灵小柔", json={
            "text": "你好。", "output_format": "opus"})
        self.assertEqual(el_opus.status_code, 400, el_opus.text)
        opus = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "response_format": "opus"})
        self.assertEqual(opus.status_code, 200, opus.text)
        self.assertIn("audio", opus.headers["content-type"])
        bad_temp = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "temperature": "hot"})
        self.assertEqual(bad_temp.status_code, 400, bad_temp.text)
        sampled = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "response_format": "pcm",
            "temperature": 0.3, "top_k": 20, "top_p": 0.9,
            "max_new_tokens": 40, "repetition_penalty": 1.1, "language": "Chinese"})
        self.assertEqual(sampled.status_code, 200, sampled.text)
        self.assertEqual(self.backend.native[-1][2], "")
        self.assertEqual(self.backend.native[-1][4]["temperature"], 0.3)
        self.assertEqual(self.backend.native[-1][4]["top_k"], 20)
        self.assertEqual(self.backend.native[-1][4]["max_new_tokens"], 40)
        self.assertEqual(self.backend.native[-1][4]["language"], "Chinese")
        streamed = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "stream": True, "response_format": "pcm"})
        self.assertEqual(streamed.status_code, 200, streamed.text)
        self.assertTrue(streamed.content)
        blocked = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "voice": "灵小柔", "stream": True, "response_format": "mp3"})
        self.assertEqual(blocked.status_code, 400, blocked.text)
        self.assertIn("pcm or wav", blocked.json()["detail"])
        batch = self.client.post("/v1/audio/speech/batch", json={
            "response_format": "pcm",
            "items": [
                {"input": "你好。", "voice": "灵小柔"},
                {"input": "你好。", "instructions": "明亮的年轻女声，说普通话"},
            ],
        })
        self.assertEqual(batch.status_code, 200, batch.text)
        rows = batch.json()["items"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["audio"])
        self.assertEqual(rows[1]["index"], 1)
        self.assertEqual(self.backend.designs[-1][0], "明亮的年轻女声，说普通话")
        empty = self.client.post("/v1/audio/speech/batch", json={"items": []})
        self.assertEqual(empty.status_code, 400, empty.text)
        raw = base64.b64encode(_wav()).decode("ascii")
        xvec = self.client.post("/v1/audio/speech", json={
            "input": "你好。", "response_format": "pcm",
            "ref_audio": "data:audio/wav;base64," + raw,
        })
        self.assertEqual(xvec.status_code, 200, xvec.text)
        self.assertEqual(self.backend.clones[-1][3], "")
        self.assertTrue(self.backend.clones[-1][4]["x_vector_only"])
        el_ref = self.client.post("/v1/text-to-speech/灵小柔", json={
            "text": "你好。", "ref_audio": "data:audio/wav;base64," + raw})
        self.assertEqual(el_ref.status_code, 400, el_ref.text)
        preview = self.client.post("/v1/text-to-voice/design", json={
            "voice_description": "明亮的年轻女声，说普通话", "text": "你好。"})
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertIsNone(preview.json()["previews"][0]["plan"])
        with self.client.websocket_connect("/v1/audio/speech/stream") as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "session.ready")
            ws.send_json({
                "type": "session.config", "voice": "nope", "language": "Chinese",
                "chunk_size": 8, "instructions": "warm",
            })
            err = ws.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertIn("unknown voice", err["message"])
            again = ws.receive_json()
            self.assertEqual(again["type"], "session.ready")
            ws.send_json({"type": "session.update", "session": {"voice": "灵小柔"}})
            ws.send_json({"type": "input.done", "text": "你好。"})
            self.assertTrue(ws.receive_bytes())
            done = ws.receive_json()
            self.assertEqual(done["type"], "session.done")
        self.assertEqual(self.backend.native[-1][2], "warm")
        self.assertEqual(self.backend.native[-1][4]["language"], "Chinese")

    def test_models_report_the_loaded_rate(self):
        tts_el._state["sample_rate"] = 44100
        client = TestClient(tts_el.build_app(
            ["tts", "tts_clone", "tts_design"], module="ming",
            openai_alias=True, task_legacy=True))
        got = client.get("/v1/models")
        self.assertEqual(got.status_code, 200, got.text)
        self.assertEqual(got.json()["data"][0]["meta"]["sample_rate"], 44100)


class _FakeBackend:
    sample_rate = 24000
    uses_shared_pack = True
    prefer_design_speak = False
    card_instruction_is_direction = False

    def __init__(self):
        self.native = []
        self.clones = []
        self.designs = []

    def presets(self):
        return []

    def speak_native(self, meta, text, instruction=None, ctx=None, settings=None, sampling=None):
        self.native.append((meta, text, instruction, settings, sampling))
        return np.zeros(240, dtype="float32"), 24000

    def clone(self, text, prompt_audio, prompt_sr, prompt_text, **kw):
        self.clones.append((text, prompt_sr, prompt_text, prompt_text, kw.get("sampling")))
        return np.zeros(240, dtype="float32"), 24000

    def design(self, instruction, text, **kw):
        self.designs.append((instruction, text))
        return np.zeros(240, dtype="float32"), 24000, {}


if __name__ == "__main__":
    unittest.main()
