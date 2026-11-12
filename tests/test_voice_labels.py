import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from wrapper.caps import tts_el


MODULES = ("breeze", "breezeov", "firered", "fireredov", "ming", "mingov",
           "qwen3_voice", "qwen3_voiceov")


class VoiceLabelsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.shared = self.root / "shared"
        state, module = dict(tts_el._state), dict(tts_el._MODULE)
        self.addCleanup(lambda: (tts_el._state.clear(), tts_el._state.update(state),
                                 tts_el._MODULE.clear(), tts_el._MODULE.update(module)))
        self.wav = tts_el._wav_bytes(np.zeros(24000, dtype="float32"), 24000)

    def client(self, module="breeze"):
        store = tts_el.VoiceStore(self.root / module, [], shared_root=self.shared,
                                  shared_presets=[])
        tts_el.install(object(), store=store, sample_rate=24000)
        app = tts_el.build_app(["tts", "tts_clone", "tts_design"], module=module,
                               openai_alias=True)
        return TestClient(app), store

    def clone(self, client, labels=None, **fields):
        data = {"name": "克隆示例", "ref_text": "逐字转写。", **fields}
        if labels is not None:
            data["labels"] = labels
        return client.post("/v1/voices/add", data=data,
                           files={"files": ("reference.wav", self.wav, "audio/wav")})

    def preview(self, store):
        gid, _wav, _sr = store.put_preview(
            "温和的声音", np.zeros(24000, dtype="float32"), 24000, "试听文本。")
        return gid

    def test_creation_labels_persist_across_engines_and_reloads(self):
        saved = {}
        for module in MODULES:
            with self.subTest(writer=module):
                client, store = self.client(module)
                labels = {"language": "zh", "gender": "female", "age": "young",
                          "accent": "mandarin", "use_case": "narration", "test": module}
                response = self.clone(client, json.dumps(labels, ensure_ascii=False))
                self.assertEqual(response.status_code, 200, response.text)
                clone = response.json()["voice_id"]
                response = client.post("/v1/text-to-voice", json={
                    "generated_voice_id": self.preview(store), "voice_name": "设计示例",
                    "voice_description": "保存的描述", "labels": labels})
                self.assertEqual(response.status_code, 200, response.text)
                design = response.json()["voice_id"]
                self.assertEqual(response.json()["labels"], {**labels, "voice_scope": "shared"})
                saved.update({clone: labels, design: labels})
                self.assertEqual(store.prompt(clone)[1], "逐字转写。")
                self.assertEqual(store.prompt(design)[1], "试听文本。")
                self.assertEqual(store.get(design)["description"], "保存的描述")
        for module in MODULES:
            with self.subTest(reader=module):
                client, store = self.client(module)
                listed = {v["voice_id"]: v for v in client.get("/v1/voices").json()["voices"]}
                compatible = {v["voice_id"]: v for v in client.get("/v1/audio/voices").json()["voices"]}
                for vid, labels in saved.items():
                    expected = {**labels, "voice_scope": "shared"}
                    self.assertEqual(store.get(vid)["labels"], labels)
                    self.assertEqual(listed[vid]["labels"], expected)
                    self.assertEqual(compatible[vid]["labels"], expected)
                    self.assertEqual(client.get("/v1/voices/" + vid).json()["labels"], expected)

    def test_omitted_and_empty_labels_preserve_legacy_creation(self):
        client, store = self.client()
        for labels in (None, "", "{}"):
            response = self.clone(client, labels)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(store.get(response.json()["voice_id"]).get("labels", {}), {})
        for fields in ({}, {"labels": None}, {"labels": {}}):
            response = client.post("/v1/text-to-voice", json={
                "generated_voice_id": self.preview(store), "voice_name": "设计示例", **fields})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(store.get(response.json()["voice_id"]).get("labels", {}), {})

    def test_transcript_in_labels_remains_supported_and_is_saved(self):
        client, store = self.client()
        for key in ("transcript", "ref_text"):
            labels = {key: "标签中的逐字转写。", "language": "zh"}
            response = self.clone(client, json.dumps(labels, ensure_ascii=False), ref_text="")
            self.assertEqual(response.status_code, 200, response.text)
            vid = response.json()["voice_id"]
            self.assertEqual(store.prompt(vid)[1], labels[key])
            self.assertEqual(store.get(vid)["labels"], labels)

    def test_empty_transcript_label_keeps_reference_text_fallback(self):
        client, store = self.client()
        for empty in (None, False, 0, ""):
            response = self.clone(client, json.dumps({"transcript": empty, "ref_text": "备用转写。"}),
                                  ref_text="")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(store.prompt(response.json()["voice_id"])[1], "备用转写。")

    def test_invalid_labels_fail_without_creating_a_voice(self):
        client, store = self.client()
        gid = self.preview(store)
        for value in ("{bad", "[]", "null", "42"):
            response = self.clone(client, value)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("labels", response.json()["detail"])
        for value in ([], 42, True, "{bad"):
            response = client.post("/v1/text-to-voice", json={
                "generated_voice_id": gid, "voice_name": "设计示例", "labels": value})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("labels", response.json()["detail"])
        self.assertEqual(store.list(), [])

    def test_edit_can_preserve_replace_and_clear_creation_labels(self):
        client, store = self.client()
        response = self.clone(client, '{"language":"zh","custom":3}')
        self.assertEqual(response.status_code, 200, response.text)
        vid = response.json()["voice_id"]
        self.assertEqual(store.get(vid)["labels"], {"language": "zh", "custom": "3"})
        endpoint = "/v1/voices/" + vid + "/edit"
        self.assertEqual(client.post(endpoint, data={"name": "重命名"}).status_code, 200)
        self.assertEqual(store.get(vid)["labels"]["language"], "zh")
        for labels in ({"gender": "male"}, {}):
            response = client.post(endpoint, data={"name": "重命名", "labels": json.dumps(labels)})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(store.get(vid)["labels"], labels)


if __name__ == "__main__":
    unittest.main()
