import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from wrapper.caps import breeze, firered, ming_backend, qwen3_voice, tts_el


def _write_clones(shared, private):
    store = tts_el.VoiceStore(private, [], shared_root=shared, shared_presets=[])
    for index in range(12):
        store.add_clone("speaker %d" % index, b"sample", "reference")


class SharedVoiceLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shared = self.root / "shared"
        previous = dict(tts_el._state)
        self.addCleanup(lambda: (tts_el._state.clear(), tts_el._state.update(previous)))
        native = self.root / "qwen-native"
        qwen3_voice.bake_native(native, generate=lambda *args: (np.zeros(1600), 16000))
        self.backends = {
            "firered": firered.FireRedBackend(None),
            "breeze": breeze.BreezeBackend(None, None, None, None),
            "ming": ming_backend.MingBackend(object()),
            "qwen": qwen3_voice.QwenVoiceBackend(object(), native),
        }
        self.stores = {name: self.install(name) for name in self.backends}
        self.wav = (tts_el.voices_root() / "zh-f" / "prompt.wav").read_bytes()

    def install(self, name):
        with patch.object(tts_el, "VOICE_DIR", self.root / name), \
                patch.object(tts_el, "SHARED_VOICE_DIR", str(self.shared)):
            return tts_el.install(self.backends[name])

    def save_pair(self):
        first = self.stores["breeze"]
        clone = first.add_clone("my clone", self.wav, "reference words")
        generated, _wav, _sr = first.put_preview(
            "calm voice", np.zeros(1600, dtype="float32"), 16000, "design preview")
        design = first.promote_preview(generated, "my design")
        return clone["voice_id"], design["voice_id"]

    def test_saved_samples_and_ids_are_shared_by_all_four_engines(self):
        clone, design = self.save_pair()
        for name, store in self.stores.items():
            with self.subTest(name=name):
                self.assertTrue({clone, design} <= {row["voice_id"] for row in store.list()})
                self.assertEqual(store.prompt(clone)[0].read_bytes(), self.wav)
                self.assertEqual(store.prompt(clone)[1], "reference words")
                self.assertEqual(store.prompt(design)[1], "design preview")
                self.assertFalse((store.root / clone).exists())
        self.stores["ming"].edit(clone, name="renamed")
        self.stores["qwen"].edit_settings(clone, {"speed": 1.2})
        for store in self.stores.values():
            self.assertEqual(store.get(clone)["name"], "renamed")
            self.assertEqual(store.settings(clone)["speed"], 1.2)
        self.stores["firered"].delete(clone)
        self.assertTrue(all(store.get(clone) is None for store in self.stores.values()))

    def test_restart_preserves_shared_voices_and_other_native_catalogs(self):
        clone, design = self.save_pair()
        self.install("firered")
        self.assertTrue((self.root / "ming" / "US_James" / "meta.json").is_file())
        self.assertTrue((self.root / "qwen" / "Vivian" / "meta.json").is_file())
        for name, store in self.stores.items():
            self.assertIsNotNone(store.get(clone))
            self.assertIsNotNone(store.get(design))
            ids = {row["voice_id"] for row in store.list()}
            self.assertEqual("US_James" in ids, name == "ming")
            self.assertEqual("Vivian" in ids, name == "qwen")
            if name != "ming":
                self.assertIsNone(store.get("US_James"))
        self.assertFalse((self.shared / "pending").exists())
        self.assertFalse((self.shared / "history").exists())
        for store in self.stores.values():
            self.assertEqual(tts_el._history_dir(store), store.root / "history")
            with self.assertRaises(tts_el.HTTPException):
                store.delete("zh-f")

    def test_el_saved_voice_speaks_with_each_selected_engine(self):
        clone, design = self.save_pair()
        for name, backend in self.backends.items():
            with self.subTest(name=name):
                store = self.install(name)
                with patch.object(backend, "clone", return_value=(np.zeros(1600), 16000)) as speak:
                    client = TestClient(tts_el.build_app(
                        ["tts", "tts_clone", "tts_design"], module="ming", openai_alias=True))
                    for vid in (clone, design):
                        response = client.post("/v1/text-to-speech/" + vid,
                                               json={"text": "new words", "output_format": "wav"})
                        self.assertEqual(response.status_code, 200, response.text[:300])
                    self.assertEqual(speak.call_count, 2)
                    self.assertEqual(speak.call_args.args[3], "design preview")
                    card = client.get("/v1/voices/" + clone).json()
                    self.assertEqual(card["labels"]["voice_scope"], "shared")
                    if store.native_ids:
                        card = client.get("/v1/voices/" + sorted(store.native_ids)[0]).json()
                        self.assertEqual(card["labels"]["voice_scope"], "model")
                    other = client.get("/v1/audio/voices").json()["voices"]
                    self.assertIn(clone, {row["voice_id"] for row in other})

    def test_incomplete_saved_samples_are_rejected_without_redesign(self):
        clone, design = self.save_pair()
        store = self.install("ming")
        store.prompt(clone)[0].with_name("prompt.txt").unlink()
        store.prompt(design)[0].unlink()
        with patch.object(self.backends["ming"], "design") as redesign:
            for vid in (clone, design):
                with self.assertRaises(tts_el.HTTPException) as raised:
                    tts_el._ensure_frozen(vid)
                self.assertEqual(raised.exception.status_code, 400)
            redesign.assert_not_called()

    def test_invalid_directory_configuration_is_rejected_before_seeding(self):
        native = self.root / "ming" / "US_James" / "meta.json"
        before = native.read_bytes()
        with patch.object(tts_el, "VOICE_DIR", self.root / "ming"), \
                patch.object(tts_el, "SHARED_VOICE_DIR", str(self.root / "ming")):
            with self.assertRaises(ValueError):
                tts_el.install(self.backends["ming"])
        self.assertEqual(native.read_bytes(), before)

    def test_independent_writers_publish_complete_cards(self):
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_write_clones, args=(str(self.shared), str(self.root / ("writer-%d" % i))))
                     for i in range(2)]
        try:
            for process in processes:
                process.start()
            deadline = time.monotonic() + 20
            while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
                for row in self.stores["breeze"].list():
                    if row.get("category") == "cloned":
                        wav, transcript, _meta = self.stores["breeze"].prompt(row["voice_id"])
                        self.assertEqual(wav.read_bytes(), b"sample")
                        self.assertEqual(transcript, "reference")
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
        rows = [row for row in self.stores["breeze"].list() if row.get("category") == "cloned"]
        self.assertEqual(len(rows), 24)
