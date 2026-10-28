"""One directory: an OpenAI card is an EL card, and the reverse."""
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException

from wrapper.caps.tts_el import VoiceStore
from wrapper.caps.tts_voice import VoiceError, VoiceLibrary


def _wav():
    buf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    buf.close()
    with wave.open(buf.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)
    data = Path(buf.name).read_bytes()
    os.unlink(buf.name)
    return data


class OneLibraryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.lib = VoiceLibrary(self.root)
        self.store = VoiceStore(self.root, [{
            "voice_id": "zh-f", "name": "zh-f", "category": "premade", "source": "clone",
        }])
        prem = self.root / "zh-f"
        prem.mkdir()
        (prem / "meta.json").write_text(
            '{"voice_id":"zh-f","name":"zh-f","category":"premade","source":"clone","origin":"pack"}',
            encoding="utf-8")
        (prem / "prompt.wav").write_bytes(_wav())
        (prem / "prompt.txt").write_text("共用包这句。", encoding="utf-8")

    def test_openai_cards_are_visible_to_el(self):
        wav = _wav()
        self.lib.add_clone("接口甲", "接口甲", wav, "这是一段参考。")
        self.lib.add_design("接口乙", "接口乙", "山东话，青年女声。")
        ids = {row["voice_id"] for row in self.store.list()}
        self.assertIn("接口甲", ids)
        self.assertIn("接口乙", ids)
        path, transcript, meta = self.store.prompt("接口甲")
        self.assertTrue(path.is_file())
        self.assertEqual(transcript.strip(), "这是一段参考。")
        self.assertEqual(meta["source"], "clone")
        design = self.store.get("接口乙")
        self.assertEqual(design["source"], "design")
        self.assertIn("山东话", design["instruction"])
        self.assertEqual(self.store.delete("接口甲"), {"status": "ok"})
        self.assertEqual(self.store.delete("接口乙"), {"status": "ok"})
        self.assertIsNone(self.lib.get("接口甲"))
        self.assertIsNone(self.lib.get("接口乙"))

    def test_el_clone_is_visible_to_openai(self):
        meta = self.store.add_clone("乙的显示名", _wav(), "EL 存下来的逐字稿。")
        vid = meta["voice_id"]
        card = self.lib.get(vid)
        self.assertIsNotNone(card)
        self.assertEqual(card["source"], "clone")
        turn = self.lib.turn_from_card(card, "你好。")
        self.assertEqual(turn.task, "clone")
        self.assertEqual(turn.prompt_text, "EL 存下来的逐字稿。")
        self.assertTrue(Path(turn.prompt_wav).is_file())
        self.assertEqual(self.lib.delete(vid), {"status": "ok"})
        self.assertIsNone(self.store.get(vid))

    def test_el_design_is_visible_to_openai(self):
        vid = "des_probe"
        self.store._write(vid, {
            "voice_id": vid, "name": "设计显示名", "category": "generated",
            "source": "design", "instruction": "山东话，青年女声。", "frozen": True,
        }, wav_bytes=_wav(), transcript="今天天气很好。")
        card = self.lib.get(vid)
        turn = self.lib.turn_from_card(card, "你好。")
        self.assertEqual(turn.task, "design")
        self.assertEqual(turn.style, "山东话，青年女声。")
        self.assertEqual(self.lib.delete(vid), {"status": "ok"})

    def test_premade_is_refused_by_both(self):
        with self.assertRaises(VoiceError) as openai_err:
            self.lib.delete("zh-f")
        self.assertEqual(openai_err.exception.status_code, 400)
        with self.assertRaises(HTTPException) as el_err:
            self.store.delete("zh-f")
        self.assertEqual(el_err.exception.status_code, 400)
        self.assertTrue((self.root / "zh-f" / "meta.json").is_file())

    def test_pending_preview_is_not_a_voice(self):
        scratch = self.root / "pending" / "gen_deadbeef"
        scratch.mkdir(parents=True)
        (scratch / "meta.json").write_text('{"generated_voice_id":"gen_deadbeef"}', encoding="utf-8")
        self.assertNotIn("pending", {row["voice_id"] for row in self.lib.list()})
        self.assertNotIn("pending", {row["voice_id"] for row in self.store.list()})


if __name__ == "__main__":
    unittest.main()
