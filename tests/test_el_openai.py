"""Breeze and FireRed open the shared OpenAI routes on the existing voice table."""
import os
import unittest
from unittest import mock

from fastapi import HTTPException

from wrapper.caps import breeze, firered, tts_el


class ElOpenAIEntryTest(unittest.TestCase):
    def test_both_ask_for_the_shared_openai_routes(self):
        for module in (breeze, firered):
            seen = {}

            def fake_run(*_args, **kwargs):
                seen["kwargs"] = kwargs

            original = module.tts_el.run
            module.tts_el.run = fake_run
            try:
                module.run(["tts", "tts_clone", "tts_design"])
            finally:
                module.tts_el.run = original
            self.assertTrue(seen["kwargs"].get("openai_alias"), module.__name__)
            self.assertTrue(seen["kwargs"].get("task_legacy"), module.__name__)

    def test_breeze_and_firered_have_no_embedding_only_clone(self):
        for backend in (breeze.BreezeBackend, firered.FireRedBackend):
            self.assertTrue(backend.requires_ref_transcript, backend.__name__)
            with mock.patch.object(tts_el, "_backend", return_value=backend):
                with self.assertRaises(HTTPException) as caught:
                    tts_el._require_ref_transcript("")
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn("ref_text", caught.exception.detail)

    def test_stream_is_per_sentence_and_the_socket_is_pcm(self):
        for backend in (breeze.BreezeBackend, firered.FireRedBackend):
            self.assertTrue(backend.streams_while_generating, backend.__name__)
            self.assertTrue(backend.socket_pcm_only, backend.__name__)

    def test_openai_routes_use_response_format_and_el_keeps_output_format(self):
        from fastapi.testclient import TestClient

        for module, base in ((breeze, "breeze"), (firered, "firered")):
            with mock.patch.dict(os.environ, {"AUDIO_BASE": base}, clear=False):
                app = module.build_app(["tts", "tts_clone", "tts_design"])
            with TestClient(app) as client:
                rows = client.get("/api/engine-spec").json()["endpoints"]
            available = {(row["method"], row["path"]) for row in rows if row["available"]}
            for route in (
                ("POST", "/v1/audio/speech"),
                ("POST", "/v1/audio/speech/batch"),
                ("GET", "/v1/audio/voices"),
                ("POST", "/v1/audio/voices"),
                ("DELETE", "/v1/audio/voices/{voice_id}"),
                ("WS", "/v1/audio/speech/stream"),
                ("POST", "/v1/audio/speech/clone"),
                ("GET", "/v1/audio/tasks"),
            ):
                self.assertIn(route, available, module.__name__)
            speech = next(row for row in rows
                          if row["method"] == "POST" and row["path"] == "/v1/audio/speech")
            speech_names = {item["name"]: item for item in speech["parameters"]}
            self.assertIn("response_format", speech_names)
            self.assertNotIn("output_format", speech_names)
            self.assertIn("opus", speech_names["response_format"]["enum"])
            el = next(row for row in rows if row["path"] == "/v1/text-to-speech/{voice_id}")
            el_names = {item["name"]: item for item in el["parameters"]}
            self.assertIn("output_format", el_names)
            self.assertNotIn("response_format", el_names)
            self.assertNotIn("opus", el_names["output_format"]["enum"])
        with self.assertRaises(HTTPException):
            tts_el._output_format("opus")
        self.assertEqual(tts_el._output_format("opus", openai=True).kind, "opus")

    def test_style_stays_out_of_the_spoken_instruction(self):
        styled = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.9,
                  "use_speaker_boost": True, "speed": 1.0}
        for base in (tts_el.INFERENCE_CFG, tts_el.DESIGN_CFG):
            kn = tts_el.resolve_settings(styled, base, "低沉一点", text="门口见。",
                                         include_speed_direction=False,
                                         include_style_direction=False)
            self.assertNotIn("语气夸张", kn.instruction)
            self.assertIn("低沉一点", kn.instruction)
        plain = tts_el.resolve_settings(None, tts_el.CFG_SCALE, "Speak clearly and naturally.",
                                        text="See you at the door.",
                                        include_speed_direction=False,
                                        include_style_direction=False)
        self.assertEqual(plain.instruction, "Speak clearly and naturally.")
        self.assertEqual(plain.cfg, tts_el.CFG_SCALE)
