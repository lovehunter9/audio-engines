import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient

from wrapper.s2st import server


class Runtime:
    def __init__(self):
        self.model_name = "test-cascade"
        self.model_repo = "test/cascade"
        self.state = {"ready": True, "error": None}


class Producer:
    def __init__(self, model, directory, binary=None):
        self.directory = Path(directory)


def wav(pcm=b"\1\0" * 160):
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    return out.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.closed = []
        closed = self.closed

        async def pipeline(source, producer, emit, speech_url, *, streaming=False):
            try:
                first = True
                async for pcm in source:
                    if first:
                        await emit({"type": "translation_delta", "text": "A complete sentence."})
                        first = False
                    await emit(
                        {"type": "audio", "pcm_base64": base64.b64encode(b"\2\0" * 8).decode()}
                    )
                await emit(
                    {
                        "type": "done",
                        "translation": "A complete sentence.",
                        "source_samples": source.samples,
                    }
                )
            finally:
                closed.append(True)

        self.patches = [
            patch.dict(
                os.environ,
                {"AUDIO_BASE": "s2st-tencent", "ENGINE_ARGS": "--cache-dir " + self.folder.name},
            ),
            patch.object(server, "Runtime", Runtime),
            patch.object(server, "Tencent", Producer),
            patch.object(server, "run_pipeline", pipeline),
        ]
        for item in self.patches:
            item.start()
        # No lifespan: model inference is mocked, while all public transport/validation is real.
        self.client = TestClient(server.build_app(["audio_s2s"]))

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    def test_nonstream_wav_has_no_delivery_wait_silence(self):
        response = self.client.post(
            "/v1/audio/speech-translations",
            files={"file": ("source.wav", wav(), "audio/wav")},
            data={"direction": "en2zh"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        with wave.open(io.BytesIO(base64.b64decode(result["audio_base64"])), "rb") as reader:
            self.assertEqual(reader.getframerate(), 24000)
            self.assertEqual(reader.readframes(reader.getnframes()), b"\2\0" * 8)
        self.assertEqual(result["source_samples"], 160)

    def test_http_stream_keeps_all_events_and_terminal_ack(self):
        response = self.client.post(
            "/v1/audio/speech-translations",
            files={"file": ("source.wav", wav(), "audio/wav")},
            data={"direction": "zh2en", "stream": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        rows = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([r["type"] for r in rows], ["translation_delta", "audio", "done"])
        self.assertEqual(rows[-1]["source_samples"], 160)

    def test_live_pcm_produces_audio_before_end(self):
        with self.client.websocket_connect("/v1/audio/speech-translations/stream") as socket:
            socket.send_json({"direction": "zh2en"})
            self.assertEqual(socket.receive_json()["type"], "ready")
            socket.send_bytes(b"\1\0" * 160)
            self.assertEqual(socket.receive_json()["type"], "translation_delta")
            self.assertEqual(socket.receive_json()["type"], "audio")
            socket.send_json({"type": "end"})
            result = socket.receive_json()
            self.assertEqual(result["type"], "done")
            self.assertEqual(result["source_samples"], 160)

    def test_disconnect_cancels_pending_input_consumer(self):
        with self.client.websocket_connect("/v1/audio/speech-translations/stream") as socket:
            socket.send_json({"direction": "en2zh"})
            socket.receive_json()
            socket.send_bytes(b"\1\0" * 160)
            socket.receive_json()
            socket.receive_json()
        self.assertTrue(self.closed)

    def test_invalid_direction_and_partial_samples_fail_clearly(self):
        response = self.client.post(
            "/v1/audio/speech-translations",
            files={"file": ("source.wav", wav(), "audio/wav")},
            data={"direction": "en2de"},
        )
        self.assertEqual(response.status_code, 400)
        with self.client.websocket_connect("/v1/audio/speech-translations/stream") as socket:
            socket.send_json({"direction": "zh2en"})
            socket.receive_json()
            socket.send_bytes(b"x")
            error = socket.receive_json()
            self.assertEqual(error["type"], "error")
            self.assertIn("whole PCM16", error["message"])


if __name__ == "__main__":
    unittest.main()
