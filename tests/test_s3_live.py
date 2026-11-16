import threading
import time
import unittest
import numpy as np
from wrapper.caps.s3_live import LiveWave


class Cases(unittest.TestCase):
    def test_end_cannot_be_inferred_from_an_exact_nonfinal_prefix(self):
        wave = LiveWave()
        wave.configure(10240, 2600)
        wave.append(np.zeros(12840))
        result = []
        t = threading.Thread(target=lambda: result.append(len(wave)))
        t.start()
        time.sleep(0.04)
        self.assertTrue(t.is_alive())
        wave.append([1])
        t.join(1)
        self.assertEqual(result, [12841])
        self.assertEqual(len(wave[:12840]), 12840)
        result = []
        t = threading.Thread(target=lambda: result.append(len(wave)))
        t.start()
        time.sleep(0.04)
        self.assertTrue(t.is_alive())
        wave.finish()
        t.join(1)
        self.assertEqual(result, [12841])

    def test_end_flushes_a_partial_final_block_without_synthetic_samples(self):
        wave = LiveWave()
        wave.configure(10240, 2600)
        x = np.linspace(-0.3, 0.6, 1000, dtype=np.float32)
        wave.append(x)
        wave.finish()
        self.assertEqual(len(wave), 1000)
        np.testing.assert_array_equal(wave[:1000], x)
        np.testing.assert_array_equal(wave.snapshot(), x)

    def test_speaker_reads_do_not_advance_past_the_next_native_block(self):
        wave = LiveWave()
        wave.configure(10240, 2600)
        wave.append(np.zeros(24000))
        wave[:12840]
        wave[:12840]
        self.assertEqual(len(wave), 24000)

    def test_disconnect_unblocks_a_waiting_prefix(self):
        wave = LiveWave()
        errors = []

        def read():
            try:
                len(wave)
            except RuntimeError as e:
                errors.append(str(e))

        t = threading.Thread(target=read)
        t.start()
        wave.cancel()
        t.join(1)
        self.assertFalse(t.is_alive())
        self.assertTrue(errors)

    def test_overlong_audio_is_rejected_without_truncation(self):
        wave = LiveWave(maximum_samples=10)
        wave.append(np.arange(10))
        with self.assertRaises(ValueError):
            wave.append([11])
        wave.finish()
        np.testing.assert_array_equal(wave.snapshot(), np.arange(10, dtype=np.float32))


class StreamProductTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from types import SimpleNamespace
        from wrapper.caps.s3_live import mount_stream

        class Engine:
            def translate_live(self, incoming, source, target, emit):
                emit("text", source + " to " + target)
                while True:
                    packet = incoming.read()
                    if packet is None:
                        return
                    emit("audio", packet)

        self.lock = threading.Lock()
        app = FastAPI()
        self.runtime = SimpleNamespace(state={"ready": True, "error": None, "engine": Engine()})
        mount_stream(app, self.runtime, self.lock, str)
        self.client = TestClient(app)

    def test_duplex_output_precedes_source_end_and_next_session_still_works(self):
        for _ in range(2):
            with self.client.websocket_connect("/v1/audio/translations/stream") as ws:
                self.assertEqual(ws.receive_json()["event"], "ready")
                ws.send_json(
                    {"source_lang": "Chinese", "target_lang": "English", "sample_rate": 16000}
                )
                self.assertFalse(ws.receive_json()["finished"])
                ws.send_bytes(b"\x01\x00" * 5120)
                self.assertEqual(ws.receive_bytes(), b"\x01\x00" * 5120)
                ws.send_json({"event": "end"})
                self.assertTrue(ws.receive_json()["finished"])
            self.assertFalse(self.lock.locked())

    def test_disconnect_cancels_source_wait_and_releases_the_model(self):
        with self.client.websocket_connect("/v1/audio/translations/stream") as ws:
            ws.receive_json()
            ws.send_json({"source_lang": "Chinese", "target_lang": "English"})
            ws.receive_json()
        self.assertFalse(self.lock.locked())

    def test_invalid_source_rate_is_rejected_before_running_inference(self):
        with self.client.websocket_connect("/v1/audio/translations/stream") as ws:
            ws.receive_json()
            ws.send_json({"sample_rate": 48000})
            self.assertIn("16000", ws.receive_json()["error"])
        self.assertFalse(self.lock.locked())


if __name__ == "__main__":
    unittest.main()
