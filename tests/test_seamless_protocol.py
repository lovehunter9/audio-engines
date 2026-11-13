import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from wrapper.caps import seamless


class FakeEngine:
    def __init__(self):
        self.sessions = []
        self.speech_languages = {"cmn", "eng"}

    def session(self, target, *, text_only=False):
        session = {'target': target, 'text_only': text_only, 'calls': 0}
        self.sessions.append(session)
        return session

    def push(self, session, samples, target, finished):
        session['calls'] += 1
        return [('text', '结束' if finished else '开始'),
                ('samples', np.full(160, 0.1, dtype=np.float32))], finished


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.state = patch.dict(seamless._state, ready=True, error=None, engine=self.engine)
        self.state.start()
        self.client = TestClient(seamless.build_app(['audio_s2s']))

    def tearDown(self):
        self.client.close()
        self.state.stop()

    def test_output_before_end_and_session_reset(self):
        for _ in range(2):
            with self.client.websocket_connect('/v1/audio/translations/stream') as ws:
                self.assertEqual(ws.receive_json()['sample_rate'], 16000)
                ws.send_json({'source_lang': 'English', 'target_lang': 'Chinese', 'sample_rate':16000})
                ws.send_bytes(b'\0\0' * 5120)
                self.assertFalse(ws.receive_json()['finished'])
                self.assertTrue(ws.receive_bytes())
                ws.send_json({'event':'end'})
                self.assertFalse(ws.receive_json()['finished'])
                self.assertTrue(ws.receive_bytes())
                self.assertTrue(ws.receive_json()['finished'])
        self.assertEqual(len(self.engine.sessions), 2)
        self.assertTrue(all(s['calls'] == 2 for s in self.engine.sessions))

    def test_text_only_stream_has_no_audio_frames(self):
        with self.client.websocket_connect('/v1/audio/translations/stream') as ws:
            ws.receive_json()
            ws.send_json({'source_lang':'English', 'target_lang':'Chinese', 'response_format':'text'})
            ws.send_bytes(b'\0\0' * 5120)
            self.assertEqual(ws.receive_json()['text'], '开始')
            ws.send_json({'event':'end'})
            self.assertEqual(ws.receive_json()['text'], '结束')
            self.assertTrue(ws.receive_json()['finished'])
        self.assertTrue(self.engine.sessions[0]['text_only'])

    def test_streamed_subword_boundaries_are_not_replaced_with_spaces(self):
        for fragments, expected in [(["inter", "net"], "internet"),
                                    (["first", " second"], "first second")]:
            values = iter(fragments)
            self.engine.push = lambda session, samples, target, finished: ([("text", next(values))], finished)
            state = seamless._new_states()
            state.source = [0.0] * 5120
            first, _ = seamless._drain(state, "cmn", "eng", False)
            last, done = seamless._drain(state, "cmn", "eng", True)
            self.assertTrue(done)
            self.assertEqual("".join(value for kind, value in first + last if kind == "text"), expected)

    def test_invalid_language_and_empty_upload_are_rejected(self):
        response=self.client.post('/v1/audio/translations', files={'file':('a.wav',b'')},
                                  data={'source_lang':'English','target_lang':'Chinese'})
        self.assertEqual(response.status_code,400)
        response=self.client.post('/v1/audio/translations', files={'file':('a.wav',b'a')},
                                  data={'source_lang':'unknown','target_lang':'Chinese'})
        self.assertEqual(response.status_code,400)
        self.assertEqual(len(self.engine.sessions),0)


if __name__=='__main__':
    unittest.main()
