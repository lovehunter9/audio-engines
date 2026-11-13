import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from wrapper.caps import s3_s2st


class FakeEngine:
    speech_languages={'cmn','eng'}

    def session(self,target,*,text_only=False):
        return {'calls':0}

    def push(self,state,samples,target,finished):
        state['calls']+=1
        return [('text','完成' if finished else '开始'),('samples',np.ones(240,dtype=np.float32)*0.1)],finished


class ProtocolTests(unittest.TestCase):
    def test_live_pcm_and_finish_use_the_advertised_rate(self):
        with patch.dict(s3_s2st._state,ready=True,error=None,engine=FakeEngine()):
            with TestClient(s3_s2st.build_app(['audio_s2s'])) as client:
                with client.websocket_connect('/v1/audio/translations/stream') as ws:
                    ready=ws.receive_json()
                    self.assertEqual(ready['input_sample_rate'],16000)
                    self.assertEqual(ready['sample_rate'],24000)
                    ws.send_bytes(b'\0\0'*2560)
                    self.assertFalse(ws.receive_json()['finished'])
                    self.assertEqual(len(ws.receive_bytes()),480)
                    ws.send_json({'event':'end'})
                    self.assertFalse(ws.receive_json()['finished'])
                    self.assertTrue(ws.receive_bytes())
                    self.assertTrue(ws.receive_json()['finished'])


if __name__=='__main__':
    unittest.main()
