import asyncio
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from wrapper.s2st.youdao import Youdao
from wrapper.s2st.youdao_worker import close_idle_http_connections


class LoopbackTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_does_not_reuse_speech_connections_or_change_pcm(self):
        app = FastAPI()
        peers = []

        @app.post("/speech")
        async def speech(request: Request):
            peers.append(request.client.port)

            async def pcm():
                yield b"\x01\x00" * 100
                await asyncio.sleep(.01)
                yield b"\x02\x00" * 100

            return StreamingResponse(pcm(), media_type="application/octet-stream")

        close_idle_http_connections(app)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(.01)
            self.assertTrue(server.started)
            async with httpx.AsyncClient(trust_env=False) as client:
                for _ in range(2):
                    response = await client.post(f"http://127.0.0.1:{listener.getsockname()[1]}/speech")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["connection"], "close")
                    self.assertEqual(response.content, b"\x01\x00" * 100 + b"\x02\x00" * 100)
            self.assertEqual(len(peers), 2)
            self.assertNotEqual(*peers)
        finally:
            server.should_exit = True
            await asyncio.to_thread(thread.join, 3)
            listener.close()
        self.assertFalse(thread.is_alive())

    async def test_simultaneous_generator_and_public_cleanup_close_once(self):
        events = []

        async def close():
            events.append('close')
            await asyncio.sleep(.01)

        async def wait_closed():
            events.append('wait_closed')

        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url):
                return SimpleNamespace(raise_for_status=lambda: None,
                    json=lambda: {'active_speech_sessions': 0},
                    text='llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n')

        with tempfile.TemporaryDirectory() as directory:
            adapter = Youdao(Path(directory)/'session', directory)
            adapter.socket = SimpleNamespace(close=close, wait_closed=wait_closed)
            with patch('wrapper.s2st.youdao.httpx.AsyncClient', Client):
                await asyncio.gather(adapter.close(), adapter.close())
            self.assertIsNone(adapter.socket)
            self.assertEqual(events, ['close', 'wait_closed'])
