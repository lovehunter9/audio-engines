import asyncio
import unittest

import websockets
from websockets.exceptions import ConnectionClosed

from wrapper.s2st.youdao_receive import BoundedReceiver


class UpstreamReceiveTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, drain):
        async def serve(socket):
            for index in range(100):
                await socket.send(str(index))
            try:
                await socket.recv()
            except ConnectionClosed:
                pass

        async with websockets.serve(serve, '127.0.0.1', 0, ping_interval=.02, ping_timeout=.05) as server:
            port = server.sockets[0].getsockname()[1]
            async with websockets.connect(f'ws://127.0.0.1:{port}', ping_interval=None, max_queue=16) as socket:
                incoming = BoundedReceiver(socket) if drain else None
                try:
                    await asyncio.sleep(.25)
                    if not drain:
                        with self.assertRaises(ConnectionClosed) as failure:
                            while True:
                                await socket.recv()
                        self.assertEqual(failure.exception.rcvd.code, 1011)
                        self.assertIn('keepalive ping timeout', failure.exception.rcvd.reason)
                        return
                    values = [await asyncio.wait_for(incoming.recv(), 1) for _ in range(100)]
                    self.assertEqual(values, [str(index) for index in range(100)])
                    self.assertLessEqual(incoming.peak_bytes, incoming.byte_limit)
                    await socket.send('all received')
                finally:
                    if incoming is not None:
                        await incoming.close()

    async def test_slow_consumer_reproduces_keepalive_failure_with_default_queue(self):
        await self.exercise(False)

    async def test_independent_bounded_receiver_preserves_order_and_heartbeats(self):
        await self.exercise(True)

    async def test_limit_failure_is_explicit_after_prior_events(self):
        queue = asyncio.Queue()
        await queue.put('abc')
        await queue.put('def')

        class Socket:
            async def recv(self):
                return await queue.get()

        receiver = BoundedReceiver(Socket(), byte_limit=5)
        try:
            await asyncio.sleep(.01)
            self.assertEqual(await receiver.recv(), 'abc')
            with self.assertRaisesRegex(BufferError, 'bounded delivery queue'):
                await receiver.recv()
            self.assertEqual(receiver.peak_bytes, 3)
        finally:
            await receiver.close()

    async def test_empty_receiver_cancellation_releases_pending_read(self):
        stopped = asyncio.Event()

        class Socket:
            async def recv(self):
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        receiver = BoundedReceiver(Socket())
        await asyncio.sleep(0)
        await receiver.close()
        self.assertTrue(stopped.is_set())
        self.assertTrue(receiver.task.done())
