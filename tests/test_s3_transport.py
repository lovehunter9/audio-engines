import asyncio
from collections import deque
from contextvars import ContextVar
from types import SimpleNamespace
import unittest

from wrapper.caps.s3_inproc import AsyncNativeClient
from wrapper.caps.s3_native_compat import _reduce_local_tensor


class Core:
    def __init__(self):
        self.pending = deque()
        self.events = []
        self.scheduler = SimpleNamespace(has_requests=lambda: bool(self.pending))
        self.failure = None

    def step_fn(self):
        if self.failure:
            raise self.failure
        result = self.pending.popleft()
        self.events.append(('step', result))
        return {0: result}, True, None

    def post_step(self, executed):
        self.events.append(('post', executed))

    def update_request_with_data(self, data):
        self.events.append(('data', data))


class Native:
    def __init__(self):
        self.engine_core = Core()
        self.closed = False

    def add_request(self, request):
        self.engine_core.events.append(('add', request.request_id))
        self.engine_core.pending.append(request.request_id)

    def collective_rpc(self, method, timeout, args, kwargs):
        self.engine_core.events.append(('rpc', method))
        return list(args)

    def shutdown(self):
        self.closed = True


class AsyncTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_allocator_context_is_preserved_in_the_execution_thread(self):
        marker = ContextVar("native_allocator")
        marker.set("original allocator")
        native = Native()
        original = native.engine_core.step_fn
        def step():
            if marker.get() != "original allocator":
                raise RuntimeError("Native execution lost its allocator")
            return original()
        native.engine_core.step_fn = step
        client = AsyncNativeClient(native, lambda: None)
        marker.set("different caller context")
        try:
            await client.add_request_async(SimpleNamespace(request_id="allocated"))
            self.assertEqual(await asyncio.wait_for(client.get_output_async(), 2), "allocated")
        finally:
            client.shutdown()

    async def test_native_requests_outputs_and_external_data_keep_their_order(self):
        native = Native()
        client = AsyncNativeClient(native, lambda: None)
        try:
            self.assertEqual(await client.collective_rpc_async('mode', args=('text',)), ['text'])
            await client.update_request_with_data_async('embedding')
            await client.add_request_async(SimpleNamespace(request_id='one'))
            self.assertEqual(await asyncio.wait_for(client.get_output_async(), 2), 'one')
            await client.add_request_async(SimpleNamespace(request_id='two'))
            self.assertEqual(await asyncio.wait_for(client.get_output_async(), 2), 'two')
            events = native.engine_core.events
            self.assertEqual(events[:3], [('rpc', 'mode'), ('data', 'embedding'), ('add', 'one')])
            self.assertEqual(sum(event == ('post', True) for event in events), 2)
        finally:
            client.shutdown()
        self.assertTrue(native.closed)
        self.assertFalse(client.thread.is_alive())

    async def test_original_executor_failure_is_forwarded_and_later_requests_fail(self):
        native = Native()
        client = AsyncNativeClient(native, lambda: None)
        native.engine_core.failure = RuntimeError('native executor failed')
        await client.add_request_async(SimpleNamespace(request_id='bad'))
        with self.assertRaisesRegex(RuntimeError, 'native executor failed'):
            await asyncio.wait_for(client.get_output_async(), 2)
        with self.assertRaisesRegex(RuntimeError, 'no longer running'):
            await client.update_request_with_data_async('later')


class DeviceTransportTests(unittest.TestCase):
    def test_same_process_transport_preserves_the_original_device_tensor(self):
        tensor = SimpleNamespace(is_cuda=True, device=SimpleNamespace(index=0))
        rebuild, metadata = _reduce_local_tensor(tensor)
        metadata = list(metadata)
        metadata[6] = 0
        self.assertIs(rebuild(*metadata), tensor)
        with self.assertRaises(KeyError):
            rebuild(*metadata)


if __name__ == '__main__':
    unittest.main()
