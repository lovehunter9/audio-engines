"""Async transport around the original in-process vLLM EngineCore."""
import asyncio
from concurrent.futures import Future
from contextvars import copy_context
import queue
import threading
from types import SimpleNamespace

GPU_EXECUTION = threading.RLock()


class AsyncNativeClient:
    engine_ranks_managed = [0]

    def __init__(self, native, synchronize):
        self.native = native
        self.synchronize = synchronize
        self.resources = SimpleNamespace(engine_dead=False)
        self.commands = queue.Queue()
        self.outputs = queue.Queue()
        context = copy_context()
        self.thread = threading.Thread(target=lambda: context.run(self._run), name="S3NativeCore", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.resources.engine_dead:
                try:
                    item = self.commands.get(timeout=.002)
                except queue.Empty:
                    item = None
                while item is not None:
                    function, args, kwargs, future = item
                    if future.set_running_or_notify_cancel():
                        try:
                            with GPU_EXECUTION:
                                result = function(*args, **kwargs)
                            future.set_result(result)
                        except BaseException as error:
                            future.set_exception(error)
                    try:
                        item = self.commands.get_nowait()
                    except queue.Empty:
                        item = None
                with GPU_EXECUTION:
                    core = self.native.engine_core
                    if core.scheduler.has_requests():
                        outputs, executed, _ = core.step_fn()
                        core.post_step(executed)
                        self.synchronize()
                    else:
                        outputs = None
                if outputs:
                    output = outputs.get(0)
                    if output is not None:
                        self.outputs.put(output)
        except BaseException as error:
            self.resources.engine_dead = True
            self.outputs.put(error)
            while True:
                try:
                    *_, future = self.commands.get_nowait()
                    if not future.done():
                        future.set_exception(error)
                except queue.Empty:
                    break

    async def _call(self, function, *args, **kwargs):
        if self.resources.engine_dead:
            raise RuntimeError("Original S3 EngineCore is no longer running")
        future = Future()
        self.commands.put((function, args, kwargs, future))
        return await asyncio.wrap_future(future)

    async def get_output_async(self, timeout=None):
        try:
            result = await asyncio.to_thread(self.outputs.get, True, timeout)
        except queue.Empty:
            from vllm.v1.engine import EngineCoreOutputs
            return EngineCoreOutputs()
        if isinstance(result, BaseException):
            raise result
        return result

    async def add_request_async(self, request):
        request.client_index = 0
        await self._call(self.native.add_request, request)

    async def update_request_with_data_async(self, external_data):
        await self._call(self.native.engine_core.update_request_with_data, external_data)

    async def abort_requests_async(self, request_ids):
        if request_ids and not self.resources.engine_dead:
            await self._call(self.native.abort_requests, request_ids)

    async def collective_rpc_async(self, method, timeout=None, args=(), kwargs=None):
        return await self._call(self.native.collective_rpc, method, timeout, args, kwargs)

    async def get_supported_tasks_async(self):
        return await self._call(self.native.get_supported_tasks)

    def dp_engines_running(self):
        return False

    def shutdown(self):
        if self.resources.engine_dead:
            return
        self.resources.engine_dead = True
        self.thread.join(timeout=5)
        with GPU_EXECUTION:
            self.native.shutdown()
        self.outputs.put(RuntimeError("Original S3 EngineCore stopped"))

    def __getattr__(self, name):
        if name.endswith("_async"):
            method = getattr(self.native, name[:-6])
            async def call(*args, **kwargs):
                return await self._call(method, *args, **kwargs)
            return call
        raise AttributeError(name)


def install():
    import torch
    from vllm.v1.engine.core_client import EngineCoreClient, InprocClient
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    def create(vllm_config, executor_class, log_stats, client_addresses=None,
               client_count=1, client_index=0, driver_tensor_queue_union=None,
               output_processor=None):
        if client_count != 1 or client_index != 0 or client_addresses:
            raise ValueError("S3 native in-process transport serves one local client")
        parallel = vllm_config.parallel_config
        if parallel.world_size != 1 or parallel.data_parallel_size != 1:
            raise ValueError("S3 native in-process transport requires one GPU")
        with GPU_EXECUTION:
            native = InprocClient(vllm_config, UniProcExecutor, log_stats,
                                  driver_tensor_queue_union=driver_tensor_queue_union)
        return AsyncNativeClient(native, torch.cuda.synchronize)

    EngineCoreClient.make_async_mp_client = staticmethod(create)
    if not getattr(torch.cuda.graph, "_s3_guarded", False):
        original = torch.cuda.graph
        class Graph(original):
            _s3_guarded = True
            def __enter__(self):
                GPU_EXECUTION.acquire()
                try:
                    torch.cuda.synchronize()
                    return super().__enter__()
                except BaseException:
                    GPU_EXECUTION.release()
                    raise
            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    GPU_EXECUTION.release()
        torch.cuda.graph = Graph


def use_original_eager_pcm(pcm):
    import inspect
    solver = pcm.fm.solve_euler
    if solver.__module__ != "s3_syncgraphs_entry":
        raise RuntimeError("The pinned official FM graph wrapper changed")
    original = inspect.getclosurevars(solver).nonlocals["original"]
    pcm.fm.solve_euler = original
    vocoder = pcm.voc.generator.inference_cache
    if vocoder.__module__ != "s3_vocgraph_entry":
        raise RuntimeError("The pinned official vocoder graph wrapper changed")
    graph_call = inspect.getclosurevars(vocoder).nonlocals["graph_call"]
    pcm.voc.generator.inference_cache = inspect.getclosurevars(graph_call).nonlocals["original"]
