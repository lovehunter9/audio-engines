import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import time

import httpx


@dataclass(frozen=True)
class Worker:
    name: str
    command: tuple[str, ...]
    health_url: str
    environment: dict[str, str] = field(default_factory=dict)
    timeout: float = 600


class Workers:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.processes = {}
        self.logs = []
        self.specifications = {}

    async def start(self, worker):
        if worker.name in self.processes:
            raise ValueError("duplicate worker: " + worker.name)
        self.directory.mkdir(parents=True, exist_ok=True)
        output = (self.directory / (worker.name + ".log")).open("ab", buffering=0)
        self.logs.append(output)
        environment = os.environ.copy()
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_PRELOAD",
            "CUDA_HOME",
            "CUDA_PATH",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
            "CPATH",
            "CMAKE_PREFIX_PATH",
        ):
            environment.pop(key, None)
        environment.update(worker.environment)
        process = await asyncio.create_subprocess_exec(
            *worker.command,
            env=environment,
            start_new_session=True,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.processes[worker.name] = process
        self.specifications[worker.name] = worker
        deadline = time.monotonic() + worker.timeout
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                while time.monotonic() < deadline:
                    if process.returncode is not None:
                        raise RuntimeError(
                            "worker exited: %s (%s)" % (worker.name, process.returncode)
                        )
                    try:
                        response = await client.get(worker.health_url)
                        if response.status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.25)
            raise TimeoutError("worker did not become ready: " + worker.name)
        except BaseException:
            await self.close()
            raise

    def assert_alive(self):
        for name, process in self.processes.items():
            if process.returncode is not None:
                raise RuntimeError("worker exited: %s (%s)" % (name, process.returncode))

    async def close(self):
        for process in reversed(list(self.processes.values())):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except asyncio.TimeoutError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        for output in self.logs:
            output.close()
        self.logs.clear()
        self.processes.clear()
        self.specifications.clear()
