"""Own only packaged child services; all inference stays in the frozen modules."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

from .layout import ROOT, PACKAGE, prepare
from .integrity import verify


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@asynccontextmanager
async def services(family, cache, models, role='s2st'):
    if role not in ('s2st', 's2t'):
        raise ValueError('Unknown fixed task role')
    verify(PACKAGE)
    manifest = prepare(family, cache, models)
    bundle = PACKAGE / family
    sys.path.insert(0, str(bundle))
    ming = load('locked_ming_lifetime', bundle / 'ming_services.py')
    directory = ROOT / 'runs' / ('service-' + family + '-' + str(time.time_ns()))
    options = SimpleNamespace(output=directory, scheme=family, kv='f16', asr_allocation='bounded',
                              t3po_context=8192, cases=[], input_mode='asap')
    @asynccontextmanager
    async def selected_context():
        if role == 's2st':
            async with ming.ming_services(options) as value:
                yield value
            return
        # Standalone Benchmark excludes MING exactly as the sealed text driver.
        directory.mkdir()
        state = {'state': 'starting_standalone_translation'}
        def save():
            (directory / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        yield dict(state=state, save=save, common_env=dict(os.environ))

    async with selected_context() as context:
        if family == 'youdao':
            sys.path.insert(0, str(ROOT / 'sources/Confucius4-T3PO'))
            upstream = load('locked_youdao_lifetime', bundle / 'upstream_services.py')
            async with upstream.upstream_services(options, directory, context['state'], context['save'], context['common_env']) as value:
                def alive():
                    return all(p.poll() is None for p in value['processes'].values()) and (
                        role == 's2t' or context['worker'].poll() is None)
                yield {'directory': directory, 'manifest': manifest, 'bundle': bundle, 'role': role,
                       'client_env': value['app_env'], 'is_alive': alive}
            return
        # Exact actual frontend/config launch from the measured Tencent harness.
        reference = ROOT / 'runs/official-tencent-filter-active-control2/measurements/config.toml'
        text = reference.read_text()
        if text.count('max_sessions = 4') != 1:
            raise RuntimeError('Packaged official configuration differs from the sealed reference')
        config = directory / 'config.toml'
        config.write_text(text.replace('max_sessions = 4', 'max_sessions = 1', 1))
        env = {key: value for key, value in context['common_env'].items() if key != 'PYTHONPATH'}
        if role == 's2t':
            from runtime_cuda_libraries import library_directories
            for key in ['PYTHONPATH', 'CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']:
                env.pop(key, None)
            env.update(LD_LIBRARY_PATH=':'.join(library_directories()), OMP_NUM_THREADS='4')
        command = [str(ROOT / 'envs/tencent-model/bin/python'), str(bundle / 'live_tencent_app.py'),
                   '--config', str(config), '--output', str(directory / 'app-runtime')]
        (directory / 'frontend-command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (directory / 'app.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Locked Tencent frontend exited during startup')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:25280/health', timeout=2) as response:
                            if json.load(response).get('ready', True):
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Locked Tencent frontend readiness timeout')
                    await asyncio.sleep(1)
                yield {'directory': directory, 'manifest': manifest, 'bundle': bundle, 'role': role, 'client_env': env,
                       'is_alive': lambda: process.poll() is None and (role == 's2t' or context['worker'].poll() is None)}
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
