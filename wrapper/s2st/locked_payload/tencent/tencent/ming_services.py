"""Load frozen MING and the verified Tencent backend together on003."""
from contextlib import asynccontextmanager
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
ROOT = Path('/home/olares/s2st-original-check-20261123')

@asynccontextmanager
async def ming_services(options):
    from runtime_cuda_libraries import library_directories
    from tencent_revised_rpc import NativePool
    out = options.output
    if not out.resolve().is_relative_to((ROOT / 'runs').resolve()):
        raise RuntimeError('Output must remain in the private run directory')
    out.mkdir(exist_ok=False)
    state = {'scheme': options.scheme, 'state': 'starting_frozen_MING', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

    def save():
        (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    python = str(ROOT / 'envs/tencent-model/bin/python')
    env = dict(os.environ)
    for name in ['PYTHONPATH', 'CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']:
        env.pop(name, None)
    sdk = ROOT / 'envs/cuda130-toolchain/nvidia/cu13'
    headers = ROOT / 'envs/python-build-headers/include'
    includes = ':'.join((str(path) for path in [headers / 'python3.12', headers, headers / 'x86_64-linux-gnu/python3.12']))
    env.update(PYTHONPATH=':'.join((str(ROOT / name) for name in ['scripts', 'envs/ming-overlay', 'sources/Ming-frozen'])), LD_LIBRARY_PATH=':'.join(library_directories()), OMP_NUM_THREADS='4', CUDA_HOME=str(sdk), PATH=str(sdk / 'bin') + ':' + env.get('PATH', ''), CXX='/usr/bin/g++', NVCC_CCBIN='/usr/bin/g++', C_INCLUDE_PATH=includes, CPATH=includes, CPLUS_INCLUDE_PATH=includes, TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'cache/ming-inductor'), TRITON_CACHE_DIR=str(ROOT / 'cache/ming-triton'))
    command = [python, str(ROOT / 'scripts/ming_independent_fragment_worker.py'), '--model', str(ROOT / 'models/Ming'), '--cache', str(out / 'ming-cache'), '--port', '18091']
    (out / 'configuration.json').write_text(json.dumps({'MING_command': command, 'MING_profile': 'Same independent-fragment seed0, native ceiling1000, eager-flow as frozen baseline', 'Tencent_backend': 'conformant r4; original Q4_K_M, CUDA all layers and embeddings, unchanged16384 unified/q8KV'}, indent=2) + '\n')
    log = (out / 'ming-worker.log').open('w')
    worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    pool = None
    application = None
    app_log = None
    try:
        deadline = time.monotonic() + 300
        while True:
            if worker.poll() is not None:
                raise RuntimeError('Frozen MING exited before becoming ready')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18091/health', timeout=2) as response:
                    state['MING_health'] = json.load(response)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Frozen MING startup timeout')
                await asyncio.sleep(1)
        state['state'] = 'warming_frozen_MING'
        save()
        warm_env = {key: value for key, value in env.items() if key != 'PYTHONPATH'}
        with (out / 'warmup.log').open('w') as warm:
            process = await asyncio.create_subprocess_exec(python, str(ROOT / 'scripts/warm_frozen_ming.py'), str(out / 'warmup'), env=warm_env, stdout=warm, stderr=subprocess.STDOUT)
            if await asyncio.wait_for(process.wait(), 240):
                raise RuntimeError('Frozen MING warmup failed')
        yield dict(output=out, state=state, save=save, common_env=env, worker=worker)
    except BaseException as error:
        state.update(state='resource_probe_failed', error=repr(error))
        save()
        raise
    finally:
        if application and application.poll() is None:
            application.terminate()
            try:
                application.wait(timeout=15)
            except subprocess.TimeoutExpired:
                application.kill()
                application.wait(timeout=5)
        if app_log:
            app_log.close()
        if pool:
            await pool.close()
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=12)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        log.close()
