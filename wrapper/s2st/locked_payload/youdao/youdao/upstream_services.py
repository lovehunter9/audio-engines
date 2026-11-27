"""Run the unchanged R2T2/T3PO services with frozen MING already resident."""
from contextlib import asynccontextmanager
import asyncio
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request
ROOT = Path('/home/olares/s2st-original-check-20261123')

@asynccontextmanager
async def upstream_services(options, out, state, save, common_env):
    from runtime_cuda_libraries import library_directories
    from official_runtime_source_identity import verify
    verify(ROOT / 'preflight/official-runtime-source-files-20261125.json')
    python = ROOT / 'envs/tencent-model/bin/python'
    asr_python = ROOT / 'envs/youdao-asr/bin/python'
    env = dict(common_env, HF_HOME=str(ROOT / 'cache/hf'), XDG_CACHE_HOME=str(ROOT / 'cache/xdg'), PYTHONUNBUFFERED='1')
    for name in ['CUDA_HOME', 'CUDA_PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS']:
        env.pop(name, None)
    env['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    native = ROOT / 'results/revised-youdao-native-20261125'
    libraries = [*library_directories(), str(native), str(ROOT / 'sources/quantized-003-snapshot/youdao/sources/llama.cpp/build/bin')]
    mt_env = dict(env, LD_LIBRARY_PATH=':'.join(libraries))
    for name in list(mt_env):
        if name.startswith('LLAMA_ARG_'):
            mt_env.pop(name)
    asr_env = dict(env, PYTHONPATH=str(Path(__file__).parent) + ':' + str(ROOT / 'sources/Confucius4-R2T2'), PATH=str(asr_python.parent) + ':' + env['PATH'], S2ST_R2T2_GPU_MEMORY_UTILIZATION='0.25' if options.asr_allocation == 'bounded' else '0.2', VLLM_CACHE_ROOT=str(ROOT / 'cache/youdao-official-asr-v014'))
    asr_env['S2ST_R6_GPU_QUEUE'] = str(out / 'gpu-stage-fifo')
    config = (ROOT / 'sources/Confucius4-T3PO/inference/config.py').read_text()
    controlled = set(re.findall('_(?:env|int|float|bool)\\("([A-Z0-9_]+)"', config))
    app_env = {key: value for key, value in env.items() if key not in controlled}
    app_env.update(PYTHONPATH=str(Path(__file__).parent) + ':' + str(ROOT / 'sources/Confucius4-T3PO'), PATH=str(python.parent) + ':' + env['PATH'], VLLM_BASE_URL='http://127.0.0.1:25324/v1', VLLM_MODEL='Confucius4-T3PO', TRANSLATION_BACKEND='auto', LATENCY_MODE='native', ASR_RECV_TIMEOUT_SECONDS='900.0', ASR_WS_URL='ws://127.0.0.1:25321/asr_stream_api_v1', ASR_SECRET_KEY='test0102', HOST='127.0.0.1', PORT='25322', MAX_ACTIVE_SESSIONS='1', HISTORY_WINDOW='128')
    generation = json.loads((ROOT / 'models/t3po-bf16/generation_config.json').read_text())
    commands = {'mt': [str(native / 'llama-server'), '-m', str(ROOT / 'models/quantized/Confucius4-T3PO-Q5_K_M.gguf'), '--alias', 'Confucius4-T3PO', '--host', '127.0.0.1', '--port', '25320', '-c', str(options.t3po_context), '-ngl', '999', '-np', '1', '-t', '12', '-fa', 'on', '--no-context-shift', '--repeat-penalty', str(generation['repetition_penalty']), '--repeat-last-n', '32768', '--temp', str(generation['temperature']), '--top-k', str(generation['top_k']), '--top-p', str(generation['top_p']), '--min-p', '0', '--override-tensor', 'token_embd.weight=CUDA0'], 'asr': [str(asr_python), str(ROOT / 'scripts/r2t2-resource-budget/ws_server.py'), '--asr_model_path', str(ROOT / 'models/r2t2-bf16'), '--vad_model_path', str(ROOT / 'models/FireRedVAD/Stream-VAD'), '--ip', '127.0.0.1', '--port', '25321'], 'admission': [str(python), str(Path(__file__).parent / 'r6_stage_proxy.py'), '--role', 't3po', '--port', '25324', '--upstream', 'http://127.0.0.1:25320', '--queue', str(out / 'gpu-stage-fifo')], 'app': ['/bin/bash', str(Path(__file__).parent / 'start_bridge.sh')]}
    if options.asr_allocation == 'single':
        commands['asr'][1] = str(Path(__file__).parent / 'r2t2_ws_allocation.py')
    if options.asr_allocation == 'bounded':
        commands['asr'][1] = str(Path(__file__).parent / 'r2t2_ws_capacity.py')
        asr_env.update(S2ST_ASR_CONTEXT_CAPACITY='8192', S2ST_ASR_CAPACITY_RECORD=str(out / 'actual-ASR-capacity.json'))
    if options.kv == 'q8_0':
        commands['mt'] += ['-ctk', 'q8_0', '-ctv', 'q8_0']
    from youdao_gpu_client import frontend_environment
    mt_env = frontend_environment(mt_env)
    asr_env = frontend_environment(asr_env)
    (out / 'youdao-configuration.json').write_text(json.dumps({'commands': commands, 'app_selected_environment': {key: app_env[key] for key in ['ASR_RECV_TIMEOUT_SECONDS', 'LATENCY_MODE', 'MAX_ACTIVE_SESSIONS', 'TRANSLATION_BACKEND', 'VLLM_MODEL']}, 'profile_name': '完锁适配版' if options.asr_allocation == 'bounded' else '完锁版', 'native_KV_precision': options.kv, 'native_context_tokens': options.t3po_context, 'context_shift': False, 'ASR_GPU_cache_fraction': float(asr_env['S2ST_R2T2_GPU_MEMORY_UTILIZATION']), 'ASR_original_context_preserved': options.asr_allocation != 'bounded', 'ASR_original_generation_preserved': True, 'ASR_allocation': options.asr_allocation, 'ASR_allocation_changes': {'max_num_seqs': 1, 'max_num_batched_tokens': 512, **({'max_model_len': 8192} if options.asr_allocation == 'bounded' else {})} if options.asr_allocation in ['single', 'bounded'] else {}, 'MING_source_weights_parameters_unchanged': True, 'source_and_inference_policy_changes': []}, indent=2) + '\n')
    processes = {}
    logs = {}
    snapshots = []

    def snapshot(stage):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        snapshots.append({'stage': stage, 'GPU_used_bytes': memory.used, 'GPU_total_bytes': memory.total, 'processes': [{'pid': row.pid, 'used_bytes': row.usedGpuMemory} for row in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]})
        (out / 'resource-stages.json').write_text(json.dumps(snapshots, indent=2) + '\n')

    async def ready(name, port, path):
        deadline = time.monotonic() + 600
        while True:
            exited = [key for key, process in processes.items() if process.poll() is not None]
            if exited:
                raise RuntimeError('Service exited during startup: ' + ','.join(exited))
            if name == 'asr':
                text = (out / 'asr.log').read_text(errors='replace')
                if 'ASR model warmup complete' not in text or 'model initialization complete.' not in text:
                    if time.monotonic() > deadline:
                        raise TimeoutError('ASR startup/warmup timeout')
                    await asyncio.sleep(1)
                    continue
            try:
                if path:
                    with urllib.request.urlopen('http://127.0.0.1:' + str(port) + path, timeout=2) as response:
                        if response.status == 200:
                            return
                else:
                    with socket.create_connection(('127.0.0.1', port), timeout=2):
                        return
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Service startup timeout: ' + name)
            await asyncio.sleep(1)
    from live_resource_observer import ResourceObserver
    observer = ResourceObserver(out / 'resource-summary.json')
    observing = asyncio.create_task(observer.run())
    try:
        for name, environment, port, path in [('mt', mt_env, 25320, '/health'), ('admission', env, 25324, '/health'), ('asr', asr_env, 25321, None), ('app', app_env, 25322, '/api/v1/health')]:
            state.update(state='starting_youdao_' + name)
            save()
            logs[name] = (out / (name + '.log')).open('w')
            processes[name] = subprocess.Popen(commands[name], env=environment, cwd=out, stdout=logs[name], stderr=subprocess.STDOUT)
            await ready(name, port, path)
            snapshot(name + '_ready')
        state.update(state='Youdao_ASR_MT_and_frozen_MING_resident_together', resources=snapshots, results=[])
        save()
        yield dict(state=state, commands=commands, processes=processes, app_env=app_env)
    finally:
        observing.cancel()
        try:
            await observing
        except asyncio.CancelledError:
            pass
        observer.save()
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                process.terminate()
        for process in reversed(list(processes.values())):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs.values():
            log.close()
