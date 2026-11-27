"""Measure the locked Tencent frontend through its unmodified original CLI."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import wave

from runtime_cuda_libraries import library_directories

ROOT = Path('/home/olares/s2st-original-check-20261123')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


async def record_client(command, output, expected_samples, timeout, env=None):
    """Observe stdout only; do not replace the author's sending or EOF logic."""
    start = time.monotonic()
    epoch = first = last = done = None
    text = ''
    process = None
    try:
        with (output / 'client.stderr.log').open('w') as stderr, (output / 'received-events.jsonl').open('w', buffering=1) as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=stderr, env=env)

            async def consume():
                nonlocal epoch, first, last, done, text
                async for line in process.stdout:
                    event = json.loads(line)
                    now = time.monotonic()
                    if event.get('event') == 'ready':
                        if epoch is not None:
                            raise RuntimeError('Duplicate original ready event')
                        epoch = now
                    log.write(json.dumps({'request_seconds': now-start,
                                          'received_seconds': None if epoch is None else now-epoch,
                                          'event': event}, ensure_ascii=False) + '\n')
                    if event.get('event') in ['chunk', 'flush'] and event.get('text'):
                        if epoch is None:
                            raise RuntimeError('Text before original ready event')
                        first = now-start if first is None else first
                        last = now-start
                        text += event['text']
                    elif event.get('event') == 'done':
                        if done is not None:
                            raise RuntimeError('Duplicate original final event')
                        done = event
                return await process.wait()

            code = await asyncio.wait_for(consume(), timeout)
        if code or done is None or epoch is None:
            raise RuntimeError('Original CLI did not complete: ' + str(code))
        if done.get('full_text') != text:
            raise RuntimeError('Emitted deltas differ from the original final text')
        if done.get('input_audio', {}).get('output_samples') != expected_samples:
            raise RuntimeError('Original CLI did not acknowledge every source sample')
        seconds = expected_samples / 16000
        return {'execution_completed': True, 'translated_text': text,
                'source_samples': expected_samples, 'source_samples_consumed': expected_samples,
                'source_seconds': seconds, 'request_to_source_epoch_seconds': epoch-start,
                'request_to_first_text_seconds': first, 'request_to_last_text_seconds': last,
                'request_to_done_seconds': time.monotonic()-start,
                'standalone_text_RTF': None if last is None else last/seconds,
                'native_complete_flag': done.get('complete'),
                'native_skipped_audio_ms': done.get('skipped_untranslated_audio_ms'),
                'native_unresolved_audio_ms': done.get('unresolved_audio_ms'),
                'original_final_event': done}
    finally:
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


async def main(options):
    if not options.output.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Output outside owned experiment root')
    options.output.mkdir(exist_ok=False)
    out = options.output
    raw = options.inputs.read_bytes()
    inputs = json.loads(raw)[options.population]
    if options.cases:
        selected = set(options.cases)
        inputs = [row for row in inputs if row['case_id'] in selected]
        if len(inputs) != len(selected):
            raise ValueError('Unknown case in sealed input manifest')
    expected_count = 777 if options.population == 'short_segments' else 20
    if not options.cases and len(inputs) != expected_count:
        raise ValueError('Incomplete benchmark input population')
    state = {'state': 'starting_original_frontend', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'population': options.population, 'input_mode': options.input_mode,
             'expected_cases': len(inputs), 'results': [], 'TTS_attached': False,
             'official_client_modified': False, 'inference_manifest_sha256': hashlib.sha256(raw).hexdigest(),
             'scope': 'Standalone translation; no MING process or input transcript/reference supplied'}
    save(out/'state.json', state)
    reference = ROOT/'runs/official-tencent-filter-active-control2/measurements/config.toml'
    config = reference.read_text()
    if config.count('max_sessions = 4') != 1:
        raise RuntimeError('Unexpected sealed reference configuration')
    (out/'config.toml').write_text(config.replace('max_sessions = 4', 'max_sessions = 1', 1))
    save(out/'profile-difference.json', {'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
         'changes': ['max_sessions4→1, shared serialized native backend'], 'all_other_config_text_identical': True})
    env = dict(os.environ)
    for key in ['PYTHONPATH', 'CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']:
        env.pop(key, None)
    env.update(LD_LIBRARY_PATH=':'.join(library_directories()), OMP_NUM_THREADS='4')
    python = str(ROOT/'envs/tencent-model/bin/python')
    application = None
    with (out/'app.log').open('w') as log:
        try:
            launch = time.monotonic()
            application = subprocess.Popen([python, str(Path(__file__).parent/'live_tencent_app.py'),
                '--config', str(out/'config.toml'), '--output', str(out/'app-runtime')], env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = launch+240
            while True:
                if application.poll() is not None:
                    raise RuntimeError('Original frontend exited during startup')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:25280/health', timeout=2) as response:
                        if json.load(response).get('ready', True):
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Original frontend startup timeout')
                await asyncio.sleep(1)
            state.update(state='running', model_startup_seconds=time.monotonic()-launch)
            save(out/'state.json', state)
            for row in inputs:
                case = row['case_id']
                source = Path(row.get('runtime_path', row.get('audio_path')))
                source_sha = row.get('source_sha256', row.get('audio_sha256'))
                source_samples = row.get('source_samples', row.get('source_samples_acknowledged'))
                source_raw = source.read_bytes()
                if hashlib.sha256(source_raw).hexdigest() != source_sha:
                    raise RuntimeError('Input hash differs: '+case)
                with wave.open(str(source), 'rb') as wav:
                    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) != (1, 2, 16000, source_samples):
                        raise RuntimeError('Input PCM shape differs: '+case)
                destination = out/case
                destination.mkdir()
                state['current_case'] = case
                save(out/'state.json', state)
                command = [str(ROOT/'envs/tencent-model/bin/simulinfer-client'), str(source),
                    '--url', 'http://127.0.0.1:25280', '--transport', 'ws',
                    '--direction', 'zh-en' if case.startswith('zh2en') else 'en-zh',
                    '--events', str(destination/'events.json')]
                if options.input_mode == 'realtime':
                    command.append('--realtime')
                save(destination/'request.json', {'case_id': case, 'input': row, 'command': command,
                     'client_modified': False, 'TTS_attached': False})
                result = await record_client(command, destination, source_samples, 1500, env)
                result.update(case_id=case, direction=row['direction'], source_sha256=source_sha)
                save(destination/'result.json', result)
                state['results'].append(result)
                save(out/'state.json', state)
            state.update(state='completed', current_case=None, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            save(out/'state.json', state)
            save(out/'result.json', state)
            print(json.dumps({'state': 'completed', 'cases': len(state['results'])}), flush=True)
        except BaseException as error:
            state.update(state='failed', error=repr(error))
            save(out/'state.json', state)
            raise
        finally:
            if application and application.poll() is None:
                application.terminate()
                try:
                    application.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    application.kill()
                    application.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--population', choices=['full_recordings', 'short_segments'], required=True)
    parser.add_argument('--cases', nargs='*', default=[])
    parser.add_argument('--input-mode', choices=['asap', 'realtime'], default='asap')
    asyncio.run(main(parser.parse_args()))
