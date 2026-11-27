"""Host-local task I/O for the official documented PCM16 WebSocket API.

No ASR/MT policy, frontend feature extraction, reset/flush or decoder is
implemented here. Frames follow the official file example:1280 samples/80ms.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import wave
import sys

import websockets
from collections import deque
from youdao_receive import BoundedReceiver


async def run_session(options, sink):
    out = options.output
    with wave.open(str(options.source), 'rb') as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 16000):
            raise RuntimeError('Input must be original mono PCM16 16kHz')
        pcm = reader.readframes(reader.getnframes())
    duration = len(pcm) / 32000
    request = {'source': str(options.source), 'source_sha256': hashlib.sha256(options.source.read_bytes()).hexdigest(),
               'direction': options.direction, 'url': options.url,
               'audio_seconds': duration, 'transport': 'Official documented PCM16 WebSocket,80ms frames; host-local I/O, original services retain all model processing',
               'configuration':'Exact pinned official web-file session settings: native, force_break_threshold15, punct_force=true, idle_force_seconds0.8, no glossary; original services retain all processing', 'input_mode': options.input_mode, 'boundary':'Host-local canonical16k mono PCM16 API boundary with original80ms web-file pacing and identical official web init frame; browser AudioContext decoding/resampling is outside this documented API boundary. No source splice or original service changes.'}
    (out / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    started = time.monotonic()
    epoch = None
    sent = 0
    eof = None
    events = []
    terminal = None
    parts = []
    joined_text = ''
    with (out / 'transport.jsonl').open('w', buffering=1) as log:
        def record(row):
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
        async with websockets.connect(options.url, max_size=16 * 1024 * 1024, open_timeout=30) as socket:
            init_frame={'type':'init','direction':options.direction,'latency_mode':'native','force_break_threshold':15,'punct_force':True,'idle_force_seconds':0.8,'terms':[]}
            (out/'official-web-init-frame.json').write_text(json.dumps(init_frame,ensure_ascii=False,indent=2)+'\n')
            await socket.send(json.dumps(init_frame))
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), 180))
                record({'action': 'receive_before_source', 'elapsed': time.monotonic() - started, 'event': event})
                if event.get('type') == 'error':
                    raise RuntimeError(str(event))
                if event.get('type') == 'init_ok':
                    break
            epoch = time.monotonic()
            received_times = deque()
            class TimestampedSocket:
                async def recv(self):
                    value = await socket.recv()
                    received_times.append((time.monotonic(), sent))
                    return value
            incoming = BoundedReceiver(TimestampedSocket())
            if sink:
                sink.begin_source(epoch)
            async def send():
                nonlocal sent, eof
                for offset in range(0, len(pcm), 2560):
                    part = pcm[offset:offset + 2560]
                    if sink:
                        import numpy as np
                        sink.feed_source(np.frombuffer(part, dtype='<i2').astype(np.float32) / 32768)
                    before = time.monotonic()
                    await socket.send(part)
                    sent += len(part) // 2
                    record({'action': 'send_audio', 'before_seconds': before - epoch,
                            'after_seconds': time.monotonic() - epoch, 'source_samples': sent,
                            'bytes': len(part), 'sha256': hashlib.sha256(part).hexdigest()})
                    # Same relative file timer as the official80ms example.
                    if options.input_mode == 'realtime':
                        await asyncio.sleep(.08)
                await socket.send(json.dumps({'type': 'end'}))
                eof = time.monotonic() - epoch
                record({'action': 'send_end', 'seconds': eof, 'source_samples': sent})
            async def receive():
                nonlocal terminal, joined_text
                while True:
                    event = json.loads(await incoming.recv())
                    received_at, source_sent_at_receive = received_times.popleft()
                    row = {'received_seconds': received_at - epoch, 'source_samples_sent': source_sent_at_receive, 'event': event}
                    record({'action': 'receive', **row})
                    events.append(row)
                    if event.get('type') == 'translation':
                        from inference.translation import join_translation_segments
                        parts.append(event['text'])
                        updated = join_translation_segments(parts, options.direction)
                        if not updated.startswith(joined_text):
                            raise RuntimeError('Official segment joiner revised committed text')
                        delta = updated[len(joined_text):]
                        joined_text = updated
                        if sink and delta:
                            await sink.text(delta)
                    if event.get('type') == 'error':
                        raise RuntimeError(str(event))
                    if event.get('type') == 'ended':
                        terminal = event
                        return
            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), duration + 600)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                queue_stats = incoming.stats()
                await incoming.close()
            if sink:
                await sink.finish(duration)
    mt = [x for x in events if x['event'].get('type') == 'translation']
    asr = [x for x in events if x['event'].get('type') == 'asr']
    stats = terminal.get('stats', {}) if terminal else {}
    result = {'scope': 'OfficialBF16 R2T2→T3PO full host API task, input=' + options.input_mode + ', S2T',
              'execution_completed': terminal is not None, 'source_buffer_empty': stats.get('buffer_units') == 0,
              'source_seconds': duration, 'source_samples_sent': sent,
              'external_receive_queue':queue_stats,
              'source_eof_seconds': eof, 'wall_seconds': time.monotonic() - epoch,
              'first_text_seconds': mt[0]['received_seconds'] if mt else None,
              'last_text_seconds': mt[-1]['received_seconds'] if mt else None,
              'last_asr_seconds': asr[-1]['received_seconds'] if asr else None,
              'request_to_source_epoch_seconds':epoch-options.request_started,
              'request_to_last_text_seconds':None if not mt else epoch-options.request_started+mt[-1]['received_seconds'],
              'max_between_text_commits_seconds': max((b['received_seconds'] - a['received_seconds'] for a, b in zip(mt, mt[1:])), default=0),
              'translated_text': joined_text,
              'source_text': ''.join(x['event'].get('text', '') for x in asr), 'stats': stats,
              'timing_limit': ('Original80ms relative pacing limits walltime; not compute throughput' if options.input_mode == 'realtime' else 'ASAP input, same official model processing; no1x input throttle'), 'input_mode': options.input_mode,
              'events': events}
    if sink:
        result.update(scope='OfficialBF16 R2T2→T3PO plus unchanged frozenMING, input=' + options.input_mode,
                      ming=json.loads((out / 'ming/result.json').read_text()))
        result['request_to_last_pcm_seconds']=epoch-options.request_started+result['ming']['last_pcm_seconds']
        result['total_request_s2st_rtf']=result['request_to_last_pcm_seconds']/duration
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['events', 'translated_text', 'source_text']}, ensure_ascii=False), flush=True)
    if not result['execution_completed'] or sent * 2 != len(pcm):
        raise RuntimeError('Native audio session did not complete')


async def run(options):
    options.request_started=time.monotonic()
    options.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, '/home/olares/s2st-original-check-20261123/sources/Confucius4-T3PO')
    sink = None
    try:
        if options.with_ming:
            from frozen_ming_sink import FrozenMingSink
            sink = FrozenMingSink(options.output / 'ming', 'zh-en' if options.direction == 'zh2en' else 'en-zh')
            await sink.start()
        await run_session(options, sink)
    finally:
        if sink:
            await sink.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direction', choices=['zh2en', 'en2zh'], required=True)
    parser.add_argument('--url', default='ws://127.0.0.1:25322/ws/simul-demo')
    parser.add_argument('--with-ming', action='store_true')
    parser.add_argument('--input-mode', choices=['realtime','asap'], default='realtime')
    options = parser.parse_args()
    try:
        asyncio.run(run(options))
    except BaseException as error:
        options.output.mkdir(parents=True, exist_ok=True)
        (options.output / 'failure.json').write_text(json.dumps({'error': repr(error)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
