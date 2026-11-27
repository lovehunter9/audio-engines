"""Subscribe to the unchanged original file client's stdout; add frozen MING.

All upstream transport, framing, processing and end-of-input remain in the
author's CLI. This external composition does not import or patch that client.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import wave
import sys

import numpy as np

ROOT = Path('/home/olares/s2st-original-check-20261123')
# Resolve the identical frozen sink from the original script directory.
sys.path.insert(0, str(ROOT/'scripts'))


async def run(args):
    from frozen_ming_sink import FrozenMingSink
    with wave.open(str(args.source), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32) / 32768
    sink = FrozenMingSink(args.output / 'ming', args.direction)
    command = [str(ROOT / 'envs/tencent-model/bin/simulinfer-client'), str(args.source),
               '--url', 'http://127.0.0.1:25280', '--transport', 'ws',
               '--direction', args.direction, '--events', str(args.output / 'events.json')]
    if args.input_mode == 'realtime':
        command.append('--realtime')
    (args.output / 'composition.json').write_text(json.dumps({
        'command': command, 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'official_client_modified': False, 'upstream_monkeypatches': [],
        'external_coordinator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'composition': 'Original CLI JSON stdout -> unchanged frozen MING; source cloning uses only already-consumed PCM',
        'input_mode': args.input_mode,
    }, indent=2) + '\n')
    process = None
    text = ''
    fed = 0
    done = None
    epoch = None
    start = time.monotonic()
    first_text = last_text = None
    try:
        await sink.start()
        with (args.output / 'original-client.stderr.log').open('w') as stderr, (args.output / 'received-events.jsonl').open('w', buffering=1) as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=stderr)
            async for line in process.stdout:
                event = json.loads(line)
                now = time.monotonic()
                print(line.decode().rstrip(), flush=True)
                if event.get('event') == 'ready':
                    epoch = now
                    sink.begin_source(epoch)
                log.write(json.dumps({'received_seconds': None if epoch is None else now - epoch, 'event': event}, ensure_ascii=False) + '\n')
                if event.get('event') in ['chunk', 'flush']:
                    end = min(len(samples), round(event['end_ms'] * 16))
                    sink.feed_source(samples[fed:end])
                    fed = end
                    delta = event.get('text', '')
                    if delta:
                        text += delta
                        first_text = first_text if first_text is not None else now - epoch
                        last_text = now - epoch
                        await sink.text(delta)
                elif event.get('event') == 'done':
                    done = event
                    if event.get('full_text') != text:
                        raise RuntimeError('Original final text differs from emitted deltas')
            code = await process.wait()
            if code or not done:
                raise RuntimeError('Original client did not complete')
            if done.get('input_audio', {}).get('output_samples') != len(samples):
                raise RuntimeError('Original client did not acknowledge all source PCM')
            if fed < len(samples):
                # Official chunk boundaries use integer milliseconds, while
                sink.feed_source(samples[fed:])
                fed = len(samples)
            result = await sink.finish(len(samples) / 16000)
            result.update(input_mode=args.input_mode, source_samples=len(samples), source_samples_consumed=fed,
                          translation=text, first_text_seconds=first_text, last_text_seconds=last_text,
                          request_to_source_epoch_seconds=epoch-start,
                          request_to_last_text_seconds=None if last_text is None else epoch-start+last_text,
                          request_to_last_pcm_seconds=None if sink.last_audio is None else epoch-start+sink.last_audio,
                          total_request_s2st_rtf=None if sink.last_audio is None else (epoch-start+sink.last_audio)/(len(samples)/16000),
                          total_request_text_rtf=None if last_text is None else (epoch-start+last_text)/(len(samples)/16000),
                          total_request_seconds=time.monotonic() - start,
                          upstream_final_event=done, upstream_complete_as_reported=done.get('complete'),
                          semantic_acceptance='Pending; original unresolved/skipped audio fields preserved')
            (args.output / 'composition-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    finally:
        if process and process.returncode is None:
            process.terminate()
            await process.wait()
        await sink.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direction', choices=['zh-en', 'en-zh'], required=True)
    parser.add_argument('--input-mode', choices=['realtime', 'asap'], default='realtime')
    asyncio.run(run(parser.parse_args()))
