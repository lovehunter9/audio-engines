"""Live PCM transport using the same original services and frozen speech sink."""
import asyncio
from collections import deque
import json
import time

from .file_session import FileSession


class LiveSession(FileSession):
    def __init__(self, context, family, direction):
        if direction not in ('zh2en', 'en2zh'):
            raise ValueError('Unknown translation direction')
        if context['role'] != 's2st':
            raise ValueError('The live composite endpoint requires the S2ST role')
        self.context, self.family, self.direction = context, family, direction
        self.directory = context['directory'] / ('live-' + str(time.time_ns()))
        self.directory.mkdir()
        self.source_samples = 0
        self.incoming = asyncio.Queue(maxsize=8)
        self.ended = False

    async def feed(self, pcm):
        if self.ended or not pcm or len(pcm) % 2:
            raise ValueError('Expected nonempty mono PCM16 bytes before end')
        await self.incoming.put(pcm)

    async def end(self):
        if self.ended:
            raise ValueError('Duplicate end record')
        self.ended = True
        await self.incoming.put(None)

    async def frames(self):
        size = 3200 if self.family == 'tencent' else 2560
        pending = bytearray()
        while True:
            block = await self.incoming.get()
            if block is None:
                if pending:
                    yield bytes(pending)
                return
            pending.extend(block)
            while len(pending) >= size:
                yield bytes(pending[:size])
                del pending[:size]

    async def run(self):
        import numpy as np
        import websockets
        from frozen_ming_sink import FrozenMingSink

        sink = FrozenMingSink(self.directory / 'ming', 'zh-en' if self.direction == 'zh2en' else 'en-zh')
        sender = receiver = None
        fifo = None
        started = time.monotonic()
        epoch = None
        fed = 0
        text = ''
        last_text = None
        terminal = None
        upstream_path = self.directory / ('received-events.jsonl' if self.family == 'tencent' else 'transport.jsonl')
        url = 'ws://127.0.0.1:25280/translate/ws' if self.family == 'tencent' else 'ws://127.0.0.1:25322/ws/simul-demo'
        try:
            await sink.start()
            with (self.directory / 'source.f32').open('w+b') as source, upstream_path.open('w', buffering=1) as log:
                kwargs = {'max_size': 16 * 1024**2, 'open_timeout': 30}
                if self.family == 'tencent':
                    kwargs['ping_interval'] = None
                async with websockets.connect(url, **kwargs) as socket:
                    init = ({'direction': 'zh-en' if self.direction == 'zh2en' else 'en-zh',
                             'input_format': 'f32le', 'input_sample_rate': 16000, 'input_channels': 1,
                             'voice': '', 'audio_transport': 'binary'} if self.family == 'tencent' else
                            {'type': 'init', 'direction': self.direction, 'latency_mode': 'native',
                             'force_break_threshold': 15, 'punct_force': True, 'idle_force_seconds': .8, 'terms': []})
                    (self.directory / 'official-init-frame.json').write_text(json.dumps(init, indent=2) + '\n')
                    await socket.send(json.dumps(init))
                    while True:
                        event = json.loads(await asyncio.wait_for(socket.recv(), 180))
                        log.write(json.dumps({'event': event}, ensure_ascii=False) + '\n')
                        kind = event.get('event', event.get('type'))
                        if kind == 'error':
                            raise RuntimeError(str(event))
                        if kind == ('ready' if self.family == 'tencent' else 'init_ok'):
                            break
                    epoch = time.monotonic()
                    sink.begin_source(epoch)
                    received = deque()
                    if self.family == 'youdao':
                        from youdao_receive import BoundedReceiver
                        class TimestampedSocket:
                            async def recv(inner):
                                value = await socket.recv()
                                received.append(time.monotonic())
                                return value
                        fifo = BoundedReceiver(TimestampedSocket())

                    async def send():
                        async for pcm in self.frames():
                            samples = np.frombuffer(pcm, '<i2').astype(np.float32) / 32768
                            source.seek(0, 2)
                            source.write(samples.astype('<f4').tobytes())
                            source.flush()
                            if self.family == 'youdao':
                                sink.feed_source(samples)
                            self.source_samples += len(samples)
                            await socket.send(samples.astype('<f4').tobytes() if self.family == 'tencent' else pcm)
                        if not self.source_samples:
                            raise ValueError('No source audio was received')
                        await socket.send(json.dumps({'eof': True} if self.family == 'tencent' else {'type': 'end'}))

                    async def receive():
                        nonlocal terminal, fed, text, last_text
                        parts = []
                        while True:
                            event = json.loads(await (fifo.recv() if fifo else socket.recv()))
                            now = received.popleft() if fifo else time.monotonic()
                            log.write(json.dumps({'received_seconds': now - epoch, 'event': event}, ensure_ascii=False) + '\n')
                            kind = event.get('event', event.get('type'))
                            if kind == 'error':
                                raise RuntimeError(str(event))
                            delta = ''
                            if self.family == 'tencent' and kind in ('chunk', 'flush'):
                                end = min(self.source_samples, round(event['end_ms'] * 16))
                                source.seek(fed * 4)
                                sink.feed_source(np.frombuffer(source.read((end - fed) * 4), '<f4'))
                                fed = end
                                delta = event.get('text', '')
                                text += delta
                            elif self.family == 'youdao' and kind == 'translation':
                                from inference.translation import join_translation_segments
                                parts.append(event['text'])
                                joined = join_translation_segments(parts, self.direction)
                                if not joined.startswith(text):
                                    raise RuntimeError('Official segment joiner revised committed text')
                                delta, text = joined[len(text):], joined
                            if delta:
                                last_text = now - epoch
                                await sink.text(delta)
                            if kind == ('done' if self.family == 'tencent' else 'ended'):
                                terminal = event
                                return

                    sender, receiver = asyncio.create_task(send()), asyncio.create_task(receive())
                    await asyncio.gather(sender, receiver)
                    if self.family == 'tencent':
                        if terminal.get('full_text') != text or terminal.get('input_audio', {}).get('output_samples') != self.source_samples:
                            raise RuntimeError('Original final text/source acknowledgement differs')
                        if fed < self.source_samples:
                            source.seek(fed * 4)
                            sink.feed_source(np.frombuffer(source.read(), '<f4'))
                    duration = self.source_samples / 16000
                    speech = await sink.finish(duration)
                    result = {'execution_completed': True, 'source_samples': self.source_samples, 'source_seconds': duration,
                              'translated_text': text, 'ming': speech, 'original_final_event': terminal,
                              'request_to_source_epoch_seconds': epoch - started,
                              'request_to_last_text_seconds': None if last_text is None else epoch - started + last_text,
                              'request_to_last_pcm_seconds': epoch - started + sink.last_audio,
                              'total_request_s2st_rtf': (epoch - started + sink.last_audio) / duration,
                              'input_mode': 'live', 'caller_supplied_source_tail': True,
                              'transport_boundary': 'Mono PCM16@16k source, reframed without resampling/padding into original100ms/80ms binary messages; original services retain all processing.'}
                    (self.directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                    return result
        finally:
            for task in (sender, receiver):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (sender, receiver) if task is not None), return_exceptions=True)
            if fifo:
                await fifo.close()
            await sink.close()
