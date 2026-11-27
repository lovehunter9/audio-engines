"""File I/O adapter: execute the exact measured client without replacing its flow."""
import asyncio
import base64
import json
from pathlib import Path
import time
import wave

from .layout import ROOT


def check_source(path):
    with wave.open(str(path), 'rb') as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('The locked profile requires mono PCM16 WAV at 16000 Hz')
        frames = reader.getnframes()
        if frames <= 0 or len(reader.readframes(frames)) != frames * 2:
            raise ValueError('Empty or truncated source WAV')
    return frames


class FileSession:
    def __init__(self, context, family, source, direction, input_mode):
        if direction not in ('zh2en', 'en2zh') or input_mode not in ('asap', 'realtime'):
            raise ValueError('Unknown direction or input pacing')
        self.context, self.family = context, family
        self.source, self.direction, self.input_mode = Path(source), direction, input_mode
        self.source_samples = check_source(source)
        self.directory = context['directory'] / ('request-' + str(time.time_ns()))
        self.process = None

    def command(self):
        python = str(ROOT / 'envs/tencent-model/bin/python')
        if self.family == 'tencent':
            self.directory.mkdir()
            return [python, str(self.context['bundle'] / 'tencent_original_cli_ming_flush.py'),
                    '--source', str(self.source), '--output', str(self.directory),
                    '--direction', 'zh-en' if self.direction == 'zh2en' else 'en-zh',
                    '--input-mode', self.input_mode]
        return [python, str(self.context['bundle'] / 'youdao_official_webfile_host_client.py'),
                '--source', str(self.source), '--output', str(self.directory),
                '--direction', self.direction, '--input-mode', self.input_mode, '--with-ming']

    async def run(self):
        if self.context['role'] == 's2t':
            return await self.run_text()
        command = self.command()
        # Logs belong to this invocation. No retry, input rewriting or selective
        log_path = self.directory.parent / (self.directory.name + '.client.log')
        with log_path.open('wb') as log:
            self.process = await asyncio.create_subprocess_exec(*command, env=self.context['client_env'], stdout=log, stderr=asyncio.subprocess.STDOUT)
            try:
                code = await self.process.wait()
                if code:
                    raise RuntimeError('Locked original client failed; inspect request log')
                name = 'composition-result.json' if self.family == 'tencent' else 'result.json'
                result = json.loads((self.directory / name).read_text())
                speech = result.get('ming', result)
                if not speech['complete'] or not speech['tts_source_matches_mt'] or speech['failures']:
                    raise RuntimeError('Frozen MING completeness check failed')
                if self.family == 'tencent':
                    if result['source_samples_consumed'] != self.source_samples:
                        raise RuntimeError('Source sample acknowledgement differs')
                elif result['source_samples_sent'] != self.source_samples or not result['execution_completed']:
                    raise RuntimeError('Original source transport did not complete')
                return result
            finally:
                if self.process.returncode is None:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), 15)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()

    async def run_text(self):
        self.directory.mkdir()
        if self.family == 'tencent':
            from locked_text_observer import record_client
            command = [str(ROOT / 'envs/tencent-model/bin/simulinfer-client'), str(self.source),
                       '--url', 'http://127.0.0.1:25280', '--transport', 'ws',
                       '--direction', 'zh-en' if self.direction == 'zh2en' else 'en-zh',
                       '--events', str(self.directory / 'events.json')]
            if self.input_mode == 'realtime':
                command.append('--realtime')
            result = await record_client(command, self.directory, self.source_samples, 1500, self.context['client_env'])
            (self.directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            return result
        # Import the unchanged official host API client rather than reimplement
        command = [str(ROOT / 'envs/tencent-model/bin/python'),
                   str(self.context['bundle'] / 'youdao_official_webfile_host_client.py'),
                   '--source', str(self.source), '--output', str(self.directory / 'text'),
                   '--direction', self.direction, '--input-mode', self.input_mode]
        with (self.directory / 'client.log').open('wb') as log:
            self.process = await asyncio.create_subprocess_exec(*command, env=self.context['client_env'], stdout=log, stderr=asyncio.subprocess.STDOUT)
            try:
                if await self.process.wait():
                    raise RuntimeError('Original standalone Youdao client failed')
                result = json.loads((self.directory / 'text/result.json').read_text())
                if not result['execution_completed'] or result['source_samples_sent'] != self.source_samples:
                    raise RuntimeError('Standalone source transport did not complete')
                (self.directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                return result
            finally:
                if self.process.returncode is None:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), 15)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()

    async def events(self, running):
        """Observe exact emitted PCM and text; never feed anything back to models."""
        upstream_name = 'received-events.jsonl' if self.family == 'tencent' else 'transport.jsonl'
        upstream = self.directory / ('text/' + upstream_name if self.family == 'youdao' and self.context['role'] == 's2t' else upstream_name)
        speech_log = self.directory / 'ming/tts-events.jsonl'
        pcm_path = self.directory / 'ming/translated.pcm'
        positions = {'upstream': 0, 'speech': 0}
        pending_audio = []
        parts, joined_text = [], ''
        while True:
            emitted = False
            for kind, path in [('upstream', upstream), ('speech', speech_log)]:
                if not path.is_file():
                    continue
                with path.open('rb') as reader:
                    reader.seek(positions[kind])
                    while True:
                        line = reader.readline()
                        if not line or not line.endswith(b'\n'):
                            break
                        positions[kind] = reader.tell()
                        record = json.loads(line)
                        if kind == 'speech' and record.get('event') == 'audio':
                            pending_audio.append(record)
                            continue
                        emitted = True
                        yield {'type': kind, 'data': record}
                        if kind == 'upstream':
                            event = record.get('event', {})
                            delta = ''
                            if self.family == 'tencent' and event.get('event') in ('chunk', 'flush'):
                                delta = event.get('text', '')
                            elif self.family == 'youdao' and event.get('type') == 'translation':
                                from inference.translation import join_translation_segments
                                parts.append(event['text'])
                                updated = join_translation_segments(parts, self.direction)
                                if not updated.startswith(joined_text):
                                    raise RuntimeError('Official segment joiner revised committed output')
                                delta, joined_text = updated[len(joined_text):], updated
                            if delta:
                                yield {'type': 'translation_delta', 'text': delta}
            while pending_audio and pcm_path.is_file():
                record = pending_audio[0]
                with pcm_path.open('rb') as reader:
                    reader.seek(record['pcm_offset'])
                    pcm = reader.read(record['pcm_bytes'])
                if len(pcm) != record['pcm_bytes']:
                    # The frozen writer's buffer is deliberately unchanged.
                    break
                pending_audio.pop(0)
                emitted = True
                yield dict(record, type='audio', encoding='pcm_s16le_base64',
                           pcm_base64=base64.b64encode(pcm).decode())
            if running.done() and not emitted:
                if pending_audio:
                    raise RuntimeError('Declared frozen PCM bytes are missing at task end')
                return
            await asyncio.sleep(.02)
