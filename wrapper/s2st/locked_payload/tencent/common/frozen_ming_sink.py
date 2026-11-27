"""Same frozen MING speech, source-voice and FIFO playout for either upstream."""
import base64
import dataclasses
import json
from pathlib import Path
import time
import wave


class FrozenMingSink:
    def __init__(self, directory, direction):
        from frozen_s2st.speech import install, settings, fingerprint
        from frozen_s2st.playout import OnlinePlayout
        install()
        from simulinfer.tts import Speech
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.settings = settings('http://127.0.0.1:18091/v1')
        self.direction = direction
        self.epoch = None
        self.events = []
        self.text_received = ''
        self.audio_bytes = 0
        self.first_audio = self.last_audio = None
        self.tts_done = None
        self.pcm = (self.directory / 'translated.pcm').open('wb')
        self.log = (self.directory / 'tts-events.jsonl').open('w', buffering=1)
        chinese = direction == 'en-zh'
        self.player = OnlinePlayout(self.directory, self.settings.tts_speed if chinese else self.settings.tts_speed_zh_en,
                                    prebuffer=2.0 if chinese else 1.0, target_queue=5.0 if chinese else 4.0,
                                    min_speed=.95 if chinese else 1.0)
        self.speech = Speech(self.settings, self.emit, direction)
        self.speech._bench_case_dir = self.directory
        (self.directory / 'frozen-policy.json').write_text(json.dumps({
            'tts_settings': {k: v for k, v in dataclasses.asdict(self.settings).items() if k.startswith('tts_')},
            'policy_sha256': fingerprint(self.settings),
            'source': 'Seven frozen TTS delivery files, unchanged; same source voice, seed, numbers and playout settings for both schemes',
            'gpu_vad_patch_scope': 'This external TTS sink process only; official upstream audio/MT service is a different process and is not patched',
        }, ensure_ascii=False, indent=2) + '\n')

    async def start(self):
        await self.speech.start()

    def begin_source(self, epoch):
        if self.epoch is not None:
            raise RuntimeError('Source epoch was already set')
        self.epoch = epoch

    def feed_source(self, samples):
        self.speech.feed_source(samples)

    async def text(self, delta):
        self.text_received += delta
        await self.speech.text(delta)

    async def emit(self, event):
        row = dict(event)
        row['stream_seconds'] = None if self.epoch is None else time.monotonic() - self.epoch
        if event['event'] == 'audio':
            pcm = base64.b64decode(row.pop('data'))
            self.player.audio(event['seq'], pcm)
            row.update(pcm_offset=self.audio_bytes, pcm_bytes=len(pcm))
            self.pcm.write(pcm)
            self.audio_bytes += len(pcm)
            if self.first_audio is None:
                self.first_audio = row['stream_seconds']
            self.last_audio = row['stream_seconds']
        elif event['event'] == 'tts_start':
            if abs(event['speed'] - self.player.base_speed) >= 1e-9:
                raise RuntimeError('MING and playout base speed differ')
        elif event['event'] == 'tts_end':
            await self.player.complete(event['seq'], self.epoch)
        elif event['event'] == 'tts_done':
            self.tts_done = event
        elif event['event'] == 'voice_clone' and self.speech.source_snapshot:
            references = self.directory / 'references'; references.mkdir(exist_ok=True)
            path = references / (str(event['revision']) + '.wav')
            path.write_bytes(base64.b64decode(self.speech.source_snapshot.audio.split(',', 1)[1]))
            row['reference_file'] = str(path.relative_to(self.directory))
        self.events.append(row)
        self.log.write(json.dumps(row, ensure_ascii=False) + '\n')

    async def finish(self, source_seconds):
        await self.speech.finish()
        state = getattr(self.speech, '_complete_words', None)
        integrity = bool(state and state.received == state.committed == self.text_received)
        failed = [x for x in self.events if x.get('event') in ['tts_error', 'tts_attempt_failed', 'tts_unavailable']]
        playout = self.player.finish(self.epoch, source_seconds, self.audio_bytes)
        self.pcm.flush()
        with wave.open(str(self.directory / 'translated.wav'), 'wb') as writer:
            writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(self.settings.tts_sample_rate)
            writer.writeframes((self.directory / 'translated.pcm').read_bytes())
        result = {'complete': bool(self.tts_done and self.tts_done.get('complete') and integrity and not failed),
                  'tts_source_matches_mt': integrity, 'first_pcm_seconds': self.first_audio,
                  'last_pcm_seconds': self.last_audio, 'audio_bytes': self.audio_bytes,
                  'source_seconds': source_seconds, 'online_playout': playout,
                  'source_paced_last_pcm_rtf': None if self.last_audio is None else self.last_audio / source_seconds,
                  'metric_limit': 'Real1x upstream plus unchanged MING; input-paced wallRTF, not isolated compute throughput. FIFO replay includes delivery gaps.',
                  'failures': failed}
        (self.directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        if not result['complete']:
            raise RuntimeError('Frozen MING failed completeness or delivery integrity')
        return result

    async def close(self):
        await self.speech.close()
        self.pcm.close()
        self.log.close()
        if self.player.writer is not None:
            self.player.writer.close()
        self.player.log.close()
