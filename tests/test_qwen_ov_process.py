import contextlib
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wrapper import tasks
from wrapper.caps import qwen3_voice, tts_el


class OVProcessTest(unittest.TestCase):
    @contextlib.contextmanager
    def children(self, source):
        spawn = subprocess.Popen
        procs = []
        paths = []

        def start(args, **kw):
            paths.append(Path(args[-1]))
            proc = spawn([sys.executable, '-c', source, args[-1]], **kw)
            procs.append(proc)
            return proc

        try:
            with patch.object(subprocess, 'Popen', start):
                yield procs, paths
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    def context_when_ready(self, paths):
        class Ctx:
            def checkpoint(self):
                if paths and paths[-1].with_suffix('.ready').exists():
                    raise tasks.Cancelled('test cancellation')
        return Ctx()

    def assert_reaped(self, procs, paths):
        self.assertTrue(procs)
        self.assertTrue(all(p.poll() is not None for p in procs))
        self.assertTrue(all(not p.parent.exists() for p in paths))

    def test_stream_cancel_before_first_frame_reaps_child(self):
        source = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).with_suffix('.ready').touch(); time.sleep(60)"
        with self.children(source) as (procs, paths):
            with self.assertRaises(tasks.Cancelled):
                list(qwen3_voice._ov_stream('repo', 'clone', {}, self.context_when_ready(paths)))
            self.assert_reaped(procs, paths)

    def test_cancel_while_reading_partial_frame_reaps_child(self):
        source = '''import json,pathlib,struct,sys,time
p=pathlib.Path(sys.argv[1]);j=json.loads(p.read_text())
with open(j['frames'],'ab',buffering=0) as f:f.write(struct.pack('<II',24000,5)+b'xx')
p.with_suffix('.ready').touch();time.sleep(60)
'''
        with self.children(source) as (procs, paths):
            with self.assertRaises(tasks.Cancelled):
                list(qwen3_voice._ov_stream('repo', 'clone', {}, self.context_when_ready(paths)))
            self.assert_reaped(procs, paths)

    def test_whole_generation_cancel_reaps_child(self):
        source = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).with_suffix('.ready').touch(); time.sleep(60)"
        with self.children(source) as (procs, paths):
            with self.assertRaises(tasks.Cancelled):
                qwen3_voice._ov_generate('repo', 'clone', {}, self.context_when_ready(paths))
            self.assert_reaped(procs, paths)

    def test_cancel_before_start_does_not_spawn(self):
        stop = threading.Event()
        stop.set()
        with patch.object(subprocess, 'Popen') as spawn:
            for stream in [False, True]:
                with self.assertRaises(tasks.Cancelled):
                    result = (qwen3_voice._ov_stream if stream else qwen3_voice._ov_generate)(
                        'repo', 'clone', {}, tasks.stream_ctx(stop))
                    if stream:
                        list(result)
            spawn.assert_not_called()

    def test_total_deadline_even_when_frames_keep_arriving(self):
        source = '''import json,struct,sys,time
j=json.load(open(sys.argv[1]))
with open(j['frames'],'ab',buffering=0) as f:
 while True:f.write(struct.pack('<II',24000,1)+struct.pack('<f',0.25));time.sleep(0.01)
'''
        with self.children(source) as (procs, paths), patch.object(qwen3_voice, '_OV_JOB_SECONDS', .25):
            with self.assertRaisesRegex(TimeoutError, 'generation exceeded'):
                list(qwen3_voice._ov_stream('repo', 'clone', {}))
            self.assert_reaped(procs, paths)

    def test_idle_deadline_precedes_shared_watchdog(self):
        with self.children('import time;time.sleep(60)') as (procs, paths), \
                patch.object(tts_el, '_STALL_SECONDS', 30.15):
            with self.assertRaisesRegex(TimeoutError, 'no audio'):
                list(qwen3_voice._ov_stream('repo', 'clone', {}))
            self.assert_reaped(procs, paths)

    def test_close_after_audio_reaps_child(self):
        source = '''import json,struct,sys,time
j=json.load(open(sys.argv[1]))
with open(j['frames'],'ab',buffering=0) as f:f.write(struct.pack('<II',24000,1)+struct.pack('<f',0.25))
time.sleep(60)
'''
        with self.children(source) as (procs, paths):
            stream = qwen3_voice._ov_stream('repo', 'clone', {})
            wave, sr = next(stream)
            self.assertEqual((wave.tolist(), sr), ([.25], 24000))
            stream.close()
            self.assert_reaped(procs, paths)

    def test_child_ignoring_term_is_killed_before_return(self):
        source = "import pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).with_suffix('.ready').touch();time.sleep(60)"
        with self.children(source) as (procs, paths):
            with self.assertRaises(tasks.Cancelled):
                list(qwen3_voice._ov_stream('repo', 'clone', {}, self.context_when_ready(paths)))
            self.assertEqual(procs[0].returncode, -9)
            self.assert_reaped(procs, paths)

    def test_terminal_frame_does_not_allow_child_to_wait_forever(self):
        source = '''import json,struct,sys,time
j=json.load(open(sys.argv[1]))
with open(j['frames'],'ab',buffering=0) as f:f.write(struct.pack('<II',0,0))
time.sleep(60)
'''
        with self.children(source) as (procs, paths), patch.object(qwen3_voice, '_OV_JOB_SECONDS', .25):
            with self.assertRaises(TimeoutError):
                list(qwen3_voice._ov_stream('repo', 'clone', {}))
            self.assert_reaped(procs, paths)

    def test_success_keeps_exact_pcm(self):
        source = '''import json,struct,sys
j=json.load(open(sys.argv[1]))
with open(j['frames'],'ab',buffering=0) as f:
 f.write(struct.pack('<II',24000,2)+struct.pack('<ff',0.25,-0.5))
 f.write(struct.pack('<II',0,0))
'''
        with self.children(source) as (procs, paths):
            pieces = list(qwen3_voice._ov_stream('repo', 'clone', {}))
            np.testing.assert_array_equal(pieces[0][0], np.array([.25, -.5], dtype='float32'))
            self.assertEqual(pieces[0][1], 24000)
            self.assert_reaped(procs, paths)

    def test_nonzero_child_error_is_preserved(self):
        with self.children("import sys;print('specific child error',file=sys.stderr);sys.exit(7)") as (procs, paths):
            with self.assertRaisesRegex(RuntimeError, 'specific child error'):
                list(qwen3_voice._ov_stream('repo', 'clone', {}))
            self.assert_reaped(procs, paths)

    def test_context_reaches_all_openvino_streaming_modes(self):
        for mode in ['native', 'clone', 'design']:
            stop = threading.Event()

            def hook(*_args):
                stop.set()
                yield np.ones(8, dtype='float32'), 24000

            with self.subTest(mode=mode), patch.dict(os.environ, {'AUDIO_BASE': 'qwen3voiceov'}), \
                    patch.object(qwen3_voice, '_OV_STREAM_HOOK', hook):
                backend = qwen3_voice.QwenVoiceBackend(object(), '/tmp/native-test')
                backend._prompt_path = lambda *_: '/tmp/missing-cancel-reference.wav'
                kw = {'ctx': tasks.stream_ctx(stop)}
                stream = (backend.iter_speak_native({'voice_id': 'Vivian', 'category': 'premade'}, 'test', **kw)
                          if mode == 'native' else backend.iter_clone('test', np.ones(8), 24000, 'ref', **kw)
                          if mode == 'clone' else backend.iter_design('a voice', 'test', **kw))
                with self.assertRaises(tasks.Cancelled):
                    list(stream)


if __name__ == '__main__':
    unittest.main()
