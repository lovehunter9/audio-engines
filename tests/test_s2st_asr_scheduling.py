import json
import os
import time
import tempfile
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from wrapper.s2st import gpu_queue
from wrapper.s2st.r2t2_worker import recognition_stage


class RecognitionSchedulingTests(unittest.TestCase):
    def check_mode(self, overlap):
        attempted, entered, done = (threading.Event() for _ in range(3))
        failures = []
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, 'DEFAULT', Path(directory) / 'queue'):
            def recognize():
                attempted.set()
                try:
                    with recognition_stage(overlap):
                        entered.set()
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    done.set()
            with gpu_queue.stage('ming'):
                thread = threading.Thread(target=recognize, daemon=True)
                thread.start()
                self.assertTrue(attempted.wait(2))
                self.assertEqual(entered.wait(.1 if not overlap else 2), overlap)
            self.assertTrue(done.wait(2))
            thread.join(2)
            self.assertFalse(failures)
            self.assertTrue(entered.is_set())
            self.assertEqual((Path(directory) / 'queue.json').read_text(), '[]')

    def test_default_preserves_serial_exclusion(self):
        self.check_mode(False)

    def test_opt_in_asr_can_enter_while_ming_holds_its_stage(self):
        self.check_mode(True)

    def test_opt_in_does_not_bypass_other_stages(self):
        attempted, entered = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, 'DEFAULT', Path(directory) / 'queue'):
            def translate():
                attempted.set()
                with gpu_queue.stage('t3po'):
                    entered.set()
            with gpu_queue.stage('ming'):
                with recognition_stage(True):
                    pass
                thread = threading.Thread(target=translate, daemon=True)
                thread.start()
                self.assertTrue(attempted.wait(2))
                self.assertFalse(entered.wait(.1))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(entered.is_set())


    def test_overlap_waits_for_an_aged_ming_ticket_and_resumes_after_release(self):
        scheduled, entered, done = (threading.Event() for _ in range(3))
        metadata, failures = {}, []
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, 'DEFAULT', Path(directory) / 'queue'):
            def recognize():
                try:
                    context = recognition_stage(True, metadata)
                    scheduled.set()
                    with context:
                        entered.set()
                except BaseException as error:
                    failures.append(error)
                finally:
                    done.set()
            with gpu_queue.stage('ming'):
                path=Path(directory)/'queue.json'
                entries=json.loads(path.read_text())
                entries[0]['enqueued']=time.monotonic()-gpu_queue.MING_PROTECTION_SECONDS-1
                path.write_text(json.dumps(entries))
                thread=threading.Thread(target=recognize,daemon=True)
                thread.start()
                self.assertTrue(scheduled.wait(2))
                self.assertFalse(entered.wait(.1))
                self.assertTrue(metadata['ming_deadline_protected'])
                self.assertGreaterEqual(metadata['ming_stage_age_seconds'],20)
            self.assertTrue(done.wait(2))
            thread.join(2)
            self.assertFalse(failures)
            self.assertTrue(entered.is_set())
            self.assertIsNone(gpu_queue.ming_priority_age())
            self.assertEqual(path.read_text(),'[]')

    def test_dead_owner_does_not_block_overlap(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, 'DEFAULT', Path(directory) / 'queue'):
            path=Path(directory)/'queue.json'
            path.write_text(json.dumps([dict(label='ming',pid=os.getpid(),start='not-this-process',enqueued=time.monotonic()-100)]))
            metadata={}
            with recognition_stage(True,metadata) as wait:
                self.assertEqual(wait,0)
            self.assertFalse(metadata['ming_deadline_protected'])

    def test_aged_translation_ticket_does_not_disable_asr_overlap(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, 'DEFAULT', Path(directory) / 'queue'):
            with gpu_queue.stage('t3po'):
                path=Path(directory)/'queue.json'
                entries=json.loads(path.read_text())
                entries[0]['enqueued']=time.monotonic()-100
                path.write_text(json.dumps(entries))
                metadata={}
                with recognition_stage(True, metadata) as wait:
                    self.assertEqual(wait, 0)
                self.assertFalse(metadata['ming_deadline_protected'])
