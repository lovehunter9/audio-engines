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
