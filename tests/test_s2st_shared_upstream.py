import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from wrapper.s2st import gpu_queue


class SharedUpstreamTests(unittest.TestCase):
    def test_upstream_readers_overlap_and_a_queued_ming_blocks_new_readers(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, "SHARED_UPSTREAM", True):
            path = Path(directory) / "queue"
            tickets = [gpu_queue.Ticket(label, path) for label in ["r2t2", "t3po", "ming", "r2t2"]]
            first, second, ming, late = tickets
            try:
                self.assertTrue(first.try_acquire())
                self.assertTrue(second.try_acquire())
                self.assertFalse(ming.try_acquire())
                self.assertFalse(late.try_acquire())
                first.close()
                self.assertFalse(ming.try_acquire())
                second.close()
                self.assertTrue(ming.try_acquire())
                self.assertFalse(late.try_acquire())
                ming.close()
                self.assertTrue(late.try_acquire())
            finally:
                for ticket in tickets:
                    if not ticket.fd.closed:
                        ticket.close()
            self.assertEqual(json.loads(path.with_suffix(".json").read_text()), [])

    def test_default_policy_and_unknown_stage_remain_exclusive(self):
        for enabled, label in [(False, "t3po"), (True, "other")]:
            with self.subTest(enabled=enabled, label=label), tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, "SHARED_UPSTREAM", enabled):
                path = Path(directory) / "queue"
                first = gpu_queue.Ticket("r2t2", path)
                second = gpu_queue.Ticket(label, path)
                try:
                    self.assertTrue(first.try_acquire())
                    self.assertFalse(second.try_acquire())
                    first.close()
                    self.assertTrue(second.try_acquire())
                finally:
                    for ticket in [first, second]:
                        if not ticket.fd.closed:
                            ticket.close()

    def test_reader_lock_is_shared_between_processes_but_ming_is_exclusive(self):
        code = """import sys
from wrapper.s2st import gpu_queue
gpu_queue.configure(sys.argv[1], shared_upstream=True)
with gpu_queue.stage('r2t2'):
 print('acquired',flush=True)
 sys.stdin.readline()
"""
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, "SHARED_UPSTREAM", True):
            path = Path(directory) / "queue"
            child = subprocess.Popen([sys.executable, "-u", "-c", code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            tickets = []
            try:
                self.assertEqual(child.stdout.readline().strip(), "acquired")
                mt = gpu_queue.Ticket("t3po", path)
                ming = gpu_queue.Ticket("ming", path)
                tickets.extend([mt, ming])
                self.assertTrue(mt.try_acquire())
                self.assertFalse(ming.try_acquire())
                mt.close()
                self.assertFalse(ming.try_acquire())
                child.stdin.write("done\n")
                child.stdin.flush()
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertTrue(ming.try_acquire())
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
                for ticket in tickets:
                    if not ticket.fd.closed:
                        ticket.close()
                for stream in [child.stdin, child.stdout, child.stderr]:
                    stream.close()

    def test_invalid_bypass_flags_fail_before_model_loading(self):
        from wrapper.s2st import server
        for family, args in [
            ("s2st-tencent", "--share-upstream-gpu"),
            ("s2st-youdao", "--share-upstream-gpu --overlap-asr"),
            ("s2st-youdao", "--share-upstream-gpu --overlap-translation"),
        ]:
            with self.subTest(family=family, args=args), patch.dict(os.environ, {"AUDIO_BASE": family, "ENGINE_ARGS": args}):
                with self.assertRaisesRegex(ValueError, "MING-exclusive"):
                    server.build_app(["audio_s2s"])


class SharedUpstreamCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_reader_removes_only_its_own_ticket(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gpu_queue, "DEFAULT", Path(directory)/"queue"), patch.object(gpu_queue, "SHARED_UPSTREAM", True):
            entered = asyncio.Event()
            async def reader():
                async with gpu_queue.async_stage("r2t2"):
                    entered.set()
            with gpu_queue.stage("ming"):
                task = asyncio.create_task(reader())
                for _ in range(100):
                    if len(json.loads((Path(directory)/"queue.json").read_text())) == 2:
                        break
                    await asyncio.sleep(.005)
                self.assertFalse(entered.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual([x["label"] for x in json.loads((Path(directory)/"queue.json").read_text())], ["ming"])
            self.assertEqual(json.loads((Path(directory)/"queue.json").read_text()), [])
