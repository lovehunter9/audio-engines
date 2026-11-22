"""Math library threads follow the container's CPU limit, not the host's core count."""
import os
import tempfile
import unittest

from wrapper import app, cgroup


def _cpu_max(body):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "cpu.max"), "w") as f:
        f.write(body)
    return d


class CpuQuotaTest(unittest.TestCase):
    def test_a_fractional_limit_rounds_up(self):
        self.assertEqual(cgroup.cpu_quota(root=_cpu_max("250000 100000\n")), 3)

    def test_max_and_a_missing_file_are_no_limit(self):
        self.assertIsNone(cgroup.cpu_quota(root=_cpu_max("max 100000\n")))
        self.assertIsNone(cgroup.cpu_quota(root=tempfile.mkdtemp()))

    def test_a_tiny_limit_still_allows_one_thread(self):
        self.assertEqual(cgroup.cpu_quota(root=_cpu_max("10000 100000\n")), 1)


class CapThreadsTest(unittest.TestCase):
    def test_a_limit_below_the_host_sets_every_pool(self):
        env = {}
        self.assertEqual(app.cap_threads(quota=4, cpus=24, environ=env), 4)
        self.assertEqual({k: env[k] for k in app.THREAD_VARS}, dict.fromkeys(app.THREAD_VARS, "4"))

    def test_a_value_the_deployment_set_wins(self):
        env = {"OMP_NUM_THREADS": "8"}
        app.cap_threads(quota=4, cpus=24, environ=env)
        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        self.assertEqual(env["MKL_NUM_THREADS"], "4")

    def test_no_limit_or_a_limit_at_the_host_leaves_the_defaults(self):
        for quota in (0, 24, 32):
            env = {}
            with self.subTest(quota=quota):
                self.assertIsNone(app.cap_threads(quota=quota, cpus=24, environ=env))
                self.assertEqual(env, {})


if __name__ == "__main__":
    unittest.main()
