"""The container's memory account, read from files that may not be there.

🔴 Every reader here has to keep "could not be read" apart from a number, because the
bound built on it treats them as opposite answers: unreadable means there is no host
bound at all, while zero means the container has nothing left and every call must be as
small as it can be.
"""
import os
import tempfile
import unittest

from wrapper import cgroup


class ReadTest(unittest.TestCase):
    def _root(self, **files):
        d = tempfile.mkdtemp()
        for name, body in files.items():
            with open(os.path.join(d, name.replace("_", ".", 1)), "w") as f:
                f.write(body)
        return d

    def test_it_reads_the_limit_the_usage_and_what_is_reclaimable(self):
        root = self._root(memory_current="2000\n", memory_max="8000\n",
                          memory_stat="anon 1200\ninactive_file 800\nfile 900\n")
        got = cgroup.read(root=root, meminfo="/nonexistent")
        self.assertEqual((got["current"], got["max"], got["reclaimable"]), (2000, 8000, 800))

    def test_the_literal_max_is_no_limit_not_a_number(self):
        # 🔴 cgroup v2 writes "max" for no limit. Read as a number it would be an
        # exception; read as zero it would be a container with nothing left.
        root = self._root(memory_current="10\n", memory_max="max\n")
        self.assertIsNone(cgroup.read(root=root, meminfo="/nonexistent")["max"])

    def test_files_that_are_not_there_are_None_rather_than_zero(self):
        got = cgroup.read(root=self._root(), meminfo="/nonexistent")
        self.assertEqual([got["current"], got["max"], got["reclaimable"], got["kernel"]],
                         [None] * 4)

    def test_what_the_kernel_cannot_reclaim_is_read_and_not_subtracted(self):
        """🔴 The two ways a container gets tight look identical in `current` and end
        differently: page cache is given back under pressure, the driver's memory is not.

        A card shared in software charges GPU allocations here, under `kernel`. Measured
        on a container at its limit: 14.0 GB kernel against 2.2 anon and 0.15 file -- so
        reaching `max` was a kill, not a reclaim. Read so the two can be told apart;
        deliberately NOT part of `headroom`, which stays the conservative arithmetic.
        """
        root = self._root(memory_current="16000\n", memory_max="17000\n",
                          memory_stat="anon 2200\ninactive_file 150\nkernel 14000\n")
        got = cgroup.read(root=root, meminfo="/nonexistent")
        self.assertEqual(got["kernel"], 14000)
        # The bound did not move: still max - (current - reclaimable).
        self.assertEqual(cgroup.headroom(got), 17000 - (16000 - 150))

    def test_a_stat_file_without_the_key_is_no_reading(self):
        root = self._root(memory_stat="anon 5\nslab 6\n")
        self.assertIsNone(cgroup.read(root=root, meminfo="/nonexistent")["reclaimable"])

    def test_a_stat_line_that_is_not_a_number_does_not_raise(self):
        root = self._root(memory_stat="inactive_file lots\n")
        self.assertIsNone(cgroup.read(root=root, meminfo="/nonexistent")["reclaimable"])


class HeadroomTest(unittest.TestCase):
    def test_page_cache_is_not_memory_this_container_cannot_give_back(self):
        # 🔴 An engine that just read a checkpoint out of a cache volume sits near its
        # limit under no pressure. Counting that as used collapses the bound for good.
        self.assertEqual(
            cgroup.headroom({"current": 7000, "max": 8000, "reclaimable": 3000}), 4000)

    def test_a_container_at_its_limit_has_nothing_left_and_says_zero(self):
        self.assertEqual(
            cgroup.headroom({"current": 8000, "max": 8000, "reclaimable": 0}), 0)

    def test_over_the_limit_is_zero_rather_than_negative(self):
        self.assertEqual(
            cgroup.headroom({"current": 9000, "max": 8000, "reclaimable": 0}), 0)

    def test_no_limit_of_its_own_falls_back_to_the_machine(self):
        # It can still run the machine out of memory, so the machine is the wall then.
        self.assertEqual(
            cgroup.headroom({"current": 10, "max": None, "available": 4242}), 4242)

    def test_nothing_readable_is_None_not_zero(self):
        self.assertIsNone(
            cgroup.headroom({"current": None, "max": None, "available": None}))


if __name__ == "__main__":
    unittest.main()
