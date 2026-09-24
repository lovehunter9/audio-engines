"""The container's memory account, read from cgroup v2. 🔴 Crossing the GPU's limit raises an
exception this engine catches and retries; crossing this is an OOMKill: exit 137, mid-request."""
import os

# 🔴 v2 only. A v1 host keeps the limit at `memory/memory.limit_in_bytes`, where none of this
# looks, so a container that HAS a limit is indistinguishable from one that has none.
ROOT = "/sys/fs/cgroup"
MEMINFO = "/proc/meminfo"


def _read_int(path):
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    # "max" means no limit, which is not a number and must not be read as one.
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _machine_available(meminfo=MEMINFO):
    """MemAvailable in bytes, or None. The wall to measure against when the container has
    no limit of its own -- it can still run the machine out of memory."""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


# ⚠️ GPU memory a driver could not fit on the card is charged here too, under `kernel`, on cards
# shared in software: measured at +3.36 GiB in one interval, never returned, in no process's RSS.
def _stat(root, key):
    try:
        with open(os.path.join(root, "memory.stat")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] == key:
                    return int(parts[1])
    except (OSError, ValueError):
        pass
    return None


def read(root=ROOT, meminfo=MEMINFO):
    """{current, max, available, reclaimable, kernel} in bytes; a key is None when unreadable."""
    return {"current": _read_int(os.path.join(root, "memory.current")),
            "max": _read_int(os.path.join(root, "memory.max")),
            "available": _machine_available(meminfo),
            # 🔴 `memory.current` counts page cache, so an engine that just read a 3.4 GiB
            # checkpoint sits at its limit under no pressure. `active_file` is left out: undercount.
            "reclaimable": _stat(root, "inactive_file"),
            # 🔴 Read but not subtracted anywhere, on purpose: this is the part the kernel CANNOT
            # reclaim, so it tells "tight because of page cache" apart from "tight because of the
            # driver" -- two situations that look identical in `current` and end differently. At a
            # container's limit with 14.0 GB here, 2.2 anon and 0.15 file, hitting `max` is a kill
            # rather than a reclaim, and only `torch.cuda.empty_cache()` can give any of it back.
            "kernel": _stat(root, "kernel")}


def headroom(snapshot):
    """Bytes left before this container is killed: its own limit when there is one, the machine's
    free memory when there is not. 🔴 None is a third answer: nothing read, not "no room left"."""
    cur, cap = snapshot.get("current"), snapshot.get("max")
    if cur is not None and cap is not None:
        cur -= snapshot.get("reclaimable") or 0
        return max(0, cap - cur)
    return snapshot.get("available")


def batch_headroom(snapshot, gpu_mode=""):
    """Bytes a batch may spend. iGPU shares the machine's RAM, so the container account
    does not see the other residents. Arc and anything else keep `headroom` as it is."""
    room = headroom(snapshot)
    if (gpu_mode or "").strip().lower() != "intel":
        return room
    avail = snapshot.get("available")
    if room is None or avail is None:
        return room
    return min(room, avail)
