"""Portable Philox normal patches matching the native CUDA audio sampler."""
import numpy as np


class PatchNoise:
    def __init__(self, seed):
        self.seed = int(seed) & ((1 << 64) - 1)
        self.offset = 0

    def normal(self, patch_size, channels):
        count = int(patch_size) * int(channels)
        if not 0 < count <= 256:
            raise ValueError("FireRed CUDA patch sampler supports up to 256 values")
        mask = np.uint64(0xffffffff)
        c0 = np.full(count, self.offset, dtype=np.uint64)
        c1 = np.zeros(count, dtype=np.uint64)
        c2 = np.arange(count, dtype=np.uint64)
        c3 = np.zeros(count, dtype=np.uint64)
        k0, k1 = np.uint64(self.seed & 0xffffffff), np.uint64(self.seed >> 32)
        for round_index in range(10):
            p0 = c0 * np.uint64(0xd2511f53)
            p1 = c2 * np.uint64(0xcd9e8d57)
            c0, c1, c2, c3 = (p1 >> np.uint64(32)) ^ c1 ^ k0, p1 & mask, (p0 >> np.uint64(32)) ^ c3 ^ k1, p0 & mask
            if round_index != 9:
                k0 = (k0 + np.uint64(0x9e3779b9)) & mask
                k1 = (k1 + np.uint64(0xbb67ae85)) & mask
        self.offset += 1
        u = (c0.astype(np.float32) + np.float32(1)) * np.float32(2 ** -32)
        angle = (c1.astype(np.float32) + np.float32(1)) * np.float32(2 ** -32 * 2 * np.pi)
        values = np.sqrt(np.float32(-2) * np.log(u)) * np.sin(angle)
        return values.reshape(1, int(patch_size), int(channels))
