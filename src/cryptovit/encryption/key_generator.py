from __future__ import annotations

import hashlib
import logging

import numpy as np

from cryptovit.encryption.chaotic_map import HyperchaosSystem

logger = logging.getLogger(__name__)


class KeyGenerator:
    def __init__(self, key: str | bytes, transient_steps: int = 1000):
        if isinstance(key, str):
            key = key.encode("utf-8")
        self._master_key = key

        perm_key = hashlib.sha256(key + b"perm").digest()
        self._perm_sys = HyperchaosSystem(perm_key, transient_steps=transient_steps)

        diff_key = hashlib.sha256(key + b"diff").digest()
        self._diff_sys = HyperchaosSystem(diff_key, transient_steps=transient_steps)

        logger.debug("KeyGenerator initialised.")

    def pixel_permutation(self, n: int) -> np.ndarray:
        return self._perm_sys.generate_indices(n)

    def channel_permutations(self, n: int, channels: int) -> np.ndarray:
        return np.stack([self._perm_sys.generate_permutation(n) for _ in range(channels)])

    def inverse_permutation(self, perm: np.ndarray) -> np.ndarray:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm), dtype=perm.dtype)
        return inv

    def diffusion_keystream(self, n: int) -> np.ndarray:
        return self._diff_sys.generate_bytes(n)

    def channel_keystreams(self, n: int, channels: int) -> np.ndarray:
        return np.stack([self._diff_sys.generate_bytes(n) for _ in range(channels)], axis=1)

    def dna_rule_sequence(self, n: int) -> np.ndarray:
        raw = self._diff_sys.generate(n)
        return ((np.abs(raw) * 1000).astype(np.uint64) % 8).astype(np.uint8)

    def reset(self) -> None:
        perm_key = hashlib.sha256(self._master_key + b"perm").digest()
        diff_key = hashlib.sha256(self._master_key + b"diff").digest()
        self._perm_sys.reset(perm_key)
        self._diff_sys.reset(diff_key)
        logger.debug("KeyGenerator reset.")
