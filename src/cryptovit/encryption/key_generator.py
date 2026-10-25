"""
cryptovit/encryption/key_generator.py
=======================================
High-level key generator that wraps the HyperchaosSystem.

Provides a clean interface for producing:
  * Permutation index arrays (for pixel/block scrambling)
  * Byte key-streams (for DNA diffusion)
  * Rule selection sequences (for DNA operation selection)

Usage
-----
    kg = KeyGenerator("my_secret_key")
    perm_indices = kg.pixel_permutation(H * W)      # shape (H*W,)
    keystream    = kg.diffusion_keystream(H * W)    # shape (H*W,) uint8
    rule_seq     = kg.dna_rule_sequence(H * W)      # shape (H*W,) in [0, 7]
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from cryptovit.encryption.chaotic_map import HyperchaosSystem

logger = logging.getLogger(__name__)


class KeyGenerator:
    """Derives all encryption sub-keys from a single master key.

    Two independent ``HyperchaosSystem`` instances are created -
    one for permutation and one for diffusion - using slightly different
    key derivations to ensure independence.

    Parameters
    ----------
    key : str or bytes
        Master secret key.
    transient_steps : int
        Number of warm-up steps for each chaotic system.
    """

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
        """Return a permutation of [0, n) for pixel-level scrambling.

        Parameters
        ----------
        n : int
            Total number of pixels (H * W for grayscale, H * W * C or
            H * W after flattening per channel).

        Returns
        -------
        (n,) int64 array - permutation of [0, n)
        """
        return self._perm_sys.generate_indices(n)

    def channel_permutations(self, n: int, channels: int) -> np.ndarray:
        """Independent uniform permutations of [0, n), one per channel.

        Used by encryption scheme ``"v2"``; each channel draws its own
        permutation from the continuing permutation stream.

        Returns
        -------
        (channels, n) int64 array
        """
        return np.stack([self._perm_sys.generate_permutation(n) for _ in range(channels)])

    def inverse_permutation(self, perm: np.ndarray) -> np.ndarray:
        """Return the inverse of a permutation array.

        Parameters
        ----------
        perm : (n,) array - a permutation of [0, n)

        Returns
        -------
        inv_perm : (n,) array - inverse permutation such that
                   inv_perm[perm[i]] == i for all i
        """
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm), dtype=perm.dtype)
        return inv

    def diffusion_keystream(self, n: int) -> np.ndarray:
        """Return a uint8 key-stream for XOR / DNA diffusion.

        Parameters
        ----------
        n : int
            Number of bytes required.

        Returns
        -------
        (n,) uint8 array
        """
        return self._diff_sys.generate_bytes(n)

    def channel_keystreams(self, n: int, channels: int) -> np.ndarray:
        """Independent uint8 key-streams, one per channel (scheme ``"v2"``).

        Returns
        -------
        (n, channels) uint8 array - column ``c`` is the key-stream of channel c
        """
        return np.stack([self._diff_sys.generate_bytes(n) for _ in range(channels)], axis=1)

    def dna_rule_sequence(self, n: int) -> np.ndarray:
        """Return a sequence of DNA rule indices in [0, 7].

        Parameters
        ----------
        n : int
            Number of rule indices required.

        Returns
        -------
        (n,) uint8 array with values in [0, 7]
        """
        raw = self._diff_sys.generate(n)
        return ((np.abs(raw) * 1000).astype(np.uint64) % 8).astype(np.uint8)

    def reset(self) -> None:
        """Reset both chaotic systems to initial state (re-derive from master key)."""
        perm_key = hashlib.sha256(self._master_key + b"perm").digest()
        diff_key = hashlib.sha256(self._master_key + b"diff").digest()
        self._perm_sys.reset(perm_key)
        self._diff_sys.reset(diff_key)
        logger.debug("KeyGenerator reset.")
