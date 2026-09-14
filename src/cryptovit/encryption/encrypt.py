"""
cryptovit/encryption/encrypt.py
=================================
Top-level encryption / decryption API.

Pipeline (encrypt)
------------------
    image (H,W,C uint8)
      → KeyGenerator(key)
        → permute_image(image, perm_indices)          [Phase 3]
          → diffuse(permuted, keystream, rule_seq)    [Phase 4]
            → encrypted (H,W,C uint8)

Pipeline (decrypt)
------------------
    encrypted (H,W,C uint8)
      → KeyGenerator(key)
        → inverse_diffuse(encrypted, keystream, rule_seq)   [Phase 4 undo]
          → inverse_permute_image(undiffused, perm_indices) [Phase 3 undo]
            → image (H,W,C uint8)

The key generator is always reset before use so that encrypt and decrypt
produce identical key-streams regardless of call order.

Usage
-----
    from cryptovit.encryption import encrypt, decrypt

    encrypted, meta = encrypt(image, key="secret")
    restored = decrypt(encrypted, key="secret", meta=meta)
    assert np.array_equal(image, restored)
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from cryptovit.encryption.diffusion import diffuse, inverse_diffuse
from cryptovit.encryption.key_generator import KeyGenerator
from cryptovit.encryption.permutation import (
    inverse_permute_image,
    permute_image,
)

logger = logging.getLogger(__name__)


def encrypt(
    image: np.ndarray,
    key: str | bytes,
    transient_steps: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """Encrypt an image using hyperchaotic permutation + DNA diffusion.

    Parameters
    ----------
    image : (H, W, C) or (H, W) uint8 numpy array
        Input plaintext image.
    key : str or bytes
        Secret key.
    transient_steps : int
        Number of warm-up steps for the chaotic system.

    Returns
    -------
    encrypted : same shape and dtype as ``image``
    meta : dict with keys:
        ``shape``     — original image shape
        ``key_hash``  — SHA-256 of the key (for verification, not reconstruction)
    """
    import hashlib

    if image.dtype != np.uint8:
        raise TypeError(f"Image must be uint8; got {image.dtype}.")

    H, W = image.shape[:2]
    n = H * W

    # --- Key generation ---
    kg = KeyGenerator(key, transient_steps=transient_steps)

    # Phase 3: Permutation
    perm = kg.pixel_permutation(n)
    permuted = permute_image(image, perm)

    # Phase 4: Diffusion
    ks   = kg.diffusion_keystream(n)
    rules = kg.dna_rule_sequence(n)
    encrypted = diffuse(permuted, ks, rules)

    key_bytes = key.encode("utf-8") if isinstance(key, str) else key
    meta = {
        "shape": image.shape,
        "key_hash": hashlib.sha256(key_bytes).hexdigest(),
    }

    logger.debug("Encrypted image shape=%s", image.shape)
    return encrypted, meta


def decrypt(
    encrypted: np.ndarray,
    key: str | bytes,
    meta: Dict | None = None,
    transient_steps: int = 1000,
) -> np.ndarray:
    """Decrypt an image produced by ``encrypt``.

    Parameters
    ----------
    encrypted : (H, W, C) or (H, W) uint8 numpy array
    key : str or bytes
        The same key used during encryption.
    meta : dict, optional
        Metadata returned by ``encrypt`` (used only for shape verification).
    transient_steps : int

    Returns
    -------
    image : same shape and dtype as ``encrypted``
    """
    if encrypted.dtype != np.uint8:
        raise TypeError(f"Encrypted array must be uint8; got {encrypted.dtype}.")

    H, W = encrypted.shape[:2]
    n = H * W

    # Re-derive the same key-streams (reset ensures identical sequence)
    kg = KeyGenerator(key, transient_steps=transient_steps)

    # Must regenerate in the same order as encrypt
    perm   = kg.pixel_permutation(n)
    ks     = kg.diffusion_keystream(n)
    rules  = kg.dna_rule_sequence(n)

    # Undo diffusion
    undiffused = inverse_diffuse(encrypted, ks, rules)

    # Undo permutation
    restored = inverse_permute_image(undiffused, perm)

    logger.debug("Decrypted image shape=%s", restored.shape)
    return restored


def verify_roundtrip(
    image: np.ndarray,
    key: str | bytes,
    transient_steps: int = 1000,
) -> bool:
    """Encrypt then decrypt and verify pixel-perfect reconstruction.

    Parameters
    ----------
    image : uint8 numpy array
    key : str or bytes

    Returns
    -------
    bool — True if reconstruction is exact, False otherwise.
    """
    encrypted, meta = encrypt(image, key, transient_steps)
    restored = decrypt(encrypted, key, meta, transient_steps)
    ok = bool(np.array_equal(image, restored))
    if ok:
        logger.info("Round-trip verification PASSED.")
    else:
        diff = np.sum(image != restored)
        logger.error("Round-trip verification FAILED! %d pixels differ.", diff)
    return ok
