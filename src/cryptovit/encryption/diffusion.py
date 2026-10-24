"""
cryptovit/encryption/diffusion.py
===================================
DNA-based diffusion stage of the CryptoViT encryption pipeline.

Algorithm (forward / encrypt)
------------------------------
1. Convert the permuted image to DNA representation (per channel).
2. Convert the key-stream bytes to DNA using a secondary rule.
3. Apply DNA XOR between the image DNA and the key-stream DNA.
4. Decode the result back to uint8 pixel values.
5. Apply a chained byte XOR pass for extra diffusion strength.

The chaining ensures that each output pixel depends on all previous
output pixels (avalanche effect), giving high NPCR and UACI values.

Algorithm (inverse / decrypt)
------------------------------
The reverse: undo the chained XOR, then undo the DNA XOR.

All operations are lossless; decrypt(encrypt(img, kg), kg) == img exactly.

Parameters
----------
The ``KeyGenerator`` instance provides both the pixel key-stream
(for DNA XOR) and the rule selector (for DNA encoding rules).
"""

from __future__ import annotations

import logging

import numpy as np

from cryptovit.encryption import dna as dna_ops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward diffusion
# ---------------------------------------------------------------------------

def diffuse(
    image: np.ndarray,
    keystream: np.ndarray,
    rule_seq: np.ndarray,
) -> np.ndarray:
    """Apply DNA XOR diffusion to an image.

    Parameters
    ----------
    image : (H, W, C) or (H, W) uint8 array
        Permuted image to diffuse.
    keystream : (H * W) uint8 array
        Key-stream bytes from ``KeyGenerator.diffusion_keystream``.
    rule_seq : (H * W) uint8 array [values 0–7]
        Per-pixel DNA rule from ``KeyGenerator.dna_rule_sequence``.

    Returns
    -------
    diffused : same shape and dtype as ``image``
    """
    _validate_inputs(image, keystream, rule_seq)
    H, W = image.shape[:2]
    n = H * W
    is_2d = image.ndim == 2

    if is_2d:
        image = image[:, :, np.newaxis]
    C = image.shape[2]

    result = np.empty_like(image)
    for c in range(C):
        channel = image[:, :, c].reshape(-1)       # (n,)
        ks = keystream[:n] if keystream.ndim == 1 else keystream[:n, c]
        rules = rule_seq[:n].astype(np.uint8)

        # 1. Encode image pixels to DNA   shape: (n, 4)
        img_dna = dna_ops.encode(channel, rules)

        # 2. Encode key-stream to DNA using complementary rule
        key_rules = (rules + 4) % 8                # shifted rule for key-stream
        key_dna = dna_ops.encode(ks, key_rules)

        # 3. DNA XOR
        xored_dna = dna_ops.dna_xor(img_dna, key_dna)

        # 4. Decode back to uint8
        diffused_channel = dna_ops.decode(xored_dna, rules)

        # 5. Chained byte XOR for diffusion avalanche
        diffused_channel = _chain_xor(diffused_channel, ks)

        result[:, :, c] = diffused_channel.reshape(H, W)

    return result[:, :, 0] if is_2d else result


# ---------------------------------------------------------------------------
# Inverse diffusion
# ---------------------------------------------------------------------------

def inverse_diffuse(
    diffused: np.ndarray,
    keystream: np.ndarray,
    rule_seq: np.ndarray,
) -> np.ndarray:
    """Reverse the DNA diffusion stage.

    Parameters
    ----------
    diffused : (H, W, C) or (H, W) uint8 array
    keystream : (H * W) uint8 array  — same as used during encryption
    rule_seq  : (H * W) uint8 array  — same as used during encryption

    Returns
    -------
    image : same shape and dtype
    """
    _validate_inputs(diffused, keystream, rule_seq)
    H, W = diffused.shape[:2]
    n = H * W
    is_2d = diffused.ndim == 2

    if is_2d:
        diffused = diffused[:, :, np.newaxis]
    C = diffused.shape[2]

    result = np.empty_like(diffused)
    for c in range(C):
        channel = diffused[:, :, c].reshape(-1)
        ks = keystream[:n] if keystream.ndim == 1 else keystream[:n, c]
        rules = rule_seq[:n].astype(np.uint8)

        # 5 (undo). Reverse chained XOR
        un_chained = _inverse_chain_xor(channel, ks)

        # 4 (undo). Re-encode to DNA using same rules
        img_dna = dna_ops.encode(un_chained, rules)

        # 3 (undo). Re-encode key-stream and XOR again (XOR is self-inverse)
        key_rules = (rules + 4) % 8
        key_dna = dna_ops.encode(ks, key_rules)
        restored_dna = dna_ops.dna_xor(img_dna, key_dna)

        # 2 (undo). Decode back to uint8
        restored_channel = dna_ops.decode(restored_dna, rules)

        result[:, :, c] = restored_channel.reshape(H, W)

    return result[:, :, 0] if is_2d else result


# ---------------------------------------------------------------------------
# Chained XOR helpers
# ---------------------------------------------------------------------------

def _chain_xor(data: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Forward chained XOR: out[i] = data[i] XOR key[i] XOR out[i-1].

    out[-1] is initialised to key[-1] (last key byte) to avoid
    a trivially predictable starting condition.
    """
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.uint8)
    # Unrolling out[i] = d[i]^k[i]^out[i-1] gives a running XOR of (d^k),
    # seeded with key[-1]; this is the vectorised form of the sequential loop.
    mixed = np.bitwise_xor(data.astype(np.uint8), key[:n].astype(np.uint8))
    return np.bitwise_xor.accumulate(mixed) ^ np.uint8(key[n - 1])


def _inverse_chain_xor(data: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Inverse of ``_chain_xor``: out[i] = data[i] ^ key[i] ^ data[i-1]."""
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.uint8)
    data = data.astype(np.uint8)
    previous = np.concatenate(([np.uint8(key[n - 1])], data[:-1]))
    return data ^ key[:n].astype(np.uint8) ^ previous


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    image: np.ndarray,
    keystream: np.ndarray,
    rule_seq: np.ndarray,
) -> None:
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}")
    H, W = image.shape[:2]
    n = H * W
    if len(keystream) < n:
        raise ValueError(
            f"keystream too short: need {n}, got {len(keystream)}"
        )
    if len(rule_seq) < n:
        raise ValueError(
            f"rule_seq too short: need {n}, got {len(rule_seq)}"
        )
