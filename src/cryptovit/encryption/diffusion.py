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
    rule_seq : (H * W) uint8 array [values 0-7]
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
        channel = image[:, :, c].reshape(-1)
        ks = keystream[:n] if keystream.ndim == 1 else keystream[:n, c]
        rules = rule_seq[:n].astype(np.uint8)

        img_dna = dna_ops.encode(channel, rules)

        key_rules = (rules + 4) % 8
        key_dna = dna_ops.encode(ks, key_rules)

        xored_dna = dna_ops.dna_xor(img_dna, key_dna)

        diffused_channel = dna_ops.decode(xored_dna, rules)

        diffused_channel = _chain_xor(diffused_channel, ks)

        result[:, :, c] = diffused_channel.reshape(H, W)

    return result[:, :, 0] if is_2d else result


def inverse_diffuse(
    diffused: np.ndarray,
    keystream: np.ndarray,
    rule_seq: np.ndarray,
) -> np.ndarray:
    """Reverse the DNA diffusion stage.

    Parameters
    ----------
    diffused : (H, W, C) or (H, W) uint8 array
    keystream : (H * W) uint8 array  - same as used during encryption
    rule_seq  : (H * W) uint8 array  - same as used during encryption

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

        un_chained = _inverse_chain_xor(channel, ks)

        img_dna = dna_ops.encode(un_chained, rules)

        key_rules = (rules + 4) % 8
        key_dna = dna_ops.encode(ks, key_rules)
        restored_dna = dna_ops.dna_xor(img_dna, key_dna)

        restored_channel = dna_ops.decode(restored_dna, rules)

        result[:, :, c] = restored_channel.reshape(H, W)

    return result[:, :, 0] if is_2d else result


def _chain_xor(data: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Forward chained XOR: out[i] = data[i] XOR key[i] XOR out[i-1].

    out[-1] is initialised to key[-1] (last key byte) to avoid
    a trivially predictable starting condition.
    """
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.uint8)
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
        raise ValueError(f"keystream too short: need {n}, got {len(keystream)}")
    if len(rule_seq) < n:
        raise ValueError(f"rule_seq too short: need {n}, got {len(rule_seq)}")
