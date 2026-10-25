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
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.uint8)
    mixed = np.bitwise_xor(data.astype(np.uint8), key[:n].astype(np.uint8))
    return np.bitwise_xor.accumulate(mixed) ^ np.uint8(key[n - 1])


def _inverse_chain_xor(data: np.ndarray, key: np.ndarray) -> np.ndarray:
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
