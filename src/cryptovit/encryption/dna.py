from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


_ENCODE_TABLE = np.array(
    [
        [0, 1, 3, 2],
        [0, 2, 3, 1],
        [1, 0, 2, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 3, 1, 0],
        [3, 0, 2, 1],
        [3, 1, 2, 0],
    ],
    dtype=np.uint8,
)

_DECODE_TABLE = np.zeros((8, 4), dtype=np.uint8)
for _r in range(8):
    for _bits in range(4):
        _base = _ENCODE_TABLE[_r, _bits]
        _DECODE_TABLE[_r, _base] = _bits

_DNA_XOR = np.array(
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=np.uint8,
)

_DNA_ADD = np.array(
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ],
    dtype=np.uint8,
)

_DNA_SUB = np.array(
    [
        [0, 3, 2, 1],
        [1, 0, 3, 2],
        [2, 1, 0, 3],
        [3, 2, 1, 0],
    ],
    dtype=np.uint8,
)


def encode(
    pixels: np.ndarray,
    rule: int | np.ndarray,
) -> np.ndarray:
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)

    b3 = (pixels >> 6) & 0x3
    b2 = (pixels >> 4) & 0x3
    b1 = (pixels >> 2) & 0x3
    b0 = pixels & 0x3

    if isinstance(rule, int):
        tbl = _ENCODE_TABLE[rule]
        d3 = tbl[b3]
        d2 = tbl[b2]
        d1 = tbl[b1]
        d0 = tbl[b0]
    else:
        rule = rule.astype(np.uint8)
        d3 = _ENCODE_TABLE[rule, b3]
        d2 = _ENCODE_TABLE[rule, b2]
        d1 = _ENCODE_TABLE[rule, b1]
        d0 = _ENCODE_TABLE[rule, b0]

    return np.stack([d3, d2, d1, d0], axis=-1)


def decode(
    dna: np.ndarray,
    rule: int | np.ndarray,
) -> np.ndarray:
    d3, d2, d1, d0 = dna[..., 0], dna[..., 1], dna[..., 2], dna[..., 3]

    if isinstance(rule, int):
        tbl = _DECODE_TABLE[rule]
        b3 = tbl[d3]
        b2 = tbl[d2]
        b1 = tbl[d1]
        b0 = tbl[d0]
    else:
        rule = rule.astype(np.uint8)
        b3 = _DECODE_TABLE[rule, d3]
        b2 = _DECODE_TABLE[rule, d2]
        b1 = _DECODE_TABLE[rule, d1]
        b0 = _DECODE_TABLE[rule, d0]

    return ((b3 << 6) | (b2 << 4) | (b1 << 2) | b0).astype(np.uint8)


def dna_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _DNA_XOR[a, b]


def dna_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _DNA_ADD[a, b]


def dna_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _DNA_SUB[a, b]
