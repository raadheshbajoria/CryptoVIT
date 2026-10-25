"""
cryptovit/encryption/dna.py
=============================
DNA encoding / decoding and DNA arithmetic operations.

Background
----------
DNA computing represents 8-bit pixel values using four DNA bases:
A, T, C, G.  Each base encodes 2 bits, so one pixel (8 bits) becomes
a sequence of 4 DNA symbols.

There are 8 valid DNA encoding rules (bijective mappings from 2-bit
pairs to {A, C, G, T}).  We follow the standard rules from the
image-encryption literature (Chai et al. 2017):

    Rule | 00 | 01 | 10 | 11
    -----|----|----|----|----|
      0  |  A |  T |  G |  C
      1  |  A |  C |  G |  T
      2  |  T |  A |  C |  G
      3  |  T |  G |  C |  A
      4  |  C |  A |  T |  G
      5  |  C |  G |  T |  A
      6  |  G |  A |  C |  T
      7  |  G |  T |  C |  A   (complementary Watson-Crick rule)

DNA arithmetic
--------------
DNA addition, subtraction, and XOR tables are defined on the four bases
(treating A=0, T=1, C=2, G=3 in cyclic group Z_4):

    XOR:  A^A=A  A^T=T  A^C=C  A^G=G
          T^A=T  T^T=A  T^C=G  T^G=C
          C^A=C  C^T=G  C^C=A  C^G=T
          G^A=G  G^T=C  G^C=T  G^G=A

    ADD:  A+A=A  A+T=T  A+C=C  A+G=G
          T+A=T  T+T=C  T+C=G  T+G=A
          C+A=C  C+T=G  C+C=A  C+G=T
          G+A=G  G+T=A  G+C=T  G+G=C

Implementation notes
--------------------
* All operations are vectorised using NumPy for speed.
* The DNA symbols are stored as uint8 integers (0=A, 1=T, 2=C, 3=G)
  internally; no string conversion is needed.
* Input images are uint8 (0-255); each pixel is split into four
  2-bit groups -> four DNA bases.
"""

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
    """Encode uint8 pixel values into DNA bases.

    Parameters
    ----------
    pixels : (...) uint8 array
        Pixel values (0-255).  Can be any shape.
    rule : int or (pixels.shape) uint8 array
        DNA encoding rule index (0-7) or a per-pixel rule array.

    Returns
    -------
    dna : (..., 4) uint8 array
        Four DNA bases (0-3) per pixel, MSB first.
        Shape is ``pixels.shape + (4,)``.
    """
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
    """Decode DNA bases back to uint8 pixel values.

    Parameters
    ----------
    dna : (..., 4) uint8 array
        Four DNA bases per pixel.
    rule : int or (...) uint8 array
        Same rule(s) used during encoding.

    Returns
    -------
    pixels : (...) uint8 array
    """
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
    """Element-wise DNA XOR of two DNA arrays.

    Parameters
    ----------
    a, b : same-shape uint8 arrays (values 0-3)

    Returns
    -------
    result : same-shape uint8 array
    """
    return _DNA_XOR[a, b]


def dna_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise DNA addition (mod 4)."""
    return _DNA_ADD[a, b]


def dna_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise DNA subtraction (mod 4)."""
    return _DNA_SUB[a, b]
