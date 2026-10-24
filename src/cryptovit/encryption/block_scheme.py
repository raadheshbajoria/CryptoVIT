"""
cryptovit/encryption/block_scheme.py
======================================
Block-wise "learnable" image encryption (schemes ``block<B>``, e.g. ``block2``).

Why this exists
---------------
The default scheme mixes every pixel (permutation + DNA-XOR + chained XOR).
That protects the picture strongly but also destroys what a Vision
Transformer learns from, and models trained on it classify poorly. This scheme
keeps the structure a ViT can use while still hiding the picture:

  1. channel permutation      - the colour channels are reordered,
  2. block negative/positive  - each B x B block is optionally inverted
                                (value v -> 255 - v) by a key bit,
  3. in-block pixel shuffle   - the B*B pixels inside every block are shuffled
                                with one key-derived permutation,
  4. block shuffle            - the blocks themselves are permuted.

Every step is a bijection, so decryption is exact. Steps 1-3 are linear or
affine in the pixel values and step 4 only moves whole patches, so a ViT whose
patch size equals the block size can learn the fixed scrambling (the patch
embedding absorbs 1-3, the learnable positional embedding absorbs 4).

What it is and is not
---------------------
It gives *visual* protection: the picture is unrecognisable and the key is
needed to restore it. It is NOT cryptographically strong: there is no
diffusion (changing one pixel changes one pixel of the output), pixel-value
histograms are preserved (up to the inversion), and the same scrambling is used
for every image under one key. It trades security for classifiability and is
meant for the research question "how much accuracy survives encryption".

All streams come from the hyperchaotic system seeded with SHA-256(key + tag).
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import NamedTuple, Optional, Tuple

import numpy as np

from cryptovit.encryption.chaotic_map import HyperchaosSystem

logger = logging.getLogger(__name__)

_SCHEME_PATTERN = re.compile(r"^block(\d+)$")


def parse_block_scheme(scheme: str) -> Optional[int]:
    """Block size for a scheme name like ``"block2"``; ``None`` if not a block scheme."""
    match = _SCHEME_PATTERN.match(scheme)
    if not match:
        return None
    size = int(match.group(1))
    if size < 1:
        raise ValueError(f"Block size must be >= 1, got {size} in scheme '{scheme}'.")
    return size


class BlockStreams(NamedTuple):
    """Key-derived parameters for one (key, image size, block size)."""
    block_perm: np.ndarray     # (n_blocks,) permutation of block positions
    pixel_perm: np.ndarray     # (B*B,) permutation of pixels inside a block
    channel_perm: np.ndarray   # (C,) permutation of colour channels
    flips: np.ndarray          # (n_blocks,) bool, True -> invert that block


@lru_cache(maxsize=8)
def derive_block_streams(
    key: bytes, height: int, width: int, channels: int, block: int,
    transient_steps: int = 1000,
) -> BlockStreams:
    """Derive the scrambling parameters. Cached; arrays are read-only."""
    if height % block or width % block:
        raise ValueError(
            f"Image size {height}x{width} is not divisible by block size {block}."
        )
    n_blocks = (height // block) * (width // block)
    seed = hashlib.sha256(key + b"block").digest()
    system = HyperchaosSystem(seed, transient_steps=transient_steps)
    # Fixed draw order: block permutation, pixel permutation, channels, flips.
    block_perm = system.generate_permutation(n_blocks)
    pixel_perm = system.generate_permutation(block * block)
    channel_perm = system.generate_permutation(channels)
    flips = system.generate_uniform(n_blocks) >= 0.5
    streams = BlockStreams(block_perm, pixel_perm, channel_perm, flips)
    for arr in streams:
        arr.setflags(write=False)
    return streams


def _to_blocks(image: np.ndarray, block: int) -> np.ndarray:
    """(H, W, C) -> (n_blocks, B, B, C) in raster block order."""
    h, w, c = image.shape
    blocks = image.reshape(h // block, block, w // block, block, c)
    return blocks.transpose(0, 2, 1, 3, 4).reshape(-1, block, block, c)


def _from_blocks(blocks: np.ndarray, h: int, w: int, block: int) -> np.ndarray:
    c = blocks.shape[-1]
    grid = blocks.reshape(h // block, w // block, block, block, c)
    return grid.transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def _check_image(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    if image.dtype != np.uint8:
        raise TypeError(f"Image must be uint8; got {image.dtype}.")
    if image.ndim == 2:
        return image[:, :, np.newaxis], True
    if image.ndim != 3:
        raise ValueError(f"Image must be (H, W) or (H, W, C); got shape {image.shape}.")
    return image, False


def block_encrypt(
    image: np.ndarray, key: bytes, block: int, transient_steps: int = 1000
) -> np.ndarray:
    """Scramble ``image`` (uint8, HxW or HxWxC) with the block scheme."""
    img, was_gray = _check_image(image)
    h, w, c = img.shape
    s = derive_block_streams(key, h, w, c, block, transient_steps)

    out = img[:, :, s.channel_perm]                              # 1. channels
    blocks = _to_blocks(out, block)                              # (nb, B, B, C)
    blocks = np.where(s.flips[:, None, None, None], 255 - blocks, blocks).astype(np.uint8)  # 2.
    flat = blocks.reshape(len(blocks), block * block, c)         # 3. pixels in block
    flat = flat[:, s.pixel_perm, :]
    blocks = flat.reshape(-1, block, block, c)[s.block_perm]     # 4. block positions
    result = _from_blocks(blocks, h, w, block)
    return result[:, :, 0] if was_gray else result


def block_decrypt(
    encrypted: np.ndarray, key: bytes, block: int, transient_steps: int = 1000
) -> np.ndarray:
    """Exact inverse of ``block_encrypt`` for the same key and block size."""
    img, was_gray = _check_image(encrypted)
    h, w, c = img.shape
    s = derive_block_streams(key, h, w, c, block, transient_steps)

    blocks = _to_blocks(img, block)
    undone = np.empty_like(blocks)
    undone[s.block_perm] = blocks                                # undo 4.
    flat = undone.reshape(len(undone), block * block, c)
    unshuffled = np.empty_like(flat)
    unshuffled[:, s.pixel_perm, :] = flat                        # undo 3.
    blocks = unshuffled.reshape(-1, block, block, c)
    blocks = np.where(s.flips[:, None, None, None], 255 - blocks, blocks).astype(np.uint8)  # undo 2.
    out = _from_blocks(blocks, h, w, block)
    inverse_channels = np.empty_like(s.channel_perm)
    inverse_channels[s.channel_perm] = np.arange(c)
    out = out[:, :, inverse_channels]                            # undo 1.
    return out[:, :, 0] if was_gray else out
