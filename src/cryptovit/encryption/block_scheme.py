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
    match = _SCHEME_PATTERN.match(scheme)
    if not match:
        return None
    size = int(match.group(1))
    if size < 1:
        raise ValueError(f"Block size must be >= 1, got {size} in scheme '{scheme}'.")
    return size


class BlockStreams(NamedTuple):
    block_perm: np.ndarray
    pixel_perm: np.ndarray
    channel_perm: np.ndarray
    flips: np.ndarray


@lru_cache(maxsize=8)
def derive_block_streams(
    key: bytes,
    height: int,
    width: int,
    channels: int,
    block: int,
    transient_steps: int = 1000,
) -> BlockStreams:
    if height % block or width % block:
        raise ValueError(f"Image size {height}x{width} is not divisible by block size {block}.")
    n_blocks = (height // block) * (width // block)
    seed = hashlib.sha256(key + b"block").digest()
    system = HyperchaosSystem(seed, transient_steps=transient_steps)
    block_perm = system.generate_permutation(n_blocks)
    pixel_perm = system.generate_permutation(block * block)
    channel_perm = system.generate_permutation(channels)
    flips = system.generate_uniform(n_blocks) >= 0.5
    streams = BlockStreams(block_perm, pixel_perm, channel_perm, flips)
    for arr in streams:
        arr.setflags(write=False)
    return streams


def _to_blocks(image: np.ndarray, block: int) -> np.ndarray:
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
    img, was_gray = _check_image(image)
    h, w, c = img.shape
    s = derive_block_streams(key, h, w, c, block, transient_steps)

    out = img[:, :, s.channel_perm]
    blocks = _to_blocks(out, block)
    blocks = np.where(s.flips[:, None, None, None], 255 - blocks, blocks).astype(np.uint8)
    flat = blocks.reshape(len(blocks), block * block, c)
    flat = flat[:, s.pixel_perm, :]
    blocks = flat.reshape(-1, block, block, c)[s.block_perm]
    result = _from_blocks(blocks, h, w, block)
    return result[:, :, 0] if was_gray else result


def block_decrypt(
    encrypted: np.ndarray, key: bytes, block: int, transient_steps: int = 1000
) -> np.ndarray:
    img, was_gray = _check_image(encrypted)
    h, w, c = img.shape
    s = derive_block_streams(key, h, w, c, block, transient_steps)

    blocks = _to_blocks(img, block)
    undone = np.empty_like(blocks)
    undone[s.block_perm] = blocks
    flat = undone.reshape(len(undone), block * block, c)
    unshuffled = np.empty_like(flat)
    unshuffled[:, s.pixel_perm, :] = flat
    blocks = unshuffled.reshape(-1, block, block, c)
    blocks = np.where(s.flips[:, None, None, None], 255 - blocks, blocks).astype(np.uint8)
    out = _from_blocks(blocks, h, w, block)
    inverse_channels = np.empty_like(s.channel_perm)
    inverse_channels[s.channel_perm] = np.arange(c)
    out = out[:, :, inverse_channels]
    return out[:, :, 0] if was_gray else out
