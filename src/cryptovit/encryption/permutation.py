from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def permute_image(
    image: np.ndarray,
    perm: np.ndarray,
) -> np.ndarray:
    _check_uint8(image)
    H, W = image.shape[:2]
    C = 1 if image.ndim == 2 else image.shape[2]
    perms = _channel_perms(perm, H * W, C)

    if image.ndim == 2:
        return image.reshape(-1)[perms[0]].reshape(H, W)

    result = np.empty_like(image)
    for c in range(C):
        flat = image[:, :, c].reshape(-1)
        result[:, :, c] = flat[perms[c]].reshape(H, W)
    return result


def inverse_permute_image(
    scrambled: np.ndarray,
    perm: np.ndarray,
) -> np.ndarray:
    _check_uint8(scrambled)
    H, W = scrambled.shape[:2]
    n = H * W
    C = 1 if scrambled.ndim == 2 else scrambled.shape[2]
    perms = _channel_perms(perm, n, C)

    def invert(p: np.ndarray) -> np.ndarray:
        inv = np.empty(n, dtype=p.dtype)
        inv[p] = np.arange(n, dtype=p.dtype)
        return inv

    if scrambled.ndim == 2:
        return scrambled.reshape(-1)[invert(perms[0])].reshape(H, W)

    result = np.empty_like(scrambled)
    for c in range(C):
        flat = scrambled[:, :, c].reshape(-1)
        result[:, :, c] = flat[invert(perms[c])].reshape(H, W)
    return result


def permute_blocks(
    image: np.ndarray,
    block_size: int,
    perm: np.ndarray,
) -> np.ndarray:
    _check_uint8(image)
    H, W, C = image.shape if image.ndim == 3 else (*image.shape, 1)
    bH, bW = H // block_size, W // block_size
    n_blocks = bH * bW
    if len(perm) != n_blocks:
        raise ValueError(f"Permutation length {len(perm)} != number of blocks {n_blocks}.")

    is_2d = image.ndim == 2
    if is_2d:
        image = image[:, :, np.newaxis]

    blocks = image.reshape(bH, block_size, bW, block_size, C)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(n_blocks, block_size, block_size, C)

    blocks = blocks[perm]

    blocks = blocks.reshape(bH, bW, block_size, block_size, C)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(H, W, C)

    return blocks[:, :, 0] if is_2d else blocks


def inverse_permute_blocks(
    scrambled: np.ndarray,
    block_size: int,
    perm: np.ndarray,
) -> np.ndarray:
    n_blocks = len(perm)
    inv_perm = np.empty(n_blocks, dtype=perm.dtype)
    inv_perm[perm] = np.arange(n_blocks, dtype=perm.dtype)
    return permute_blocks(scrambled, block_size, inv_perm)


def _channel_perms(perm: np.ndarray, n: int, channels: int) -> list:
    perm = np.asarray(perm)
    if perm.ndim == 1:
        rows = [perm] * channels
    elif perm.ndim == 2:
        if perm.shape[0] != channels:
            raise ValueError(
                f"Got {perm.shape[0]} per-channel permutations for an image "
                f"with {channels} channel(s)."
            )
        rows = list(perm)
    else:
        raise ValueError(f"perm must be 1-D or 2-D, got {perm.ndim}-D.")
    for row in rows:
        if len(row) != n:
            raise ValueError(f"Permutation length {len(row)} does not match image size {n}.")
    return rows


def _check_uint8(image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise TypeError(
            f"Image must be uint8, got {image.dtype}. "
            "Convert with image.astype(np.uint8) before calling."
        )
