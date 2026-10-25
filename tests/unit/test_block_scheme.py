"""Tests for the block-wise learnable encryption scheme (block<B>)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cryptovit.encryption.block_scheme import (
    block_decrypt,
    block_encrypt,
    derive_block_streams,
    parse_block_scheme,
)

KEY = b"block-test-key"
STEPS = 50


def _img(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize(
    "shape, block",
    [
        ((28, 28, 3), 2),
        ((28, 28, 3), 4),
        ((28, 28, 3), 7),
        ((28, 28, 3), 14),
        ((8, 8), 2),
        ((4, 6, 3), 2),
        ((6, 6, 3), 1),
        ((2, 2, 3), 2),
    ],
)
def test_round_trip_is_exact(shape, block):
    img = _img(shape, seed=sum(shape))
    enc = block_encrypt(img, KEY, block, STEPS)
    assert enc.shape == img.shape and enc.dtype == np.uint8
    assert np.array_equal(block_decrypt(enc, KEY, block, STEPS), img)


def test_scrambles_deterministically_and_depends_on_key():
    img = _img((28, 28, 3))
    a = block_encrypt(img, KEY, 2, STEPS)
    assert not np.array_equal(a, img)
    assert np.array_equal(a, block_encrypt(img, KEY, 2, STEPS))
    assert not np.array_equal(a, block_encrypt(img, b"other-key", 2, STEPS))


def test_wrong_key_does_not_decrypt():
    img = _img((28, 28, 3))
    enc = block_encrypt(img, KEY, 2, STEPS)
    assert not np.array_equal(block_decrypt(enc, b"wrong", 2, STEPS), img)


def test_inputs_are_not_modified():
    img = _img((8, 8, 3))
    before = img.copy()
    enc = block_encrypt(img, KEY, 2, STEPS)
    assert np.array_equal(img, before)
    enc_before = enc.copy()
    block_decrypt(enc, KEY, 2, STEPS)
    assert np.array_equal(enc, enc_before)


def test_blocks_are_only_moved_and_inverted_not_mixed():
    """Every output block holds exactly the values of one input block, or of its
    negative; nothing is mixed between blocks (the property that keeps patches
    learnable)."""
    block = 2
    img = _img((8, 8, 3), seed=4)
    enc = block_encrypt(img, KEY, block, STEPS)

    def blocks(a):
        h, w, c = a.shape
        return [
            tuple(sorted(a[y : y + block, x : x + block].ravel().tolist()))
            for y in range(0, h, block)
            for x in range(0, w, block)
        ]

    allowed = Counter(blocks(img))
    negatives = Counter(blocks(255 - img))
    for ob in blocks(enc):
        assert ob in allowed or ob in negatives


def test_no_diffusion_is_a_documented_limitation():
    """One changed input pixel changes exactly one output pixel."""
    img = _img((28, 28, 3), seed=7)
    changed = img.copy()
    changed[5, 9, 1] ^= 1
    diff = block_encrypt(img, KEY, 2, STEPS) != block_encrypt(changed, KEY, 2, STEPS)
    assert diff.sum() == 1


def test_streams_are_valid_cached_and_read_only():
    s = derive_block_streams(KEY, 28, 28, 3, 2, STEPS)
    assert sorted(s.block_perm.tolist()) == list(range(196))
    assert sorted(s.pixel_perm.tolist()) == [0, 1, 2, 3]
    assert sorted(s.channel_perm.tolist()) == [0, 1, 2]
    assert s.flips.dtype == bool and len(s.flips) == 196
    assert derive_block_streams(KEY, 28, 28, 3, 2, STEPS) is s
    with pytest.raises(ValueError):
        s.block_perm[0] = 0


def test_flips_are_roughly_balanced_and_blocks_move():
    s = derive_block_streams(KEY, 28, 28, 3, 2, STEPS)
    assert 0.3 < s.flips.mean() < 0.7
    assert (s.block_perm == np.arange(196)).sum() <= 8


def test_block_grid_matches_vit_patch_grid():
    assert (28 // 2) ** 2 == (224 // 16) ** 2 == 196


def test_parse_block_scheme():
    assert parse_block_scheme("block2") == 2
    assert parse_block_scheme("block14") == 14
    assert parse_block_scheme("v1") is None
    assert parse_block_scheme("v2") is None
    assert parse_block_scheme("blockx") is None
    with pytest.raises(ValueError):
        parse_block_scheme("block0")


def test_invalid_inputs():
    with pytest.raises(ValueError):
        block_encrypt(_img((28, 28, 3)), KEY, 5, STEPS)
    with pytest.raises(TypeError):
        block_encrypt(_img((8, 8, 3)).astype(np.float32), KEY, 2, STEPS)
    with pytest.raises(ValueError):
        block_encrypt(_img((8, 8, 3, 1)), KEY, 2, STEPS)
