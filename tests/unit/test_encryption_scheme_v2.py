"""Tests for encryption scheme "v2" (independent per-channel permutations and
keystreams, uniform chaos-driven shuffle) and its coexistence with "v1"."""

from __future__ import annotations

import numpy as np
import pytest

from cryptovit.encryption import decrypt, encrypt
from cryptovit.encryption.encrypt import SCHEMES, verify_roundtrip
from cryptovit.encryption.key_generator import KeyGenerator
from cryptovit.encryption.permutation import inverse_permute_image, permute_image

STEPS = 100


def _img(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_default_scheme_is_v1_and_unchanged():
    img = _img((8, 8, 3))
    default, meta = encrypt(img, "k", STEPS)
    explicit, _ = encrypt(img, "k", STEPS, scheme="v1")
    assert np.array_equal(default, explicit)
    assert meta["scheme"] == "v1"


@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 3, 3), (5, 7), (1, 8, 3), (32, 32, 3)])
def test_v2_round_trip(shape):
    img = _img(shape, seed=sum(shape))
    enc, meta = encrypt(img, "key", STEPS, scheme="v2")
    assert meta["scheme"] == "v2"
    assert np.array_equal(decrypt(enc, "key", meta, STEPS), img)
    assert verify_roundtrip(img, "key", STEPS, scheme="v2")


def test_v2_differs_from_v1():
    img = _img((16, 16, 3))
    v1, _ = encrypt(img, "k", STEPS, scheme="v1")
    v2, _ = encrypt(img, "k", STEPS, scheme="v2")
    assert not np.array_equal(v1, v2)


def test_v2_is_deterministic_and_key_dependent():
    img = _img((16, 16, 3))
    a, _ = encrypt(img, "k", STEPS, scheme="v2")
    b, _ = encrypt(img, "k", STEPS, scheme="v2")
    c, _ = encrypt(img, "other", STEPS, scheme="v2")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_v2_wrong_key_does_not_decrypt():
    img = _img((16, 16, 3))
    enc, meta = encrypt(img, "right", STEPS, scheme="v2")
    assert not np.array_equal(decrypt(enc, "wrong", meta, STEPS), img)


def test_decrypt_reads_scheme_from_meta_and_explicit_scheme_wins():
    img = _img((8, 8, 3))
    enc, meta = encrypt(img, "k", STEPS, scheme="v2")
    assert np.array_equal(decrypt(enc, "k", meta, STEPS), img)
    assert np.array_equal(decrypt(enc, "k", None, STEPS, scheme="v2"), img)
    assert not np.array_equal(decrypt(enc, "k", None, STEPS, scheme="v1"), img)


def test_decrypt_without_meta_defaults_to_v1():
    img = _img((8, 8, 3))
    enc, _ = encrypt(img, "k", STEPS)
    assert np.array_equal(decrypt(enc, "k", None, STEPS), img)


def test_unknown_scheme_is_rejected():
    img = _img((4, 4, 3))
    with pytest.raises(ValueError):
        encrypt(img, "k", STEPS, scheme="v3")
    enc, _ = encrypt(img, "k", STEPS)
    with pytest.raises(ValueError):
        decrypt(enc, "k", None, STEPS, scheme="nope")
    assert SCHEMES == ("v1", "v2")


def test_identical_channels_leak_in_v1_but_not_v2():
    gray = _img((16, 16), seed=5)
    triple = np.stack([gray, gray, gray], axis=-1)
    v1, _ = encrypt(triple, "k", STEPS, scheme="v1")
    v2, _ = encrypt(triple, "k", STEPS, scheme="v2")
    assert np.array_equal(v1[..., 0], v1[..., 1])
    assert not np.array_equal(v2[..., 0], v2[..., 1])
    assert not np.array_equal(v2[..., 1], v2[..., 2])


class TestUniformPermutation:
    def test_channel_permutations_are_valid_and_distinct(self):
        perms = KeyGenerator("k", transient_steps=STEPS).channel_permutations(500, 3)
        assert perms.shape == (3, 500)
        for p in perms:
            assert sorted(p.tolist()) == list(range(500))
        assert not np.array_equal(perms[0], perms[1])

    def test_mean_adjacent_jump_is_close_to_uniform_random(self):
        n = 1024
        p = KeyGenerator("k", transient_steps=STEPS).channel_permutations(n, 1)[0]
        mean_jump = np.mean(np.abs(np.diff(p)))
        assert mean_jump == pytest.approx(n / 3, rel=0.10)

    def test_few_fixed_points(self):
        n = 1024
        p = KeyGenerator("k", transient_steps=STEPS).channel_permutations(n, 1)[0]
        assert (p == np.arange(n)).sum() <= 8

    def test_deterministic_per_key(self):
        a = KeyGenerator("k", transient_steps=STEPS).channel_permutations(64, 3)
        b = KeyGenerator("k", transient_steps=STEPS).channel_permutations(64, 3)
        c = KeyGenerator("z", transient_steps=STEPS).channel_permutations(64, 3)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_channel_keystreams_are_independent_columns(self):
        ks = KeyGenerator("k", transient_steps=STEPS).channel_keystreams(256, 3)
        assert ks.shape == (256, 3) and ks.dtype == np.uint8
        assert not np.array_equal(ks[:, 0], ks[:, 1])


class TestPerChannelPermutationFunctions:
    def test_2d_perm_round_trip(self):
        img = _img((6, 5, 3))
        perms = np.stack([np.random.default_rng(i).permutation(30) for i in range(3)])
        scrambled = permute_image(img, perms)
        assert not np.array_equal(scrambled, img)
        assert np.array_equal(inverse_permute_image(scrambled, perms), img)

    def test_1d_perm_still_applies_to_every_channel(self):
        img = _img((4, 4, 3))
        perm = np.random.default_rng(0).permutation(16)
        stacked = np.stack([perm, perm, perm])
        assert np.array_equal(permute_image(img, perm), permute_image(img, stacked))

    def test_wrong_number_of_channel_perms_is_rejected(self):
        img = _img((4, 4, 3))
        perms = np.stack([np.arange(16), np.arange(16)])
        with pytest.raises(ValueError):
            permute_image(img, perms)

    def test_wrong_length_is_rejected(self):
        img = _img((4, 4, 3))
        with pytest.raises(ValueError):
            permute_image(img, np.arange(15))
