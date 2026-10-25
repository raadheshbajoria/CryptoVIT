import numpy as np
import pytest

from cryptovit.encryption import decrypt, encrypt
from cryptovit.encryption.diffusion import (
    _chain_xor,
    _inverse_chain_xor,
    diffuse,
)
from cryptovit.encryption.encrypt import _key_streams
from cryptovit.encryption.key_generator import KeyGenerator
from cryptovit.encryption.permutation import permute_image


def _chain_xor_reference(data, key):
    n = len(data)
    out = np.empty(n, dtype=np.uint8)
    prev = key[n - 1] if n > 0 else np.uint8(0)
    for i in range(n):
        out[i] = data[i] ^ key[i] ^ prev
        prev = out[i]
    return out


def _inverse_chain_xor_reference(data, key):
    n = len(data)
    out = np.empty(n, dtype=np.uint8)
    prev = key[n - 1] if n > 0 else np.uint8(0)
    for i in range(n):
        out[i] = data[i] ^ key[i] ^ prev
        prev = data[i]
    return out


@pytest.mark.parametrize("n", [0, 1, 2, 17, 1024])
def test_chain_xor_matches_sequential_reference(n):
    rng = np.random.default_rng(n)
    data = rng.integers(0, 256, n, dtype=np.uint8)
    key = rng.integers(0, 256, n, dtype=np.uint8)
    assert np.array_equal(_chain_xor(data, key), _chain_xor_reference(data, key))
    assert np.array_equal(_inverse_chain_xor(data, key), _inverse_chain_xor_reference(data, key))


@pytest.mark.parametrize("n", [1, 2, 17, 1024])
def test_chain_xor_round_trip(n):
    rng = np.random.default_rng(100 + n)
    data = rng.integers(0, 256, n, dtype=np.uint8)
    key = rng.integers(0, 256, n, dtype=np.uint8)
    assert np.array_equal(_inverse_chain_xor(_chain_xor(data, key), key), data)


def test_encrypt_matches_uncached_derivation():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    key = "equivalence-key"

    kg = KeyGenerator(key, transient_steps=100)
    perm = kg.pixel_permutation(64)
    keystream = kg.diffusion_keystream(64)
    rules = kg.dna_rule_sequence(64)
    expected = diffuse(permute_image(image, perm), keystream, rules)

    actual, _ = encrypt(image, key, transient_steps=100)
    assert np.array_equal(actual, expected)


def test_key_streams_are_cached_and_read_only():
    _key_streams.cache_clear()
    a = _key_streams(b"k", 16, 50)
    b = _key_streams(b"k", 16, 50)
    assert a[0] is b[0]
    assert _key_streams.cache_info().hits == 1
    with pytest.raises(ValueError):
        a[0][0] = 99


def test_cache_does_not_leak_across_keys_or_sizes():
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    e1, m1 = encrypt(img, "key-a", transient_steps=50)
    e2, m2 = encrypt(img, "key-b", transient_steps=50)
    assert not np.array_equal(e1, e2)
    assert np.array_equal(decrypt(e1, "key-a", m1, transient_steps=50), img)
    assert np.array_equal(decrypt(e2, "key-b", m2, transient_steps=50), img)
    assert not np.array_equal(decrypt(e1, "key-b", m1, transient_steps=50), img)
