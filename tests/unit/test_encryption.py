from __future__ import annotations

import numpy as np
import pytest


class TestHyperchaosSystem:
    def _sys(self, key="testkey"):
        from cryptovit.encryption.chaotic_map import HyperchaosSystem

        return HyperchaosSystem(key, transient_steps=100)

    def test_generate_length(self):
        sys = self._sys()
        seq = sys.generate(500)
        assert len(seq) == 500

    def test_generate_bytes_dtype(self):
        sys = self._sys()
        b = sys.generate_bytes(256)
        assert b.dtype == np.uint8
        assert len(b) == 256

    def test_generate_bytes_range(self):
        sys = self._sys()
        b = sys.generate_bytes(1000)
        assert b.min() >= 0
        assert b.max() <= 255

    def test_generate_indices_is_permutation(self):
        sys = self._sys()
        n = 100
        idx = sys.generate_indices(n)
        assert sorted(idx.tolist()) == list(range(n)), "Not a valid permutation"

    def test_different_keys_produce_different_output(self):
        from cryptovit.encryption.chaotic_map import HyperchaosSystem

        s1 = HyperchaosSystem("key_one", transient_steps=100)
        s2 = HyperchaosSystem("key_two", transient_steps=100)
        seq1 = s1.generate_bytes(100)
        seq2 = s2.generate_bytes(100)
        assert not np.array_equal(seq1, seq2), "Different keys should give different output"

    def test_same_key_produces_same_output(self):
        from cryptovit.encryption.chaotic_map import HyperchaosSystem

        s1 = HyperchaosSystem("same_key", transient_steps=100)
        s2 = HyperchaosSystem("same_key", transient_steps=100)
        assert np.array_equal(s1.generate_bytes(200), s2.generate_bytes(200))

    def test_sensitivity_test_high_divergence(self):
        sys = self._sys("testkey")
        divergence = sys.sensitivity_test(n=500, delta=0.1)
        assert divergence > 0.5, f"Sensitivity too low: {divergence}"


class TestKeyGenerator:
    def _kg(self, key="secretkey"):
        from cryptovit.encryption.key_generator import KeyGenerator

        return KeyGenerator(key, transient_steps=100)

    def test_pixel_permutation_is_valid(self):
        kg = self._kg()
        perm = kg.pixel_permutation(1000)
        assert sorted(perm.tolist()) == list(range(1000))

    def test_inverse_permutation(self):
        kg = self._kg()
        n = 500
        perm = kg.pixel_permutation(n)
        inv = kg.inverse_permutation(perm)
        idx = np.arange(n)
        assert np.array_equal(idx[perm][inv], idx)

    def test_diffusion_keystream_length_and_dtype(self):
        kg = self._kg()
        ks = kg.diffusion_keystream(1024)
        assert len(ks) == 1024
        assert ks.dtype == np.uint8

    def test_dna_rule_sequence_range(self):
        kg = self._kg()
        rules = kg.dna_rule_sequence(500)
        assert rules.min() >= 0
        assert rules.max() <= 7


class TestPermutation:
    def _make_image(self, H=8, W=8, C=3):
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (H, W, C), dtype=np.uint8)

    def test_permute_changes_image(self):
        from cryptovit.encryption.permutation import permute_image

        img = self._make_image()
        perm = np.random.permutation(64)
        perm = perm.astype(np.int64)
        permuted = permute_image(img, perm)
        assert not np.array_equal(img, permuted) or np.array_equal(perm, np.arange(64))

    def test_roundtrip_pixel_permutation(self):
        from cryptovit.encryption.permutation import inverse_permute_image, permute_image

        img = self._make_image()
        perm = np.random.default_rng(1).permutation(64).astype(np.int64)
        scrambled = permute_image(img, perm)
        restored = inverse_permute_image(scrambled, perm)
        assert np.array_equal(img, restored), "Permutation round-trip failed"

    def test_permute_grayscale(self):
        from cryptovit.encryption.permutation import inverse_permute_image, permute_image

        img = np.random.default_rng(2).integers(0, 256, (8, 8), dtype=np.uint8)
        perm = np.random.default_rng(2).permutation(64).astype(np.int64)
        assert np.array_equal(img, inverse_permute_image(permute_image(img, perm), perm))

    def test_wrong_dtype_raises(self):
        from cryptovit.encryption.permutation import permute_image

        img = np.ones((4, 4, 3), dtype=np.float32)
        perm = np.arange(16)
        with pytest.raises(TypeError):
            permute_image(img, perm)

    def test_roundtrip_block_permutation(self):
        from cryptovit.encryption.permutation import inverse_permute_blocks, permute_blocks

        img = self._make_image(H=16, W=16, C=3)
        perm = np.random.default_rng(3).permutation(16).astype(np.int64)
        scrambled = permute_blocks(img, block_size=4, perm=perm)
        restored = inverse_permute_blocks(scrambled, block_size=4, perm=perm)
        assert np.array_equal(img, restored)

    def test_roundtrip_block_permutation_2d(self):
        from cryptovit.encryption.permutation import inverse_permute_blocks, permute_blocks

        img = np.random.default_rng(4).integers(0, 256, (16, 16), dtype=np.uint8)
        perm = np.random.default_rng(4).permutation(16).astype(np.int64)
        scrambled = permute_blocks(img, block_size=4, perm=perm)
        restored = inverse_permute_blocks(scrambled, block_size=4, perm=perm)
        assert np.array_equal(img, restored)


class TestDNA:
    def test_encode_decode_all_rules(self):
        from cryptovit.encryption.dna import decode, encode

        pixels = np.arange(256, dtype=np.uint8)
        for rule in range(8):
            encoded = encode(pixels, rule)
            decoded = decode(encoded, rule)
            assert np.array_equal(pixels, decoded), f"Round-trip failed for rule {rule}"

    def test_encode_shape(self):
        from cryptovit.encryption.dna import encode

        pixels = np.zeros((10, 10), dtype=np.uint8)
        enc = encode(pixels, rule=0)
        assert enc.shape == (10, 10, 4)

    def test_xor_self_inverse(self):
        from cryptovit.encryption.dna import dna_xor

        for b in range(4):
            arr = np.array([[b]], dtype=np.uint8)
            result = dna_xor(arr, arr)
            assert result[0, 0] == 0, f"XOR of base {b} with itself should be A(0)"

    def test_dna_values_in_range(self):
        from cryptovit.encryption.dna import encode

        pixels = np.random.randint(0, 256, (100,), dtype=np.uint8)
        enc = encode(pixels, rule=3)
        assert enc.min() >= 0
        assert enc.max() <= 3


class TestDiffusion:
    def _make_data(self, n=64):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        ks = rng.integers(0, 256, (n,), dtype=np.uint8)
        rules = (rng.random(n) * 8).astype(np.uint8)
        return image, ks, rules

    def test_diffusion_roundtrip(self):
        from cryptovit.encryption.diffusion import diffuse, inverse_diffuse

        image, ks, rules = self._make_data()
        diffused = diffuse(image, ks, rules)
        restored = inverse_diffuse(diffused, ks, rules)
        assert np.array_equal(image, restored), "Diffusion round-trip failed"

    def test_diffused_differs_from_original(self):
        from cryptovit.encryption.diffusion import diffuse

        image, ks, rules = self._make_data()
        diffused = diffuse(image, ks, rules)
        assert not np.array_equal(image, diffused), "Diffusion should change the image"

    def test_diffused_dtype(self):
        from cryptovit.encryption.diffusion import diffuse

        image, ks, rules = self._make_data()
        diffused = diffuse(image, ks, rules)
        assert diffused.dtype == np.uint8


class TestEncryptDecrypt:
    def _make_image(self, H=32, W=32):
        return np.random.default_rng(99).integers(0, 256, (H, W, 3), dtype=np.uint8)

    def test_roundtrip_exact(self):
        from cryptovit.encryption import decrypt, encrypt

        img = self._make_image()
        enc, meta = encrypt(img, key="test_key_roundtrip", transient_steps=100)
        dec = decrypt(enc, key="test_key_roundtrip", meta=meta, transient_steps=100)
        assert np.array_equal(img, dec), "Encrypt->Decrypt must be pixel-perfect"

    def test_encrypted_shape_preserved(self):
        from cryptovit.encryption import encrypt

        img = self._make_image()
        enc, _ = encrypt(img, key="shape_test", transient_steps=100)
        assert enc.shape == img.shape

    def test_encrypted_dtype_uint8(self):
        from cryptovit.encryption import encrypt

        img = self._make_image()
        enc, _ = encrypt(img, key="dtype_test", transient_steps=100)
        assert enc.dtype == np.uint8

    def test_encrypted_differs_from_plaintext(self):
        from cryptovit.encryption import encrypt

        img = self._make_image()
        enc, _ = encrypt(img, key="diff_test", transient_steps=100)
        assert not np.array_equal(img, enc)

    def test_different_keys_different_encryption(self):
        from cryptovit.encryption import encrypt

        img = self._make_image()
        enc1, _ = encrypt(img, key="key_A", transient_steps=100)
        enc2, _ = encrypt(img, key="key_B", transient_steps=100)
        assert not np.array_equal(enc1, enc2)

    def test_verify_roundtrip(self):
        from cryptovit.encryption import verify_roundtrip

        img = self._make_image(H=16, W=16)
        ok = verify_roundtrip(img, key="verify_test", transient_steps=100)
        assert ok, "verify_roundtrip should return True"

    def test_wrong_key_fails_roundtrip(self):
        from cryptovit.encryption import decrypt, encrypt

        img = self._make_image(H=16, W=16)
        enc, meta = encrypt(img, key="correct_key", transient_steps=100)
        wrong_dec = decrypt(enc, key="wrong_key", meta=meta, transient_steps=100)
        assert not np.array_equal(img, wrong_dec), (
            "Decryption with wrong key should not restore original image"
        )
