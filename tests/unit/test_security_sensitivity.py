from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from cryptovit.encryption import encrypt
from cryptovit.evaluation.security_metrics import (
    compute_all_security_metrics,
    compute_entropy,
    expected_uniform_entropy,
    plaintext_sensitivity,
)


def _ideal_cipher(image, key):
    seed = int.from_bytes(hashlib.sha256(image.tobytes() + key.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).integers(0, 256, image.shape, dtype=np.uint8)


def _pixelwise_cipher(image, key):
    return image ^ np.uint8(0x5A)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)


class TestExpectedEntropy:
    def test_matches_measured_entropy_of_uniform_noise(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        measured = compute_entropy(img)["overall"]
        assert measured == pytest.approx(expected_uniform_entropy(32 * 32), abs=0.05)

    def test_small_images_cannot_reach_7_99(self):
        assert expected_uniform_entropy(32 * 32) < 7.9
        assert expected_uniform_entropy(256 * 256) > 7.99

    def test_approaches_log2_levels_for_large_n(self):
        assert expected_uniform_entropy(10**9) == pytest.approx(math.log2(256), abs=1e-6)

    def test_rejects_non_positive_sample_count(self):
        with pytest.raises(ValueError):
            expected_uniform_entropy(0)

    def test_compute_entropy_reports_reference(self, image):
        ent = compute_entropy(image)
        assert ent["expected_for_uniform_noise"] == pytest.approx(expected_uniform_entropy(32 * 32))


class TestPlaintextSensitivity:
    def test_ideal_cipher_scores_near_theoretical_values(self, image):
        res = plaintext_sensitivity(image, "k", _ideal_cipher, n_trials=5)
        assert res["npcr"] == pytest.approx(99.6, abs=0.6)
        assert res["uaci"] == pytest.approx(33.46, abs=1.0)

    def test_no_diffusion_cipher_scores_near_zero(self, image):
        res = plaintext_sensitivity(image, "k", _pixelwise_cipher, n_trials=5)
        assert res["npcr"] == pytest.approx(100 / image.size, rel=1e-6)
        assert res["npcr"] < 0.1

    def test_deterministic_for_a_seed(self, image):
        a = plaintext_sensitivity(image, "k", _ideal_cipher, n_trials=4, seed=3)
        b = plaintext_sensitivity(image, "k", _ideal_cipher, n_trials=4, seed=3)
        assert a == b

    def test_does_not_modify_input(self, image):
        before = image.copy()
        plaintext_sensitivity(image, "k", _ideal_cipher, n_trials=3)
        assert np.array_equal(image, before)

    def test_works_with_the_real_cipher(self, image):
        cipher = lambda img, k: encrypt(img, k, transient_steps=50)[0]  # noqa: E731
        res = plaintext_sensitivity(image, "k", cipher, n_trials=3)
        assert 0.0 <= res["npcr_min"] <= res["npcr"] <= res["npcr_max"] <= 100.0
        assert 0.0 <= res["uaci"] <= 100.0

    def test_rejects_non_positive_trials(self, image):
        with pytest.raises(ValueError):
            plaintext_sensitivity(image, "k", _ideal_cipher, n_trials=0)

    def test_grayscale_image(self):
        img = np.random.default_rng(0).integers(0, 256, (16, 16), dtype=np.uint8)
        res = plaintext_sensitivity(img, "k", _ideal_cipher, n_trials=2)
        assert res["n_trials"] == 2


def test_compute_all_includes_both_sensitivities_when_key_given(image):
    enc_fn = lambda img, k: (_ideal_cipher(img, k), {})  # noqa: E731
    enc, _ = enc_fn(image, "k")
    out = compute_all_security_metrics(image, enc, key="k", encrypt_fn=enc_fn)
    assert "key_sensitivity" in out
    assert "plaintext_sensitivity" in out
    assert "npcr_plain_vs_enc" in out and "uaci_plain_vs_enc" in out


def test_compute_all_without_key_has_no_sensitivity_sections(image):
    out = compute_all_security_metrics(image, _ideal_cipher(image, "k"))
    assert "plaintext_sensitivity" not in out
    assert "key_sensitivity" not in out
