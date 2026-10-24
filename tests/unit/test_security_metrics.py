"""
tests/unit/test_security_metrics.py
===================================
Unit tests for image encryption security metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptovit.evaluation.security_metrics import (
    adjacent_pixel_correlation,
    compute_all_security_metrics,
    compute_entropy,
    histogram_analysis,
    key_sensitivity,
    measure_encryption_time,
    npcr,
    uaci,
)


class TestSecurityMetrics:
    def test_compute_entropy_uniform_random(self):
        rng = np.random.default_rng(42)
        # Random uint8 image should have high entropy (close to 8.0)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ent = compute_entropy(img)
        assert "overall" in ent
        assert "channel_0" in ent
        assert ent["overall"] > 7.0

    def test_compute_entropy_flat_image(self):
        # Image with single value has zero entropy
        flat = np.zeros((32, 32, 3), dtype=np.uint8)
        ent = compute_entropy(flat)
        assert ent["overall"] == 0.0

    def test_histogram_analysis(self):
        rng = np.random.default_rng(10)
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        hist = histogram_analysis(img)
        assert "chi_squared_overall" in hist
        assert hist["chi_squared_overall"] >= 0.0

    def test_adjacent_pixel_correlation(self):
        rng = np.random.default_rng(20)
        img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        corr = adjacent_pixel_correlation(img, n_samples=500, seed=42)
        assert -1.0 <= corr["horizontal"] <= 1.0
        assert -1.0 <= corr["vertical"] <= 1.0
        assert -1.0 <= corr["diagonal"] <= 1.0

    def test_npcr_identical_images(self):
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        assert npcr(img, img) == 0.0

    def test_npcr_completely_different_images(self):
        img1 = np.zeros((16, 16, 3), dtype=np.uint8)
        img2 = np.ones((16, 16, 3), dtype=np.uint8)
        assert npcr(img1, img2) == 100.0

    def test_uaci_identical_images(self):
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        assert uaci(img, img) == 0.0

    def test_uaci_extremes(self):
        img1 = np.zeros((16, 16, 3), dtype=np.uint8)
        img2 = np.full((16, 16, 3), 255, dtype=np.uint8)
        assert uaci(img1, img2) == 100.0

    def test_npcr_uaci_shape_mismatch_raises(self):
        img1 = np.zeros((16, 16, 3), dtype=np.uint8)
        img2 = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            npcr(img1, img2)
        with pytest.raises(ValueError):
            uaci(img1, img2)

    def test_key_sensitivity(self):
        def fake_encrypt(img, key):
            rng = np.random.default_rng(hash(key) % (2**32))
            return rng.integers(0, 256, img.shape, dtype=np.uint8)

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        res = key_sensitivity(img, "test_secret_key", fake_encrypt)
        assert "npcr" in res
        assert "uaci" in res
        assert 0.0 <= res["npcr"] <= 100.0
        assert 0.0 <= res["uaci"] <= 100.0

    def test_measure_encryption_time(self):
        def fake_enc(img, key):
            return img.copy(), {"key_hash": "dummy"}

        def fake_dec(enc, key, meta):
            return enc.copy()

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        times = measure_encryption_time(img, fake_enc, fake_dec, "key", n_trials=2)
        assert "encrypt_ms" in times
        assert "decrypt_ms" in times
        assert times["encrypt_ms"] >= 0.0

    def test_compute_all_security_metrics(self):
        rng = np.random.default_rng(77)
        plain = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        enc = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        all_metrics = compute_all_security_metrics(plain, enc)
        assert "entropy" in all_metrics
        assert "histogram" in all_metrics
        assert "correlation_encrypted" in all_metrics
        assert "npcr_plain_vs_enc" in all_metrics
        assert "uaci_plain_vs_enc" in all_metrics
