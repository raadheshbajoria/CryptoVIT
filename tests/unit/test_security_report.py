"""Tests for the security report tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "security_report_under_test", ROOT / "scripts" / "security_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_histogram_distance_extremes():
    mod = _load()
    a = np.full((4, 4, 3), 10, np.uint8)
    b = np.full((4, 4, 3), 200, np.uint8)
    assert mod.histogram_distance(a, a) == 0.0
    assert mod.histogram_distance(a, b) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert mod.histogram_distance(img, np.flip(img, axis=0)) == 0.0


def test_mixture_distance_recognises_inversion_but_not_noise():
    mod = _load()
    from cryptovit.encryption.block_scheme import block_encrypt

    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:28, 0:28]
    img = np.stack([(xx * 2) % 120, (yy * 2) % 120, (xx + yy) % 120], -1).astype(np.uint8)
    noise = rng.integers(0, 256, img.shape, dtype=np.uint8)
    blocked = block_encrypt(img, b"k", 2, 50)
    assert mod.mixture_distance(img, blocked) < 0.25
    assert mod.mixture_distance(img, blocked) < mod.mixture_distance(img, noise)
    assert mod.mixture_distance(img, (255 - img).astype(np.uint8)) == pytest.approx(
        mod.mixture_distance(img, img)
    )


def test_measure_scheme_reports_the_known_properties():
    mod = _load()
    yy, xx = np.mgrid[0:28, 0:28]
    smooth = np.stack([(xx * 3 + yy) % 256, (xx + yy * 3) % 256, (xx + yy) * 2 % 256], -1).astype(
        np.uint8
    )
    images = [smooth, np.roll(smooth, 5, axis=0)]
    block = mod.measure_scheme(images, "block2", "k", trials=3)
    pixel = mod.measure_scheme(images, "v1", "k", trials=3)

    assert block["pt_npcr"] == pytest.approx(100 / smooth.size, rel=1e-6)
    assert block["hist_tv"] < pixel["hist_tv"]
    assert block["key_npcr"] > 80 and pixel["key_npcr"] > 80


def test_markdown_lists_both_schemes_and_is_plain_ascii():
    mod = _load()
    fake = {
        s: dict(
            entropy=7.5,
            entropy_ref=7.8,
            corr_plain=0.9,
            corr_enc=0.1,
            key_npcr=99.0,
            key_uaci=33.0,
            pt_npcr=1.0,
            pt_uaci=0.1,
            hist_tv=0.2,
            hist_mix=0.1,
        )
        for s in mod.SCHEMES
    }
    text = mod.build_markdown(fake, 100)
    assert "per-pixel (v1)" in text and "block (2x2)" in text
    assert "not a security proof" in text
