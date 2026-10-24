"""Tests for the demo-pack helpers: image selection and picture building."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_demo_pack_under_test", ROOT / "scripts" / "make_demo_pack.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rec(index, label, plain_pred, plain_conf, enc_pred, enc_conf):
    return dict(index=index, label=label, plain_pred=plain_pred, plain_conf=plain_conf,
                enc_pred=enc_pred, enc_conf=enc_conf)


def test_picks_the_image_both_models_get_right_with_best_lowest_confidence():
    mod = _load()
    records = [
        rec(1, "melanoma", "melanoma", 0.9, "melanoma", 0.4),     # min 0.4
        rec(2, "melanoma", "melanoma", 0.7, "melanoma", 0.6),     # min 0.6  <- best
        rec(3, "melanoma", "melanoma", 0.95, "nevus", 0.9),       # encrypted wrong
        rec(4, "melanoma", "nevus", 0.9, "melanoma", 0.9),        # plaintext wrong
    ]
    out = mod.choose_images(records, ["melanoma"])
    assert out["safe"]["melanoma"]["index"] == 2


def test_minimum_confidence_filters_and_missing_classes_are_left_out():
    mod = _load()
    records = [rec(1, "melanoma", "melanoma", 0.9, "melanoma", 0.2)]
    assert mod.choose_images(records, ["melanoma"], min_confidence=0.3)["safe"] == {}
    assert "melanoma" in mod.choose_images(records, ["melanoma"], min_confidence=0.1)["safe"]
    assert "nevus" not in mod.choose_images(records, ["melanoma", "nevus"])["safe"]


def test_failure_is_plaintext_right_encrypted_wrong_most_confident_mistake():
    mod = _load()
    records = [
        rec(1, "melanoma", "melanoma", 0.9, "nevus", 0.5),
        rec(2, "melanoma", "melanoma", 0.8, "nevus", 0.8),        # <- most visible mistake
        rec(3, "melanoma", "nevus", 0.9, "nevus", 0.99),          # plaintext also wrong
        rec(4, "other", "other", 0.9, "nevus", 0.99),             # class not requested
    ]
    assert mod.choose_images(records, ["melanoma"])["failure"]["index"] == 2
    assert mod.choose_images([rec(1, "melanoma", "melanoma", 0.9, "melanoma", 0.9)],
                             ["melanoma"])["failure"] is None


def test_demo_filenames_sort_safe_images_first_and_failure_last():
    mod = _load()
    names = [
        mod.demo_filename("safe_melanoma_test1", "safe", 1),
        mod.demo_filename("safe_nevi_test2", "safe", 2),
        mod.demo_filename("safe_bcc_test3", "safe", 3),
        mod.demo_filename("failure_bcc_test4", "failure", 4),
    ]
    assert names[-1].startswith("9_failure")
    assert sorted(names) == names                     # the dropdown sorts alphabetically


def test_slug():
    assert _load().slug("Basal cell carcinoma!") == "basal_cell_carcinoma"
    assert _load().slug("melanocytic nevi") == "melanocytic_nevi"


def test_enlarge_keeps_pixels_crisp():
    mod = _load()
    img = Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8))
    big = mod.enlarge(img, 8)
    assert big.size == (16, 8)
    arr = np.array(big)
    assert (arr[:, :8] == [255, 0, 0]).all() and (arr[:, 8:] == [0, 255, 0]).all()


def test_triptych_layout_and_validation():
    mod = _load()
    panels = [Image.new("RGB", (224, 224), c) for c in ("red", "green", "blue")]
    strip = mod.triptych(panels, ["a", "b", "c"], caption="caption")
    assert strip.width == 16 + 3 * (224 + 16)
    assert strip.height > 224 + 28
    with pytest.raises(ValueError):
        mod.triptych(panels, ["only one"])


def test_selection_markdown_is_plain_ascii(tmp_path):
    mod = _load()
    rows = [dict(rec(5, "melanoma", "melanoma", 0.8, "melanoma", 0.7),
                 role="safe", file_stem="safe_melanoma_test5")]
    mod.write_selection_md(rows, tmp_path, "D", 2, 90)
    text = (tmp_path / "SELECTION.md").read_text(encoding="utf-8")
    assert "safe_melanoma_test5" in text and text.isascii()
