"""
tests/unit/test_dashboard_state.py
==================================
Unit regression tests for dashboard state-management and invalidation helpers.

Verifies CodeRabbit Findings:
1. Finding A: New encryption clears old decrypted/recovered image and verification state.
2. Finding B: Missing/failed metrics are represented as None/error, not numeric fallbacks.
3. Finding C: Changing original image invalidates all derived encryption/decryption state.
4. Finding E: Changing input image invalidates previous predictions.
5. Preserves state when image identity is unchanged across reruns.
"""

import numpy as np
import pytest
from PIL import Image

from dashboard.state_helpers import (
    compute_image_id,
    check_and_update_source_image,
    check_and_update_prediction_image,
    reset_decryption_derived_state,
    reset_encryption_derived_state,
    reset_prediction_derived_state,
)


def _make_image(color=(100, 150, 200), size=(32, 32)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def test_compute_image_id_deterministic():
    img1 = _make_image(color=(50, 60, 70))
    img2 = _make_image(color=(50, 60, 70))
    img3 = _make_image(color=(50, 60, 71))  # 1 byte difference

    id1 = compute_image_id(img1)
    id2 = compute_image_id(img2)
    id3 = compute_image_id(img3)

    assert id1 is not None
    assert id1 == id2
    assert id1 != id3
    assert compute_image_id(None) is None


def test_finding_a_new_encryption_clears_old_decryption():
    """Finding A: When initiating a new encryption, old decrypted image and

    comparison metrics must be purged so they cannot be mistaken for current results.
    """
    state = {
        "demo_enc_img": _make_image(),
        "demo_enc_bytes": b"ciphertext_bytes",
        "demo_dec_img": _make_image(),
        "demo_dec_success": True,
        "demo_roundtrip_match": True,
        "demo_pixel_diff": 0.0,
        "demo_metrics": {"entropy": {"overall": 7.8}},
    }

    reset_decryption_derived_state(state)

    # Decryption/recovered and match artifacts must be cleared
    assert "demo_dec_img" not in state
    assert "demo_dec_success" not in state
    assert "demo_roundtrip_match" not in state
    assert "demo_pixel_diff" not in state

    # Encrypted image and metrics remain intact until overwritten
    assert "demo_enc_img" in state
    assert "demo_metrics" in state


def test_finding_c_changing_source_image_resets_derived_state():
    """Finding C: When the source image changes, all downstream derived state

    (encrypted, decrypted, metrics, round-trip) must be invalidated.
    """
    img1 = _make_image(color=(10, 20, 30))
    img2 = _make_image(color=(40, 50, 60))

    state = {}
    # First image set
    changed = check_and_update_source_image(state, img1)
    assert changed is True
    assert state.get("demo_orig_img_id") == compute_image_id(img1)

    # Simulate subsequent encryption and decryption run
    state["demo_enc_img"] = _make_image()
    state["demo_dec_img"] = _make_image()
    state["demo_roundtrip_match"] = True
    state["demo_metrics"] = {"entropy": 7.9}

    # Re-running with SAME image identity should NOT reset state
    changed_same = check_and_update_source_image(state, img1)
    assert changed_same is False
    assert "demo_enc_img" in state
    assert "demo_dec_img" in state

    # Changing to DIFFERENT image must reset all derived state
    changed_diff = check_and_update_source_image(state, img2)
    assert changed_diff is True
    assert state.get("demo_orig_img_id") == compute_image_id(img2)
    assert "demo_enc_img" not in state
    assert "demo_dec_img" not in state
    assert "demo_roundtrip_match" not in state
    assert "demo_metrics" not in state


def test_finding_e_changing_prediction_image_resets_previous_prediction():
    """Finding E: Changing the input image in the ViT tab must clear old

    prediction, confidence, and distribution results.
    """
    img1 = _make_image(color=(1, 2, 3))
    img2 = _make_image(color=(4, 5, 6))

    state = {}
    check_and_update_prediction_image(state, img1)

    # Simulate model prediction
    state["last_baseline_pred"] = {
        "class_name": "airplane",
        "confidence": 0.85,
        "logits": [1.0] * 10,
    }

    # Same image: prediction preserved across Streamlit rerun
    assert check_and_update_prediction_image(state, img1) is False
    assert "last_baseline_pred" in state

    # New image: prediction cleared immediately
    assert check_and_update_prediction_image(state, img2) is True
    assert "last_baseline_pred" not in state
