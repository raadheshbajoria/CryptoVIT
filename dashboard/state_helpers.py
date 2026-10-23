"""
dashboard/state_helpers.py
==========================
Pure functional state-management helpers for the Streamlit dashboard.

Extracts state invalidation and identity verification logic so it can be:
1. Tested deterministically without a running Streamlit server.
2. Maintained with clear preconditions and invariants.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional
from PIL import Image


def compute_image_id(img: Optional[Image.Image]) -> Optional[str]:
    """Compute a stable SHA-256 identifier from PIL image dimensions and raw pixel bytes."""
    if img is None:
        return None
    # Use image size, mode, and raw RGB bytes to form stable identity
    rgb = img.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{rgb.size[0]}x{rgb.size[1]}_{rgb.mode}_".encode("ascii"))
    h.update(rgb.tobytes())
    return h.hexdigest()


# Session state keys associated with encryption/decryption derived results
ENCRYPTION_DERIVED_KEYS = [
    "demo_enc_img",
    "demo_enc_bytes",
    "demo_enc_key",
    "demo_key_hash",
    "demo_dec_img",
    "demo_dec_success",
    "demo_roundtrip_match",
    "demo_pixel_diff",
    "demo_metrics",
    "demo_metrics_error",
    "demo_enc_error",
    "demo_dec_error",
]

# Session state keys associated with standard ViT baseline predictions
PREDICTION_DERIVED_KEYS = [
    "last_baseline_pred",
    "last_baseline_pred_error",
]


def reset_encryption_derived_state(state: Dict[str, Any]) -> None:
    """Clear all results derived from a previous encryption or decryption operation."""
    for key in ENCRYPTION_DERIVED_KEYS:
        state.pop(key, None)


def reset_decryption_derived_state(state: Dict[str, Any]) -> None:
    """Clear only the decrypted image and round-trip comparison results,

    leaving the encrypted image and key hash intact.
    Used when a new encryption operation is initiated.
    """
    for key in [
        "demo_dec_img",
        "demo_dec_success",
        "demo_roundtrip_match",
        "demo_pixel_diff",
        "demo_dec_error",
    ]:
        state.pop(key, None)


def reset_prediction_derived_state(state: Dict[str, Any]) -> None:
    """Clear all results derived from a previous model prediction."""
    for key in PREDICTION_DERIVED_KEYS:
        state.pop(key, None)


def check_and_update_source_image(
    state: Dict[str, Any],
    new_img: Optional[Image.Image],
    id_key: str = "demo_orig_img_id",
    img_key: str = "demo_orig_img",
) -> bool:
    """Check if the source image changed. If changed or new, update state

    and invalidate all derived encryption and decryption results.
    Returns True if the image was changed/updated, False otherwise.
    """
    if new_img is None:
        if id_key in state or img_key in state:
            state.pop(id_key, None)
            state.pop(img_key, None)
            reset_encryption_derived_state(state)
            return True
        return False

    new_id = compute_image_id(new_img)
    old_id = state.get(id_key)

    if new_id != old_id:
        state[id_key] = new_id
        state[img_key] = new_img
        reset_encryption_derived_state(state)
        return True

    return False


def check_and_update_prediction_image(
    state: Dict[str, Any],
    new_img: Optional[Image.Image],
    id_key: str = "tab3_img_id",
    img_key: str = "tab3_img",
) -> bool:
    """Check if the prediction input image changed. If changed or new, update state

    and invalidate all prior predictions.
    Returns True if the image was changed/updated, False otherwise.
    """
    if new_img is None:
        if id_key in state or img_key in state:
            state.pop(id_key, None)
            state.pop(img_key, None)
            reset_prediction_derived_state(state)
            return True
        return False

    new_id = compute_image_id(new_img)
    old_id = state.get(id_key)

    if new_id != old_id:
        state[id_key] = new_id
        state[img_key] = new_img
        reset_prediction_derived_state(state)
        return True

    return False
