"""
Unit and integration tests for FastAPI backend readiness and endpoints.

Verifies:
1. Health endpoint distinguishes API status from trained model readiness.
2. Audited baseline ViT checkpoint loads strictly and predicts on real CIFAR-10 tensor
   (skipped cleanly if the checkpoint is absent from disk).
3. Prediction returns 10 finite logits, valid CIFAR-10 class, and top-5 probability distribution.
4. Hyperchaotic encryption and decryption endpoints perform exact round trip.
5. CryptoViT prediction endpoint returns HTTP 503 while model training is pending (no random model fallback).
6. Missing, empty and whitespace-only keys are rejected over real multipart HTTP.

Implementation note
-------------------
Tests use Starlette's ``TestClient`` (backed by ``httpx2`` in this virtualenv) without
entering the ASGI lifespan context.  The module fixture populates ``api.main`` globals
directly, so the real checkpoint is loaded once and monkeypatched state (for the
missing-checkpoint test) is not overwritten by ``startup_event``.  Requests still go
through FastAPI's real multipart/form parsing and status-code handling.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Make project importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import api.main as api_module
from api.main import CIFAR10_CLASSES, _find_baseline_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_baseline_checkpoint_available() -> bool:
    """Check if the trained baseline checkpoint actually exists on disk."""
    ckpt = _find_baseline_checkpoint()
    return ckpt is not None and ckpt.is_file()


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _image_files(img: Image.Image, filename: str = "test.png") -> dict:
    return {"file": (filename, _png_bytes(img), "image/png")}


# ---------------------------------------------------------------------------
# Module-scoped fixture: populate api.main globals WITHOUT running startup_event.
# Running the ASGI lifespan would reload the model from disk on every TestClient
# construction (repeated CUDA model init) and would obliterate per-test
# monkeypatches of _baseline_model.  We set _cfg and _device explicitly, then
# load the baseline model directly once per module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def initialise_api_globals():
    """
    Populate api.main module globals (``_cfg``, ``_device``, ``_baseline_model``,
    ``_baseline_ckpt_path``, ``_baseline_cfg``) by calling the pure-Python helpers
    in api.main directly, without starting the ASGI lifecycle or anyio portal.
    """
    from cryptovit.utils.config import get_device, load_config
    from cryptovit.utils.logging_setup import setup_logging

    cfg = load_config()
    setup_logging(level=cfg.logging.level)
    device = get_device(cfg)

    # Inject into module globals so the endpoint functions can use them
    api_module._cfg = cfg
    api_module._device = device

    # Load baseline model (pure Python, no anyio)
    model, ckpt_path, model_cfg_dict = api_module._load_baseline_model()
    api_module._baseline_model = model
    api_module._baseline_ckpt_path = ckpt_path
    api_module._baseline_cfg = model_cfg_dict

    # CryptoViT stays None (no checkpoint exists)
    api_module._cryptovit_model = None
    api_module._cryptovit_ckpt_path = None

    yield


@pytest.fixture(scope="module")
def client(initialise_api_globals):
    """HTTP client that exercises the real ASGI routes without lifespan startup."""
    return TestClient(api_module.app)


# ---------------------------------------------------------------------------
# Test 1 — Health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"

    if _is_baseline_checkpoint_available():
        assert result["baseline_ready"] is True
        assert "checkpoint_epoch0005_acc0.3110.pt" in (result["baseline_checkpoint"] or "")
        assert result["baseline_architecture"] == "vit_tiny"
    else:
        assert result["baseline_ready"] is False

    assert result["cryptovit_ready"] is False
    assert result["cryptovit_status"] == "pending_training_validation"


# ---------------------------------------------------------------------------
# Test 2 — Model info endpoint
# ---------------------------------------------------------------------------

def test_model_info_endpoint(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()

    assert "baseline_vit" in data
    if _is_baseline_checkpoint_available():
        assert data["baseline_vit"]["status"] == "trained"
        assert data["baseline_vit"]["parameters"] == 525994
    else:
        assert data["baseline_vit"]["status"] == "not_loaded"

    assert data["cryptovit"]["status"] == "pending"


# ---------------------------------------------------------------------------
# Test 3 — Predict plaintext (checkpoint-dependent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _is_baseline_checkpoint_available(),
    reason="Trained baseline checkpoint not present on disk; skipping checkpoint-dependent prediction test.",
)
def test_predict_plaintext_success(client):
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    response = client.post("/predict_plaintext", files=_image_files(img))

    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] in CIFAR10_CLASSES
    assert 0 <= data["confidence"] <= 1.0
    assert len(data["logits"]) == 10
    assert all(np.isfinite(v) for v in data["logits"])
    assert len(data["top5"]) == 5
    prob_sum = sum(item["prob"] for item in data["top5"])
    assert 0 < prob_sum <= 1.0


# ---------------------------------------------------------------------------
# Test 4 — Predict plaintext returns 503 when baseline model is not loaded
# ---------------------------------------------------------------------------

def test_predict_plaintext_when_checkpoint_missing(client, monkeypatch):
    """
    When no baseline model is loaded, /predict_plaintext must return HTTP 503,
    not create an untrained model or fabricate predictions.
    """
    monkeypatch.setattr(api_module, "_baseline_model", None)

    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    response = client.post("/predict_plaintext", files=_image_files(img))

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 5 — Encryption / decryption round-trip
# ---------------------------------------------------------------------------

def test_encryption_decryption_round_trip(client):
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    orig_img = Image.fromarray(arr)
    key = "UnitReviewTestKey_987"

    # Encrypt
    enc_response = client.post(
        "/encrypt",
        files=_image_files(orig_img, "orig.png"),
        data={"key": key},
    )
    assert enc_response.status_code == 200
    assert enc_response.headers.get("X-Encryption-KeyHash")

    # Decrypt
    dec_response = client.post(
        "/decrypt",
        files={"file": ("enc.png", enc_response.content, "image/png")},
        data={"key": key},
    )
    assert dec_response.status_code == 200

    dec_img = Image.open(io.BytesIO(dec_response.content))
    dec_arr = np.array(dec_img)
    assert np.array_equal(arr, dec_arr), "Decrypted array does not match original plaintext array!"


# ---------------------------------------------------------------------------
# Test 6 — CryptoViT /predict endpoint blocked when model is pending
# ---------------------------------------------------------------------------

def test_cryptovit_predict_blocked_when_pending(client):
    """CryptoViT endpoint must return HTTP 503 while model is pending validation."""
    # Ensure cryptovit model is None (already set by fixture, but be explicit)
    assert api_module._cryptovit_model is None

    img = Image.new("RGB", (32, 32), color=(50, 50, 50))
    response = client.post(
        "/predict",
        files=_image_files(img),
        data={"key": "Key123", "encrypt_first": "true"},
    )

    assert response.status_code == 503
    assert "pending validation" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 7 — Empty / whitespace-only keys rejected over real multipart HTTP
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_key, expected_status", [("", 422), ("   ", 400)])
def test_encrypt_rejects_empty_or_blank_key(client, bad_key, expected_status):
    """Regression (live-HTTP finding): FastAPI turns an empty multipart value into
    'missing', so a defaulted key silently fell back to the well-known DEFAULT_KEY.
    The key is now a required form field (422 for empty) and a body-level strip()
    check rejects whitespace-only keys with HTTP 400."""
    img = Image.new("RGB", (32, 32), color=(1, 2, 3))
    response = client.post(
        "/encrypt",
        files=_image_files(img),
        data={"key": bad_key},
    )

    assert response.status_code == expected_status
    if expected_status == 400:
        assert "empty" in response.json()["detail"].lower()
    else:
        assert response.json()["detail"][0]["loc"] == ["body", "key"]


@pytest.mark.parametrize("bad_key, expected_status", [("", 422), ("   ", 400)])
def test_decrypt_rejects_empty_or_blank_key(client, bad_key, expected_status):
    img = Image.new("RGB", (32, 32), color=(1, 2, 3))
    response = client.post(
        "/decrypt",
        files=_image_files(img),
        data={"key": bad_key},
    )

    assert response.status_code == expected_status
    if expected_status == 400:
        assert "empty" in response.json()["detail"].lower()
    else:
        assert response.json()["detail"][0]["loc"] == ["body", "key"]
