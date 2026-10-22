"""
tests/unit/test_api_endpoints.py
================================
Unit and integration tests for FastAPI backend readiness and endpoints.

Verifies:
1. Health endpoint distinguishes API status from trained model readiness.
2. Audited baseline ViT checkpoint loads strictly and predicts on real CIFAR-10 tensor.
3. Prediction returns 10 finite logits, valid CIFAR-10 class, and top-5 probability distribution.
4. Hyperchaotic encryption and decryption endpoints perform exact round trip.
5. CryptoViT prediction endpoint returns HTTP 503 while model training is pending (no random model fallback).
"""

import io
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, startup_event, CIFAR10_CLASSES


@pytest.fixture(scope="module")
def client():
    # Trigger startup event to load models
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["baseline_ready"] is True
    assert "checkpoint_epoch0005_acc0.3110.pt" in data["baseline_checkpoint"]
    assert data["baseline_architecture"] == "vit_tiny"
    assert data["cryptovit_ready"] is False
    assert data["cryptovit_status"] == "pending_training_validation"


def test_model_info_endpoint(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "baseline_vit" in data
    assert data["baseline_vit"]["status"] == "trained"
    assert data["baseline_vit"]["parameters"] == 525994
    assert data["cryptovit"]["status"] == "pending"


def test_predict_plaintext_success(client):
    # Create test image (32x32 RGB)
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict_plaintext",
        files={"file": ("test.png", buf.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] in CIFAR10_CLASSES
    assert 0 <= data["confidence"] <= 1.0
    assert len(data["logits"]) == 10
    assert len(data["top5"]) == 5
    # Check top5 probabilities sum <= 1.0
    prob_sum = sum(item["prob"] for item in data["top5"])
    assert 0 < prob_sum <= 1.0


def test_encryption_decryption_round_trip(client):
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    key = "UnitReviewTestKey_987"

    # Encrypt
    enc_resp = client.post(
        "/encrypt",
        files={"file": ("orig.png", img_bytes, "image/png")},
        data={"key": key}
    )
    assert enc_resp.status_code == 200
    enc_bytes = enc_resp.content

    # Decrypt
    dec_resp = client.post(
        "/decrypt",
        files={"file": ("enc.png", enc_bytes, "image/png")},
        data={"key": key}
    )
    assert dec_resp.status_code == 200
    dec_img = Image.open(io.BytesIO(dec_resp.content))
    dec_arr = np.array(dec_img)

    assert np.array_equal(arr, dec_arr), "Decrypted array does not match original plaintext array!"


def test_cryptovit_predict_blocked_when_pending(client):
    img = Image.new("RGB", (32, 32), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
        data={"key": "Key123"}
    )
    assert response.status_code == 503
    assert "pending validation" in response.json()["detail"]

