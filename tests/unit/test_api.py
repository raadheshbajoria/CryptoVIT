from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _make_test_image_bytes(H: int = 32, W: int = 32) -> bytes:
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


class TestAPIEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "baseline_loaded" in data
        assert "cryptovit_loaded" in data

    def test_model_info(self, client):
        resp = client.get("/model_info")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_encrypt_and_decrypt_roundtrip(self, client):
        img_bytes = _make_test_image_bytes(32, 32)

        enc_resp = client.post(
            "/encrypt",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"key": "api_test_key"},
        )
        assert enc_resp.status_code == 200
        assert enc_resp.headers["content-type"] == "image/png"
        assert "X-Encryption-KeyHash" in enc_resp.headers
        enc_bytes = enc_resp.content

        dec_resp = client.post(
            "/decrypt",
            files={"file": ("enc.png", enc_bytes, "image/png")},
            data={"key": "api_test_key"},
        )
        assert dec_resp.status_code == 200
        assert dec_resp.headers["content-type"] == "image/png"

        orig_img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        dec_img = np.array(Image.open(io.BytesIO(dec_resp.content)).convert("RGB"))
        assert np.array_equal(orig_img, dec_img)

    def test_metrics_after_encrypt(self, client):
        img_bytes = _make_test_image_bytes(32, 32)
        client.post(
            "/encrypt",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"key": "api_test_key"},
        )
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        data = metrics_resp.json()
        assert "entropy" in data
        assert "key_hash" in data

    def test_predict_endpoints(self, client):
        img_bytes = _make_test_image_bytes(32, 32)

        plain_resp = client.post(
            "/predict_plaintext",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert plain_resp.status_code == 200
        data = plain_resp.json()
        assert "class_name" in data
        assert "confidence" in data
        assert "top5" in data

        enc_pred_resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"key": "api_test_key", "encrypt_first": "true"},
        )
        assert enc_pred_resp.status_code == 200
        enc_data = enc_pred_resp.json()
        assert "class_name" in enc_data
        assert "confidence" in enc_data
