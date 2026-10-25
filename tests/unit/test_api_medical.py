"""API tests for the medical (DermaMNIST) configuration: class names, variant
status, trained-checkpoint loading, /compare, /results and key behaviour."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from cryptovit.inference.pipeline import (
    build_variant_model,
    find_best_checkpoint,
)
from cryptovit.utils import load_config


def _png(seed=0, size=28) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _files(data: bytes):
    return {"file": ("img.png", data, "image/png")}


@pytest.fixture
def medical_client(tmp_path, monkeypatch):
    """Client configured for DermaMNIST with an empty experiments directory."""
    monkeypatch.setenv("CRYPTOVIT_CONFIG", "dermamnist")
    monkeypatch.setenv("CRYPTOVIT_EXPERIMENTS", str(tmp_path))
    monkeypatch.setenv("CRYPTOVIT_DEVICE", "cpu")
    with TestClient(app) as client:
        yield client, tmp_path


def test_health_reports_the_medical_task(medical_client):
    client, _ = medical_client
    data = client.get("/health").json()
    assert data["dataset"] == "dermamnist"
    assert len(data["class_names"]) == 7 and "melanoma" in data["class_names"]
    assert data["image_size"] == 28
    assert set(data["variants"]) == {"A", "B", "C", "D"}
    assert data["variants"]["A"]["loaded"] and not data["variants"]["A"]["trained"]
    assert data["variants"]["B"]["loaded"] is False
    assert "not a medical device" in data["disclaimer"].lower()


def test_predictions_use_skin_lesion_class_names(medical_client):
    client, _ = medical_client
    names = client.get("/health").json()["class_names"]

    plain = client.post("/predict_plaintext", files=_files(_png())).json()
    assert plain["class_name"] in names and plain["variant"] == "A"
    assert len(plain["top5"]) == 5

    enc = client.post("/predict", files=_files(_png()), data={"key": "k"}).json()
    assert enc["class_name"] in names and enc["variant"] == "D"
    assert sum(item["prob"] for item in enc["top5"]) <= 1.0 + 1e-6


def test_larger_uploads_are_reduced_to_the_native_size(medical_client):
    client, _ = medical_client
    resp = client.post("/encrypt", files=_files(_png(size=200)), data={"key": "k"})
    assert resp.status_code == 200
    assert Image.open(io.BytesIO(resp.content)).size == (28, 28)


def test_decrypt_right_key_restores_and_wrong_key_does_not(medical_client):
    client, _ = medical_client
    original = _png(seed=3)
    enc = client.post("/encrypt", files=_files(original), data={"key": "right"}).content

    ok = client.post("/decrypt", files=_files(enc), data={"key": "right"}).content
    bad = client.post("/decrypt", files=_files(enc), data={"key": "wrong"})
    assert bad.status_code == 200
    orig_arr = np.array(Image.open(io.BytesIO(original)).convert("RGB"))
    assert np.array_equal(np.array(Image.open(io.BytesIO(ok)).convert("RGB")), orig_arr)
    assert not np.array_equal(
        np.array(Image.open(io.BytesIO(bad.content)).convert("RGB")), orig_arr
    )


def test_compare_includes_only_loaded_variants(medical_client):
    client, _ = medical_client
    data = client.post("/compare", files=_files(_png()), data={"key": "k"}).json()
    assert data["plaintext"]["variant"] == "A"
    assert set(data["encrypted"]) == {"D"}


def test_trained_checkpoint_is_loaded_and_used(tmp_path, monkeypatch):
    cfg = load_config(overrides=["config=dermamnist"])
    ckpt_dir = tmp_path / "ablation" / "variant_B" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.manual_seed(0)
    model = build_variant_model("B", cfg)
    torch.save(
        {"model_state_dict": model.state_dict()}, ckpt_dir / "checkpoint_epoch0003_acc0.5000.pt"
    )

    monkeypatch.setenv("CRYPTOVIT_CONFIG", "dermamnist")
    monkeypatch.setenv("CRYPTOVIT_EXPERIMENTS", str(tmp_path))
    monkeypatch.setenv("CRYPTOVIT_DEVICE", "cpu")
    with TestClient(app) as client:
        status = client.get("/health").json()["variants"]
        assert status["B"]["loaded"] and status["B"]["trained"]
        assert not status["A"]["trained"]
        cmp = client.post("/compare", files=_files(_png()), data={"key": "k"}).json()
        assert set(cmp["encrypted"]) == {"B", "D"}
        assert cmp["encrypted"]["B"]["trained"] is True
        one = client.post(
            "/predict", files=_files(_png()), data={"key": "k", "variant": "b"}
        ).json()
        assert one["variant"] == "B"


def test_several_configs_can_be_combined(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTOVIT_CONFIG", "dermamnist,block2")
    monkeypatch.setenv("CRYPTOVIT_EXPERIMENTS", str(tmp_path))
    monkeypatch.setenv("CRYPTOVIT_DEVICE", "cpu")
    with TestClient(app) as client:
        data = client.get("/health").json()
        assert data["dataset"] == "dermamnist"
        assert data["encryption_scheme"] == "block2"


def test_results_endpoint(medical_client):
    client, tmp = medical_client
    assert client.get("/results").status_code == 404

    out = tmp / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result_A.json").write_text(
        json.dumps({"accuracy": 0.5, "balanced_accuracy": 0.4, "auc_macro_ovr": None})
    )
    rows = client.get("/results").json()
    assert rows[0]["variant"] == "A" and rows[0]["balanced_accuracy"] == 0.4
    assert "label" in rows[0]


def test_find_best_checkpoint_ignores_last_and_picks_highest(tmp_path):
    for name in (
        "checkpoint_epoch0001_acc0.3000.pt",
        "checkpoint_epoch0005_acc0.6000.pt",
        "checkpoint_epoch0007_acc0.5500.pt",
        "last.pt",
        "checkpoint_periodic_epoch0010.pt",
    ):
        (tmp_path / name).write_bytes(b"x")
    assert find_best_checkpoint(tmp_path).name == "checkpoint_epoch0005_acc0.6000.pt"
    assert find_best_checkpoint(tmp_path / "missing") is None
