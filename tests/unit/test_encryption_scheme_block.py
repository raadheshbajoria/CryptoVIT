"""The block<B> scheme as seen through encrypt()/decrypt(), the dataset loaders,
the inference service and the API (it must work everywhere a scheme name does)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from cryptovit.encryption import decrypt, encrypt
from cryptovit.encryption.block_scheme import block_encrypt
from cryptovit.utils import load_config

STEPS = 50


def _img(shape=(28, 28, 3), seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_encrypt_decrypt_round_trip_and_meta():
    img = _img()
    enc, meta = encrypt(img, "key", STEPS, scheme="block2")
    assert meta["scheme"] == "block2"
    assert enc.shape == img.shape and enc.dtype == np.uint8
    assert not np.array_equal(enc, img)
    assert np.array_equal(decrypt(enc, "key", meta, STEPS), img)          # scheme from meta
    assert np.array_equal(decrypt(enc, "key", None, STEPS, scheme="block2"), img)


def test_matches_the_block_module_and_differs_from_other_schemes():
    img = _img(seed=3)
    enc, _ = encrypt(img, "key", STEPS, scheme="block2")
    assert np.array_equal(enc, block_encrypt(img, b"key", 2, STEPS))
    assert not np.array_equal(enc, encrypt(img, "key", STEPS, scheme="v1")[0])
    assert not np.array_equal(enc, encrypt(img, "key", STEPS, scheme="block4")[0])


def test_wrong_key_and_wrong_scheme_do_not_decrypt():
    img = _img(seed=4)
    enc, meta = encrypt(img, "right", STEPS, scheme="block2")
    assert not np.array_equal(decrypt(enc, "wrong", meta, STEPS), img)
    assert not np.array_equal(decrypt(enc, "right", None, STEPS, scheme="block4"), img)
    assert not np.array_equal(decrypt(enc, "right", None, STEPS, scheme="v1"), img)


@pytest.mark.parametrize("scheme", ["block0", "blocks", "block", "block-2", "v3"])
def test_invalid_scheme_names_are_rejected(scheme):
    with pytest.raises(ValueError):
        encrypt(_img((8, 8, 3)), "key", STEPS, scheme=scheme)


def test_block_size_must_divide_the_image():
    with pytest.raises(ValueError):
        encrypt(_img((28, 28, 3)), "key", STEPS, scheme="block5")


def test_v1_and_v2_are_unchanged_by_the_new_scheme():
    img = _img((8, 8, 3), seed=9)
    for scheme in ("v1", "v2"):
        enc, meta = encrypt(img, "key", STEPS, scheme=scheme)
        assert np.array_equal(decrypt(enc, "key", meta, STEPS), img)


def test_dataset_loader_uses_the_block_scheme():
    import torch
    from torch.utils.data import Dataset

    from cryptovit.datasets import build_encrypted_loaders

    class Fake(Dataset):
        def __init__(self, n, seed):
            rng = np.random.default_rng(seed)
            self.imgs = rng.integers(0, 256, (n, 28, 28, 3), dtype=np.uint8)
            self.targets = [i % 7 for i in range(n)]

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return Image.fromarray(self.imgs[i]), self.targets[i]

    base = ["dataset.image_size=28", "dataset.num_workers=0", "dataset.pin_memory=false",
            "encryption.transient_steps=50", "training.batch_size=8",
            "evaluation.batch_size=8"]
    first = {}
    for scheme in ("v1", "block2"):
        cfg = load_config(overrides=["config=dermamnist"] + base + [f"encryption.scheme={scheme}"])
        train_l, _, _ = build_encrypted_loaders(
            cfg, Fake(16, 1), Fake(8, 2), "unit-key", seed=7, base_val=Fake(8, 3)
        )
        first[scheme] = train_l.dataset.dataset[0][0]
    assert first["v1"].shape == first["block2"].shape == (3, 28, 28)
    assert not torch.equal(first["v1"], first["block2"])


def test_api_encrypts_and_decrypts_with_the_block_scheme(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTOVIT_CONFIG", "dermamnist,block2")
    monkeypatch.setenv("CRYPTOVIT_EXPERIMENTS", str(tmp_path))
    monkeypatch.setenv("CRYPTOVIT_DEVICE", "cpu")

    def png(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    original = _img((28, 28, 3), seed=5)
    with TestClient(app) as client:
        assert client.get("/health").json()["encryption_scheme"] == "block2"
        enc = client.post("/encrypt", files={"file": ("a.png", png(original), "image/png")},
                          data={"key": "k"})
        assert enc.status_code == 200
        dec = client.post("/decrypt", files={"file": ("e.png", enc.content, "image/png")},
                          data={"key": "k"})
        restored = np.array(Image.open(io.BytesIO(dec.content)).convert("RGB"))
        assert np.array_equal(restored, original)
        cmp = client.post("/compare", files={"file": ("a.png", png(original), "image/png")},
                          data={"key": "k"})
        assert cmp.status_code == 200
        # metrics (entropy etc.) are computed for the block scheme too
        assert client.get("/metrics").status_code == 200
