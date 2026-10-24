"""Dashboard tests: the page stays plain text (no emojis/symbols) and the full
flow works against a faked API (no server needed)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "dashboard" / "app.py"
DEMO_IMAGES = sorted((ROOT / "demo_images").glob("*.png"))

CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis",
           "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"]


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _pred(variant, name, cls, trained=True):
    return {"variant": variant, "model": name, "trained": trained,
            "class_id": CLASSES.index(cls), "class_name": cls, "confidence": 0.61,
            "top5": [{"class": cls, "prob": 0.61}, {"class": "melanoma", "prob": 0.2}]}


class FakeResponse:
    def __init__(self, json_data=None, content=b"", headers=None, status=200):
        self._json, self.content = json_data, content
        self.headers, self.status_code = headers or {}, status
        self.text = ""

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@pytest.fixture
def fake_api(monkeypatch):
    """Patch requests.get/post with an in-memory API. Returns a state dict."""
    state = {"results": None, "original": None, "calls": []}
    noise = _png(np.random.default_rng(0).integers(0, 256, (28, 28, 3), dtype=np.uint8))

    def fake_get(url, **kwargs):
        endpoint = url.split("8000", 1)[-1]
        state["calls"].append(("GET", endpoint))
        if endpoint == "/health":
            return FakeResponse({
                "status": "ok", "device": "cpu", "dataset": "dermamnist",
                "class_names": CLASSES, "image_size": 28,
                "variants": {v: {"loaded": True, "trained": v != "D"} for v in "ABCD"},
            })
        if endpoint == "/results" and state["results"] is not None:
            return FakeResponse(state["results"])
        return FakeResponse(status=404)

    def fake_post(url, files=None, data=None, **kwargs):
        endpoint = url.split("8000", 1)[-1]
        state["calls"].append(("POST", endpoint, dict(data or {})))
        if endpoint == "/encrypt":
            return FakeResponse(content=noise, headers={"X-Encryption-KeyHash": "ab" * 32})
        if endpoint == "/compare":
            return FakeResponse({
                "plaintext": _pred("A", "Plain ViT (plaintext image)", "melanocytic nevi"),
                "encrypted": {
                    "B": _pred("B", "Plain ViT (encrypted image)", "melanocytic nevi"),
                    "D": _pred("D", "CryptoViT", "vascular lesions", trained=False),
                },
            })
        if endpoint == "/decrypt":
            wrong = (data or {}).get("key") == "wrong_key"
            return FakeResponse(content=noise if wrong else state["original"])
        return FakeResponse(status=404)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return state


def _open_app(state):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_source_is_plain_ascii():
    text = APP.read_bytes().decode("utf-8")
    offenders = sorted({c for c in text if ord(c) > 127})
    assert not offenders, "non-ASCII characters in dashboard: " + ", ".join(
        f"U+{ord(c):04X}" for c in offenders
    )


def test_page_without_an_image(fake_api):
    at = _open_app(fake_api)
    assert "Medical" in at.title[0].value
    assert any("not a medical device" in w.value.lower() for w in at.warning)
    assert any("pick a demo image" in i.value.lower() for i in at.info)

    # the status row, university line and footer were removed on request
    labels = [m.label for m in at.metric]
    for removed in ("Dataset", "Device", "Trained models", "Classes"):
        assert removed not in labels
    all_text = " ".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption]
        + [i.value for i in at.info]
    )
    assert "Bhopal" not in all_text
    assert "Final Year" not in all_text
    assert "Classes:" not in all_text


@pytest.mark.skipif(not DEMO_IMAGES, reason="no demo_images to select")
def test_full_flow_with_a_demo_image(fake_api):
    fake_api["original"] = _png(np.array(Image.open(DEMO_IMAGES[0]).convert("RGB")))
    at = _open_app(fake_api)
    at.sidebar.selectbox[0].select_index(1).run()
    assert not at.exception, at.exception

    # comparison table: plaintext model first, then the encrypted-image models
    table = at.dataframe[0].value
    assert list(table["Variant"]) == ["A", "B", "D"]
    assert list(table["Sees"]) == ["plaintext image", "encrypted image", "encrypted image"]
    assert list(table["Trained"]) == ["yes", "yes", "untrained"]

    labels = [b.label for b in at.button]
    assert labels == ["Decrypt with the correct key", "Decrypt with a wrong key"]

    at.button[0].click().run()
    assert any("identical" in s.value for s in at.success)
    at.button[1].click().run()
    assert any("not recovered" in e.value for e in at.error)

    posts = [c for c in fake_api["calls"] if c[0] == "POST"]
    assert any(c[1] == "/decrypt" and c[2]["key"] == "wrong_key" for c in posts)
    assert any(c[1] == "/encrypt" and "key" in c[2] for c in posts)


@pytest.mark.skipif(not DEMO_IMAGES, reason="no demo_images to select")
def test_results_table_when_results_exist(fake_api):
    fake_api["original"] = _png(np.array(Image.open(DEMO_IMAGES[0]).convert("RGB")))
    fake_api["results"] = [{
        "variant": "A", "label": "Plain ViT (plaintext image)", "accuracy": 0.6,
        "balanced_accuracy": 0.5, "auc_macro_ovr": 0.8, "epochs_trained": 40,
        "num_samples": 2005, "per_class_sensitivity": [0.1, 0.2, 0.3, 0.4, 0.55, 0.9, 0.5],
    }]
    at = _open_app(fake_api)
    results_table = at.dataframe[-1].value
    assert "Balanced accuracy" in results_table.columns
    assert "Melanoma sensitivity" in results_table.columns
    assert results_table.iloc[0]["Melanoma sensitivity"] == "55.0%"
    assert not any("Results appear here" in i.value for i in at.info)

    # the per-class chart is horizontal (classes on the y axis)
    charts = at.get("vega_lite_chart")
    assert len(charts) == 1
    import json
    spec = json.loads(charts[0].proto.spec)
    encoding = spec.get("encoding") or spec["layer"][0]["encoding"]
    assert encoding["y"]["field"] == "Class"            # classes on the y axis
    assert encoding["x"]["field"] == "Sensitivity"
    assert encoding["x"]["axis"]["labelAngle"] == 0     # horizontal axis labels
    assert encoding["y"]["axis"]["labelAngle"] == 0


def test_results_placeholder_before_training_finishes(fake_api):
    at = _open_app(fake_api)
    assert any("Results appear here" in i.value for i in at.info)

