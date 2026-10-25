"""Dashboard tests against a faked API (no server needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"

CLASSES = [
    "actinic keratoses",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]


class FakeResponse:
    def __init__(self, json_data=None, status=200):
        self._json, self.status_code, self.text = json_data, status, ""

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@pytest.fixture
def fake_api(monkeypatch):
    state = {"results": None}

    def fake_get(url, **kwargs):
        endpoint = url.split("8000", 1)[-1]
        if endpoint == "/health":
            return FakeResponse(
                {
                    "status": "ok",
                    "device": "cpu",
                    "dataset": "dermamnist",
                    "class_names": CLASSES,
                    "image_size": 28,
                    "variants": {v: {"loaded": True, "trained": True} for v in "ABCD"},
                }
            )
        if endpoint == "/results" and state["results"] is not None:
            return FakeResponse(state["results"])
        return FakeResponse(status=404)

    monkeypatch.setattr(requests, "get", fake_get)
    return state


def _open_app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_page_without_an_image(fake_api):
    at = _open_app()
    assert "Medical" in at.title[0].value
    assert any("not a medical device" in w.value.lower() for w in at.warning)
    assert any("upload an image" in i.value.lower() for i in at.info)
    assert not at.sidebar.selectbox


def test_results_placeholder_before_training_finishes(fake_api):
    at = _open_app()
    assert any("Results appear here" in i.value for i in at.info)


def test_results_table_and_horizontal_chart_when_results_exist(fake_api):
    fake_api["results"] = [
        {
            "variant": "A",
            "label": "Plain ViT (plaintext image)",
            "accuracy": 0.6,
            "balanced_accuracy": 0.5,
            "auc_macro_ovr": 0.8,
            "epochs_trained": 40,
            "num_samples": 2005,
            "per_class_sensitivity": [0.1, 0.2, 0.3, 0.4, 0.55, 0.9, 0.5],
        }
    ]
    at = _open_app()

    table = at.dataframe[-1].value
    assert "Balanced accuracy" in table.columns
    assert table.iloc[0]["Melanoma sensitivity"] == "55.0%"
    assert not any("Results appear here" in i.value for i in at.info)

    charts = at.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    encoding = spec.get("encoding") or spec["layer"][0]["encoding"]
    assert encoding["y"]["field"] == "Class"
    assert encoding["x"]["field"] == "Sensitivity"
    assert encoding["x"]["axis"]["labelAngle"] == 0
