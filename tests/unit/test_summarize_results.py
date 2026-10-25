from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["actinic", "bcc", "benign", "dermatofibroma", "melanoma", "nevi", "vascular"]


def _load():
    spec = importlib.util.spec_from_file_location(
        "summarize_under_test", ROOT / "scripts" / "summarize_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(acc, bal, auc, mel):
    sens = [0.1, 0.2, 0.3, 0.4, mel, 0.6, 0.7]
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "auc_macro_ovr": auc,
        "per_class_sensitivity": sens,
        "class_names": CLASSES,
        "epochs_trained": 40,
        "train_time_s": 1800,
        "num_samples": 2005,
    }


def test_table_contains_the_measured_numbers_and_lists_plaintext_once():
    mod = _load()
    runs = [
        (
            "per-pixel (v1)",
            {"A": _result(0.52, 0.53, 0.79, 0.64), "B": _result(0.13, 0.19, 0.57, 0.1)},
        ),
        ("block2", {"A": _result(0.52, 0.53, 0.79, 0.64), "B": _result(0.46, 0.536, 0.809, 0.65)}),
    ]
    text = mod.build_markdown(runs)
    assert text.count("none (plaintext)") == 2
    assert "| block2 | B |" in text and "53.6%" in text and "0.809" in text
    assert "| per-pixel (v1) | B |" in text and "19.0%" in text
    assert "65.0%" in text
    assert "(2005 images)" in text
    assert "not a medical device" in text.lower()


def test_missing_variants_are_skipped_and_missing_values_show_na():
    mod = _load()
    partial = _result(0.5, 0.5, None, 0.5)
    text = mod.build_markdown([("block2", {"B": partial})])
    assert "| block2 | B |" in text and "| block2 | C |" not in text
    assert "n/a" in text


def test_load_results_reads_only_known_files(tmp_path):
    mod = _load()
    (tmp_path / "result_B.json").write_text(json.dumps({"accuracy": 0.4}))
    (tmp_path / "result_Z.json").write_text(json.dumps({"accuracy": 0.9}))
    assert list(mod.load_results(tmp_path)) == ["B"]
