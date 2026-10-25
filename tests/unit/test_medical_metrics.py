"""Tests for the medical-style metrics (balanced accuracy, sensitivity,
specificity, AUC), class-weighted loss and balanced-accuracy checkpointing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from cryptovit.datasets.cifar_dataset import get_dataset_targets
from cryptovit.evaluation import evaluate_model
from cryptovit.training import Trainer
from cryptovit.training.trainer import balanced_class_weights, build_class_weights


class Identity(nn.Module):
    """Model whose input already is the logits (lets tests dictate predictions)."""

    def forward(self, x, key_embedding=None):
        return x


def _logits_for(preds, n_cls, margin=5.0):
    out = torch.zeros(len(preds), n_cls)
    for i, p in enumerate(preds):
        out[i, p] = margin
    return out


def _loader(logits, labels):
    return DataLoader(TensorDataset(logits, torch.tensor(labels)), batch_size=4)


class TestEvaluateModelMedicalMetrics:
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]

    @pytest.fixture
    def results(self):
        return evaluate_model(
            Identity(),
            _loader(_logits_for(self.preds, 3), self.labels),
            torch.device("cpu"),
            num_classes=3,
        )

    def test_accuracy_and_balanced_accuracy(self, results):
        assert results["accuracy"] == pytest.approx(4 / 6)
        assert results["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)

    def test_per_class_sensitivity_is_recall(self, results):
        assert results["per_class_sensitivity"] == pytest.approx([0.5, 1.0, 0.5])
        assert results["per_class_sensitivity"] == results["per_class_recall"]

    def test_per_class_specificity(self, results):
        assert results["per_class_specificity"] == pytest.approx([0.75, 0.75, 1.0])

    def test_support(self, results):
        assert results["per_class_support"] == [2, 2, 2]

    def test_auc_matches_sklearn_binary(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 60)
        scores = torch.tensor(rng.normal(size=(60, 2)), dtype=torch.float32)
        res = evaluate_model(
            Identity(), _loader(scores, labels.tolist()), torch.device("cpu"), num_classes=2
        )
        probs = torch.softmax(scores, dim=1).numpy()
        assert res["auc_macro_ovr"] == pytest.approx(roc_auc_score(labels, probs[:, 1]))

    def test_missing_class_has_no_auc_but_others_do(self):
        labels = [0, 1, 0, 1]
        res = evaluate_model(
            Identity(), _loader(_logits_for(labels, 3), labels), torch.device("cpu"), num_classes=3
        )
        assert res["per_class_auc"][2] is None
        assert res["per_class_auc"][0] == pytest.approx(1.0)
        assert res["auc_macro_ovr"] == pytest.approx(1.0)
        assert len(res["per_class_specificity"]) == 3

    def test_results_are_json_serialisable(self, results):
        import json

        json.dumps(results)


class TestClassWeights:
    def test_balanced_weights(self):
        w = balanced_class_weights([0, 0, 0, 1], 2)
        assert w.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])

    def test_absent_class_does_not_divide_by_zero(self):
        w = balanced_class_weights([0, 0, 1, 1], 3)
        assert torch.isfinite(w).all()

    def test_build_class_weights_modes(self):
        ds = TensorDataset(torch.zeros(6, 1), torch.tensor([0, 0, 0, 0, 1, 1]))
        ds.targets = [0, 0, 0, 0, 1, 1]
        loader = DataLoader(ds, batch_size=2)
        cfg = lambda mode: OmegaConf.create(  # noqa: E731
            {"training": {"class_weights": mode}, "dataset": {"num_classes": 2}}
        )
        assert build_class_weights(cfg("none"), loader) is None
        w = build_class_weights(cfg("balanced"), loader)
        assert w[1] > w[0]
        with pytest.raises(ValueError):
            build_class_weights(cfg("weird"), loader)

    def test_get_dataset_targets_unwraps_subset_and_wrappers(self):
        class Base(Dataset):
            targets = [3, 1, 2, 0]

            def __len__(self):
                return 4

            def __getitem__(self, i):
                return 0, self.targets[i]

        class Wrapper(Dataset):
            def __init__(self):
                self.base_dataset = Base()

            def __len__(self):
                return 4

            def __getitem__(self, i):
                return self.base_dataset[i]

        assert get_dataset_targets(Base()) == [3, 1, 2, 0]
        assert get_dataset_targets(Subset(Wrapper(), [3, 0])) == [0, 3]
        with pytest.raises(TypeError):
            get_dataset_targets(TensorDataset(torch.zeros(2, 1)))


def _cfg(select_metric="accuracy"):
    return OmegaConf.create(
        {
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "accumulation_steps": 1,
                "mixed_precision": False,
                "learning_rate": 1e-2,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "warmup_epochs": 0,
                "min_lr": 1e-5,
                "label_smoothing": 0.0,
                "clip_grad_norm": 1.0,
                "save_every_n_epochs": 100,
                "keep_best_k": 2,
                "early_stopping_patience": 100,
                "select_metric": select_metric,
            },
            "logging": {"use_tensorboard": False},
        }
    )


def _data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 3, 4, 4, generator=g)
    y = torch.tensor([0] * 12 + [1] * 4)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(48, 2))


class TestTrainerSelectionMetric:
    def test_history_records_balanced_accuracy_and_score(self, tmp_path):
        t = Trainer(_model(), _cfg("balanced_accuracy"), torch.device("cpu"), tmp_path)
        h = t.fit(_data(), _data())
        assert len(h["val_balanced_acc"]) == len(h["val_score"]) == 2
        assert h["val_score"] == pytest.approx(h["val_balanced_acc"])

    def test_default_selection_uses_accuracy(self, tmp_path):
        t = Trainer(_model(), _cfg("accuracy"), torch.device("cpu"), tmp_path)
        h = t.fit(_data(), _data())
        assert h["val_score"] == pytest.approx(h["val_acc"])

    def test_invalid_select_metric_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            Trainer(_model(), _cfg("f1"), torch.device("cpu"), tmp_path)

    def test_class_weights_reach_the_loss(self, tmp_path):
        w = torch.tensor([0.5, 2.0])
        t = Trainer(_model(), _cfg(), torch.device("cpu"), tmp_path, class_weights=w)
        assert torch.equal(t.criterion.weight, w)
