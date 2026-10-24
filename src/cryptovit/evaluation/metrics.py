"""
cryptovit/evaluation/metrics.py
================================
ML evaluation metrics for CryptoViT experiments.

Provides a single ``evaluate_model`` function that returns a comprehensive
metrics dictionary suitable for logging, CSV export, and comparison tables.

Metrics computed
----------------
* Accuracy (top-1)
* Precision, Recall, F1 (macro-averaged)
* Per-class precision, recall, F1
* Confusion matrix (as nested list)
* Inference throughput (images/sec)
* Peak GPU memory (if CUDA)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
    key_embedding: Optional[torch.Tensor] = None,
) -> Dict:
    """Run inference on ``loader`` and compute comprehensive metrics.

    Parameters
    ----------
    model:
        Trained model in eval mode (this function sets eval mode internally).
    loader:
        DataLoader over the evaluation split.
    device:
        Torch device.
    use_amp:
        Whether to use automatic mixed precision during inference.
    class_names:
        Optional list of class name strings (for labelled confusion matrix).
    num_classes:
        Number of output classes. When given, per-class metrics and the
        confusion matrix always have this many entries, even if a class is
        missing from the labels or predictions. Inferred from data if None.
    key_embedding:
        Optional ``(1, key_embed_dim)`` tensor passed to the model as
        ``key_embedding=`` (for models with a key-aware positional encoding).

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1,
        per_class_precision, per_class_recall, per_class_f1,
        confusion_matrix, throughput_img_per_sec, peak_gpu_memory_mb
    """
    model.eval()
    model.to(device)
    if key_embedding is not None:
        key_embedding = key_embedding.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    # Reset peak memory counter
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    total_images = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if key_embedding is None:
                    logits = model(images)
                else:
                    logits = model(images, key_embedding=key_embedding)

            preds = logits.argmax(dim=1)
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_images += labels.size(0)

    elapsed = time.perf_counter() - t0
    throughput = total_images / elapsed

    # --- sklearn metrics ---
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float(accuracy_score(y_true, y_pred))
    # Fix the label set so every per-class list has num_classes entries in
    # class-index order, regardless of which classes appear in this split.
    labels = np.arange(num_classes) if num_classes else None
    precision_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    recall_macro    = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_macro        = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    precision_per  = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
    recall_per     = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()
    f1_per         = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()

    cm_arr = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm_arr.tolist()

    # --- Medical-style metrics -------------------------------------------
    # Sensitivity = recall of the class; specificity = TN / (TN + FP) with the
    # class treated as "positive" against all others.
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    support = cm_arr.sum(axis=1)
    tp = np.diag(cm_arr)
    fp = cm_arr.sum(axis=0) - tp
    tn = cm_arr.sum() - support - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0).tolist()
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 0))
    auc_macro, auc_per_class = _auc_scores(y_true, probs, num_classes)

    # --- Hardware ---
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    results = {
        "accuracy":               accuracy,
        "precision":              precision_macro,
        "recall":                 recall_macro,
        "f1":                     f1_macro,
        "per_class_precision":    precision_per,
        "per_class_recall":       recall_per,
        "per_class_f1":           f1_per,
        "confusion_matrix":       cm,
        "balanced_accuracy":      balanced_acc,
        "per_class_sensitivity":  recall_per,
        "per_class_specificity":  specificity,
        "per_class_support":      support.tolist(),
        "auc_macro_ovr":          auc_macro,
        "per_class_auc":          auc_per_class,
        "throughput_img_per_sec": throughput,
        "peak_gpu_memory_mb":     peak_mem_mb,
        "num_samples":            total_images,
        "class_names":            class_names or [],
    }

    _log_summary(results)
    return results


def _auc_scores(
    y_true: np.ndarray, probs: np.ndarray, num_classes: Optional[int]
) -> tuple:
    """Macro one-vs-rest AUC and per-class AUC.

    A class that never occurs (or always occurs) in ``y_true`` has no defined
    AUC; its entry is ``None`` and it is skipped in the macro average. Returns
    ``(None, [])`` if probabilities are unavailable or only one class is present.
    """
    if probs.ndim != 2 or probs.shape[0] != len(y_true) or probs.shape[0] == 0:
        return None, []
    n_cls = num_classes or probs.shape[1]
    per_class: List[Optional[float]] = []
    for c in range(n_cls):
        positive = (y_true == c)
        if positive.all() or not positive.any() or c >= probs.shape[1]:
            per_class.append(None)
        else:
            per_class.append(float(roc_auc_score(positive, probs[:, c])))
    valid = [a for a in per_class if a is not None]
    return (float(np.mean(valid)) if valid else None), per_class


def _log_summary(results: Dict) -> None:
    auc = results.get("auc_macro_ovr")
    logger.info(
        "Evaluation | acc=%.4f  bal_acc=%.4f  auc=%s  prec=%.4f  recall=%.4f  f1=%.4f | "
        "%.1f img/s | peak_mem=%.1fMB",
        results["accuracy"],
        results["balanced_accuracy"],
        f"{auc:.4f}" if auc is not None else "n/a",
        results["precision"],
        results["recall"],
        results["f1"],
        results["throughput_img_per_sec"],
        results["peak_gpu_memory_mb"],
    )


def print_metrics_table(results: Dict) -> None:
    """Pretty-print a classification report to stdout."""
    print(f"\n{'='*60}")
    print(f"  Overall  | Acc={results['accuracy']:.4f}  "
          f"P={results['precision']:.4f}  "
          f"R={results['recall']:.4f}  "
          f"F1={results['f1']:.4f}")
    if "balanced_accuracy" in results:
        auc = results.get("auc_macro_ovr")
        print(f"  Balanced accuracy: {results['balanced_accuracy']:.4f}   "
              f"Macro AUC: {auc:.4f}" if auc is not None else
              f"  Balanced accuracy: {results['balanced_accuracy']:.4f}   Macro AUC: n/a")
    print(f"  Samples  : {results['num_samples']}")
    print(f"  Speed    : {results['throughput_img_per_sec']:.1f} img/s")
    if results["peak_gpu_memory_mb"] > 0:
        print(f"  GPU Mem  : {results['peak_gpu_memory_mb']:.1f} MB (peak)")
    print(f"{'='*60}")

    class_names = results.get("class_names") or []
    if class_names:
        def at(key, i):
            vals = results.get(key) or []
            v = vals[i] if i < len(vals) else None
            return f"{v:>8.4f}" if v is not None else f"{'n/a':>8}"

        print(f"\n  {'Class':<28} {'Prec':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
        print(f"  {'-'*72}")
        for i, name in enumerate(class_names):
            print(f"  {name[:27]:<28} {at('per_class_precision', i)} "
                  f"{at('per_class_recall', i)} {at('per_class_specificity', i)} "
                  f"{at('per_class_f1', i)} {at('per_class_auc', i)}")
    print()
