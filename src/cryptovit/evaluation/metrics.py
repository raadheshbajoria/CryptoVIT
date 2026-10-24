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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
        Number of output classes (inferred from data if None).

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1,
        per_class_precision, per_class_recall, per_class_f1,
        confusion_matrix, throughput_img_per_sec, peak_gpu_memory_mb
    """
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []

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
                logits = model(images)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            total_images += labels.size(0)

    elapsed = time.perf_counter() - t0
    throughput = total_images / elapsed

    # --- sklearn metrics ---
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precision_per  = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    recall_per     = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per         = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    cm = confusion_matrix(y_true, y_pred).tolist()

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
        "throughput_img_per_sec": throughput,
        "peak_gpu_memory_mb":     peak_mem_mb,
        "num_samples":            total_images,
        "class_names":            class_names or [],
    }

    _log_summary(results)
    return results


def _log_summary(results: Dict) -> None:
    logger.info(
        "Evaluation | acc=%.4f  prec=%.4f  recall=%.4f  f1=%.4f | "
        "%.1f img/s | peak_mem=%.1fMB",
        results["accuracy"],
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
    print(f"  Samples  : {results['num_samples']}")
    print(f"  Speed    : {results['throughput_img_per_sec']:.1f} img/s")
    if results["peak_gpu_memory_mb"] > 0:
        print(f"  GPU Mem  : {results['peak_gpu_memory_mb']:.1f} MB (peak)")
    print(f"{'='*60}")

    class_names = results.get("class_names") or []
    if class_names:
        print(f"\n  {'Class':<20} {'P':>8} {'R':>8} {'F1':>8}")
        print(f"  {'-'*44}")
        for i, name in enumerate(class_names):
            p = results["per_class_precision"][i] if i < len(results["per_class_precision"]) else 0
            r = results["per_class_recall"][i]    if i < len(results["per_class_recall"])    else 0
            f = results["per_class_f1"][i]        if i < len(results["per_class_f1"])        else 0
            print(f"  {name:<20} {p:>8.4f} {r:>8.4f} {f:>8.4f}")
    print()
