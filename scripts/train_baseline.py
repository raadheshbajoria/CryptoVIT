#!/usr/bin/env python
"""
scripts/train_baseline.py
==========================
Phase 1 — Train the plaintext ViT baseline on CIFAR-10 or CIFAR-100.

Usage
-----
    python scripts/train_baseline.py
    python scripts/train_baseline.py dataset.name=cifar100 dataset.num_classes=100
    python scripts/train_baseline.py training.epochs=30 model.name=vit_tiny

All key=value pairs override the base.yaml config via OmegaConf.

Outputs
-------
  experiments/
    baseline_<dataset>_<model>/
      checkpoints/    — best-K + periodic .pt files
      tensorboard/    — optional TensorBoard event files
      results.json    — final evaluation metrics
      history.json    — per-epoch train/val loss and accuracy
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure the src package is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cryptovit.datasets import get_class_names, get_dataloaders
from cryptovit.evaluation import evaluate_model, print_metrics_table
from cryptovit.models import build_model
from cryptovit.training import Trainer
from cryptovit.utils.config import get_device, load_config
from cryptovit.utils.logging_setup import setup_logging
from cryptovit.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Configuration                                                     #
    # ------------------------------------------------------------------ #
    overrides = sys.argv[1:]          # accept key=value pairs from CLI
    cfg = load_config(overrides=overrides if overrides else None)
    log_file = PROJECT_ROOT / cfg.paths.experiments_dir / f"baseline_{cfg.dataset.name}_{cfg.model.name}.log"
    setup_logging(level=cfg.logging.level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("CryptoViT — Phase 1 Plaintext Baseline")
    logger.info("Dataset : %s", cfg.dataset.name)
    logger.info("Model   : %s", cfg.model.name)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 2. Reproducibility                                                   #
    # ------------------------------------------------------------------ #
    set_seed(cfg.project.seed)
    device = get_device(cfg)
    logger.info("Device  : %s", device)

    # ------------------------------------------------------------------ #
    # 3. Experiment directory                                              #
    # ------------------------------------------------------------------ #
    exp_name = f"baseline_{cfg.dataset.name}_{cfg.model.name}"
    exp_dir = PROJECT_ROOT / cfg.paths.experiments_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment dir: %s", exp_dir)

    # ------------------------------------------------------------------ #
    # 4. Data                                                              #
    # ------------------------------------------------------------------ #
    logger.info("Loading dataset …")
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg,
        data_root=PROJECT_ROOT / cfg.paths.raw_dir,
        seed=cfg.project.seed,
    )
    class_names = get_class_names(cfg.dataset.name)
    logger.info(
        "Train batches=%d  Val batches=%d  Test batches=%d",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # ------------------------------------------------------------------ #
    # 5. Model                                                             #
    # ------------------------------------------------------------------ #
    logger.info("Building model …")
    model = build_model(cfg)
    logger.info("Parameters: %.2fM", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    # ------------------------------------------------------------------ #
    # 6. Train                                                             #
    # ------------------------------------------------------------------ #
    trainer = Trainer(model, cfg, device, exp_dir)
    t0 = time.time()
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - t0
    logger.info("Training finished in %.1f minutes.", train_time / 60)

    # ------------------------------------------------------------------ #
    # 7. Save history                                                      #
    # ------------------------------------------------------------------ #
    history_path = exp_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history: %s", history_path)

    # ------------------------------------------------------------------ #
    # 8. Load best checkpoint and evaluate on test set                     #
    # ------------------------------------------------------------------ #
    trainer.load_best_checkpoint()
    logger.info("Evaluating on test set …")
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    results = evaluate_model(
        model, test_loader, device,
        use_amp=use_amp,
        class_names=class_names,
        num_classes=cfg.dataset.num_classes,
    )
    print_metrics_table(results)

    # Append training meta-info
    results["train_time_seconds"] = train_time
    results["model_name"]  = cfg.model.name
    results["dataset_name"] = cfg.dataset.name

    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved test results: %s", results_path)

    logger.info("Done! Best val_acc = %.4f | Test acc = %.4f",
                max(history["val_acc"]), results["accuracy"])


if __name__ == "__main__":
    main()
