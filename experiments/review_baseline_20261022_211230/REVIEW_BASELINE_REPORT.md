# CryptoViT: Standard ViT Baseline Training Report for Monday Review

**Date of Report:** 22 October 2026  
**Target Review Date:** Monday, 26 October 2026  
**Project:** CryptoViT (Chaos-Assisted Vision Transformer for Privacy-Preserving Classification)  
**Author / ML Engineer:** Senior ML Assistant  
**Git Branch:** `master` | **HEAD:** `8000df8755926557159a7a520197b6e8d4d44959`

---

## 1. Executive Summary

This report documents the environment setup, dataset verification, unit test results, and empirical training of the standard **Vision Transformer (`VisionTransformer`)** baseline on CIFAR-10. 

No CryptoViT modifications or ablation scripts were executed. A dedicated, opt-in script (`scripts/train_review_baseline.py`) was employed to run a reproducible review training loop without altering default project configurations or existing workflow files.

### Key Outcomes:
- **Environment:** Dedicated Python 3.11 virtual environment (`.venv`) established with PyTorch 2.5.1 + CUDA 12.1.
- **Hardware Utilized:** NVIDIA GeForce RTX 2050 Laptop GPU (4GB VRAM) running CUDA acceleration and automatic mixed precision (AMP).
- **Unit Tests:** 71/71 tests passing in `tests/unit` (100% pass rate).
- **Dataset:** CIFAR-10 successfully downloaded and verified (50,000 train images, 10,000 official test images across 10 classes).
- **Training Evidence:** 5 real epochs completed on a strictly disjoint 5,000-sample training subset and 1,000-sample validation subset.
- **Accuracy Progression:** Training accuracy improved from **19.52%** to **31.62%**; validation accuracy improved from **23.20%** to **31.10%**.
- **Final Test Set Evaluation:** Evaluated once against the untouched official 10,000-image test set, achieving **31.39% accuracy** (random guess baseline is 10.00%).

---

## 2. Environment & Hardware Configuration

- **OS:** Windows 11 (build 10.0.26100)
- **Python Version:** 3.11.9 (`C:\Program Files\Python311\python.exe`)
- **Virtual Environment:** `.\.venv` (isolated, ignored by git)
- **Key Installed Packages:**
  - `torch`: 2.5.1+cu121
  - `torchvision`: 0.20.1+cu121
  - `timm`: 1.0.30
  - `omegaconf`: 2.4.0
  - `einops`: 0.8.2
  - `scikit-learn`: 1.9.1
  - `matplotlib`: 3.11.2
  - `pytest`: 9.1.1
- **Detected GPU:** NVIDIA GeForce RTX 2050 (Driver: 591.74 / 32.0.15.9174, VRAM: 4096 MB)
- **Compute Device:** `cuda`

---

## 3. Dataset Verification (CIFAR-10)

- **Storage Location:** `data/raw/cifar-10-python.tar.gz` and `data/raw/cifar-10-batches-py/`
- **Class Count:** 10 (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`)
- **Verification Run (`scripts/phase1_dataset_check.py`):**
  - Training raw count: 50,000 images
  - Official test raw count: 10,000 images
  - Balanced distribution: 1,000 samples per class in test set
  - Normalization verification:
    - Empirical computed mean: `[0.4914, 0.4822, 0.4465]`
    - Empirical computed std: `[0.2413, 0.2378, 0.2564]`

---

## 4. Test Suite Execution

All unit tests inside `tests/unit` were run using the project virtual environment interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

**Results:**
- `tests/unit/test_encryption.py`: **29 passed** (hyperchaotic map, permutation, DNA encoding/decoding)
- `tests/unit/test_vit.py`: **18 passed** (PatchEmbedding, Attention, TransformerBlock, VisionTransformer forward/backward)
- `tests/unit/test_config.py`: **9 passed** (OmegaConf hierarchy, overrides, path resolutions)
- `tests/unit/test_seed.py`: **5 passed** (deterministic torch/numpy seeds)
- `tests/unit/test_dataset.py`: **10 passed** (data transforms, normalizations, class names)
- **Total:** **71 passed in 6.76s (0 failures, 0 errors)**

---

## 5. Baseline Training Specification & Split Hygiene

To ensure CPU/GPU friendliness for the review milestone while preventing data leakage:
- **Model:** Standard `VisionTransformer` (`vit_tiny`) from `src/cryptovit/models/vit.py`
  - Input resolution: 96 × 96
  - Patch size: 16 × 16 (yielding $(96/16)^2 = 36$ patches + 1 `[CLS]` token = 37 tokens)
  - Embedding dimension: 96
  - Depth: 4 Transformer encoder layers
  - Attention heads: 3 (head dimension = 32)
  - Parameter count: **525,994 (0.53M parameters)**
- **Data Split Protocol:**
  - Official 50,000 training set partitioned using fixed `seed=42`:
    - Training subset: **5,000 samples**
    - Validation subset: **1,000 samples**
    - **Overlap verification:** Disjoint verification confirmed $\text{train\_indices} \cap \text{val\_indices} = \emptyset$ (0 overlap).
  - Official 10,000 test set: Held out strictly; never used during training, tuning, or epoch selection.
- **Optimization:**
  - Optimizer: AdamW (`lr=3e-4`, `weight_decay=0.05`)
  - Learning rate schedule: Linear warmup (1 epoch) followed by cosine annealing decay to `1e-6`
  - Loss: Cross-entropy with `label_smoothing=0.1`
  - Mixed precision: PyTorch AMP enabled (`float16`)
  - Epochs: 5

---

## 6. Empirical Training History

Training completed in **177.32 seconds** (~2.95 minutes).

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2.1744 | 19.52% | 2.1082 | 23.20% | 3.000e-4 |
| 2 | 2.0819 | 24.78% | 2.0511 | 28.00% | 2.562e-4 |
| 3 | 2.0182 | 28.28% | 1.9878 | 28.80% | 1.505e-4 |
| 4 | 1.9796 | 30.52% | 1.9619 | 30.60% | 4.479e-5 |
| 5 | **1.9591** | **31.62%** | **1.9555** | **31.10%** | 1.000e-6 |

**Observation:**
Both training and validation losses decreased monotonically across all 5 epochs, while training and validation accuracies increased steadily from ~19% to ~31%. This confirms that the model is actively learning discriminative visual representations and not memorizing or diverging.

---

## 7. Official Test Set Evaluation

The checkpoint from Epoch 5 was selected on the basis of best validation accuracy (31.10%) and evaluated once on the 10,000-image official test set:

- **Test Accuracy (Top-1):** **31.39%** (3,139 / 10,000 correct)
- **Macro Precision:** 0.3123
- **Macro Recall:** 0.3139
- **Macro F1 Score:** 0.2945
- **Throughput:** 435.6 images/sec
- **Peak GPU Memory:** 82.2 MB

### Per-Class Test Performance:
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| airplane | 0.3932 | 0.4710 | 0.4286 |
| automobile | 0.3428 | 0.2410 | 0.2830 |
| bird | 0.2720 | 0.0340 | 0.0604 |
| cat | 0.2477 | 0.1890 | 0.2144 |
| deer | 0.3808 | 0.1980 | 0.2605 |
| dog | 0.2903 | 0.3010 | 0.2955 |
| frog | 0.3167 | 0.4520 | 0.3725 |
| horse | 0.2679 | 0.3110 | 0.2878 |
| ship | 0.3369 | 0.5360 | 0.4137 |
| truck | 0.2753 | 0.4060 | 0.3281 |

---

## 8. Saved Artifacts and File Locations

All run outputs were stored inside a dedicated timestamped directory and confirmed ignored by Git:

**Output Directory:**  
`experiments/review_baseline_20261022_211230/`

1. **Model Checkpoint:**  
   `experiments/review_baseline_20261022_211230/checkpoints/checkpoint_epoch0005_acc0.3110.pt` (Size: 2.11 MB)
2. **Training History Log:**  
   `experiments/review_baseline_20261022_211230/history.json`
3. **Loss & Accuracy Plot:**  
   `experiments/review_baseline_20261022_211230/loss_acc_curves.png`
4. **Official Test Evaluation Metrics:**  
   `experiments/review_baseline_20261022_211230/test_metrics.json`
5. **Confusion Matrix Plot:**  
   `experiments/review_baseline_20261022_211230/confusion_matrix.png`
6. **Environment & Run Metadata:**  
   `experiments/review_baseline_20261022_211230/run_metadata.json`
7. **Sample Predictions:**  
   `experiments/review_baseline_20261022_211230/sample_predictions.json`
8. **Detailed Execution Console Log:**  
   `experiments/review_baseline_20261022_211230/training.log`

---

## 9. Verification of Reload and Inference

The saved checkpoint was reloaded into a newly initialized `VisionTransformer` instance:
- `torch.load` successfully restored all model parameters.
- Inference on a live batch of test images verified output tensor shape `(64, 10)`.
- Class predictions mapped correctly to CIFAR-10 class labels (`airplane`, `automobile`, `ship`, etc.).

---

## 10. Scientific Limitations and Context for Monday Review

1. **Limited Training Volume:** The model was trained on 5,000 samples for 5 epochs (total 25,000 sample-passes) to serve as a fast, verified proof-of-pipeline for the review. Standard ViTs trained from scratch on CIFAR-10 typically require 100–300 epochs and the full 50,000 images or pre-training on ImageNet to reach 75%–85%+ accuracy because ViTs lack inductive bias (like translation equivariance found in CNNs).
2. **Performance vs Random Chance:** A random guess baseline on CIFAR-10 is **10.00%**. Achieving **31.39%** test accuracy after only 5 epochs on a 10% data subset confirms that the pipeline, attention blocks, loss functions, and optimizers are functioning as mathematically intended.
3. **CryptoViT Scope:** This milestone strictly validated the unencrypted plaintext ViT baseline. Encryption mechanisms (hyperchaotic map, DNA diffusion) remain ready in unit tests and will be trained in the next phase.

---

## 11. Command to Reproduce

To reproduce this exact training run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/train_review_baseline.py --epochs 5 --batch-size 64 --image-size 96 --train-samples 5000 --val-samples 1000 --seed 42
```

