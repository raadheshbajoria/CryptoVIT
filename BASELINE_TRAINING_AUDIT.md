# CryptoViT Standard ViT Baseline — Independent Training Audit

**Audit date:** 2026-10-22  
**Scope:** Existing review-baseline run and saved checkpoint only. No training was performed. No source code, tests, configuration, existing report, results, or checkpoint was changed. This report is the only file created by this audit.

## 1. Executive verdict

**Verdict: Partially verified.**

The saved evidence strongly supports that the standard ViT baseline actually trained for five epochs: the log records five train/validation epochs and checkpoint saves; the checkpoint stores epoch 5, validation accuracy 0.311, model and optimizer state; every one of its 56 optimizer parameter states records step 395; and every one of its 56 model tensors differs from a fresh same-architecture initialization. The exact checkpoint loaded into a fresh standard ViT with strict key matching, and real CIFAR-10 test images produced finite, correctly shaped logits in evaluation/no-gradient mode.

An independent, full-precision pass over the local official CIFAR-10 test split reproduced **3,139 correct of 10,000, accuracy 31.39%, and macro recall 0.3139**. Macro precision and F1 were very close but not identical to the saved AMP-run values. The saved precision, recall, F1, and accuracy are internally consistent with the saved confusion matrix. The original review run enabled CUDA AMP; the independent full-set pass did not, so this is a plausible explanation for small prediction-level/metric differences, but the cause was not independently isolated.

The training/validation/test partitioning is sound in the script: train and validation are disjoint subsets from the official training split; the official test dataset is constructed separately and is not used for model selection. The deterministic first 20 train and validation indices in metadata exactly match a reconstruction from the saved seed and script logic. The full indices were not saved, so the complete membership is supported by the script and reproducible reconstruction, not independently compared to a stored complete index manifest.

This is suitable for a **carefully qualified preliminary baseline demonstration**: the weights load, inference works, and the held-out accuracy was independently reproduced. It is not a mature benchmark or evidence of CryptoViT performance. Five epochs on 5,000 training samples is a short/subset baseline, and the macro precision/F1 discrepancy should be disclosed. The baseline does not establish any encryption security property.

## 2. Repository state, environment, and hardware

### Git state at audit start

- Branch: `master`
- HEAD: `8000df8755926557159a7a520197b6e8d4d44959`
- The worktree was **not clean** when this audit began. Pre-existing state was:
  - Modified tracked `.gitignore` (two added lines ignore `experiments/review_baseline_*/`).
  - Untracked `scripts/train_review_baseline.py`.
  - Untracked `CRYPTOVIT_EXISTING_IMPLEMENTATION_AUDIT.md` from the previous task.
- These pre-existing changes were preserved. This audit did not edit them.
- The new `BASELINE_TRAINING_AUDIT.md` is the only file created by this task.

The experiment directory is ignored by the pre-existing `experiments/review_baseline_*/` rule. The review script itself is not ignored; it was already an untracked user/worktree file at the start. `.venv/` is ignored by the existing environment rules.

### Current `.venv` runtime observed

Checked using `.venv\Scripts\python.exe`, without installing or changing packages:

| Item | Observed value |
|---|---|
| Python | 3.11.9, Windows `win32` |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121; import succeeded |
| PyTorch CUDA build | 12.1 |
| `torch.cuda.is_available()` | `True` |
| GPU | NVIDIA GeForce RTX 2050 |
| GPU memory reported by PyTorch | 4.0 GiB |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| pytest | 9.1.1 |
| OmegaConf | 2.4.0 |
| Pillow | 12.3.0 |

The runtime agrees with the run metadata for Python, PyTorch, CUDA availability, and GPU name. `run_metadata.json` does **not** record the torchvision version or driver version. The driver version stated in the existing review report was not independently queried in this audit. The operating system was verified as Windows, but the exact OS build was not independently checked.

### Reported artifacts present at audit start

| Artifact | Present | Size |
|---|---:|---:|
| `scripts/train_review_baseline.py` | Yes | 13,878 bytes |
| `history.json` | Yes | 579 bytes |
| `test_metrics.json` | Yes | 2,728 bytes |
| `run_metadata.json` | Yes | 6,148 bytes |
| `sample_predictions.json` | Yes | 1,238 bytes |
| `loss_acc_curves.png` | Yes | 93,420 bytes |
| `confusion_matrix.png` | Yes | 118,911 bytes |
| `training.log` | Yes | 7,162 bytes |
| `REVIEW_BASELINE_REPORT.md` | Yes | 9,373 bytes |
| `checkpoints/checkpoint_epoch0005_acc0.3110.pt` | Yes | 6,384,302 bytes |
| `data/raw/cifar-10-python.tar.gz` | Yes | 170,498,071 bytes |
| `data/raw/cifar-10-batches-py/` | Yes | Directory |

The artifact and metadata files were inspected before loading checkpoint weights. The checkpoint is a PyTorch ZIP-format archive, with `data.pkl`, byte-order metadata, and tensor-storage entries. It was subsequently loaded with `weights_only=True` for audit checks.

## 3. Training script review

The inspected entry point is the pre-existing untracked [`scripts/train_review_baseline.py`](scripts/train_review_baseline.py). Its default run parameters are five epochs, batch size 64, 96×96 input, 5,000 training examples, 1,000 validation examples, seed 42, and learning rate `3e-4` ([argument defaults](scripts/train_review_baseline.py#L184)).

### Model and training mechanics

- The script sets the model name to `vit_tiny`, with 96-dimensional embeddings, depth 4, 3 heads, patch size 16, 10 outputs, and 96×96 input ([configuration](scripts/train_review_baseline.py#L198)).
- It calls the repository model factory, `build_model(cfg)`, not an unrelated implementation ([`build_model`](src/cryptovit/models/__init__.py#L46), [script call](scripts/train_review_baseline.py#L250)). The factory returns the standard `VisionTransformer` for the `vit_tiny` name.
- It calls `Trainer.fit(train_loader, val_loader)` ([script](scripts/train_review_baseline.py#L261)). `Trainer` runs image batches through the model, calculates cross-entropy, invokes scaled backward propagation, and steps AdamW ([backward/update](src/cryptovit/training/trainer.py#L314)).
- Validation is run separately each epoch. The trainer records history and saves new best checkpoints when validation accuracy improves ([validation and checkpoint selection](src/cryptovit/training/trainer.py#L270)).
- The script reloads the best checkpoint before evaluating the official test loader ([script](scripts/train_review_baseline.py#L276)).
- Checkpoints serialize epoch, model state, optimizer state, validation accuracy, and additional training accuracy ([`CheckpointManager.save`](src/cryptovit/training/trainer.py#L77)).

### Output path safety

The script creates `experiments/review_baseline_<timestamp>/` with a timestamp accurate to one second and `exist_ok=True` ([script](scripts/train_review_baseline.py#L215)). This run wrote into the named directory and all expected artifacts are present. The script does not explicitly refuse an existing directory with the same timestamp, so a same-second collision could overwrite files. This is a preventive robustness risk; there is no evidence it happened to this run.

## 4. Dataset and split-integrity findings

`create_subset_dataloaders` loads the CIFAR-10 training split with a stochastic training transform and again with deterministic evaluation transforms; it loads the CIFAR-10 official test split separately, all with `download=False` ([function](scripts/train_review_baseline.py#L55)). It generates a seeded permutation over the 50,000 official training indices, assigns the first 5,000 to train and the next 1,000 to validation, and asserts no overlap ([split construction](scripts/train_review_baseline.py#L84)). Validation is wrapped around the evaluation-transform dataset, not the augmented training-transform dataset. The official test loader wraps the separately loaded test partition ([test loader](scripts/train_review_baseline.py#L115)).

I reconstructed the split in memory using the saved seed 42 and the script’s `torch.randperm(50000, generator=...)` algorithm:

- Train count / unique indices: **5,000**
- Validation count / unique indices: **1,000**
- Train/validation intersection: **empty**
- First 20 train indices: **exact match** with the metadata sample
- First 20 validation indices: **exact match** with the metadata sample

Metadata stores only the first 20 indices from each subset, not all 6,000. Thus exact full-index equality cannot be checked against a saved full manifest. The script is deterministic and the recorded samples match its reconstruction.

The official test split is not passed to `Trainer.fit`; it is used after best-checkpoint selection. The script then re-reads the first test batch after full test evaluation to write up to 10 sample predictions. This is a small repeated inference on the initial test examples, not a use for training or checkpoint/model selection.

### Preprocessing

Training preprocessing uses `build_train_transform`: resize to 96×96, optional random crop with padding 4 and horizontal flip, tensor conversion, and CIFAR-10 normalization ([transform](src/cryptovit/datasets/cifar_dataset.py#L59)). Validation and test use `build_eval_transform`: resize to 96×96, tensor conversion, and the same configured normalization ([transform](src/cryptovit/datasets/cifar_dataset.py#L114)). The independent evaluation used the same evaluation-transform factory. No preprocessing discrepancy was observed.

## 5. Artifact and training-history verification

### History checks

All five history arrays (`train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`) have exactly five numeric, finite values. Their changes are consistent with the log:

- Train loss: 2.17436 → 1.95907 (decrease 0.21529)
- Validation loss: 2.10817 → 1.95546 (decrease 0.15270)
- Train accuracy: 0.1952 → 0.3162
- Validation accuracy: 0.2320 → 0.3110
- Learning rate: `3e-4` → `1e-6`
- Highest recorded validation accuracy: **0.311 at epoch 5**

The history, metadata, test metrics and checkpoint all identify epoch 5 / validation accuracy 0.311 as the selected best. The checkpoint filename agrees.

### Log evidence

`training.log` records the CUDA device, 79 train batches, 4 validation batches, 40 test batches, model construction and 525,994 parameters. It records one train/validation summary per epoch and a best-checkpoint save at each improving epoch. It then records reloading the epoch-5 best checkpoint, the full test-set evaluation, saved metrics, and a separate checkpoint reload/inference step.

The log does not contain every individual optimizer-step line. The loaded checkpoint adds stronger corroboration: all 56 optimizer parameter states have step count 395. With 79 training batches per epoch, batch size 64, five epochs, and gradient accumulation 1, this is consistent with 79×5 successful optimizer steps. Every one of the 56 model tensors also differs from a newly initialized same-architecture instance. Together with changing losses/accuracies and the log, this strongly supports actual training rather than only file creation.

### Prediction artifact

All 10 sample records use valid CIFAR-10 indices 0–9, and the saved class strings match the CIFAR-10 label order. However, the JSON contains only `pred_idx`, `pred_class`, `true_idx`, and `true_class`; it has **no logits, probabilities, confidence, or other class-score output**. Therefore class-score validity is not assessable from that artifact.

## 6. Checkpoint inspection and independent reload

The exact named checkpoint was loaded in the `.venv` with `torch.load(..., map_location="cpu", weights_only=True)`. Its top-level keys are `epoch`, `model_state_dict`, `optimizer_state_dict`, `train_acc`, and `val_acc`.

| Check | Observed result |
|---|---|
| Recorded epoch / validation accuracy | 5 / 0.311 |
| Model state entries | 56 tensors |
| Classifier weight shape | `(10, 96)` |
| Checkpoint architecture indicators | Standard `pos_embed` and `patch_embed.projection`; no `pos_encoding` CryptoViT keys |
| Strict load to fresh recorded `vit_tiny` `VisionTransformer` | Succeeded |
| Missing keys / unexpected keys | None / none |
| Model evaluation mode | `model.training == False` after `eval()` |
| State tensors different from fresh initialization | 56 of 56 |
| Adam state | 56 parameter states; step 395 for each |

A real local CIFAR-10 test batch of shape `(16,3,96,96)` produced logits of shape `(16,10)`. All logits were finite; predicted indices were within 0–9; inference ran inside `torch.inference_mode()` and logits did not require gradients.

These checks prove that the checkpoint is structurally loadable and usable for inference with the recorded standard ViT architecture. Parameter differences by themselves do not prove correct training, but here they align with the optimizer state, history, logs, and independent evaluation.

## 7. Independent held-out evaluation

The evaluation implementation sets the model to eval mode, uses no-grad inference, applies optional autocast, chooses top-1 predictions by `argmax`, and computes accuracy, macro precision, macro recall, macro F1, and confusion matrix with scikit-learn ([`evaluate_model`](src/cryptovit/evaluation/metrics.py#L42)). With CIFAR-10’s test split containing all ten labels, the macro averages use the ten classes. The label names and order match torchvision CIFAR-10.

I loaded the exact checkpoint and local CIFAR-10 official test split with `download=False`, using the saved 96×96 evaluation preprocessing, and evaluated the full set once in memory. This pass used full-precision model inference on CUDA (no AMP); it did not save or overwrite any metrics file.

| Metric | Existing `test_metrics.json` | Independent FP32 pass | Difference |
|---|---:|---:|---:|
| Evaluated examples | 10,000 | 10,000 | 0 |
| Correct predictions | 3,139 (from confusion-matrix diagonal) | 3,139 | 0 |
| Accuracy | 0.313900 | 0.313900 | 0 |
| Macro precision | 0.312347800 | 0.312595287 | +0.000247487 |
| Macro recall | 0.313900 | 0.313900 | 0 |
| Macro F1 | 0.294463874 | 0.294470314 | +0.000006440 |

The saved confusion matrix sums to 10,000 and its diagonal sums to 3,139. Recomputing macro precision, recall, and F1 from that matrix reproduces the values in `test_metrics.json` exactly to the saved precision. The log also records these same saved values.

**Interpretation:** the reported headline accuracy is independently reproduced exactly; count and recall also match. Precision and F1 are not bit-for-bit reproduced in the independent full-precision pass. Since the original evaluation script sets AMP true when CUDA is used and the independent full-set pass was FP32, different near-boundary predictions are a plausible cause; that explanation is not proven without another full-set AMP run. Accordingly, the reported precision/F1 are internally supported by the saved confusion matrix and log, but were not independently reproduced exactly. The overall saved test metrics are credible but should be presented with this limitation, rather than described as fully identical independent reproduction.

## 8. Unit-test verification

The requested test command was run using the project virtual environment and with bytecode/cache generation disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pytest tests/unit -q -p no:cacheprovider
```

**Observed result: 71 passed in 6.28 seconds; no failures or collection errors.**

Breakdown:

| Test file | Passed |
|---|---:|
| `tests/unit/test_config.py` | 9 |
| `tests/unit/test_dataset.py` | 10 |
| `tests/unit/test_encryption.py` | 29 |
| `tests/unit/test_seed.py` | 5 |
| `tests/unit/test_vit.py` | 18 |
| **Total** | **71** |

No network-dependent integration tests were run.

## 9. Status summary

Status terms are used as requested: **Verified by direct execution**, **Supported by artifacts but not independently reproduced**, **Implemented but not verified**, **Failed verification**, and **Not assessed**.

| Finding | Status | Evidence |
|---|---|---|
| Five training epochs completed | **Supported by artifacts but not independently reproduced** | Five epoch summaries, improving histories, checkpoints and optimizer step count 395; audit did not retrain. |
| Standard ViT model architecture used | **Verified by direct execution** | Exact checkpoint strictly loaded into factory-built standard ViT; classifier shape `(10,96)`, no CryptoViT positional keys. |
| Train/validation disjointness | **Verified by direct execution** | Reconstructed 5,000/1,000 split from seed; zero overlap and metadata sample matches. |
| Official test split excluded from training/selection | **Supported by artifacts but not independently reproduced** | Script constructs separate test dataset and only evaluates it after best-validation checkpoint reload; log agrees. |
| Checkpoint can be loaded and inferred | **Verified by direct execution** | Strict state load; finite `(16,10)` logits on real CIFAR images with no gradients. |
| Accuracy 31.39%, 3,139/10,000 | **Verified by direct execution** | Independent full-precision held-out pass exactly matched count and accuracy. |
| Saved precision and F1 | **Supported by artifacts but not independently reproduced** | They agree with saved confusion matrix and log; independent FP32 results differ slightly. |
| Saved class-score values in sample JSON | **Failed verification** | Artifact contains class indices/names only, no scores or probabilities. |
| CUDA AMP evaluation’s exact macro precision/F1 | **Not assessed** | Independent complete test-set pass used FP32; original run used AMP. |
| CUDA driver version / exact Windows build during original run | **Not assessed** | Not captured in metadata and not queried as part of this audit. |
| Encryption security | **Not assessed** | This audit concerns the standard baseline, not cipher analysis. No security claim follows from classification results. |

## 10. Correctness issues and unexplained discrepancies

1. **AMP explains a plausible, not confirmed, small metric discrepancy.** The independent full-set evaluation used FP32 while the original CUDA run used AMP. Accuracy and recall match; macro precision differs by about `2.47e-4` and F1 by about `6.44e-6`. Do not claim exact macro-metric reproduction without evaluating under the original precision mode.
2. **Sample predictions do not store class scores.** The existing report/artifact can demonstrate predicted/true labels but cannot substantiate confidence or probability values.
3. **Only sample split indices are preserved.** The 20-element index samples match a full deterministic reconstruction; saving all selected indices would improve auditability for future runs.
4. **Timestamped output directory is not collision-proof.** The script uses second-level timestamps and `exist_ok=True`; a same-second collision could overwrite output files. This was not observed in the existing run.
5. **Environment metadata is incomplete.** The run records Python and PyTorch but not torchvision or driver versions; the exact historical environment cannot be wholly reconstructed from metadata.
6. **Provenance remains limited.** Logs, JSON files, optimizer state and checkpoint all corroborate each other, but ordinary local artifacts are not signed or otherwise tamper-evident. The audit establishes consistency and executable behavior, not cryptographic provenance.

No defect was found in the train/validation/test separation of this specific review baseline. These findings do not certify unrelated CryptoViT encrypted-data or ablation workflows.

## 11. Is this baseline suitable for preliminary academic review?

**Yes, with qualifications.** It is demonstrably a loadable, trained standard ViT baseline with a held-out CIFAR-10 accuracy that was independently reproduced at 31.39%. Its architecture and split are documented, and all 71 unit tests currently pass in the project `.venv`.

Present it as an early, five-epoch, 5,000-sample baseline—not as a converged or state-of-the-art classifier. State that the full-precision audit reproduced accuracy/recall exactly but had a small precision/F1 mismatch versus the saved AMP evaluation. The current sample-prediction JSON has no confidence scores. Do not infer or claim encryption security, encrypted inference quality, or superiority of CryptoViT from this baseline.

Whether it is ready for a **live Monday demo** depends on the demo path: this audit verified direct checkpoint inference only. It did not launch the API or dashboard, and the training script’s built-in inference check occurs only after a training run. Thus the saved checkpoint is ready for a controlled inference demonstration using a known compatible loader, but an end-to-end API/dashboard demo is **not verified here**.

## 12. Before training CryptoViT for scientific comparison

1. **Keep this standard baseline fixed** as the plaintext reference and record its exact config, checkpoint hash, split seed, and dependency versions.
2. **Use disjoint encrypted train/validation subsets** from CIFAR’s official training data and reserve the official test set for final evaluation only. The existing ablation loader reviewed previously used the test loader as validation and test; fix that before interpreting CryptoViT comparison results.
3. **Wire the key embedding consistently** through CryptoViT training, validation, test evaluation, and inference. The generic trainer/evaluator currently call models with images only; CryptoViT then falls back to unconditioned positional embeddings, while the API can supply a key embedding.
4. **Make planned ablation variants genuinely distinct** and document exact model dimensions/hyperparameters; in the existing ablation script, variants D and E build the same model.
5. **Record complete split indices, exact software/hardware environment, precision mode, config, checkpoint, and results** for each run. Avoid using official test metrics to tune architecture or hyperparameters.
6. **Add tests for encrypted split integrity, key-conditioning, and metric calculations** before expensive training. Run network/data integration tests only when intentionally requested.
7. **Treat security as a separate research evaluation.** This baseline is plaintext classification and provides no evidence that the custom image transformation is cryptographically secure.

## 13. Commands and checks executed

### Read-only inspection

- Git branch, HEAD, status, diff, tracked/untracked file state, `.gitignore`, script, configs and artifact sizes.
- Read-only artifact inspection of JSON, logs, prior report, and checkpoint ZIP archive directory.
- `.venv` checks of Python, PyTorch, torchvision, CUDA availability/build, GPU name and memory.
- In-memory split reconstruction, history/finite-value checks, confusion-matrix consistency checks, and checkpoint optimizer-state inspection.

### Test suite

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pytest tests/unit -q -p no:cacheprovider
```

Result: **71 passed in 6.28 seconds**.

### Checkpoint and inference

Used `.venv\Scripts\python.exe` with an inline, in-memory Python snippet. It loaded the exact checkpoint with `weights_only=True`, reconstructed the saved ViT dimensions, loaded the model with `strict=True`, compared it to fresh initialization, checked optimizer step state, loaded a real CIFAR-10 batch with `download=False`, and ran inference in `torch.inference_mode()`. It wrote no files.

### Full test-set evaluation

Used `.venv\Scripts\python.exe` with an inline Python snippet to load the exact checkpoint and local CIFAR-10 official test split with `download=False`, use the repository’s `build_eval_transform`, run one full-set FP32 pass on CUDA, and calculate count, accuracy, macro precision, macro recall and macro F1. It wrote no files. Result: **10,000 samples, 3,139 correct, accuracy 0.3139, precision 0.312595287, recall 0.3139, F1 0.294470314**.

No training, installation, dataset download, integration test, API, or dashboard launch was performed. No checkpoint, metric, log, image, or existing report was overwritten.

