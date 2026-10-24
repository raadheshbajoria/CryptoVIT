# CryptoViT — Work Log

All times are local (IST, UTC+05:30), 2026-10-24 unless stated. Newest entries are at the bottom.
Everything listed here was actually done; numbers are copied from real command output.

## Part 1 — Setup, audit and code improvements (before the overnight run)

| Time | What happened |
|---|---|
| 22:15 | Starting point: branch `ananjan` at `af61e56`, clean tree, 115 unit tests passing. |
| ~22:30 | Found the venv had **CPU-only PyTorch** (`2.14.0+cpu`). Installed the CUDA build (`2.11.0+cu128`); first attempt failed a hash check (corrupt download), second succeeded. GPU detected: RTX 3050 Laptop, 4 GB. |
| ~22:40 | Found `data/raw/cifar-10-python.tar.gz` was **truncated** (3.8 MB instead of 170 MB; `tar` reported "Truncated input file"). Resumed the download with `curl -C -` (the dataset server was slow, 26-100 kB/s, and dropped the connection once). |
| 22:54 | Commit `0629840` — trainer: LR schedule sized by optimiser steps (fixes gradient accumulation), no weight decay on bias/norm params, loss/acc accumulated on-device. |
| 22:15–23:00 | **Read-only audit** of the whole pipeline (encryption, datasets, models, training, evaluation) plus probes. Key findings: ablation used the **test set for model selection** (variants B–E); the key-aware positional encoding was **never given a key** in training; variant E was identical to D; README claim "NPCR > 99%" was not supported (measured plaintext-sensitivity NPCR 1.6–8%); 32x32 entropy cannot exceed ~7.82; all colour channels shared one permutation/keystream. |
| 23:02 | Commit `d8c998f` — encryption 140x faster (53 ms -> 0.37 ms per 32x32 image) by caching key streams and vectorising the chained XOR. Output verified bit-identical to the original with equivalence tests. |
| 23:05 | Commit `0e7933c` — `Trainer` and `evaluate_model` now pass the key embedding to the model; per-class metrics keep a fixed length; headless matplotlib backend for tests (a plotting test was flaky). |
| 23:07 | Commit `79a2fab` — ablation protocol fixed: shared seeded train/val split for plaintext and encrypted runs, test set used only for the final score, no crop/flip on ciphertext, normalisation stats from encrypted training images, variant D trains with the key embedding, duplicate variant E removed. |
| 23:11 | Commit `32d2af8` — measured GPU memory (batch 32: 1.75 GB; 64: 3.4 GB; 128: 6.7 GB which spills into shared memory and runs 10–70x slower), so variants C/D use micro-batches of 32 with gradient accumulation. Docs corrected: `CryptoPatchEmbedding` adds ~3.04M parameters over the standard embedding. |
| 23:12 | Commit `f90cad4` — added the standard plaintext-sensitivity NPCR/UACI test and a size-aware entropy reference; dashboard relabelled; README claims corrected. |
| 23:13 | Commit `73cca02` — end-to-end smoke tests for variants A–D on synthetic data. |
| 23:17 | Commit `34767e0` — opt-in encryption **scheme v2** (per-channel permutations/keystreams, uniform Fisher–Yates shuffle). v1 stays default and bit-identical. v2 does **not** improve plaintext sensitivity (NPCR ~14% vs ~18% for v1, ideal 99.6%) because the XOR-chain diffusion is unchanged. |
| 23:2x | CIFAR-10 download finished: 170,498,071 bytes, MD5 `c58f30108f718f92721af3b95e74349a` (matches the official checksum). Extracted: 50,000 train / 10,000 test, 5,000 per class. |
| 23:18–23:20 | 1-epoch real-data smoke run of the plain ViT: 111 s/epoch, 540 MB GPU memory, val acc 38.6%, test acc 39.5% (pipeline check only, not a result). |
| 23:31 | You approved: train **A and B at 30 epochs**, handle errors myself, keep this log. |

## Part 2 — Overnight run (A = plain ViT on plaintext, B = plain ViT on encrypted, 30 epochs)

| Time | What happened |
|---|---|
| 23:32 | Commit `5ac77bb` — training made **resumable**: `checkpoints/last.pt` written atomically after every epoch (model, optimiser, LR schedule, AMP scaler, history, early-stopping state); `run_ablation.py` resumes each variant, writes `result_<variant>.json` when a variant finishes, skips finished variants, retries a failed variant once, and logs to `experiments/ablation/run.log`. A test confirms a crashed + resumed run ends with the same weights/LR as an uninterrupted run. 197 unit tests passing. |
| 23:32:54 | **Launched** `python scripts/run_ablation.py ablation.variants=A,B training.epochs=30` as a detached process (PID 41580), stdout/stderr in `experiments/ablation/`. Settings: AdamW, lr 3e-4, cosine with 5 warm-up epochs, batch 128, mixed precision, label smoothing 0.1, early-stopping patience 20, seed 42, encryption scheme v1, key `CryptoViT_Research_Key_2025`. A = plain ViT on plaintext; B = same ViT on encrypted images; both use the same 45,000 / 5,000 train/val split and the 10,000-image test set (used once, after training). |
| 23:32 | Started a keep-awake helper (PID 36108, Windows `SetThreadExecutionState`) that lives only while the training process runs, so the laptop does not idle-sleep. It does not change any power settings. Closing the lid or choosing Sleep would still stop the run. |
| 23:33:01 | Variant A started (epoch 1 of 30). |

| 23:43 | Variant A (CIFAR-10, plain ViT) reached epoch 6/30 with val acc 53.1% (about 107 s/epoch). |
| 23:45 | **Stopped the CIFAR-10 run on your instruction** (CIFAR is not relevant to the medical goal). Process tree of PID 41580 killed; GPU memory back to 74 MiB. The partial CIFAR output stays on disk in `experiments/ablation/` (not committed, not a result). `last.pt` exists, so it could be resumed, but there is no plan to. |

## Part 3 — Medical direction (DermaMNIST)

| Time | What happened |
|---|---|
| 23:4x | Goal changed to **medical image classification**. Capstone Phase 1 review is **Monday**; plan: DermaMNIST (7 skin-lesion classes), variants A–D. |
| 23:4x | Installed `medmnist` 3.0.2 into the venv with `--no-deps` (so PyTorch/NumPy were not touched). Downloaded DermaMNIST 28x28 (18.8 MB) to `data/raw/medmnist/` (git-ignored). Official splits: train 7,007 / val 1,003 / test 2,005. Train class counts: 228, 359, 769, 80, 779, **4,693**, 99 — melanocytic nevi are 67% of the training set, so plain accuracy would hide poor melanoma detection. |

| 23:50 | Commit `c31eedf` — MedMNIST support: `medmnist_dataset.py` (adapter to RGB + int labels, official splits, class names from official metadata), `config=<name>` command-line option, config validation for the new datasets. 205 tests passing. |
| 00:05 | Commit `dffe492` — medical metrics (balanced accuracy, per-class sensitivity/specificity/support, per-class + macro one-vs-rest AUC), class-weighted loss, checkpoint selection and early stopping on **balanced accuracy**, `configs/dermamnist.yaml` (stats from the training split, no crop/flip so A–D are comparable, `class_weights: balanced`). 226 tests passing. |
| 23:51–00:14 | **Real-data checks found a bug the synthetic tests could not:** evaluating the CryptoPatchEmbedding variants (C/D) at batch 256 needs ~6.5 GB (measured) and overflowed the 4 GB GPU -> `CUDA out of memory` during validation. Fixed by capping evaluation batch size to 32 for C/D (results do not depend on eval batch size). Also set `persistent_workers` on training loaders (Windows re-spawns workers every epoch otherwise). A first D smoke run took ~5 min/epoch before the OOM; after the fix D runs at ~61 s/epoch. **The cause of that one slow epoch was not conclusively identified** (a clean profile showed 0.14 s per micro-step, data loading 74 ms/batch; power was AC, 100%, Turbo plan). |
| 00:09 | A hung profiling script (my own: it was missing the `if __name__ == "__main__"` guard that Windows data-loader workers need) was killed. Not a project bug. |
| 00:15:32 | **Launched the DermaMNIST run:** `python scripts/run_ablation.py config=dermamnist ablation.variants=A,B,C,D training.epochs=60 paths.experiments_dir=experiments/dermamnist` (PID 42800). Output in `experiments/dermamnist/ablation/` (kept apart from the abandoned CIFAR output so a resume cannot pick up the wrong checkpoint). Settings: 60 epochs, early-stopping patience 20, AdamW lr 3e-4, 5 warm-up epochs, effective batch 128 (micro-batch 32 x 4 for C/D), mixed precision, label smoothing 0.1, balanced class weights, checkpoint chosen by validation balanced accuracy, seed 42, scheme v1, official splits (train 7,007 / val 1,003 / test 2,005; test used once per variant after training). Keep-awake helper PID 18456. |

| 00:20–00:40 | **API and dashboard updated for the skin-lesion task** (while training ran on the GPU; all testing was done on CPU). New module `src/cryptovit/inference/pipeline.py` loads the best checkpoint of each variant A–D, applies the same preprocessing as training (plaintext statistics for A, statistics of the encrypted training images for B/C/D, cached in `encrypted_stats.json`), and classifies. `api/main.py` now serves class names from the dataset, `/compare` (all variants for one image), `/results` (test results from `result_*.json`) and reports which models are trained; config is selected with the environment variables `CRYPTOVIT_CONFIG=dermamnist` and `CRYPTOVIT_EXPERIMENTS=experiments/dermamnist`. |
| 00:33 | Parity check: for 150 DermaMNIST test images the API's plaintext predictions were **identical (150/150)** to the training/evaluation pipeline, using the partially trained variant A checkpoint. (The same check for the encrypted models is still to do once B/C/D are trained.) |
| 00:36 | Dashboard changes: medical title and "not a medical device" banner; key box and a "wrong key" box; demo-image picker (`demo_images/`); prediction table with the plaintext model next to the encrypted-image models; decrypt buttons for the correct and a wrong key; results table (balanced accuracy, accuracy, AUC, melanoma sensitivity) that appears when results exist. Tested headless against a live API. |
| 00:37 | **My mistake, fixed:** replacing a deprecated Streamlit argument with a Windows PowerShell text command re-encoded `dashboard/app.py` with the wrong character set and garbled every emoji and symbol. Reversed exactly with a script (character by character), verified, and re-tested. From then on only the editor tool or UTF-8-safe scripts are used for text files. |
| 00:41 | On your request the dashboard was made **plain text** (no emojis or symbols). `tests/unit/test_dashboard.py` fails if any non-ASCII character returns, and also runs the whole dashboard flow with a faked API (5 tests). |
| 00:41 | Provisional demo images exported to `demo_images/` (test-set indices: melanoma 2, benign nevus 0, basal cell carcinoma 10). **These are placeholders**; the final choice will be made from the trained models' results. |

| 00:54 | Variant A (plain ViT, plaintext) finished: early stopping after 20 epochs without improvement. Held-out test (2,005 images, used once): accuracy 52.2%, balanced accuracy 53.3%, macro AUC 0.788. Variant B started; at epoch 6/60 its validation balanced accuracy was 14.2% (chance for 7 classes is 14.3%) - early, still in warm-up. |
| 01:01 | Dashboard simplified on your request: removed the university line, the Dataset/Device/Trained models/Classes row and the footer. The per-class chart is now **horizontal** (class names on the y axis, horizontal axis labels, percentage in each label, sorted by value, melanoma in orange) so nothing needs tilting to read. Checked visually in the browser pane; tests updated (5 dashboard tests). |


## Part 4 - Block-wise encryption (Option 2)

| Time | What happened |
|---|---|
| 01:03 | Observation: variant B (plain ViT on encrypted images, per-pixel scheme) reached only about 22% validation balanced accuracy (chance 14.3%) against 53% for plaintext. You chose Option 2: add a **block-wise "learnable" scheme**. |
| 01:07 | Wrote `src/cryptovit/encryption/block_scheme.py` (commit `cbd1f66`) as a **new file** (plus 18 tests) so the encryption modules used by the running training were not touched. Scheme `block<B>` (e.g. `block2`): channel permutation, optional per-block inversion (255 - v) by key bit, one key-derived pixel shuffle inside every block, and a shuffle of the block positions. All steps are bijections (exact decryption). With B=2 on 28x28 images enlarged to 224, a block is exactly one 16x16 ViT patch (196 blocks = 196 patches). **Limits stated plainly:** no diffusion (one changed pixel changes one output pixel), pixel-value histogram preserved up to the inversion, same scrambling for every image under one key. It gives visual protection, not cryptographic strength, and trades security for classifiability. |
| next | After the current run ends: hook `block<B>` into `encrypt()/decrypt()` (the scheme name carries the block size, so config, datasets, API and dashboard need no new parameters), then train variants on `encryption.scheme=block2` and compare with the per-pixel results. |

| 01:09 | (commit `d55b8e6`) Added `configs/block2.yaml`, support for combining configs in the API (`CRYPTOVIT_CONFIG=dermamnist,block2`) and `scripts/start_demo.py` (one command that starts the API and the dashboard; `--block 2` selects the block-scheme models, `--dry-run` only prints). 5 launcher tests plus an API test. None of these files is used by the running training. |

| 01:12 | Correction: three log entries above had times I had estimated too late (they were ahead of the machine clock); they now carry the real commit times. |

| 01:22 | Started a **proxy test** on the GPU alongside the main training: a small ViT (embed 96, depth 4, 2x2 patches) on native 28x28 DermaMNIST, identical settings for plaintext, per-pixel (v1) and block2, 2 seeds each, to see whether block2 is easier to learn. Not a final result (different model and resolution from the main pipeline). |
| 01:36 | Variant B (full-size plain ViT, per-pixel encryption) finished the main run: test accuracy 12.9%, balanced accuracy 19.4%, macro AUC 0.573 (chance balanced accuracy is 14.3%). Variant C started. |
| 01:40 | You reported the laptop was very hot. GPU read **82 C at 100% load**. **I stopped the proxy test** (my own extra load) before the block2 runs happened. Partial proxy results (test balanced accuracy): plaintext 47.5% and 44.6%; per-pixel 20.5% and 19.3%; **block2 not measured**, so no conclusion about block2 yet. Suggested cooling (hard flat surface, raised back, vents clear, plugged in, fan on max) and offered to pause (training is resumable) or slow down. |
| 01:42 | You chose **keep training**. Main run continues unchanged; no additional GPU load will be added while it runs. The demo API/dashboard are idle on CPU. |

