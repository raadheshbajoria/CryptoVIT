# CryptoViT

Chaos-assisted Vision Transformer for privacy-preserving classification of encrypted images.

A skin-lesion image is scrambled with a secret key and the scrambled image is classified by a Vision
Transformer (ViT) without ever being decrypted. The project measures how much classification accuracy
survives encryption and how the choice of encryption scheme changes that.

> **Research prototype. Not a medical device and not for clinical use.** The encryption must not be
> used to protect real patient data. All experiments use the public DermaMNIST benchmark.

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [Limitations](#limitations)

## How it works

```
image (28x28 RGB) --encrypt(key)--> encrypted image --enlarge to 224x224--> normalise
      --> 196 patches of 16x16 --> Vision Transformer --> one of 7 lesion classes
```

Decryption exists only to show that encryption is lossless: with the right key the original image is
restored exactly; with a wrong key a scrambled image is returned.

### Encryption schemes (`encryption.scheme`)

| Scheme | What it does | Protection | Learnable by the ViT |
|---|---|---|---|
| `v1` (default) | 4D hyperchaotic key stream: pixel permutation, DNA-XOR substitution, chained XOR | Strong scrambling of every pixel, weak diffusion (one changed pixel changes only part of the output) | Poor (see Results) |
| `v2` | `v1` with an independent permutation and key stream per colour channel and a uniform shuffle | Removes the colour-channel leak and the non-uniform permutation of `v1`; diffusion still weak | Not evaluated |
| `block<B>` (e.g. `block2`) | Shuffles whole BxB blocks, plus a keyed channel order, per-block inversion and a pixel shuffle inside every block | Visual protection only: no diffusion, palette kept up to inversion, one scrambling per key | Good: for B=2 on 28x28 images enlarged to 224, one block is exactly one 16x16 ViT patch |

All schemes are exact bijections (`decrypt(encrypt(x)) == x`). Statistical checks (entropy, correlation,
key and plaintext sensitivity) are in `cryptovit.evaluation.security_metrics` and `docs/SECURITY.md`;
they are not a security proof. The key in the code is a public research key.

### Models compared

| Variant | Model | Input |
|---|---|---|
| A | Plain ViT | plaintext image |
| B | Plain ViT | encrypted image |
| C | ViT with `CryptoPatchEmbedding` (3x3 convolutional front end) | encrypted image |
| D | `CryptoViT`: C plus a key-aware positional encoding fed with an embedding of the key | encrypted image |

All variants share the data split, training settings and seed. The test set is used once, after
training, with the checkpoint selected on the validation set (best balanced accuracy). C and D train
with micro-batches of 32 and gradient accumulation (effective batch 128) to fit a 4 GB GPU.

### Dataset

[DermaMNIST](https://medmnist.com) (from HAM10000): 7 skin-lesion classes, 28x28 RGB images, official
split of 7,007 / 1,003 / 2,005 images. 67% of the training images are one class (melanocytic nevi), so
the loss is class-weighted and results are reported as balanced accuracy, per-class sensitivity and
AUC. License: CC BY-NC 4.0. MedMNIST is for research benchmarking and not for clinical use.

## Results

Measured once on the held-out test set (2,005 images), one training run per row (seed 42). Generated
from the saved results by `scripts/summarize_results.py`; full table with per-class sensitivity in
`docs/RESULTS.md`. Chance level is 14.3% balanced accuracy.

| Encryption | Variant | Model | Accuracy | Balanced accuracy | Macro AUC | Melanoma sensitivity | Epochs | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| none (plaintext) | A | Plain ViT, plaintext image | 52.2% | 53.3% | 0.788 | 64.1% | 58 | 34 |
| per-pixel (v1) | B | Plain ViT, encrypted image | 12.9% | 19.4% | 0.573 | 48.0% | 51 | 36 |
| per-pixel (v1) | C | CryptoPatchEmbedding, encrypted image | 31.8% | 23.9% | 0.581 | 45.7% | 54 | 53 |
| per-pixel (v1) | D | CryptoViT (patch embedding + key-aware positions), encrypted image | 9.1% | 23.0% | 0.627 | 25.6% | 34 | 42 |
| block2 | B | Plain ViT, encrypted image | 46.5% | 53.6% | 0.809 | 65.0% | 60 | 41 |
| block2 | C | CryptoPatchEmbedding, encrypted image | 49.1% | 53.0% | 0.839 | 61.9% | 50 | 62 |
| block2 | D | CryptoViT (patch embedding + key-aware positions), encrypted image | 56.1% | 54.9% | 0.815 | 71.3% | 54 | 68 |

- The encryption scheme decides whether classifying encrypted images works: per-pixel scrambling leaves
  the models at 19-24% balanced accuracy, block-wise scrambling gives 53-55%, comparable to the
  plaintext model's 53.3%.
- The custom components (variants C and D) show no measurable accuracy gain over a plain ViT on
  block-encrypted images. Differences of a few points are within noise: single seed, and the rarest
  classes have only 23-29 test images.
- The block scheme is the weaker cipher. Changing one pixel changes about 20.5% of the output for the
  per-pixel scheme and exactly one pixel for the block scheme, against about 99.6% for a good cipher.

## Installation

Python 3.10 or newer. A CUDA build of PyTorch is recommended for training.

```bash
pip install -e ".[dev,api,dashboard]"
```

## Usage

```bash
# train and evaluate the variants on DermaMNIST (downloads the ~19 MB dataset on first use)
python scripts/run_ablation.py config=dermamnist ablation.variants=A,B,C,D \
    training.epochs=60 paths.experiments_dir=experiments/dermamnist

# the same with the block scheme (copy variant_A and result_A.json from the first run to reuse A)
python scripts/run_ablation.py config=dermamnist config=block2 ablation.variants=B,D,C \
    training.epochs=60 paths.experiments_dir=experiments/dermamnist_block2

# results table and security report from the saved measurements
python scripts/summarize_results.py
python scripts/security_report.py

# sample pictures for a demo (writes demo_images/ and presentation/)
python scripts/make_demo_pack.py --block 2 --variant D --replace-demo-images

# API (port 8000) and dashboard (port 8501); add --block 2 for the block-scheme models
python scripts/start_demo.py
```

Training is resumable: state is saved after every epoch and rerunning a command continues where it
stopped; finished variants are skipped. In the dashboard, upload a 28x28 test image (the files in
`demo_images/` can be used) to see it encrypted, classified and decrypted with a right and a wrong key.

### API

`GET /health`, `POST /encrypt`, `POST /decrypt`, `POST /predict`, `POST /predict_plaintext`,
`POST /compare`, `GET /results`, `GET /metrics`, `GET /model_info`. Interactive docs at `/docs`.

## Configuration

- `configs/base.yaml` holds the defaults. `config=<name>` on the command line merges
  `configs/<name>.yaml` (`dermamnist`, `block2`, `cifar100`) and `key=value` overrides anything, for
  example `training.epochs=30` or `encryption.scheme=block2`.
- `training.class_weights` (`none` / `balanced`) and `training.select_metric` (`accuracy` /
  `balanced_accuracy`) are set for DermaMNIST in `configs/dermamnist.yaml`.
- The API reads `CRYPTOVIT_CONFIG` (comma separated, e.g. `dermamnist,block2`), `CRYPTOVIT_EXPERIMENTS`,
  `CRYPTOVIT_DEVICE` and `CRYPTOVIT_VARIANT`; `scripts/start_demo.py` sets them.

Two optional Windows helpers protect a laptop during long runs: `scripts/thermal_guard.py` pauses a job
above a temperature limit and can pace it to a duty cycle, and `scripts/emergency_pause.py` freezes it
at a hard limit until a person resumes it. They only suspend and resume the training process.

## Development

```bash
ruff check .          # lint
ruff format .         # format
pytest                # tests (offline, no dataset download)
```

The same checks run in CI (`.github/workflows/ci.yml`) on Linux and Windows.

## Project structure

```
.github/workflows/   CI
api/                 FastAPI backend
configs/             YAML configs (base, dermamnist, block2, cifar100)
dashboard/           Streamlit demo
demo_images/         sample 28x28 test images to upload in the dashboard
docs/                results, security report, review guide, work log
presentation/        enlarged pictures of the samples (original, per-pixel, block)
scripts/             training, evaluation, reporting and demo entry points
src/cryptovit/
  datasets/          CIFAR and MedMNIST loading, encrypted datasets, shared splits
  encryption/        key streams, v1/v2 schemes, block scheme
  evaluation/        accuracy, balanced accuracy, sensitivity/specificity, AUC, security metrics
  inference/         service that loads the trained variants
  models/            ViT, CryptoPatchEmbedding, key-aware positional encoding, CryptoViT
  training/          trainer (mixed precision, cosine schedule, resumable), class weights
  utils/             config, logging, seeding
tests/               unit tests
```

## Documentation

- `docs/RESULTS.md` - every result row with per-class sensitivity
- `docs/SECURITY.md` - measured security properties of the schemes
- `docs/REVIEW_GUIDE.md` - talk track, demo script and likely questions
- `docs/TRAINING_LOG.md` - timestamped log of how the project was built and trained

## Limitations

- The encryption is a research prototype: the per-pixel schemes have weak diffusion and the block scheme
  gives visual protection only. Neither replaces a standard cipher such as AES.
- Models only know this dataset's image type; photos of anything else get a meaningless answer.
- One training run per row and small rare classes: no claim of statistical significance.
- Images are 28x28; higher-resolution MedMNIST versions were not used.
- A model only works with the key it was trained with and must be retrained for another key.
