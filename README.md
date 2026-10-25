# CryptoViT

> **Chaos-Assisted Vision Transformer for Privacy-Preserving Classification of Encrypted Images**

Final-year B.Tech capstone project (research prototype).

**Important:** this is a student research prototype. It is **not a medical device, not for clinical
use, and the encryption must not be used to protect real patient data.** All experiments use the
public DermaMNIST benchmark.

## What it does

A skin-lesion image is scrambled with a secret key and the scrambled image is classified by a
Vision Transformer (ViT) **without ever being decrypted**. The project asks one question:
*how much classification accuracy survives when the model only sees encrypted images, and how
does the choice of encryption scheme change that?*

```
image (28x28 RGB) --encrypt(key)--> encrypted image --enlarge to 224x224--> normalise
      --> 14x14 = 196 patches of 16x16 --> Vision Transformer --> one of 7 lesion classes
```

The decryption path exists only to show that the encryption is lossless: with the right key the
original image is restored exactly, with a wrong key you get noise.

## Encryption schemes (`encryption.scheme`)

| Scheme | What it does | Protection | Learnable by the ViT |
|---|---|---|---|
| `v1` (default) | 4D hyperchaotic key stream: pixel permutation, DNA-XOR substitution, chained XOR | Strong scrambling of every pixel, but diffusion is weak (changing one pixel changes only part of the output) | Poor in our experiments (see Results): the output is close to noise |
| `v2` | `v1` with independent permutation and key stream per colour channel and a uniform shuffle | Fixes the colour-channel leak and the non-uniform permutation of `v1`; diffusion is still weak | Not evaluated for classification (only checked for exact decryption and the two fixes) |
| `block<B>` (e.g. `block2`) | Shuffles whole BxB blocks of the image, plus a keyed channel order, per-block inversion and a pixel shuffle inside every block | **Visual protection only**: the picture is unrecognisable, but there is no diffusion (one changed pixel changes one output pixel), the colour palette is kept up to inversion (see `SECURITY.md`) and one scrambling is used for every image under a key | Good: with B = 2 on 28x28 images enlarged to 224, one block is exactly one 16x16 ViT patch |

All schemes are exact bijections (`decrypt(encrypt(x)) == x`). Statistical checks (entropy,
correlation, key and plaintext sensitivity NPCR/UACI) are in
`cryptovit.evaluation.security_metrics`; they are not a security proof. The default key in the
code is a public research key: change it and retrain before using anything seriously.

## Models compared (the ablation)

| Variant | Model | Input |
|---|---|---|
| A | Plain ViT | plaintext image |
| B | Plain ViT | encrypted image |
| C | ViT with `CryptoPatchEmbedding` (3x3 conv front-end, about 3 M extra parameters) | encrypted image |
| D | `CryptoViT`: C plus a key-aware positional encoding fed with an embedding of the key | encrypted image |

Every variant uses the same data split, the same training settings and the same seed, and the test
set is used once, after training, with the checkpoint chosen on the validation set (best balanced
accuracy). C and D use micro-batches of 32 with gradient accumulation (same effective batch of 128)
because the convolutions run at full resolution and do not fit a 4 GB GPU at batch 128.

## Dataset

[DermaMNIST](https://medmnist.com) (from HAM10000): 7 skin-lesion classes, 28x28 RGB images, the
official split of 7,007 train / 1,003 validation / 2,005 test images. 67% of the training images are
one class (melanocytic nevi), so the loss is class-weighted and results are reported as **balanced
accuracy**, per-class sensitivity and AUC, not plain accuracy. License: CC BY-NC 4.0 (non-commercial).
MedMNIST is for research benchmarking and is not intended for clinical use.

## Results

All numbers are measured once on the held-out test set (2,005 images), one training run per row (seed 42);
the table is generated from the saved results by `scripts/summarize_results.py` (full version with
per-class sensitivity in `RESULTS.md`). Chance level for 7 balanced classes is 14.3% balanced accuracy.

| Encryption | Variant | Model | Accuracy | Balanced accuracy | Macro AUC | Melanoma sensitivity | Epochs | Train time (min) |
|---|---|---|---|---|---|---|---|---|
| none (plaintext) | A | Plain ViT, plaintext image | 52.2% | 53.3% | 0.788 | 64.1% | 58 | 34 |
| per-pixel (v1) | B | Plain ViT, encrypted image | 12.9% | 19.4% | 0.573 | 48.0% | 51 | 36 |
| per-pixel (v1) | C | CryptoPatchEmbedding, encrypted image | 31.8% | 23.9% | 0.581 | 45.7% | 54 | 53 |
| per-pixel (v1) | D | CryptoViT (patch embedding + key-aware positions), encrypted image | 9.1% | 23.0% | 0.627 | 25.6% | 34 | 42 |
| block2 | B | Plain ViT, encrypted image | 46.5% | 53.6% | 0.809 | 65.0% | 60 | 41 |
| block2 | C | CryptoPatchEmbedding, encrypted image | 49.1% | 53.0% | 0.839 | 61.9% | 50 | 62 |
| block2 | D | CryptoViT (patch embedding + key-aware positions), encrypted image | 56.1% | 54.9% | 0.815 | 71.3% | 54 | 68 |

How to read it:

* **The encryption scheme decides whether classifying encrypted images works.** Per-pixel scrambling
  leaves the models at 19-24% balanced accuracy; block-wise scrambling gives 53-55%, comparable to the
  plaintext model's 53.3% (the models never see the unscrambled image).
* **The custom components show no measurable accuracy gain** over a plain ViT in these runs: on block-encrypted
  images B, C and D are 53.6%, 53.0% and 54.9% (AUC 0.809, 0.839, 0.815). Differences of a few points are
  within noise (single seed; the rarest classes have only 23-29 test images), so no claim of
  significance is made.
* **Block scrambling is the weaker cipher** (no diffusion, palette kept up to inversion); see
  `SECURITY.md` for measured numbers: changing one pixel changes about 20.5% of the output for the
  per-pixel scheme and exactly one pixel for the block scheme, against about 99.6% for a good cipher.
  The block scheme trades security for classifiability.
* Plain accuracy is lower than balanced accuracy for several rows because the loss is class-weighted
  towards the rare classes.

## Quick start

```bash
# install (Python 3.9+; a CUDA build of PyTorch is recommended for training)
pip install -e ".[dev,api,dashboard]"

# unit tests (fast, offline; no dataset download needed)
pytest tests/unit

# train and evaluate the variants on DermaMNIST (downloads the data on first use, ~19 MB)
python scripts/run_ablation.py config=dermamnist ablation.variants=A,B,C,D \
    training.epochs=60 paths.experiments_dir=experiments/dermamnist

# same with the block scheme (reuse the plaintext baseline A by copying its folder)
python scripts/run_ablation.py config=dermamnist config=block2 ablation.variants=B,D,C \
    training.epochs=60 paths.experiments_dir=experiments/dermamnist_block2

# results table from the saved measurements, and the picture pack for a live demo
python scripts/summarize_results.py
python scripts/make_demo_pack.py --block 2 --variant D --replace-demo-images

# interactive demo (API + dashboard); add --block 2 for the block-scheme models
python scripts/start_demo.py
```

Training is resumable: state is saved after every epoch and re-running the same command continues
where it stopped; finished variants are skipped.

## Configuration

* `configs/base.yaml` holds the defaults; `config=<name>` on the command line merges
  `configs/<name>.yaml` (e.g. `dermamnist`, `block2`, `cifar100`) and `key=value` overrides
  anything (e.g. `training.epochs=30`, `encryption.scheme=block2`).
* `training.class_weights` (`none`/`balanced`) and `training.select_metric`
  (`accuracy`/`balanced_accuracy`) are set for DermaMNIST in `configs/dermamnist.yaml`.
* The API reads `CRYPTOVIT_CONFIG` (comma separated, e.g. `dermamnist,block2`),
  `CRYPTOVIT_EXPERIMENTS`, `CRYPTOVIT_DEVICE` and `CRYPTOVIT_VARIANT`;
  `scripts/start_demo.py` sets them for you.

## Training on a laptop

Two optional Windows helpers protect a laptop during long runs (they only freeze and unfreeze the
training process; they change no system settings): `scripts/thermal_guard.py` pauses the job above
80 C and resumes below 70 C and can pace it to a duty cycle; `scripts/emergency_pause.py` freezes
everything at 90 C until a person resumes it. GPU memory is released between variants because
memory kept by an earlier variant can overflow a 4 GB card and silently slow training about tenfold.

## Known limitations

* The encryption is a research prototype. The per-pixel schemes have weak diffusion; the block scheme
  gives visual protection only. Neither is a substitute for a standard cipher such as AES.
* Models only know this dataset's image type (28x28 dermatoscope images). Photos of anything else
  receive a meaningless answer.
* One training run (seed 42) per row, and rare classes have only 23-29 test images, so differences of a
  few points in balanced accuracy are within noise. No claim of statistical significance is made.
* Images are 28x28; higher-resolution MedMNIST versions exist but were not used.
* A fixed key means the model only works with that key and must be retrained for another one.

## Project structure

```
CryptoViT/
  configs/           YAML configs (base, dermamnist, block2, cifar100)
  src/cryptovit/
    datasets/        CIFAR and MedMNIST loading, encrypted datasets, shared splits
    encryption/      hyperchaotic key streams, v1/v2 schemes, block scheme, DNA diffusion
    models/          ViT, CryptoPatchEmbedding, key-aware positional encoding, CryptoViT
    training/        trainer (AMP, cosine schedule, resumable), class weights
    evaluation/      accuracy, balanced accuracy, sensitivity/specificity, AUC, security metrics
    inference/       service that loads the trained variants (used by the API)
  scripts/           run_ablation, start_demo, make_demo_pack, summarize_results, thermal guards
  api/               FastAPI backend (encrypt, decrypt, predict, compare, results)
  dashboard/         Streamlit demo
  demo_images/       test images for the live demo; presentation/ holds enlarged pictures
  tests/             unit tests (no downloads needed)
  TRAINING_LOG.md    timestamped log of how the project was built and trained
```
