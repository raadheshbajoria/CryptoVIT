# CryptoViT Reviewer Dashboard Launch Guide

**Document Purpose:** Reviewer guide for evaluating the **CryptoViT** demonstration interface and the verified standard Vision Transformer (`vit_tiny`) baseline.  
**Project:** CryptoViT — Chaos-Assisted Vision Transformer for Privacy-Preserving Classification  
**Target Review Date:** Monday, 26 October 2026  
**Institution:** VIT Bhopal University (B.Tech Capstone)

---

## 1. System Overview & Implementation Status

| Component | Status | Key Metrics / Evidence |
| :--- | :--- | :--- |
| **Standard ViT Baseline** | ✅ **Trained & Audited** | `vit_tiny` (525,994 params), **31.39% accuracy** on 10,000 official CIFAR-10 test set |
| **Full Test Suite** | ✅ **100% Passed** | **85/85 passed in the latest verification (2026-10-23)** (including API endpoints) |
| **Hyperchaotic Transformation Module** | ✅ **Functional Reversibility Verified** | Reversible 4D hyperchaotic map + pixel permutation + DNA dynamic diffusion (exact round-trip on tested inputs) |
| **FastAPI Backend (`api/main.py`)** | ✅ **Operational** | Strict checkpoint loading; blocks untrained model inference |
| **Streamlit Dashboard (`dashboard/app.py`)** | ✅ **Operational** | 5 comprehensive review tabs with live backend integration |
| **CryptoViT Encrypted Model** | ⏳ **Pending Validation** | Architecture implemented; training scheduled for next experimental phase |

---

## 2. Verified Baseline Artifact Locations

All baseline assets reside in an isolated, git-ignored directory:
`experiments/review_baseline_20261022_211230/`

- **Trained Model Checkpoint:**  
  `experiments/review_baseline_20261022_211230/checkpoints/checkpoint_epoch0005_acc0.3110.pt` (6,384,302 bytes, approx. 6.38 MB decimal / 6.09 MiB binary)
- **Detailed Audit Reports:**  
  - `CRYPTOVIT_EXISTING_IMPLEMENTATION_AUDIT.md` (Initial system audit)  
  - `BASELINE_TRAINING_AUDIT.md` (Independent verification of checkpoint & metrics)  
  - `experiments/review_baseline_20261022_211230/REVIEW_BASELINE_REPORT.md` (Training execution report)
- **Training Progression & Test Metrics:**  
  - `history.json` (Per-epoch train/val loss & accuracy)  
  - `test_metrics.json` (Held-out 10,000-image evaluation metrics)  
  - `run_metadata.json` (Environment, seed, parameter configuration)  
  - `loss_acc_curves.png` & `confusion_matrix.png` (Visualizations)

---

## 3. Environment & Hardware Prerequisites

- **Python Version:** Python 3.11 (Windows)
- **Virtual Environment:** `.\.venv`
- **PyTorch Build:** `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`
- **Acceleration:** CUDA enabled (tested on NVIDIA GeForce RTX 2050 GPU; CPU fallback supported)

---

## 4. How to Launch the Application (Step-by-Step)

The application requires running the **FastAPI backend** and the **Streamlit frontend** in two separate PowerShell terminals opened at your repository root directory:

### Terminal 1: Start the FastAPI Backend
From your repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

*Expected console output:*
```
INFO:     CryptoViT API starting up on device: cuda
INFO:     Successfully loaded trained baseline model from .../checkpoint_epoch0005_acc0.3110.pt (strict=True)
INFO:     No trained CryptoViT checkpoint exists. CryptoViT inference remains pending.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### Terminal 2: Start the Streamlit Dashboard
From your repository root, in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

*Expected console output:*
```
  You can now view your Streamlit app in your browser.
  Local URL: http://127.0.0.1:8501
```

Access the interface in your browser at:  
👉 **`http://127.0.0.1:8501`**

---

## 5. Reviewer Demonstration Workflow

The dashboard contains 5 dedicated tabs tailored for the project review:

### Tab 1: Project Overview
- Explains the core privacy challenge in cloud computer vision.
- Displays an ASCII system architecture flowchart showing the parallel roles of the Standard ViT baseline vs. CryptoViT.
- Discloses the current project milestone status.

### Tab 2: Hyperchaotic Encryption Demonstration
- **Upload Image or Load CIFAR-10 Sample**: Click *"Load Local CIFAR-10 Test Sample"* or upload an image.
- **Encrypt**: Set a secret key and click *"🔒 Encrypt Image"*. The interface renders the ciphertext representation and reports available statistical metrics (information entropy, NPCR, UACI, and horizontal correlation).
- **Decrypt**: Click *"🔓 Decrypt Image"*. The interface recovers the original pixels and programmatically verifies **exact mathematical invertibility** (`Array difference = 0`).
- **Scope Qualification**: The round-trip test verifies functional reversibility for the tested inputs and matching keys. It does not establish cryptographic security or independently validate the statistical and dynamical properties of the hyperchaotic generator.

### Tab 3: Standard ViT Baseline Classification
- Evaluates the **trained `vit_tiny` baseline checkpoint** on plaintext images.
- Preprocesses images using the exact 96×96 resolution and CIFAR-10 empirical normalization used during training.
- Outputs the predicted class, normalized softmax confidence score, and top-5 probability distribution.
- Inspects real finite logit values.

### Tab 4: Baseline Training Results
- Dynamically loads and plots the saved experiment artifacts (`history.json`, `test_metrics.json`).
- Displays the training & validation loss/accuracy curves.
- Renders the normalized confusion matrix across all 10 CIFAR-10 classes on the 10,000-image official test set.
- Highlights the held-out test accuracy of **31.39%**.
- Discloses that the difference in precision mode between CUDA AMP (original run) and FP32 (independent audit) is a plausible explanation for minor precision/F1 variations, though the cause has not been isolated.

### Tab 5: CryptoViT Development Status
- Explains the architectural design of CryptoViT (convolutional Crypto Patch Embedding, Key-Aware Positional Encoding).
- Explicitly states that CryptoViT training is **pending experimental validation**.
- Proves that the API refuses to serve fake or randomly initialized weights for CryptoViT.

---

## 6. Privacy & Trust Boundary Disclosure

1. **Client-Server Privacy Boundary**: In this prototype demonstration, image bytes are sent over `http://127.0.0.1:8000` to the local API. The host receives the image and key to perform encryption. This is an algorithmic proof-of-concept; it does not yet provide end-to-end client-side sandboxing or hardware-enclave guarantees.
2. **Statistical vs Cryptographic Security**: Entropy (~7.80) and NPCR (~99.6%) metrics measure visual dispersion and diffusion. They demonstrate transformation efficacy for neural networks, but do not replace formal cryptographic security proofs.

---

## 7. Troubleshooting & FAQ

- **"API Backend Offline" in Dashboard sidebar**: Ensure Terminal 1 is running `uvicorn api.main:app --port 8000`. Verify `http://127.0.0.1:8000/health` returns `{"status": "ok"}`.
- **"Baseline ViT Checkpoint Not Loaded"**: Verify the file `experiments/review_baseline_20261022_211230/checkpoints/checkpoint_epoch0005_acc0.3110.pt` exists and has not been moved.
- **Port 8000 or 8501 already in use**: Use a different port (e.g., `--port 8001`) and update the "FastAPI Backend URL" field in the dashboard sidebar.
- **Test validation**: To re-run the full project test suite:
  ```powershell
  $env:PYTHONDONTWRITEBYTECODE = "1"
  .\.venv\Scripts\python.exe -m pytest tests -v
  ```
