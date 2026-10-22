# CryptoViT

> **Chaos-Assisted Vision Transformer for Privacy-Preserving Classification of Encrypted Images**

Final-year B.Tech capstone project.

## Quick Start

```bash
# Install the package in editable mode
pip install -e ".[dev]"

# Run Phase 1 dataset sanity check (downloads CIFAR-10)
python scripts/phase1_dataset_check.py

# Run unit tests
pytest tests/ -v
```

## Project Structure

```
CryptoViT/
├── configs/           YAML configurations
├── data/              Dataset storage (gitignored)
├── src/cryptovit/     Main Python package
│   ├── datasets/      Dataset loading & preprocessing
│   ├── encryption/    Hyperchaotic encryption engine (Phase 3+)
│   ├── models/        ViT & CryptoViT (Phase 2+)
│   ├── training/      Training loops (Phase 2+)
│   ├── evaluation/    Metrics & security analysis (Phase 8+)
│   ├── inference/     Inference pipeline (Phase 17+)
│   └── utils/         Config, logging, seeding, visualisation
├── scripts/           Phase-specific run scripts
├── experiments/       Saved results, checkpoints, logs
├── tests/             pytest test suite
├── api/               FastAPI backend (Phase 17+)
├── dashboard/         Streamlit app (Phase 18+)
└── notebooks/         Exploratory notebooks
```

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset & preprocessing | ✅ |
| 2 | Standard ViT baseline | ⏳ |
| 3 | 4D hyperchaotic key generator | ⏳ |
| … | … | … |

## Hardware

Developed on: **NVIDIA RTX 3050 Laptop GPU (4 GB)**

## Local Review Dashboard and Standard ViT Baseline

The repository includes a local FastAPI backend and a Streamlit review dashboard. This review milestone demonstrates a trained **Standard ViT** on plaintext CIFAR-10 images, alongside an image-encryption round-trip demonstration. It does not demonstrate CryptoViT classification performance.

### Prerequisites and installation

- Python **3.11.9** was used for the verified review environment. The project metadata supports Python 3.9 and newer.
- A CUDA-capable NVIDIA GPU is optional. The reviewed setup used an NVIDIA GeForce RTX 2050 Laptop GPU with CUDA-enabled PyTorch; inference can also run on CPU, though it may be slower.
- From Windows PowerShell, enter the repository and create/activate a virtual environment if one is not already available:

  ```powershell
  Set-Location 'C:\Users\User\OneDrive\Desktop\Capstone Project\CryptoVIT'
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

- Install the project and its declared development, API, and dashboard extras:

  ```powershell
  python -m pip install -e ".[dev,api,dashboard]"
  ```

  If PowerShell activation is unavailable or the environment is already set up, use `.venv\Scripts\python.exe` explicitly in place of `python`. For a GPU-specific PyTorch setup, follow the official PyTorch installation selector for the target CUDA/runtime; do not assume a GPU build is available merely because an NVIDIA GPU is installed.

### Start the API and dashboard

Start the FastAPI backend first, from the repository root, in its own PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open while using the dashboard. In a second PowerShell terminal, also at the repository root, start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open the dashboard at **http://127.0.0.1:8501**. Both services bind to localhost for this review prototype.

Useful backend endpoints:

- **http://127.0.0.1:8000/health** — API and model readiness.
- **http://127.0.0.1:8000/model_info** — loaded model details and CryptoViT status.
- **http://127.0.0.1:8000/docs** — interactive API documentation.

The API does not define a root page, so **http://127.0.0.1:8000/** may return 404 even when the backend is running correctly.

### Dashboard tabs

1. **Project Overview** — project motivation, current milestone, and architecture/workflow.
2. **Encryption Demonstration** — upload an image or load the local CIFAR-10 test sample, choose a secret key, then encrypt and decrypt the image. The dashboard compares the original and recovered pixel arrays to report whether the round trip is exact. It also displays available statistical metrics; these are not cryptographic security proofs.
3. **Standard ViT Prediction** — load a local CIFAR-10 test image or upload a PNG/JPG image, then view the predicted CIFAR-10 class, confidence, top-five probabilities, and logits. The plaintext input is resized to 96 × 96 and normalized using CIFAR-10 statistics.
4. **Baseline Training Results** — view the saved experiment metrics, training/validation loss and accuracy curves, test confusion matrix, and per-epoch history when those artifacts are available.
5. **CryptoViT Status** — inspect the implementation overview and the explicit pending-training-and-validation status.

### Trained Standard ViT baseline

The dashboard/API use this reviewed checkpoint:

`experiments/review_baseline_20261022_211230/checkpoints/checkpoint_epoch0005_acc0.3110.pt`

It is a **Standard `vit_tiny` VisionTransformer**, with **96 × 96** input, **10 CIFAR-10 classes**, and **525,994 parameters**. The reported run trained for **5 epochs** using a **5,000-example CIFAR-10 training subset** and reported **31.39% accuracy** on the official CIFAR-10 test set. This is a preliminary, limited-subset baseline, not mature model performance. The independent baseline audit and run report are available alongside the experiment artifacts and document the evaluation details and limitations.

A successful Standard ViT prediction is for a **plaintext** image. It does not show that Standard ViT can classify encrypted images, and it is not evidence of CryptoViT performance. CryptoViT's own training and evaluation remain pending validation; its prediction endpoint returns HTTP 503 until a validated trained checkpoint is available.

### Troubleshooting and limitations

- **Dashboard unavailable / port 8501:** confirm the Streamlit terminal is still running and started from the repository root. Check for a listener with `Get-NetTCPConnection -State Listen -LocalPort 8501`. If another process owns the port, stop or reconfigure that process intentionally before starting Streamlit on a different port.
- **API unavailable / port 8000:** confirm the API terminal is still running. Check with `Get-NetTCPConnection -State Listen -LocalPort 8000`. Start FastAPI before Streamlit and keep both terminals open.
- **Dashboard says the API is offline:** verify that the sidebar backend URL is `http://127.0.0.1:8000` and that `/health` responds. The dashboard reports connection failures rather than returning a fabricated result.
- **Baseline checkpoint missing or incompatible:** check that the exact checkpoint path above exists and that startup logs show it loaded successfully with strict state-dictionary matching. Baseline inference is unavailable if no valid checkpoint loads.
- **CryptoViT prediction returns 503:** this is expected while CryptoViT's own training and validation are pending. The API deliberately does not substitute randomly initialized weights.

This is a **local review prototype**, not a deployed end-to-end zero-trust system or a cryptographic security proof. In the demonstration flow, the local API receives the uploaded plaintext image to perform encryption or baseline inference. CryptoViT requires its own valid training and evaluation, and the Standard ViT baseline remains preliminary.
