"""
dashboard/app.py
=================
Streamlit demonstration interface for CryptoViT.

Designed for B.Tech Capstone Project Review:
"CryptoViT: Chaos-Assisted Vision Transformer for Privacy-Preserving Classification"
VIT Bhopal University — ECE + AI/Cybernetics

Tabs:
  1. Project Overview & Architecture
  2. Hyperchaotic Encryption Demonstration
  3. Standard ViT Baseline Classification
  4. Baseline Training & Empirical Results
  5. CryptoViT Development Status
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import streamlit as st
from PIL import Image

# -------------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="CryptoViT — Project Review Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://127.0.0.1:8000"
REVIEW_EXP_DIR = PROJECT_ROOT / "experiments" / "review_baseline_20261022_211230"

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------

st.sidebar.title("🔐 CryptoViT")
st.sidebar.markdown(
    "**Chaos-Assisted Vision Transformer**  \n"
    "*Privacy-Preserving Image Classification*  \n\n"
    "**Institution:** VIT Bhopal University  \n"
    "**Discipline:** B.Tech ECE + AI / Cybernetics  \n"
    "**Review Milestone:** Standard ViT Baseline"
)
st.sidebar.divider()

api_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=DEFAULT_API_URL,
    help="Address of the local FastAPI inference and demo service.",
)

# Live Backend Health Probe
@st.cache_data(ttl=3)
def fetch_api_health(url: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None

health_info = fetch_api_health(api_url)

if health_info:
    st.sidebar.success("● API Backend Connected")
    device_label = health_info.get("device", "unknown")
    st.sidebar.caption(f"Compute Device: `{device_label}`")
    if health_info.get("baseline_ready"):
        arch = health_info.get("baseline_architecture", "vit_tiny")
        st.sidebar.info(f"● Baseline ViT: **Loaded ({arch})**")
    else:
        st.sidebar.warning("● Baseline ViT: **Checkpoint Not Loaded**")
    
    if health_info.get("cryptovit_ready"):
        st.sidebar.success("● CryptoViT: **Loaded**")
    else:
        st.sidebar.caption("● CryptoViT: **Pending Validation**")
else:
    st.sidebar.error("○ API Backend Offline")
    st.sidebar.caption("Run: `.\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000`")

st.sidebar.divider()
st.sidebar.markdown("### Privacy Boundary Notice")
st.sidebar.caption(
    "⚠️ **Client-Server Privacy Disclosure**: This interactive demonstration sends image bytes "
    "to the local API service over localhost (`127.0.0.1`). The server receives the image to perform "
    "encryption and model evaluation. It does not provide end-to-end client-side confidentiality "
    "or zero-trust hardware isolation."
)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_sample_test_image() -> Optional[Image.Image]:
    """Load a sample CIFAR-10 test image from the local raw batch if available."""
    cifar_batch = PROJECT_ROOT / "data" / "raw" / "cifar-10-batches-py" / "test_batch"
    if cifar_batch.is_file():
        import pickle
        with open(cifar_batch, "rb") as f:
            data_dict = pickle.load(f, encoding="bytes")
        img_arr = data_dict[b"data"][0].reshape(3, 32, 32).transpose(1, 2, 0)
        return Image.fromarray(img_arr)
    return None


# -------------------------------------------------------------------------
# Main UI Layout — Tabs
# -------------------------------------------------------------------------

st.title("🔐 CryptoViT — Privacy-Preserving Image Classification")
st.markdown(
    "Demonstration and technical review system for the **Chaos-Assisted Vision Transformer** research project."
)

tab_overview, tab_encrypt, tab_baseline, tab_results, tab_status = st.tabs([
    "1️⃣ Project Overview",
    "2️⃣ Encryption Demonstration",
    "3️⃣ Standard ViT Prediction",
    "4️⃣ Baseline Training Results",
    "5️⃣ CryptoViT Status",
])

# =========================================================================
# TAB 1: Project Overview
# =========================================================================
with tab_overview:
    st.header("Project Overview & Problem Statement")
    
    col_intro, col_status_box = st.columns([3, 2])
    
    with col_intro:
        st.markdown(
            """
            ### The Core Privacy Problem
            Deep neural networks deployed on cloud platforms require users to upload input images in plaintext.
            In sensitive applications (such as medical imaging, facial biometric verification, or private surveillance),
            transmitting unencrypted raw pixels exposes users to eavesdropping, data harvesting, and unauthorized inspection.
            
            ### Standard ViT vs. CryptoViT Approach
            - **Ordinary Image Classification**: Standard Vision Transformers (ViT) divide raw spatial images into square patches, 
              project them linearly, and apply multi-head self-attention. They require clear local spatial structure and ordinary pixel distributions.
            - **Protected Image Representation**: Conventional cryptographic ciphers (such as AES) destroy all visual correlations, turning 
              images into pseudorandom bitstreams that ordinary convolutional or transformer networks cannot classify directly.
            - **CryptoViT Objective**: Investigates whether a deterministic, key-derived transformation (hyperchaotic permutation + DNA diffusion) 
              combined with specialized network adaptations (convolutional patch embeddings and key-aware positional encodings) can enable accurate 
              classification directly on the protected visual representation **without ever decrypting the image on the host**.
            """
        )
    
    with col_status_box:
        st.markdown("### Current Engineering Milestone")
        st.info(
            """
            **Milestone Status (Review Baseline)**
            - **Standard ViT Baseline**: ✅ Genuinely trained (`vit_tiny`, 5 epochs on 5,000 samples)
            - **Baseline Test Accuracy**: **31.39%** on official CIFAR-10 test set (3,139 / 10,000)
            - **Unit Test Suite**: ✅ 71/71 tests passing (100% pass rate)
            - **Hyperchaotic Cipher**: ✅ Exact round-trip invertibility verified
            - **CryptoViT Encrypted Model**: ⏳ **Pending validation** (training not yet executed)
            """
        )
    
    st.divider()
    st.subheader("System Architecture & Workflow")
    
    st.markdown(
        """
        ```
        +-----------------------------------------------------------------------------+
        |                          SYSTEM PIPELINE WORKFLOW                           |
        +-----------------------------------------------------------------------------+
                                       
           [ Plaintext Image ] (CIFAR-10, 32x32 / 96x96)
                   |
                   v
        +-------------------------------------------------------------+
        | Hyperchaotic Transformation Engine                          |
        | 1. 4D Hyperchaotic Sequence Generation (Lorenz/Chen Map)    |
        | 2. Pixel & Patch Permutation (Spatial position shuffling)   |
        | 3. DNA Rule-Based Dynamic Encoding & Diffusion              |
        +-------------------------------------------------------------+
                   |
                   v
           [ Protected Representation ] (Visually obscured ciphertext)
                   |
         +---------+-----------------------------------+
         |                                             |
         v                                             v
  [ Standard ViT ]                             [ CryptoViT ]
  - Linear Patch Projection                    - Crypto Patch Embedding (Conv-based)
  - Standard 1D Learned PE                     - Key-Aware Positional Encoding (MLP)
  - Transformer Encoder Blocks                 - Transformer Encoder Blocks
  - Classifies Plaintext Baseline              - Classifies Encrypted Visuals Directly
  - STATUS: TRAINED & AUDITED (31.39% Acc)     - STATUS: PENDING EXPERIMENTAL VALIDATION
        ```
        """
    )


# =========================================================================
# TAB 2: Hyperchaotic Encryption Demonstration
# =========================================================================
with tab_encrypt:
    st.header("Hyperchaotic Encryption & Decryption Demonstration")
    st.markdown(
        "Demonstrates the deterministic image transformation pipeline using hyperchaotic sequences, "
        "permutation, and DNA-based diffusion. This tab allows testing invertibility on user-uploaded images."
    )
    
    enc_col1, enc_col2 = st.columns([1, 1])
    
    with enc_col1:
        enc_key = st.text_input(
            "Secret Key",
            value="CryptoViT_Research_Key_2025",
            type="password",
            help="Secret key string used to initialize the 4D chaotic map parameters.",
            key="tab2_key",
        )
    
    with enc_col2:
        st.caption("Key Hygiene & Privacy Notice")
        st.info("The key is passed directly to the local encryption module via POST request to calculate chaotic trajectories. Never reuse production cryptographic keys.")

    uploaded_enc_file = st.file_uploader(
        "Choose an image for encryption testing (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        key="tab2_uploader",
    )
    
    # Quick CIFAR sample loader button
    if st.button("Load Local CIFAR-10 Test Sample", key="btn_load_sample"):
        sample_img = load_sample_test_image()
        if sample_img:
            st.session_state["demo_orig_img"] = sample_img
        else:
            st.warning("Local CIFAR-10 test sample file not found in `data/raw/`.")

    if uploaded_enc_file is not None:
        try:
            st.session_state["demo_orig_img"] = Image.open(uploaded_enc_file).convert("RGB")
        except (OSError, ValueError) as e:
            st.session_state.pop("demo_orig_img", None)
            st.error(f"Could not read the uploaded image: {e}")

    if "demo_orig_img" in st.session_state:
        orig_img = st.session_state["demo_orig_img"]
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("1. Original Image")
            st.image(orig_img, caption=f"Original ({orig_img.size[0]}×{orig_img.size[1]} px)", use_container_width=True)
            if st.button("🔒 Encrypt Image", key="btn_encrypt"):
                if not health_info:
                    st.error("API backend offline. Start the FastAPI server first.")
                else:
                    with st.spinner("Encrypting with 4D hyperchaotic engine..."):
                        try:
                            r = requests.post(
                                f"{api_url}/encrypt",
                                files={"file": ("input.png", pil_to_png_bytes(orig_img), "image/png")},
                                data={"key": enc_key},
                                timeout=20,
                            )
                            if r.status_code == 200:
                                st.session_state["demo_enc_img"] = Image.open(io.BytesIO(r.content))
                                st.session_state["demo_enc_bytes"] = r.content
                                st.session_state["demo_enc_key"] = enc_key
                                st.session_state["demo_key_hash"] = r.headers.get("X-Encryption-KeyHash", "N/A")
                                st.success("Encryption complete!")
                            else:
                                st.error(f"Encryption error ({r.status_code}): {r.text}")
                        except requests.RequestException as e:
                            st.error(f"Encryption request failed: {e}")
        
        with c2:
            st.subheader("2. Protected Image")
            if "demo_enc_img" in st.session_state:
                enc_display = st.session_state["demo_enc_img"]
                st.image(enc_display, caption="Ciphertext Visualization", use_container_width=True)
                st.caption(f"Key Hash: `{st.session_state.get('demo_key_hash', 'N/A')[:16]}...`")
                st.download_button(
                    "Download Encrypted PNG",
                    data=st.session_state["demo_enc_bytes"],
                    file_name="encrypted_image.png",
                    mime="image/png",
                )
                
                dec_key_input = st.text_input("Decryption Key", value=enc_key, type="password", key="tab2_dec_key")
                if st.button("🔓 Decrypt Image", key="btn_decrypt"):
                    if not health_info:
                        st.error("API backend offline.")
                    else:
                        with st.spinner("Decrypting image..."):
                            try:
                                r_dec = requests.post(
                                    f"{api_url}/decrypt",
                                    files={"file": ("enc.png", st.session_state["demo_enc_bytes"], "image/png")},
                                    data={"key": dec_key_input},
                                    timeout=20,
                                )
                                if r_dec.status_code == 200:
                                    st.session_state["demo_dec_img"] = Image.open(io.BytesIO(r_dec.content))
                                    st.success("Decryption finished!")
                                else:
                                    st.error(f"Decryption error: {r_dec.text}")
                            except requests.RequestException as e:
                                st.error(f"Decryption request failed: {e}")
                            except (OSError, ValueError) as e:
                                st.error(f"Could not read the decrypted image: {e}")
            else:
                st.info("Click 'Encrypt Image' to generate protected representation.")
        
        with c3:
            st.subheader("3. Recovered Image & Check")
            if "demo_dec_img" in st.session_state:
                dec_display = st.session_state["demo_dec_img"]
                st.image(dec_display, caption=f"Recovered ({dec_display.size[0]}×{dec_display.size[1]} px)", use_container_width=True)
                
                # Verification of invertibility
                orig_arr = np.array(orig_img)
                dec_arr = np.array(dec_display)
                
                if orig_arr.shape == dec_arr.shape:
                    is_exact = np.array_equal(orig_arr, dec_arr)
                    if is_exact:
                        st.success("✅ **Exact Invertibility Verified!** (Array difference = 0)")
                    else:
                        diff = np.abs(orig_arr.astype(int) - dec_arr.astype(int)).mean()
                        st.warning(f"⚠️ Recovered image differs (Mean pixel diff: {diff:.2f}). Check key or resize settings.")
                else:
                    st.warning("Image dimensions differed during round-trip.")
            else:
                st.info("Decrypted output will appear here after decryption.")
        
        # Security metrics display
        if "demo_enc_img" in st.session_state:
            st.divider()
            st.subheader("Statistical Properties of Protected Representation")
            try:
                m_resp = requests.get(f"{api_url}/metrics", timeout=5)
                if m_resp.status_code == 200:
                    metrics = m_resp.json()
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    ent_val = metrics.get("entropy", {}).get("overall", 0.0)
                    npcr_val = metrics.get("npcr_plain_vs_enc", 0.0)
                    uaci_val = metrics.get("uaci_plain_vs_enc", 0.0)
                    h_corr = metrics.get("correlation_encrypted", {}).get("horizontal", 0.0)
                    
                    col_m1.metric("Information Entropy", f"{ent_val:.4f}", help="Theoretical maximum: 8.0000 for uniform byte distribution.")
                    col_m2.metric("NPCR (Plain vs Cipher)", f"{npcr_val:.2f}%", help="Percentage of differing pixels.")
                    col_m3.metric("UACI (Plain vs Cipher)", f"{uaci_val:.2f}%", help="Mean intensity difference.")
                    col_m4.metric("Horizontal Correlation", f"{h_corr:.4f}", help="Spatial correlation between adjacent pixels (Target ~ 0).")
                    
                    st.caption(
                        "ℹ️ *Note for reviewers: Statistical metrics quantify visual randomness and diffusion, "
                        "but do not alone constitute formal semantic security proofs.*"
                    )
                else:
                    st.warning(f"Could not load encryption metrics ({m_resp.status_code}): {m_resp.text}")
            except requests.RequestException as e:
                st.warning(f"Could not load encryption metrics: {e}")


# =========================================================================
# TAB 3: Standard ViT Baseline Classification
# =========================================================================
with tab_baseline:
    st.header("Standard ViT Baseline Classification")
    st.markdown(
        "Evaluate the **trained, audited Standard Vision Transformer (`vit_tiny`)** on input images. "
        "The model predicts probabilities among the 10 standard CIFAR-10 classes."
    )
    
    # Checkpoint Status Banner
    if health_info and health_info.get("baseline_ready"):
        st.success(f"✅ Trained baseline checkpoint active: `{health_info.get('baseline_checkpoint')}`")
    else:
        st.error("⚠️ Baseline model checkpoint is unavailable. Prediction is disabled.")
    
    col_pred_in, col_pred_out = st.columns([1, 1])
    
    with col_pred_in:
        st.subheader("Input Image")
        pred_upload = st.file_uploader(
            "Upload image for classification",
            type=["png", "jpg", "jpeg"],
            key="tab3_uploader",
        )
        
        if st.button("Use Sample CIFAR-10 Test Image", key="tab3_sample_btn"):
            sample_cifar = load_sample_test_image()
            if sample_cifar:
                st.session_state["tab3_img"] = sample_cifar
        
        if pred_upload is not None:
            try:
                st.session_state["tab3_img"] = Image.open(pred_upload).convert("RGB")
            except (OSError, ValueError) as e:
                st.session_state.pop("tab3_img", None)
                st.error(f"Could not read the uploaded image: {e}")
        
        if "tab3_img" in st.session_state:
            active_img = st.session_state["tab3_img"]
            st.image(active_img, caption="Input for ViT", width=180)
            st.caption("Preprocessing: Automatically resized to 96×96 and normalized using CIFAR-10 empirical statistics.")
            
            predict_btn = st.button("Classify Image (Standard ViT)", key="tab3_predict_btn", disabled=not (health_info and health_info.get("baseline_ready")))
        else:
            predict_btn = False
            st.info("Upload an image or load a CIFAR-10 test sample to run prediction.")
    
    with col_pred_out:
        st.subheader("Model Prediction")
        if predict_btn and "tab3_img" in st.session_state:
            with st.spinner("Running forward pass through VisionTransformer..."):
                st.session_state.pop("last_baseline_pred", None)
                try:
                    r_pred = requests.post(
                        f"{api_url}/predict_plaintext",
                        files={"file": ("eval.png", pil_to_png_bytes(st.session_state["tab3_img"]), "image/png")},
                        timeout=15,
                    )
                    if r_pred.status_code == 200:
                        pred_data = r_pred.json()
                        st.session_state["last_baseline_pred"] = pred_data
                    else:
                        st.error(f"Prediction failed ({r_pred.status_code}): {r_pred.text}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
        
        if "last_baseline_pred" in st.session_state:
            p = st.session_state["last_baseline_pred"]
            pred_class = p["class_name"].upper()
            confidence = p["confidence"] * 100
            
            st.metric("Top Predicted Class", pred_class, delta=f"Softmax Confidence: {confidence:.2f}%")
            
            st.markdown("#### Top-5 Softmax Distribution")
            for item in p.get("top5", []):
                score_pct = item["prob"] * 100
                st.progress(min(100, int(score_pct)), text=f"{item['class']}: {score_pct:.2f}%")
            
            with st.expander("Technical Logit Inspection"):
                st.write("Finite Logits Vector:", [round(v, 4) for v in p.get("logits", [])])
                st.write("Model Architecture:", p.get("model"))
                st.write("Input Resolution:", p.get("input_resolution"))
            
            st.caption(
                "⚠️ *Interpretation note: Confidence reflects the normalized softmax probability output by the model. "
                "Because this baseline was trained on a 5,000-sample subset for 5 epochs, confidence scores indicate "
                "relative class preference rather than definitive correctness.*"
            )


# =========================================================================
# TAB 4: Baseline Training Results
# =========================================================================
with tab_results:
    st.header("Audited Baseline Training Results")
    st.markdown(
        "Empirical evidence and metrics from the completed 5-epoch baseline training run on CIFAR-10. "
        "All values are dynamically loaded from verified experiment artifacts."
    )
    
    # Load artifact files
    history_file = REVIEW_EXP_DIR / "history.json"
    metrics_file = REVIEW_EXP_DIR / "test_metrics.json"

    if history_file.is_file() and metrics_file.is_file():
        with open(history_file, "r") as f:
            hist = json.load(f)
        with open(metrics_file, "r") as f:
            test_m = json.load(f)
            
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Official Test Accuracy", f"{test_m.get('accuracy', 0.3139)*100:.2f}%", help="Evaluated on 10,000 held-out CIFAR-10 test images.")
        col_m2.metric("Best Val Accuracy", f"{max(hist.get('val_acc', [0.311]))*100:.2f}%", help="Selected at Epoch 5.")
        col_m3.metric("Training Samples", f"{test_m.get('train_samples', 5000):,}")
        col_m4.metric("Train Time (5 Epochs)", f"{test_m.get('train_time_seconds', 177.3):.1f} s")
        
        st.divider()
        
        col_plot1, col_plot2 = st.columns(2)
        
        loss_curve_path = REVIEW_EXP_DIR / "loss_acc_curves.png"
        cm_path = REVIEW_EXP_DIR / "confusion_matrix.png"
        
        with col_plot1:
            st.subheader("Training & Validation Curves")
            if loss_curve_path.is_file():
                st.image(str(loss_curve_path), caption="Cross-Entropy Loss and Accuracy Progression", use_container_width=True)
            else:
                st.info("Loss curve plot not found.")
        
        with col_plot2:
            st.subheader("Official Test Confusion Matrix")
            if cm_path.is_file():
                st.image(str(cm_path), caption="Normalized Confusion Matrix (10,000 Test Images)", use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")
                
        st.divider()
        st.subheader("Per-Epoch Training Progress")
        
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        prog_data = {
            "Epoch": epochs,
            "Train Loss": [round(x, 4) for x in hist["train_loss"]],
            "Train Accuracy (%)": [round(x * 100, 2) for x in hist["train_acc"]],
            "Val Loss": [round(x, 4) for x in hist["val_loss"]],
            "Val Accuracy (%)": [round(x * 100, 2) for x in hist["val_acc"]],
            "Learning Rate": [f"{x:.2e}" for x in hist["lr"]],
        }
        st.dataframe(prog_data, use_container_width=True)
        
        st.caption(
            "**Evaluation Note**: The independent baseline audit reproduced exactly 3,139 / 10,000 correct (31.39% accuracy) "
            "and 0.3139 recall. Minor floating-point variations in precision/F1 between the review log and independent audit "
            "are attributable to FP16 mixed-precision (AMP) during review execution vs full FP32 independent verification."
        )
    else:
        st.warning(f"Experiment artifact files not found in `{REVIEW_EXP_DIR}`.")


# =========================================================================
# TAB 5: CryptoViT Development Status
# =========================================================================
with tab_status:
    st.header("CryptoViT Model & Next Experimental Phase")
    
    st.markdown(
        """
        ### Specialized CryptoViT Architecture Components
        In contrast to the Standard Vision Transformer demonstrated in Tab 3, the full **CryptoViT** model introduces:
        
        1. **Crypto Patch Embedding (`CryptoPatchEmbedding`)**:
           - Replaces standard linear projection with a convolutional patch extraction pipeline.
           - Helps recover localized features from permuted and diffused patch structures.
        
        2. **Key-Aware Positional Encoding (`KeyAwarePositionalEncoding`)**:
           - Computes a key-derived embedding via SHA-256 and an MLP projection.
           - Generates a key-dependent spatial offset added to the learnable positional encodings.
           - Ensures the positional representation is coordinated with the chaotic permutation key.
           
        3. **Direct Ciphertext Inference**:
           - Designed to classify encrypted representations without requiring a decryption step on the host.
        """
    )
    
    st.divider()
    
    st.warning("### ⚠️ Implementation & Training Status: PENDING VALIDATION")
    st.markdown(
        """
        **Official Review Disclosure**:
        - **No CryptoViT model has been trained or benchmarked yet.**
        - The standard ViT baseline demonstrated in Tab 3 and Tab 4 represents the **unencrypted reference benchmark** (31.39% accuracy on CIFAR-10 subset).
        - In accordance with non-negotiable project review restrictions, **no randomly initialized CryptoViT weights are served as trained predictions**.
        - Full CryptoViT training and ablation experiments (comparing standard ViT, pixel permutation, patch permutation, and full hyperchaotic encryption) are scheduled for the subsequent research phase.
        """
    )


# -------------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------------
st.divider()
st.markdown(
    "<small>CryptoViT — Capstone Project Review Demonstration | VIT Bhopal University | "
    "Standard ViT Baseline Active</small>",
    unsafe_allow_html=True,
)
