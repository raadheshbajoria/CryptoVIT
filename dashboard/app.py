"""
dashboard/app.py
==================
Streamlit demonstration interface for CryptoViT.

Demonstrates:
  1. Upload a plaintext image
  2. Encrypt it with hyperchaotic encryption
  3. Classify the encrypted image with CryptoViT
  4. Show security metrics panel
  5. Compare with plaintext baseline ViT

Running
-------
    streamlit run dashboard/app.py

Or with API backend:
    # Start API: uvicorn api.main:app --port 8000
    # Then: streamlit run dashboard/app.py
"""

from __future__ import annotations

import io

import requests
import streamlit as st
from PIL import Image

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="CryptoViT Demo",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# -------------------------------------------------------------------------
# Sidebar — configuration
# -------------------------------------------------------------------------

st.sidebar.title("🔐 CryptoViT")
st.sidebar.markdown(
    "**Chaos-Assisted Vision Transformer**\n\n"
    "Privacy-Preserving Image Classification\n\n"
    "*VIT Bhopal University — B.Tech ECE+AI/Cybernetics*"
)
st.sidebar.divider()

api_url = st.sidebar.text_input(
    "API URL",
    value=API_URL,
    help="URL of the CryptoViT FastAPI backend.",
)

show_security = st.sidebar.checkbox("Show security metrics", value=True)
show_comparison = st.sidebar.checkbox("Show baseline comparison", value=True)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


def _pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _api_get(endpoint: str):
    try:
        r = requests.get(f"{api_url}{endpoint}", timeout=10)
        r.raise_for_status()
        return r
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Is the server running?")
        return None


def _api_post_image(endpoint: str, img_bytes: bytes, **kwargs):
    try:
        r = requests.post(
            f"{api_url}{endpoint}",
            files={"file": ("image.png", img_bytes, "image/png")},
            data=kwargs,
            timeout=30,
        )
        r.raise_for_status()
        return r
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None


# -------------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------------

st.title("🔐 CryptoViT — Privacy-Preserving Image Classification")
st.markdown(
    "Upload an image to see it encrypted and classified by the "
    "**Chaos-Assisted Vision Transformer** pipeline."
)

# Check API health
health_resp = _api_get("/health")
if health_resp:
    h = health_resp.json()
    col1, col2, col3 = st.columns(3)
    col1.metric("Device", h.get("device", "unknown"))
    col2.metric("Baseline ViT", "✅ Loaded" if h["baseline_loaded"] else "⚠️ Not loaded")
    col3.metric("CryptoViT", "✅ Loaded" if h["cryptovit_loaded"] else "⚠️ Not loaded")
    st.divider()

# File upload
uploaded = st.file_uploader(
    "Upload an image (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Small images (32×32) are best. Larger images will be resized to 32×32 for CIFAR-scale demo.",
)

if uploaded is not None:
    img_bytes = uploaded.read()
    orig_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # ------------------------------------------------------------------ #
    # Step 1: Display original                                            #
    # ------------------------------------------------------------------ #
    col_orig, col_enc, col_pred = st.columns(3)

    with col_orig:
        st.subheader("1️⃣ Original Image")
        st.image(orig_pil, caption="Plaintext input", use_container_width=True)
        st.caption(f"Size: {orig_pil.size[0]}×{orig_pil.size[1]} px")

    # ------------------------------------------------------------------ #
    # Step 2: Encrypt                                                     #
    # ------------------------------------------------------------------ #
    with col_enc:
        st.subheader("2️⃣ Encrypted Image")
        with st.spinner("Encrypting…"):
            enc_resp = _api_post_image("/encrypt", _pil_to_bytes(orig_pil))

        if enc_resp:
            enc_pil = _bytes_to_pil(enc_resp.content)
            key_hash = enc_resp.headers.get("X-Encryption-KeyHash", "unknown")
            st.image(enc_pil, caption="Encrypted representation", use_container_width=True)
            st.caption(f"Key hash: `{key_hash[:16]}…`")
            st.download_button(
                "Download encrypted image",
                data=enc_resp.content,
                file_name="encrypted.png",
                mime="image/png",
            )
        else:
            enc_pil = None

    # ------------------------------------------------------------------ #
    # Step 3: Classify encrypted with CryptoViT                          #
    # ------------------------------------------------------------------ #
    with col_pred:
        st.subheader("3️⃣ CryptoViT Prediction")
        if enc_pil is not None:
            with st.spinner("Classifying encrypted image…"):
                pred_resp = _api_post_image(
                    "/predict", _pil_to_bytes(orig_pil),
                    encrypt_first="true",
                )
            if pred_resp:
                p = pred_resp.json()
                st.metric(
                    label="Predicted Class",
                    value=p["class_name"].upper(),
                    delta=f"Confidence: {p['confidence']*100:.1f}%",
                )
                st.markdown("**Top-5 predictions:**")
                for item in p.get("top5", []):
                    prob_pct = item["prob"] * 100
                    st.progress(int(prob_pct), text=f"{item['class']}: {prob_pct:.1f}%")

    st.divider()

    # ------------------------------------------------------------------ #
    # Security metrics panel                                              #
    # ------------------------------------------------------------------ #
    if show_security:
        st.subheader("🔒 Security Metrics")
        metrics_resp = _api_get("/metrics")
        if metrics_resp:
            m = metrics_resp.json()
            ent = m.get("entropy", {})
            corr = m.get("correlation_encrypted", {})
            npcr_val = m.get("npcr_plain_vs_enc", 0)
            uaci_val = m.get("uaci_plain_vs_enc", 0)
            chi = m.get("histogram", {})

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                "Entropy (overall)", f"{ent.get('overall', 0):.4f}",
                help=(
                    "Reference for uniform noise at this image size: "
                    f"{ent.get('expected_for_uniform_noise', 8.0):.4f}. "
                    "The usual '> 7.99' only applies to images of about "
                    "256x256 pixels or more."
                ),
            )
            mc2.metric(
                "Pixels changed vs plaintext", f"{npcr_val:.2f}%",
                help="Share of pixels that differ from the original image. "
                     "Not the standard NPCR (see plaintext sensitivity below).",
            )
            mc3.metric(
                "Mean change vs plaintext", f"{uaci_val:.2f}%",
                help="Mean absolute intensity change from the original image. "
                     "Not the standard UACI.",
            )
            mc4.metric(
                "H.Correlation",
                f"{corr.get('horizontal', 0):.4f}",
                help="Target: ≈ 0",
            )

            ks = m.get("key_sensitivity")
            ps = m.get("plaintext_sensitivity")
            if ks or ps:
                s1, s2, s3, s4 = st.columns(4)
                if ks:
                    s1.metric("Key sensitivity NPCR", f"{ks.get('npcr', 0):.2f}%",
                              help="Ciphertexts under keys differing by one bit.")
                    s2.metric("Key sensitivity UACI", f"{ks.get('uaci', 0):.2f}%")
                if ps:
                    s3.metric("Plaintext sensitivity NPCR", f"{ps.get('npcr', 0):.2f}%",
                              help="One plaintext pixel changed. Ideal ~99.6%.")
                    s4.metric("Plaintext sensitivity UACI", f"{ps.get('uaci', 0):.2f}%",
                              help="One plaintext pixel changed. Ideal ~33.46%.")

    # ------------------------------------------------------------------ #
    # Baseline comparison                                                 #
    # ------------------------------------------------------------------ #
    if show_comparison:
        st.subheader("📊 Baseline Comparison")
        with st.spinner("Classifying plaintext with baseline ViT…"):
            base_resp = _api_post_image("/predict_plaintext", _pil_to_bytes(orig_pil))

        if base_resp and enc_pil is not None:
            b = base_resp.json()
            st.markdown(f"**Baseline ViT (plaintext):** {b['class_name'].upper()} ({b['confidence']*100:.1f}%)")
            if 'pred_resp' in locals() and pred_resp is not None:
                p = pred_resp.json()
                st.markdown(f"**CryptoViT (encrypted):** {p['class_name'].upper()} ({p['confidence']*100:.1f}%)")
                agree = b["class_name"] == p["class_name"]
                if agree:
                    st.success("✅ Both models agree on the predicted class.")
                else:
                    st.warning("⚠️ Models disagree — this is expected when CryptoViT is not fully trained.")

else:
    st.info("👆 Upload an image to get started.")

# -------------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------------

st.divider()
st.markdown(
    "<small>CryptoViT — Final Year B.Tech Capstone | VIT Bhopal University | "
    "ECE + AI/Cybernetics</small>",
    unsafe_allow_html=True,
)
