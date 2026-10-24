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
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="CryptoViT Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# Public research key the shipped models were trained with. Encrypted-image
# models only work with this key; anything else shows what the key does.
DEFAULT_KEY = "CryptoViT_Research_Key_2025"

# Pre-selected test images for the live demo (PNG files), if the folder exists.
DEMO_DIR = Path(__file__).resolve().parents[1] / "demo_images"

# -------------------------------------------------------------------------
# Sidebar - configuration
# -------------------------------------------------------------------------

st.sidebar.title("CryptoViT")
st.sidebar.markdown(
    "**Chaos-Assisted Vision Transformer**\n\n"
    "Privacy-Preserving Image Classification"
)
st.sidebar.divider()

api_url = st.sidebar.text_input(
    "API URL",
    value=API_URL,
    help="URL of the CryptoViT FastAPI backend.",
)

st.sidebar.divider()
key = st.sidebar.text_input(
    "Encryption key",
    value=DEFAULT_KEY,
    help="Used to encrypt the image. The encrypted-image models were trained "
         "with the default key only.",
)
wrong_key = st.sidebar.text_input(
    "Wrong key (for the failure demo)",
    value="wrong_key",
    help="Used by the 'wrong key' buttons below.",
)

demo_files = sorted(DEMO_DIR.glob("*.png")) if DEMO_DIR.is_dir() else []
demo_choice = None
if demo_files:
    demo_choice = st.sidebar.selectbox(
        "Demo image",
        options=[None] + demo_files,
        format_func=lambda p: "- none (upload below) -" if p is None else p.stem,
        help="Pre-selected test images from the dataset.",
    )

st.sidebar.divider()
show_security = st.sidebar.checkbox("Show security metrics", value=True)
show_comparison = st.sidebar.checkbox("Show model comparison", value=True)
show_wrong_key_predictions = st.sidebar.checkbox(
    "Also classify with the wrong key", value=False
)

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
        st.error("Cannot connect to API. Is the server running?")
        return None
    except requests.exceptions.HTTPError:
        return None   # e.g. 404 before any metrics/results exist


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
        st.error("Cannot connect to API.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None


# -------------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------------

st.title("CryptoViT - Privacy-Preserving Medical Image Classification")
st.markdown(
    "The image is **scrambled with a secret key** and then classified by a "
    "Vision Transformer **without ever being decrypted**."
)
st.warning(
    "Research prototype for a student project. **Not a medical device and not "
    "for clinical use.** Do not upload real patient images.",
)

# Check API health
class_names: list = []
native_size = 32
health_resp = _api_get("/health")
if health_resp:
    h = health_resp.json()
    class_names = h.get("class_names", [])
    native_size = h.get("image_size", 32)

# Image source: sidebar demo image, otherwise an upload
uploaded = st.file_uploader(
    f"Upload an image (JPEG / PNG) - it is reduced to {native_size}x{native_size} px",
    type=["jpg", "jpeg", "png"],
    help="The models only know this dataset's image type. Photos of other "
         "things will get a meaningless answer.",
)

img_bytes = None
if demo_choice is not None:
    img_bytes = demo_choice.read_bytes()
elif uploaded is not None:
    img_bytes = uploaded.read()

if img_bytes is not None:
    orig_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # ------------------------------------------------------------------ #
    # Step 1: Display original                                            #
    # ------------------------------------------------------------------ #
    col_orig, col_enc, col_pred = st.columns(3)

    with col_orig:
        st.subheader("1. Original Image")
        shown = orig_pil.resize((256, 256), Image.NEAREST) if max(orig_pil.size) <= 64 else orig_pil
        st.image(shown, caption="Plaintext input (enlarged for display)", width="stretch")
        st.caption(f"Size: {orig_pil.size[0]}x{orig_pil.size[1]} px")

    # ------------------------------------------------------------------ #
    # Step 2: Encrypt                                                     #
    # ------------------------------------------------------------------ #
    with col_enc:
        st.subheader("2. Encrypted Image")
        with st.spinner("Encrypting..."):
            enc_resp = _api_post_image("/encrypt", _pil_to_bytes(orig_pil), key=key)

        if enc_resp:
            enc_pil = _bytes_to_pil(enc_resp.content)
            key_hash = enc_resp.headers.get("X-Encryption-KeyHash", "unknown")
            shown_enc = enc_pil.resize((256, 256), Image.NEAREST) if max(enc_pil.size) <= 64 else enc_pil
            st.image(shown_enc, caption="Encrypted representation", width="stretch")
            st.caption(f"Key hash: `{key_hash[:16]}...`")
            st.download_button(
                "Download encrypted image",
                data=enc_resp.content,
                file_name="encrypted.png",
                mime="image/png",
            )
        else:
            enc_pil = None

    # ------------------------------------------------------------------ #
    # Step 3: Classify the encrypted image (no decryption)                #
    # ------------------------------------------------------------------ #
    cmp_data = None
    with col_pred:
        st.subheader("3. Prediction on the encrypted image")
        if enc_pil is not None:
            with st.spinner("Classifying encrypted image..."):
                cmp_resp = _api_post_image("/compare", _pil_to_bytes(orig_pil), key=key)
            if cmp_resp:
                cmp_data = cmp_resp.json()
                encrypted_preds = cmp_data.get("encrypted", {})
                # Show the CryptoViT (D) result if available, otherwise the best other
                main = (encrypted_preds.get("D") or encrypted_preds.get("C")
                        or encrypted_preds.get("B"))
                if main:
                    st.caption(main["model"])
                    st.metric(
                        label="Predicted Class",
                        value=main["class_name"].upper(),
                        delta=f"Confidence: {main['confidence']*100:.1f}%",
                    )
                    st.markdown("**Top predictions:**")
                    for item in main.get("top5", []):
                        prob_pct = item["prob"] * 100
                        st.progress(int(prob_pct), text=f"{item['class']}: {prob_pct:.1f}%")

    st.divider()

    # ------------------------------------------------------------------ #
    # What the key does: decrypt with the right and a wrong key           #
    # ------------------------------------------------------------------ #
    if enc_resp:
        st.subheader("What the key does")
        st.caption(
            "Decryption is only possible with the key used to encrypt. With a "
            "wrong key the result is noise, not an error."
        )
        k1, k2 = st.columns(2)
        reference = orig_pil.resize((native_size, native_size), Image.LANCZOS).convert("RGB")

        with k1:
            if st.button("Decrypt with the correct key", width="stretch"):
                dec = _api_post_image("/decrypt", enc_resp.content, key=key)
                if dec:
                    restored = _bytes_to_pil(dec.content).convert("RGB")
                    st.image(restored.resize((256, 256), Image.NEAREST),
                             caption="Decrypted with the correct key", width="stretch")
                    if list(restored.getdata()) == list(reference.getdata()):
                        st.success("Exactly identical to the original image.")
                    else:
                        st.warning("Not identical to the original (is the key the one used to encrypt?).")
        with k2:
            if st.button("Decrypt with a wrong key", width="stretch"):
                dec = _api_post_image("/decrypt", enc_resp.content, key=wrong_key)
                if dec:
                    st.image(_bytes_to_pil(dec.content).resize((256, 256), Image.NEAREST),
                             caption=f"Decrypted with the wrong key '{wrong_key}'",
                             width="stretch")
                    st.error("The image is not recovered - only noise.")
        st.divider()

    # ------------------------------------------------------------------ #
    # Security metrics panel                                              #
    # ------------------------------------------------------------------ #
    if show_security:
        st.subheader("Security Metrics")
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
                help="Target: about 0",
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
    def _comparison_rows(data: dict) -> list:
        """Table rows: the plaintext model first, then the encrypted-image models."""
        rows = []
        plain = data.get("plaintext")
        if plain:
            rows.append(("A", plain, "plaintext image"))
        for variant in ("B", "C", "D"):
            if variant in data.get("encrypted", {}):
                rows.append((variant, data["encrypted"][variant], "encrypted image"))
        return [
            {
                "Variant": v,
                "Model": p["model"],
                "Sees": sees,
                "Prediction": p["class_name"],
                "Confidence": f"{p['confidence'] * 100:.1f}%",
                "Trained": "yes" if p.get("trained") else "untrained",
            }
            for v, p, sees in rows
        ]

    if show_comparison and cmp_data is not None:
        st.subheader("Model comparison")
        st.caption(
            "Variant A sees the plaintext image. B, C and D only ever see the "
            f"encrypted image (key: '{key}')."
        )
        st.dataframe(_comparison_rows(cmp_data), hide_index=True, width="stretch")

        preds = [r["Prediction"] for r in _comparison_rows(cmp_data) if r["Trained"] == "yes"]
        if len(preds) >= 2:
            if len(set(preds)) == 1:
                st.success("All trained models agree on this image.")
            else:
                st.warning("The trained models disagree on this image.")

        if show_wrong_key_predictions:
            wk_resp = _api_post_image("/compare", _pil_to_bytes(orig_pil), key=wrong_key)
            if wk_resp:
                st.markdown(f"**Same image encrypted with the wrong key '{wrong_key}':**")
                wk_rows = [r for r in _comparison_rows(wk_resp.json()) if r["Sees"] == "encrypted image"]
                st.dataframe(wk_rows, hide_index=True, width="stretch")
                st.caption(
                    "The encrypted-image models were trained on images scrambled "
                    "with the correct key, so their answers on other scrambles are unreliable."
                )

else:
    st.info("Pick a demo image in the sidebar or upload an image to get started.")

# -------------------------------------------------------------------------
# Test results of the trained models (shown once the training run has finished)
# -------------------------------------------------------------------------

st.divider()
st.subheader("Test results of the trained models")
results_resp = _api_get("/results")
if results_resp is None:
    st.info("Results appear here once the training run has finished.")
else:
    results = results_resp.json()
    melanoma_idx = class_names.index("melanoma") if "melanoma" in class_names else None

    def _fmt(value, pct=False):
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%" if pct else f"{value:.3f}"

    table = []
    for r in results:
        sens = r.get("per_class_sensitivity") or []
        row = {
            "Variant": r["variant"],
            "Model": r["label"],
            "Balanced accuracy": _fmt(r.get("balanced_accuracy"), pct=True),
            "Accuracy": _fmt(r.get("accuracy"), pct=True),
            "Macro AUC": _fmt(r.get("auc_macro_ovr")),
        }
        if melanoma_idx is not None and melanoma_idx < len(sens):
            row["Melanoma sensitivity"] = _fmt(sens[melanoma_idx], pct=True)
        row["Epochs"] = r.get("epochs_trained", "?")
        table.append(row)
    st.dataframe(table, hide_index=True, width="stretch")
    n_test = results[0].get("num_samples")
    st.caption(
        f"Measured once on the held-out test set ({n_test} images) after training. "
        "Balanced accuracy is the mean of the per-class sensitivities, so a model "
        "that only guesses the most common class scores low."
    )

    chosen = st.selectbox(
        "Per-class sensitivity of", [r["variant"] for r in results],
        format_func=lambda v: next(f"{r['variant']} - {r['label']}" for r in results if r["variant"] == v),
    )
    sel = next(r for r in results if r["variant"] == chosen)
    sens = sel.get("per_class_sensitivity") or []
    if sens and class_names:
        support = sel.get("per_class_support") or [None] * len(sens)
        # Horizontal bars: class names read left to right, so there is no need
        # to tilt your head. The percentage is part of each label.
        chart_df = pd.DataFrame({
            "Class": [f"{n}  ({s * 100:.0f}%)" for n, s in zip(class_names, sens)],
            "Sensitivity": sens,
            "Test images": support[: len(sens)],
            "Is melanoma": [n == "melanoma" for n in class_names[: len(sens)]],
        })
        bars = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=22)
            .encode(
                y=alt.Y(
                    "Class:N",
                    sort=alt.EncodingSortField(field="Sensitivity", order="descending"),
                    title=None,
                    axis=alt.Axis(labelLimit=520, labelFontSize=13, labelAngle=0),
                ),
                x=alt.X(
                    "Sensitivity:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Sensitivity: share of this class that was found",
                    axis=alt.Axis(format="%", tickCount=5, labelAngle=0, grid=True),
                ),
                color=alt.condition(
                    alt.datum["Is melanoma"],
                    alt.value("#e4572e"),
                    alt.value("#4c78a8"),
                ),
                tooltip=["Class", alt.Tooltip("Sensitivity:Q", format=".1%"), "Test images"],
            )
            .properties(height=42 * len(chart_df))
        )
        st.altair_chart(bars, width="stretch")
        st.caption("Melanoma is highlighted in orange.")
