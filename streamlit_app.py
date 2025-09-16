# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os, json, math
import matplotlib.pyplot as plt

st.set_page_config(page_title="🕵️ Ethereum Fraud Detection", layout="wide")
st.title("🕵️ Ethereum Fraud Detection Dashboard")
st.write("A hybrid model using tabular + Alchemy features to detect fraud types in Ethereum DeFi.")

# --- File paths ---
MODEL_PATH = "artifacts/hybrid_with_alchemy_rep_logreg.joblib"
SCALER_PATH = "artifacts/hybrid_with_alchemy_scaler.joblib"
CACHE_PATH = "artifacts/alchemy_cache.json"

# --- Load model & scaler (cached) ---
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

# --- Inspect scaler to find expected features ---
expected_n = getattr(scaler, "n_features_in_", None)
feature_names_in = getattr(scaler, "feature_names_in_", None)

st.sidebar.markdown("**Model info**")
st.sidebar.write(f"- Scaler expects **{expected_n}** features")
if feature_names_in is not None:
    st.sidebar.write(f"- Scaler has stored feature names (used automatically).")
else:
    st.sidebar.write("- Scaler does not expose feature names; using fallback order.")

# --- Fallback feature order (Alchemy features) ---
FALLBACK_FEATURE_ORDER = [
    "alchemy_tx_count",
    "alchemy_sent_count",
    "alchemy_recv_count",
    "alchemy_unique_senders",
    "alchemy_unique_receivers",
    "alchemy_avg_min_between_sent",
    "alchemy_avg_min_between_recv",
    "alchemy_time_span_mins",
    "alchemy_contract_interactions",
    "alchemy_is_contract",
    "alchemy_missing"
]

# Decide which feature order to use
if feature_names_in is not None:
    FEATURE_ORDER = list(feature_names_in)
else:
    FEATURE_ORDER = FALLBACK_FEATURE_ORDER

# Validate length
if expected_n is not None and len(FEATURE_ORDER) != expected_n:
    st.warning(
        f"Scaler expects {expected_n} features but we have {len(FEATURE_ORDER)} in the chosen FEATURE_ORDER.\n"
        "If predictions fail, you may need to update FEATURE_ORDER to exactly match your training features."
    )

# --- Load cached features ---
@st.cache_data
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

cache = load_cache()

# --- safe float coercion ---
def safe_float(x):
    if x is None:
        return 0.0
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float, np.number)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return 0.0
        return float(x)
    try:
        s = str(x).strip()
        if s.lower() in ("true", "yes"): return 1.0
        if s.lower() in ("false", "no"): return 0.0
        s2 = s.replace(",", "").replace("%", "")
        return float(s2)
    except Exception:
        return 0.0

# --- Sidebar input & helper dropdown of cached addresses ---
st.sidebar.header("🔎 Try an Address")
addresses = list(cache.keys())
if len(addresses) > 0:
    sample_addrs = addresses[:200]  # limit to first 200 for dropdown safety
else:
    sample_addrs = []
addr_choice = st.sidebar.selectbox("Or pick from cached addresses:", ["-- pick one --"] + sample_addrs)
eth_address = st.sidebar.text_input("Or paste an Ethereum address (0x...)", value="")

# If user picks from dropdown, use that value
if addr_choice and addr_choice != "-- pick one --":
    eth_address = addr_choice

# --- Prediction ---
if st.sidebar.button("Predict"):
    if not eth_address:
        st.warning("Please enter or pick an address.")
    else:
        addr = eth_address.strip()
        if addr not in cache:
            st.warning("Address not found in cache. Use a cached address from the dropdown or enrich the address externally.")
        else:
            feats = cache[addr]
            # build ordered numeric vector
            vec = [safe_float(feats.get(f, 0.0)) for f in FEATURE_ORDER]
            X = np.array([vec], dtype=float)

            # check dimensions before scaling
            if expected_n is not None and X.shape[1] != expected_n:
                st.error(
                    f"Feature length mismatch: input has {X.shape[1]} columns but scaler expects {expected_n}.\n"
                    "This means FEATURE_ORDER does not match training features. See repo README or regenerate artefacts."
                )
            else:
                try:
                    X_scaled = scaler.transform(X)
                except Exception as e:
                    st.error(f"Scaler transform failed: {e}")
                    st.write("Raw feature vector:", vec)
                    raise

                try:
                    probs = model.predict_proba(X_scaled)[0]
                    pred = model.predict(X_scaled)[0]
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    raise

                # Show neat result
                st.markdown("## Prediction")
                st.success(f"**{pred.upper()}**")

                # probabilities
                prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probs})
                prob_df = prob_df.sort_values("Probability", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(prob_df["Class"], prob_df["Probability"], color="#2b6cb0")
                for i, (cls, prob) in enumerate(zip(prob_df["Class"], prob_df["Probability"])):
                    ax.text(prob + 0.01, i, f"{prob:.2f}", va="center")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                st.pyplot(fig)

# --- Model evaluation visuals ---
st.header("📊 Model Evaluation")
col1, col2 = st.columns(2)
with col1:
    cm_path = "artifacts/plots/confusion_hybrid_with_alchemy.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.info("Confusion matrix plot not found in artifacts/plots.")
with col2:
    pr_path = "artifacts/plots/perclass_pr_hybrid_with_alchemy.png"
    if os.path.exists(pr_path):
        st.image(pr_path, caption="Per-class Precision/Recall", use_column_width=True)
    else:
        st.info("Per-class PR plot not found in artifacts/plots.")

st.markdown("---")
st.caption("Tips: Use the dropdown to select cached addresses (faster). To add new addresses, re-run enrichment in your training environment and upload the updated cache to artifacts/alchemy_cache.json.")
