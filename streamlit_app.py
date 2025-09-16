import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os, json, math
import matplotlib.pyplot as plt

st.set_page_config(page_title="🕵️ Ethereum Fraud Detection", layout="wide")
st.title("Ethereum Fraud Detection Dashboard")
st.write("A hybrid model using **tabular + Alchemy features** to detect fraud types in Ethereum DeFi.")

# --- File paths ---
MODEL_PATH = "artifacts/hybrid_with_alchemy_rep_logreg.joblib"
SCALER_PATH = "artifacts/hybrid_with_alchemy_scaler.joblib"
CACHE_PATH = "artifacts/alchemy_cache.json"

# --- Load model & scaler ---
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# --- Load cached features ---
@st.cache_resource
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

cache = load_cache()

# --- Feature order (adjust if your scaler was trained differently) ---
FEATURE_ORDER = [
    "alchemy_avg_min_between_sent",
    "alchemy_avg_min_between_recv",
    "alchemy_time_span_mins",
    "alchemy_sent_count",
    "alchemy_recv_count",
    "alchemy_unique_senders",
    "alchemy_unique_receivers",
    "alchemy_contract_interactions",
    "alchemy_is_contract",
    "alchemy_missing"
]

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

# --- Sidebar input ---
st.sidebar.header("🔎 Try an Address")
eth_address = st.sidebar.text_input("Ethereum address (from cache)", "")

if st.sidebar.button("Predict"):
    if eth_address in cache:
        feats = cache[eth_address]
        vec = [safe_float(feats.get(fname, 0.0)) for fname in FEATURE_ORDER]
        X = np.array([vec], dtype=float)
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]

        # --- Show result ---
        st.success(f"**Prediction: {pred.upper()}**")
        st.subheader("Class probabilities")
        prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probs})
        prob_df = prob_df.sort_values("Probability", ascending=True)

        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(prob_df["Class"], prob_df["Probability"], color="#2b6cb0")
        for i, (cls, prob) in enumerate(zip(prob_df["Class"], prob_df["Probability"])):
            ax.text(prob + 0.01, i, f"{prob:.2f}", va="center")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

    else:
        st.warning("Address not found in cache. (Pick one from `artifacts/alchemy_cache.json`).")

# --- Results section ---
st.header("📊 Model Evaluation")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists("artifacts/plots/confusion_hybrid_with_alchemy.png"):
        st.image("artifacts/plots/confusion_hybrid_with_alchemy.png", caption="Confusion Matrix")
with col2:
    if os.path.exists("artifacts/plots/perclass_pr_hybrid_with_alchemy.png"):
        st.image("artifacts/plots/perclass_pr_hybrid_with_alchemy.png", caption="Per-class Precision/Recall")
