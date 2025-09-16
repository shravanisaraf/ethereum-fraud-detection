import streamlit as st
import joblib
import numpy as np
import os, json

st.set_page_config(page_title="Ethereum Fraud Detection", layout="wide")
st.title("🕵️ Ethereum Fraud Detection Dashboard")

# --- Load model & scaler ---
MODEL_PATH = "artifacts/hybrid_with_alchemy_rep_logreg.joblib"
SCALER_PATH = "artifacts/hybrid_with_alchemy_scaler.joblib"
CACHE_PATH = "artifacts/alchemy_cache.json"

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

# --- User input ---
eth_address = st.text_input("Enter Ethereum address (0x...)", "")

if st.button("Predict"):
    if eth_address in cache:
        feats = cache[eth_address]
        X = np.array([list(feats.values())], dtype=float)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted fraud type: **{pred}**")
        st.write("Probabilities:", model.predict_proba(X_scaled))
    else:
        st.warning("Address not found in cache. (For demo, use cached dataset addresses.)")

# --- Show plots ---
st.header("📊 Model Results")
if os.path.exists("artifacts/plots/confusion_hybrid_with_alchemy.png"):
    st.image("artifacts/plots/confusion_hybrid_with_alchemy.png", caption="Confusion Matrix")
if os.path.exists("artifacts/plots/perclass_pr_hybrid_with_alchemy.png"):
    st.image("artifacts/plots/perclass_pr_hybrid_with_alchemy.png", caption="Per-class Precision/Recall")
