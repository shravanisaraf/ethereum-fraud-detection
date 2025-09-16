import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os, json, math
import matplotlib.pyplot as plt

st.set_page_config(page_title="🕵️ Ethereum Fraud Detection", layout="wide")
st.title("🕵️ Ethereum Fraud Detection Dashboard")
st.write("A hybrid model using **Alchemy + tabular features** to detect fraud types in Ethereum DeFi.")

# --- File paths ---
MODEL_PATH = "artifacts/ablation_aug_model.joblib"
CACHE_PATH = "artifacts/alchemy_cache.json"

# --- Load model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# --- Load cached features ---
@st.cache_data
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

cache = load_cache()

# --- Fixed feature order (13 features) ---
FEATURE_ORDER = [
  "alchemy_avg_min_between_sent",
  "alchemy_avg_min_between_recv",
  "alchemy_time_span_mins",
  "alchemy_sent_count",
  "alchemy_recv_count",
  "alchemy_unique_receivers",
  "alchemy_unique_senders",
  "alchemy_tx_count",
  "alchemy_contract_interactions",
  "alchemy_is_contract",
  "alchemy_missing",
  "tx_count",              # extra tabular feature
  "unique_senders"         # extra tabular feature
]

# --- Helper to coerce values ---
def safe_float(x):
    if x is None:
        return 0.0
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float, np.number)):
        return float(x)
    try:
        s = str(x).strip()
        if s.lower() in ("true", "yes"): return 1.0
        if s.lower() in ("false", "no"): return 0.0
        return float(s.replace(",", "").replace("%", ""))
    except Exception:
        return 0.0

# --- Sidebar input ---
st.sidebar.header("🔎 Try an Address")
addresses = list(cache.keys())
addr_choice = st.sidebar.selectbox("Pick from cached addresses:", ["-- pick one --"] + addresses[:200])
eth_address = st.sidebar.text_input("Or paste an Ethereum address (0x...)", "")

if addr_choice != "-- pick one --":
    eth_address = addr_choice

# --- Prediction button ---
if st.sidebar.button("Predict"):
    if not eth_address:
        st.warning("Please enter or pick an address.")
    elif eth_address not in cache:
        st.warning("Address not found in cache. Use one from the dropdown.")
    else:
        feats = cache[eth_address]
        vec = [safe_float(feats.get(f, 0.0)) for f in FEATURE_ORDER]
        X = np.array([vec], dtype=float)

        try:
            probs = model.predict_proba(X)[0]
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            # Show prediction
            st.markdown("## 🔮 Prediction")
            st.success(f"**{pred.upper()}**")

            # Probabilities bar chart
            prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probs})
            prob_df = prob_df.sort_values("Probability", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(prob_df["Class"], prob_df["Probability"], color="#2b6cb0")
            for i, (cls, prob) in enumerate(zip(prob_df["Class"], prob_df["Probability"])):
                ax.text(prob + 0.01, i, f"{prob:.2f}", va="center")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)
