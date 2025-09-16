# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os, json, math, traceback
import matplotlib.pyplot as plt

st.set_page_config(page_title="🕵️ Ethereum Fraud Detection", layout="wide")
st.title("🕵️ Ethereum Fraud Detection Dashboard")
st.write("A hybrid model using **Alchemy + tabular features** to detect fraud types in Ethereum DeFi.")

# ---------- CONFIG ----------
MODEL_PATH = "artifacts/ablation_aug_model.joblib"
CACHE_PATH = "artifacts/alchemy_cache.json"
LABELS_CSV = "artifacts/labels_with_alchemy.csv"  # optional fallback source for tabular cols

# 13-feature order (model that caused earlier 'expected 13'): alchemy (11) + 2 tabular extras
FEATURE_ORDER_13 = [
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
  "tx_count",            # tabular from original dataset (fallback)
  "unique_senders"       # tabular from original dataset (fallback)
]

# 11-feature Alchemy-only order (your scaler's features)
FEATURE_ORDER_11 = [
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
  "alchemy_missing"
]

# ---------- helpers ----------
def safe_float(x):
    """Coerce to float safely; booleans -> 1/0, strings parsed to float or 0.0 on failure."""
    if x is None:
        return 0.0
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return 0.0
    s = str(x).strip()
    if s == "":
        return 0.0
    if s.lower() in ("true", "yes"):
        return 1.0
    if s.lower() in ("false", "no"):
        return 0.0
    try:
        return float(s.replace(",", "").replace("%", ""))
    except Exception:
        return 0.0

def load_json_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    try:
        mdl = joblib.load(MODEL_PATH)
        return mdl
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

@st.cache_data
def load_labels_csv():
    if os.path.exists(LABELS_CSV):
        try:
            df = pd.read_csv(LABELS_CSV, dtype=str)  # load as strings, coerce later
            df = df.set_index("address", drop=False) if "address" in df.columns else df
            return df
        except Exception:
            return None
    return None

def get_value_for_feature(addr, feat_name, cache_dict, labels_df):
    """Try cache first, then labels_df if available. Return safe_float."""
    # cache often stores numeric types already
    v = None
    if addr in cache_dict:
        rec = cache_dict[addr]
        # keys in cache may be strings, booleans, or numbers
        if feat_name in rec:
            v = rec[feat_name]
        else:
            # some caches have slightly different names; try common alternates
            alt_map = {
                "tx_count": "tx_count",
                "unique_senders": "unique_senders",
                "alchemy_tx_count": "alchemy_tx_count"
            }
            # (no-op here; kept for extension)
            v = rec.get(feat_name, None)
    # fallback to labels csv row
    if (v is None or v == "") and labels_df is not None:
        try:
            if feat_name in labels_df.columns:
                row = labels_df.loc[addr] if addr in labels_df.index else None
                if row is not None:
                    v = row.get(feat_name, None)
        except Exception:
            v = None
    return safe_float(v)

def build_feature_vector(addr, feature_order, cache_dict, labels_df):
    vec = [ get_value_for_feature(addr, f, cache_dict, labels_df) for f in feature_order ]
    return np.array([vec], dtype=float)

def model_predict_safe(model, X_try):
    """Attempt to get prediction/probs; raise with original exception for upstream handling."""
    # prefer predict_proba when available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_try)
        preds = model.predict(X_try)
        return preds, probs
    # fallback: decision_function -> softmax
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X_try)
        # if binary, make into 2-column
        if dec.ndim == 1:
            dec = np.vstack([-dec, dec]).T
        # softmax
        e = np.exp(dec - np.max(dec, axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = model.predict(X_try)
        return preds, probs
    # last fallback: only predict
    preds = model.predict(X_try)
    probs = None
    return preds, probs

# ---------- load resources ----------
model = load_model()
cache = load_json_cache(CACHE_PATH)
labels_df = load_labels_csv()

# Show compact model info for debugging (non-sensitive)
model_info_lines = []
try:
    n_in = getattr(model, "n_features_in_", None)
    model_info_lines.append(f"model.n_features_in_ = {n_in}")
except Exception:
    pass
try:
    cls = getattr(model, "classes_", None)
    if cls is not None:
        model_info_lines.append(f"model.classes_ = {list(cls)}")
except Exception:
    pass

st.sidebar.markdown("### Model info")
for l in model_info_lines:
    st.sidebar.write("-", l)

# ---------- UI controls ----------
st.sidebar.header("🔎 Try an Address")
addresses = list(cache.keys())
addr_choice = st.sidebar.selectbox("Pick from cached addresses:", ["-- pick one --"] + addresses[:300])
eth_address = st.sidebar.text_input("Or paste an Ethereum address (0x...)", "")

if addr_choice != "-- pick one --":
    eth_address = addr_choice

# ---------- Prediction flow (robust) ----------
if st.sidebar.button("Predict"):
    if not eth_address:
        st.warning("Please enter or pick an address.")
    else:
        addr = eth_address.strip()
        if addr not in cache and (labels_df is None or addr not in labels_df.index):
            st.warning("Address not found in cache or labels CSV. For demo use an address from the dropdown or enrich offline and upload the cache.")
        else:
            # Try: 1) build 13-feature vector and call model
            tried = []
            success = False
            errors = []

            # build both candidate vectors
            X13 = build_feature_vector(addr, FEATURE_ORDER_13, cache, labels_df)
            X11 = build_feature_vector(addr, FEATURE_ORDER_11, cache, labels_df)

            # helper to attempt predict with diagnostics
            def try_predict(X, desc):
                try:
                    preds, probs = model_predict_safe(model, X)
                    return {"ok": True, "preds": preds, "probs": probs, "desc": desc}
                except Exception as e:
                    tb = traceback.format_exc()
                    return {"ok": False, "error": str(e), "traceback": tb, "desc": desc}

            # 1) first try X13 (model was originally used with 13 in your sessions)
            res = try_predict(X13, "13-features")
            tried.append(res)
            if res["ok"]:
                used_X = X13
                used_order = FEATURE_ORDER_13
                result = res
                success = True
            else:
                # If failed due to StandardScaler expecting 11, try X11 automatically
                err_msg = res.get("error","").lower()
                if "standardscaler" in err_msg or "expect" in err_msg or "11" in err_msg:
                    # try X11
                    res2 = try_predict(X11, "11-features (fallback)")
                    tried.append(res2)
                    if res2["ok"]:
                        used_X = X11
                        used_order = FEATURE_ORDER_11
                        result = res2
                        success = True

            # 2) If not success, try X11 as a last resort
            if not success:
                res3 = try_predict(X11, "11-features final")
                tried.append(res3)
                if res3["ok"]:
                    used_X = X11
                    used_order = FEATURE_ORDER_11
                    result = res3
                    success = True

            # 3) If still not success, show helpful diagnostics
            if not success:
                st.error("Prediction failed. Diagnostic summary below — copy this and show it to me if you want help.")
                for t in tried:
                    st.write("Attempt:", t.get("desc", "unknown"))
                    st.write("Error:", t.get("error", "none"))
                    # optionally show a truncated traceback for debugging
                    if t.get("traceback"):
                        st.text(t["traceback"].splitlines()[-6:])  # last few lines
                # show what we built
                st.write("Built feature vector lengths:")
                st.write("len(X13) =", X13.shape[1], "features (names):", FEATURE_ORDER_13)
                st.write("len(X11) =", X11.shape[1], "features (names):", FEATURE_ORDER_11)
                st.stop()

            # ---------- present result ----------
            preds = result["preds"]
            probs = result["probs"]

            # single-row predictions
            pred = preds[0] if preds is not None and len(preds)>0 else None

            # map numeric label -> class name if needed
            pred_label = None
            try:
                classes = getattr(model, "classes_", None)
                # if pred is index (int-like) and classes available
                if classes is not None:
                    try:
                        idx = int(pred)
                        # but only map if idx in range
                        if 0 <= idx < len(classes):
                            pred_label = str(classes[idx])
                        else:
                            # if pred already matches a member, use it
                            pred_label = str(pred)
                    except Exception:
                        # pred not an index, maybe string label directly
                        pred_label = str(pred)
                else:
                    pred_label = str(pred)
            except Exception:
                pred_label = str(pred)

            st.markdown("## 🔮 Prediction")
            st.success(f"**{pred_label.upper()}**")

            # probabilities: if available, pretty print
            if probs is not None:
                prob_row = probs[0] if probs.shape[0] == 1 else probs
                # build DataFrame using classes if present
                classes = getattr(model, "classes_", None)
                if classes is None:
                    classes = [str(i) for i in range(len(prob_row))]
                prob_df = pd.DataFrame({"Class": list(classes), "Probability": prob_row})
                prob_df = prob_df.sort_values("Probability", ascending=True)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(prob_df["Class"], prob_df["Probability"], color="#2b6cb0")
                for i, (cls, p) in enumerate(zip(prob_df["Class"], prob_df["Probability"])):
                    ax.text(p + 0.01, i, f"{p:.2f}", va="center")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                st.pyplot(fig)
            else:
                st.info("Model didn't provide class probabilities; only label prediction shown.")

            # show which feature ordering was used (helpful)
            st.caption(f"Features used (order) — {used_order}")
