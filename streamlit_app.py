import json
from pathlib import Path

import joblib
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        # Attempt to download model from environment-provided URL if available
        model_url = os.environ.get("MODEL_URL")
        if model_url:
            try:
                dest = path
                dest.parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(model_url, timeout=20)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
                return joblib.load(dest)
            except Exception:
                return None
        return None
    return joblib.load(path)


def load_metrics(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

st.set_page_config(page_title="Spam SMS Classifier", layout="wide")
st.title("Spam SMS Classifier — Logistic Regression")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Predict a message")
    text = st.text_area("Enter SMS text to classify", height=150)
    if st.button("Predict"):
        if model is None:
            st.error("Model artifact not found. Run `scripts/train.py` to produce `models/pipeline.joblib`.")
        else:
            pred = model.predict([text])[0]
            prob = None
            try:
                prob = float(model.predict_proba([text])[0, 1])
            except Exception:
                try:
                    score = model.decision_function([text])[0]
                    prob = float(1 / (1 + np.exp(-score)))
                except Exception:
                    prob = None

            st.write("**Prediction:**", pred)
            if prob is not None:
                st.write("**Spam probability:**", f"{prob:.3f}")

    st.markdown("---")
    st.header("About this app")
    st.write(
        "This app loads a trained scikit-learn pipeline from `models/pipeline.joblib` and displays evaluation metrics saved in `models/metrics.json`. Train a model locally with `python scripts/train.py`."
    )

with col2:
    st.header("Latest metrics")
    if metrics is None:
        st.info("No metrics found. Train a model to see metrics here.")
    else:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        st.metric("F1", f"{metrics.get('f1', 0):.3f}")

        st.subheader("Dataset sizes")
        st.write(f"Train: {metrics.get('n_train', '?')}  —  Val: {metrics.get('n_val', '?')}")

        # Simple bar chart of precision/recall/f1
        fig, ax = plt.subplots(figsize=(4, 3))
        vals = [metrics.get('precision', 0), metrics.get('recall', 0), metrics.get('f1', 0)]
        sns.barplot(x=["precision", "recall", "f1"], y=vals, ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

st.sidebar.title("Controls")
st.sidebar.write("Model path: ", str(MODEL_PATH))
st.sidebar.markdown(
    "**Model availability**: The app looks for `models/pipeline.joblib` in the repo."
)
st.sidebar.markdown(
    "If the model is not in the repo (recommended), set an environment variable `MODEL_URL` in Streamlit Cloud to a downloadable URL (S3 presigned URL, GitHub raw URL, or release asset). The app will attempt to download it at startup."
)

st.sidebar.header("Sample predictions")
if model is not None:
    sample_texts = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Hey, are we still meeting for lunch today?",
        "URGENT! Your account has been compromised. Call us now.",
    ]
    for t in sample_texts:
        prob = None
        try:
            prob = float(model.predict_proba([t])[0, 1])
        except Exception:
            try:
                score = model.decision_function([t])[0]
                prob = float(1 / (1 + np.exp(-score)))
            except Exception:
                prob = None
        label = model.predict([t])[0]
        st.sidebar.write(f"{label} ({prob:.3f}) — {t}")
else:
    st.sidebar.info("No model loaded — run training first.")
