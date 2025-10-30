import json
from pathlib import Path

import joblib
import os
import requests
"""Streamlit app for the SMS Spam classifier.

Features:
- Multi-tab layout: Predict, Dataset, Metrics
- Shows class distribution and confusion matrix computed from the dataset when a model is available
- Explain: for logistic regression, shows top tokens contributing to the prediction
- Model download fallback via MODEL_URL environment variable (set in Streamlit Cloud)
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATASET_PATH = ROOT / "dataset" / "sms_spam_no_header.csv"


@st.cache_resource
def load_model(path: Path):
    # Try local file first, then optional MODEL_URL
    if path.exists():
        return joblib.load(path)

    model_url = os.environ.get("MODEL_URL")
    if model_url:
        try:
            import requests

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


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["label", "text"])
    df = pd.read_csv(path, header=None, names=("label", "text"), encoding="latin-1")
    if df.shape[1] > 2:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
    return df


def predict_and_explain(model, text: str) -> Tuple[Optional[str], Optional[float], Optional[dict]]:
    if model is None or not text:
        return None, None, None

    label = model.predict([text])[0]
    prob = None
    try:
        prob = float(model.predict_proba([text])[0, 1])
    except Exception:
        try:
            score = model.decision_function([text])[0]
            prob = float(1 / (1 + np.exp(-score)))
        except Exception:
            prob = None

    explanation = None
    # Explain using logistic regression coefficients when available
    try:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            clf = model.named_steps["clf"]
            vec = model.named_steps.get("tfidf")
            if hasattr(clf, "coef_") and vec is not None:
                features = vec.get_feature_names_out()
                coefs = clf.coef_[0]
                # Compute token contribution for this single text (approx)
                x = vec.transform([text]).toarray()[0]
                contrib = dict()
                for i, val in enumerate(x):
                    if val != 0 and i < len(features):
                        contrib[features[i]] = float(coefs[i] * val)
                # Top positive and negative tokens
                if contrib:
                    sorted_items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
                    top_pos = sorted_items[:10]
                    top_neg = sorted_items[-10:][::-1]
                    explanation = {"top_positive": top_pos, "top_negative": top_neg}
    except Exception:
        explanation = None

    return label, prob, explanation


st.set_page_config(page_title="Spam SMS Classifier", layout="wide")

st.markdown("# 📬 Spam SMS Classifier")
st.markdown("**Logistic Regression** — small, interpretable text classifier for SMS spam detection")

model = load_model(MODEL_PATH)
df = load_dataset(DATASET_PATH)

cols = st.columns([3, 1])
with cols[1]:
    st.write("**Model status**")
    if model is None:
        st.error("Model not found. Set `MODEL_URL` in Streamlit settings or push `models/pipeline.joblib` to the repo.")
    else:
        st.success("Model loaded ✅")

    st.write("---")
    st.write("**Controls**")
    st.write("Model path:", str(MODEL_PATH))
    st.write("Set `MODEL_URL` env var in Streamlit Cloud to a downloadable model if not committing the artifact.")

tabs = st.tabs(["Predict", "Dataset", "Metrics"])

with tabs[0]:
    st.header("Predict a message")
    text = st.text_area("Message text", height=180)
    colp1, colp2 = st.columns([3, 2])
    with colp1:
        if st.button("Predict"):
            if not model:
                st.error("No model available to predict.")
            else:
                label, prob, explanation = predict_and_explain(model, text)
                st.metric("Prediction", label if label is not None else "-")
                if prob is not None:
                    st.metric("Spam probability", f"{prob:.3f}")
                if explanation:
                    st.subheader("Top contributing tokens")
                    st.write("Top positive tokens (towards spam)")
                    pos_df = pd.DataFrame(explanation.get("top_positive", []), columns=["token", "score"]).set_index("token")
                    st.dataframe(pos_df)
                    st.write("Top negative tokens (towards ham)")
                    neg_df = pd.DataFrame(explanation.get("top_negative", []), columns=["token", "score"]).set_index("token")
                    st.dataframe(neg_df)

    with colp2:
        st.subheader("Examples")
        examples = [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "Hey, are we still meeting for lunch today?",
            "URGENT! Your account has been compromised. Call us now.",
        ]
        for ex in examples:
            if st.button("Use: " + ex[:40] + "..."):
                text = ex

with tabs[1]:
    st.header("Dataset sample & distribution")
    st.write("Data source: `dataset/sms_spam_no_header.csv`")
    st.write(f"Loaded {len(df)} rows")
    if not df.empty:
        st.subheader("Class distribution")
        dist = df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=dist.index, y=dist.values, ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Sample rows")
        st.dataframe(df.sample(min(10, len(df))))
    else:
        st.info("Dataset not found in repo. Upload or set MODEL_URL and provide dataset in the repo for Dataset view.")

with tabs[2]:
    st.header("Metrics & Confusion Matrix")
    if model is None or df.empty:
        st.info("Model or dataset missing; metrics unavailable.")
    else:
        # compute predictions on whole dataset for display purposes
        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(str).tolist()
        preds = model.predict(texts)
        report = classification_report(labels, preds, output_dict=True)
        precision = report.get("spam", {}).get("precision", 0)
        recall = report.get("spam", {}).get("recall", 0)
        f1 = report.get("spam", {}).get("f1-score", 0)

        st.metric("Precision (spam)", f"{precision:.3f}")
        st.metric("Recall (spam)", f"{recall:.3f}")
        st.metric("F1 (spam)", f"{f1:.3f}")

        cm = confusion_matrix(labels, preds, labels=["ham", "spam"]) if len(set(labels)) > 1 else None
        if cm is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

st.markdown("---")
st.write("Repository: https://github.com/SchwarzHawk/IoT_HW2")
