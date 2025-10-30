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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)


ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = ROOT / "dataset" / "sms_spam_no_header.csv"
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


@st.cache_resource
def load_model(path: Path):
    """Load a joblib model. If not found, try MODEL_URL env var download."""
    if not path.exists():
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


@st.cache_data
def load_dataset(path: Path):
    # Dataset provided has no header (two columns: label, text)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 2:
            df.columns = ["col_0", "col_1"]
        return df
    except Exception:
        # fallback to default read
        return None


@st.cache_data
def top_tokens_by_class(texts, labels, top_n=20):
    vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(texts)
    tokens = np.array(vec.get_feature_names_out())
    df = pd.DataFrame(X.toarray(), columns=tokens)
    classes = np.unique(labels)
    results = {}
    for c in classes:
        mask = labels == c
        counts = df[mask].sum(axis=0)
        top = counts.sort_values(ascending=False).head(top_n)
        results[c] = top
    return results


@st.cache_data
def token_replacements_preview(texts):
    # naive placeholders from dataset — this is approximate
    tokens = {"<URL>": 0, "<EMAIL>": 0, "<PHONE>": 0, "<NUM>": 0}
    # very naive checks
    for t in texts:
        if isinstance(t, str):
            if "http" in t or "www." in t:
                tokens["<URL>"] += 1
            if "@" in t:
                tokens["<EMAIL>"] += 1
            if any(ch.isdigit() for ch in t):
                tokens["<NUM>"] += 1
    return pd.DataFrame([{"token": k, "count": v} for k, v in tokens.items()])


model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

st.set_page_config(page_title="Spam/Ham Classifier — Visualizations", layout="wide")

# Minimal CSS polish to make sidebar compact like the screenshots
st.markdown(
    "<style>\n+    .css-1v3fvcr {padding-top: 0rem;}\n+    .stSidebar .css-1d391kg {padding-top: 1.5rem;}\n+    .block-container{padding-top:1rem;}\n+    h1{font-size:36px;}\n+    </style>",
    unsafe_allow_html=True,
)

st.title("Spam/Ham Classifier — Phase 4 Visualizations")
st.caption("Interactive dashboard for data distribution, token patterns, and model performance")

# Sidebar inputs (match screenshot layout)
st.sidebar.header("Inputs")
data_files = [str(DEFAULT_DATA_PATH)]
# allow the dataset list to expand in the future
dataset_choice = st.sidebar.selectbox("Dataset CSV", data_files, index=0)
df = load_dataset(Path(dataset_choice))

label_col = st.sidebar.selectbox("Label column", options=df.columns.tolist() if df is not None else ["col_0"], index=0)
text_col = st.sidebar.selectbox("Text column", options=df.columns.tolist() if df is not None else ["col_1"], index=1 if df is not None and len(df.columns) > 1 else 0)

models_dir = st.sidebar.text_input("Models dir", value="models")
test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2, step=0.01)
seed = st.sidebar.number_input("Seed", value=42, step=1)
decision_threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01)

st.sidebar.markdown("---")
st.sidebar.write("Model availability: ")
if model is None:
    st.sidebar.warning("No model found locally. Set MODEL_URL in Streamlit Cloud or run training locally.")
else:
    st.sidebar.success("Model loaded")

st.sidebar.markdown("---")

# -------------------- Main layout --------------------
st.header("Data Overview")
cols = st.columns([3, 2])

with cols[0]:
    st.subheader("Class distribution")
    if df is None:
        st.info("Dataset not found at the expected path.")
    else:
        counts = df[label_col].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.index, y=counts.values, palette="tab10", ax=ax)
        ax.set_ylabel("count")
        st.pyplot(fig)

with cols[1]:
    st.subheader("Token replacements in cleaned text (approximate)")
    if df is not None:
        preview = token_replacements_preview(df[text_col].astype(str).tolist())
        # Streamlit does not accept a pandas Styler in st.dataframe on some runtimes;
        # show a simple table with token as the index instead.
        preview_display = preview.set_index("token")
        st.table(preview_display)
    else:
        st.write("—")

st.markdown("---")

# Top tokens by class
st.subheader("Top Tokens by Class")
top_n = st.slider("Top-N tokens", 5, 50, 20)
if df is not None:
    tokens = top_tokens_by_class(df[text_col].astype(str).tolist(), df[label_col].values, top_n=top_n)
    classes = list(tokens.keys())
    cols = st.columns(len(classes))
    for i, c in enumerate(classes):
        with cols[i]:
            st.markdown(f"**Class: {c}**")
            top = tokens[c]
            fig, ax = plt.subplots(figsize=(6, 6))
            y = top.index[::-1]
            x = top.values[::-1]
            ax.barh(y, x, color=sns.color_palette("viridis", len(x)))
            ax.set_xlabel("frequency")
            st.pyplot(fig)
else:
    st.info("No data to compute tokens.")

st.markdown("---")

# Model Performance
st.subheader("Model Performance (Test)")
if model is None or df is None:
    st.info("Model or dataset not available — train a model and ensure dataset exists to see performance plots.")
else:
    # prepare test split quickly (we don't retrain here; just quick deterministic split)
    from sklearn.model_selection import train_test_split

    X = df[text_col].astype(str).values
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed), stratify=y)

    # get prediction probabilities
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback to decision_function
        try:
            scores = model.decision_function(X_test)
            probs = 1 / (1 + np.exp(-scores))
        except Exception:
            probs = None

    if probs is not None:
        # Determine positive/negative label values from the trained pipeline if possible
        model_classes = getattr(model, "classes_", None)
        if model_classes is not None and len(model_classes) >= 2:
            neg_label, pos_label = model_classes[0], model_classes[1]
        else:
            # fallback: assume binary labels encoded as 0/1
            neg_label, pos_label = 0, 1

        # Convert probability threshold into predicted label values (same dtype as y_test)
        preds_labels = np.where(probs >= decision_threshold, pos_label, neg_label)

        # confusion matrix table using original label values
        labels_unique = np.unique(y_test)
        cm = confusion_matrix(y_test, preds_labels, labels=labels_unique)
        cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in labels_unique], columns=[f"pred_{i}" for i in labels_unique])
        st.table(cm_df)

        # For ROC and PR we need binary ground truth
        try:
            y_test_bin = (y_test == pos_label).astype(int)
        except Exception:
            # fallback: try to coerce to int
            y_test_bin = y_test.astype(int)

        # ROC / Precision-Recall
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fpr, tpr, _ = roc_curve(y_test_bin, probs)
        axes[0].plot(fpr, tpr, lw=2)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title("ROC")

        prec, rec, _ = precision_recall_curve(y_test_bin, probs)
        axes[1].plot(rec, prec, lw=2)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall")

        st.pyplot(fig)

        # Threshold sweep (compute metrics relative to positive label)
        thresholds = np.linspace(0.3, 0.75, 10)
        rows = []
        for t in thresholds:
            p_bin = (probs >= t).astype(int)
            rows.append({
                "threshold": round(float(t), 2),
                "precision": round(float(precision_score(y_test_bin, p_bin, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test_bin, p_bin, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test_bin, p_bin, zero_division=0)), 4),
            })
        st.subheader("Threshold sweep (precision/recall/f1)")
        st.dataframe(pd.DataFrame(rows))

    else:
        st.warning("Model does not provide probabilities — cannot compute ROC/PR.")

st.markdown("---")

st.subheader("Live Inference")
col_a, col_b = st.columns([1, 4])
with col_a:
    if st.button("Use spam example"):
        example = "Free entry in 2 a wkly comp to win FA Cup final tkts"
        st.session_state.setdefault("live_text", example)
    if st.button("Use ham example"):
        example2 = "Hey, are we still meeting for lunch today?"
        st.session_state.setdefault("live_text", example2)

with col_b:
    live_text = st.text_area("Enter a message to classify", value=st.session_state.get("live_text", ""), height=150)
    if st.button("Classify message"):
        if model is None:
            st.error("Model not available — run training or provide MODEL_URL.")
        else:
            lbl = model.predict([live_text])[0]
            prob = None
            try:
                prob = float(model.predict_proba([live_text])[0, 1])
            except Exception:
                try:
                    score = model.decision_function([live_text])[0]
                    prob = float(1 / (1 + np.exp(-score)))
                except Exception:
                    prob = None
            st.write("**Prediction:**", lbl)
            if prob is not None:
                st.write("**Spam probability:**", f"{prob:.3f}")

    # If logistic regression, show top contributing tokens (approximate)
    try:
        clf = getattr(model, "named_steps", None) and model.named_steps.get("clf") or None
        vec = getattr(model, "named_steps", None) and model.named_steps.get("vect") or None
    except Exception:
        clf = None
        vec = None

    if clf is not None and hasattr(clf, "coef_") and vec is not None:
        st.markdown("**Explanation (approx): top coefficients for spam class**")
        feature_names = vec.get_feature_names_out()
        coefs = clf.coef_.ravel()
        top_idx = np.argsort(coefs)[-10:][::-1]
        top_feats = [(feature_names[i], float(coefs[i])) for i in top_idx]
        st.table(pd.DataFrame(top_feats, columns=["token", "coef"]))

st.markdown("---")
st.caption("App: quick visual dashboard for dataset diagnostics and model evaluation")
