from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import simple_preprocessor


def build_pipeline(model_type: str = "logreg", random_state: int = 42) -> Pipeline:
    """Return an sklearn Pipeline for text -> classifier.

    model_type: 'logreg' or 'svm'
    """
    vectorizer = TfidfVectorizer(
        preprocessor=simple_preprocessor, ngram_range=(1, 2), max_df=0.9
    )

    if model_type == "svm":
        clf = SVC(probability=True, class_weight="balanced", random_state=random_state)
    else:
        clf = LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced", random_state=random_state
        )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_pipeline(
    texts, labels, model_type: str = "logreg", test_size: float = 0.2, random_state: int = 42
) -> Tuple[Pipeline, Dict]:
    """Train the pipeline and return (pipeline, metrics).

    Metrics contains precision, recall, f1 on the validation split and other meta.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    pipe = build_pipeline(model_type=model_type, random_state=random_state)
    pipe.fit(X_train, y_train)

    # Predict probabilities if available, else fallback to predicted labels
    try:
        probs = pipe.predict_proba(X_val)[:, 1]
    except Exception:
        # SVC with probability=True should work, but as a fallback use decision_function
        try:
            scores = pipe.decision_function(X_val)
            probs = 1 / (1 + np.exp(-scores))
        except Exception:
            probs = None

    y_pred = pipe.predict(X_val)

    precision = precision_score(y_val, y_pred, pos_label="spam")
    recall = recall_score(y_val, y_pred, pos_label="spam")
    f1 = f1_score(y_val, y_pred, pos_label="spam")

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }

    return pipe, metrics


def evaluate_pipeline(pipe: Pipeline, texts, labels) -> Dict:
    y_pred = pipe.predict(texts)
    precision = precision_score(labels, y_pred, pos_label="spam")
    recall = recall_score(labels, y_pred, pos_label="spam")
    f1 = f1_score(labels, y_pred, pos_label="spam")
    report = classification_report(labels, y_pred, output_dict=True)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "report": report}


if __name__ == "__main__":
    print("This module provides build_pipeline, train_pipeline, and evaluate_pipeline.")
