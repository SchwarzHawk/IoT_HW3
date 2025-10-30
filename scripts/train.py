#!/usr/bin/env python3
"""Simple training script for SMS spam classifier.

Usage:
  python scripts/train.py --dataset dataset/sms_spam_no_header.csv --model logreg
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path so `from src.*` imports work when running this
# script directly (python scripts/train.py). When run as a module or installed package
# this is unnecessary.
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import pandas as pd

from src.model import train_pipeline
from src.persistence import save_model


def load_dataset(path: str):
    # Dataset has no header: assume two columns: label, text
    df = pd.read_csv(path, header=None, names=("label", "text"), encoding="latin-1")
    # Some datasets have extra columns; keep only first two
    if df.shape[1] > 2:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/sms_spam_no_header.csv")
    parser.add_argument("--model", choices=("logreg", "svm"), default="logreg")
    parser.add_argument("--out", default="models/pipeline.joblib")
    parser.add_argument("--metrics", default="models/metrics.json")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise SystemExit(f"Dataset not found: {ds_path}")

    df = load_dataset(str(ds_path))
    labels = df["label"].astype(str)
    texts = df["text"].astype(str)

    pipe, metrics = train_pipeline(texts, labels, model_type=args.model)

    out_path = Path(args.out)
    save_model(pipe, str(out_path))

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
