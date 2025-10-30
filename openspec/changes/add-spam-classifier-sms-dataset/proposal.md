## Why

There is a need for a small, reproducible spam-classification capability that can be trained and validated locally and served via a lightweight API. This change will provide a concrete training pipeline, a documented data source, and a prediction API so contributors can iterate quickly and CI can run reproducible smoke-tests.

## What Changes

- **ADD** a training pipeline and example scripts that train a spam classifier from a public SMS spam dataset.
- **ADD** a serializable model artifact (preprocessing + estimator) saved with `joblib` under `models/`.
- **ADD** an inference API (FastAPI) exposing `/predict` to classify a text message and return a probability.
- **ADD** unit and integration tests plus a small CI job that runs a quick training smoke-test on the sample dataset.

## Data Source

We will use the public SMS spam dataset hosted in this repository:

https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

Assumptions and notes about the dataset:
- The CSV is expected to contain two columns per row: label (spam/ham) and text message. We'll validate the exact delimiter and column order at training runtime and document it in the training script.
- This dataset is publicly hosted; confirm licensing for redistribution if you plan to include it in the repo. For CI and examples we will download remotely at runtime (avoids storing PII in repo).

## Design Summary

- Primary model: Logistic Regression (fast, interpretable coefficients, compact). This will be the default training target and the model we recommend for initial deployment.
- Alternative/experiment: Support training a Support Vector Machine (SVM) so we can compare performance. The pipeline and training script will accept a `--model` flag (`logreg`|`svm`).
- Preprocessing: simple text cleaning (lowercasing, punctuation removal), tokenization, stopword removal (optional), and TF-IDF vectorization. The preprocessing + estimator will be composed into a single sklearn Pipeline and persisted with `joblib`.
- Evaluation: stratified train/validation split, reporting precision, recall, F1, and confusion matrix. Carry class-weighting or use stratified sampling to mitigate class imbalance.
- Reproducibility: fixed random seeds documented in the script and environment versions recorded in `models/metrics.json`.

## What Files Will Be Added

- `openspec/changes/add-spam-classifier-sms-dataset/proposal.md` (this file)
- `openspec/changes/add-spam-classifier-sms-dataset/tasks.md` (tasks checklist)
- `src/preprocessing.py` (text cleaning & transformer)
- `src/model.py` (pipeline builder and training helpers)
- `src/persistence.py` (save/load joblib helper)
- `scripts/train.py` (CLI to download dataset, train, evaluate, save)
- `src/api.py` (FastAPI inference endpoint)
- `tests/` unit + integration tests and `notebooks/example_training.ipynb`

## Impact

- Affected capabilities: new `spam` capability (adds ADDED requirements and scenarios under `openspec/changes/.../specs/spam/spec.md`).
- Affected code: new `src/` and `scripts/` files; small CI workflow addition.
- Migration: None. Model artifacts will be created under `models/` and gitignored.

## Validation / Acceptance Criteria

- Training script runs without uncaught exceptions and produces `models/pipeline.joblib` and `models/metrics.json`.
- Validation metrics printed and saved; default logistic regression model should reach a sanity threshold (precision >= 0.70 on validation) on this dataset or the run is marked for manual review.
- API `/predict` returns HTTP 200 with JSON `{ "label": "spam|ham", "probability": float }` for typical short messages.
- Unit tests for preprocessing must pass; integration test trains on a tiny sample and asserts metrics are computed.

## Rollout Plan

1. Create PR with implementation and tests.
2. CI runs lint, unit tests, and a quick training smoke test that downloads the dataset and runs training with a small subset.
3. After review and CI green, merge to `main` and tag release.
4. Archive the change after deployment and move final spec to `openspec/specs/spam/spec.md` if approved.

## Owners

- Primary: data-science / ML engineer
- Reviewer: backend engineer (API)

## Open Questions

- Do you want the dataset downloaded in CI at runtime, or should we include a small sanitized sample in `data/` for tests?
- Which exact metric thresholds should be enforced by CI for automatic pass/fail? (default: precision >= 0.70)
