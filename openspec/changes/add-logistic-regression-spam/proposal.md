## Why

We need a small, reproducible training pipeline and a lightweight inference API so contributors can train, evaluate, and serve a logistic regression model for spam classification. This enables consistent experiments, CI validation, and a simple production-ready prediction endpoint.

## What Changes

- **ADD** training scripts and a reusable training pipeline (`scripts/train.py`, `src/model.py`, `src/preprocessing.py`).
- **ADD** a preprocessing + feature pipeline (TF-IDF or count vectors) that is serializable with the model.
- **ADD** a model persistence utility that saves a pipeline (preprocessing + estimator) using `joblib` into `models/`.
- **ADD** a FastAPI-based inference endpoint (`src/api.py`) exposing a `/predict` POST that returns label and probability.
- **ADD** unit tests and a small CI job to run quick training and prediction smoke tests.

**BREAKING**: None.

## Impact

- Affected specs: `spam` capability (new ADDED requirements included in `specs/spam/spec.md`).
- Affected code: new files under `src/` and `scripts/`, plus tests under `tests/`.
- Migration: None required for existing code; model artifacts will be stored in `models/` and gitignored.

## Rollout

1. Create PR with this proposal and implementation changes.
2. CI runs unit tests + quick training smoke test (small dataset).
3. Once CI green and code reviewed, merge and tag release.
4. Archive this change after deployment.

## Owners

- Primary: data-science / ML engineer
- Reviewer: backend engineer (API)
