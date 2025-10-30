## 1. Implementation
- [ ] 1.1 Create `src/preprocessing.py` with text cleaning, tokenization and a TF-IDF feature transformer.
- [ ] 1.2 Create `src/model.py` to build a sklearn Pipeline (preprocessing + estimator) and training helpers supporting `logreg` and `svm`.
- [ ] 1.3 Create `scripts/train.py` to download the remote CSV, run training on a dataset, evaluate metrics, and persist the model with `joblib` into `models/`.
- [ ] 1.4 Add model persistence helpers in `src/persistence.py`.
- [ ] 1.5 Add `src/api.py` (FastAPI) with a `/predict` endpoint exposing label and probability.

## 2. Tests
- [ ] 2.1 Unit tests for `preprocessing.py` covering tokenization, stopword removal, and output shapes.
- [ ] 2.2 Integration test that trains on a tiny sample dataset and asserts metrics (sanity thresholds).
- [ ] 2.3 API test using FastAPI TestClient for `/predict` endpoint.

## 3. CI
- [ ] 3.1 Add a GitHub Actions workflow (or extend existing) to run linting and the tests above. Use a small subset of the remote CSV for smoke test to keep CI quick.

## 4. Documentation
- [ ] 4.1 Add README section with how to train and run the API locally.
- [ ] 4.2 Add example notebook under `notebooks/` showing training + evaluation.
