# Project Context

## Purpose
Use machine learning logistic regression to do a spam email classification

## Tech Stack
Python 3.10
scikit-learn (core ML algorithms)
pandas, numpy (data manipulation)
Jupyter / JupyterLab (exploration & notebooks)
FastAPI (lightweight REST inference API) or Flask
joblib (model serialization)
pytest (testing)
black, isort, flake8 (formatting & linting)
GitHub Actions (CI)

## Project Conventions

### Code Style
We follow common Python conventions with a small, enforceable toolchain so contributions are consistent and reviewable.

Formatter: black (opinionated formatting)
Import order: isort
Linting: flake8
Type hints: Use mypy-compatible type hints where it clarifies intent; prefer explicit types on public functions.
Docstrings: Google-style or NumPy-style docstrings for public modules/functions.
Naming: snake_case for functions and variables, PascalCase for classes, UPPER_SNAKE for constants.
Line length: 88 (black default)
Tests: pytest style, `tests/` mirrors `src/` layout.

### Architecture Patterns
This project uses a small, modular structure that separates data, model training, and inference API.

Suggested layout:
- data/                # raw and processed sample data, ingestion helpers (not for PII)
- notebooks/           # EDA and experiments
- src/
	- preprocessing.py   # cleaning and feature extraction
	- features.py        # feature transforms
	- model.py           # model training & evaluation helpers
	- persistence.py     # save/load model (joblib)
	- api.py             # FastAPI app for inference
- models/              # serialized models (gitignored)
- tests/               # unit & integration tests
- scripts/             # helpers to run training, evaluation, etc.

Pipeline pattern:
1. Ingest data (CSV, dataset loader)
2. Clean & normalize text (tokenization, stopword removal, simple TF-IDF)
3. Feature extraction (TF-IDF or count vectors)
4. Train logistic regression with cross-validation
5. Evaluate using precision/recall/F1, pick model meeting target metrics
6. Serialize model and preprocessing pipeline (joblib)
7. Serve via lightweight API for inference

### Testing Strategy
Test tiers:
- Unit tests (fast): test preprocessing steps, tokenization, feature shapes, small deterministic transforms.
- Model tests: smoke-tests for training pipeline that run on a tiny sample dataset, assert metrics (precision/recall) are computed and meet minimum sanity thresholds.
- Integration tests: run the saved model end-to-end with the API (predict endpoint) using test client.
- CI: GitHub Actions runs black --check, flake8, pytest (unit+selected integration), and optionally a small training job.

Acceptance criteria for CI:
- Linting passes
- Tests pass
- Model training script runs without uncaught exceptions on sample data

### Git Workflow
Branching / PRs:
- Protected `main` (or `master`) branch for releases
- Feature branches: `feature/<short-desc>` or `add-<change-id>`
- Small, focused PRs; reference issue IDs in PR description
- Squash or rebase merges as per team preference

Commit messages:
- Use conventional commits style for clarity (e.g., `feat: add training pipeline`, `fix: normalize text tokenizer`).

Code review:
- Each PR should have at least one reviewer and CI green before merge.

## Domain Context
Domain notes for spam classification:
- Typical datasets: Enron email dataset, SMS Spam Collection, or curated corpora.
- Class imbalance: spam is often less frequent than ham; use stratified sampling or class weights.
- Evaluation: prioritize precision (minimize false positives) for production use; but measure recall and F1 as well.
- Privacy: emails may contain PII; avoid storing raw datasets with PII in repository. Use sanitized or synthetic data for examples.
- Explainability: Logistic regression provides interpretable coefficients which can help explain decisions.

## Important Constraints
Constraints and non-functional requirements:
- No PII or sensitive data in repo or CI logs.
- Reproducible training: fix random seeds and record environment versions.
- Lightweight runtime: inference API should be deployable on modest hardware (single small VM/container).
- Model size: prefer compact preprocessing + linear model over large neural networks for efficiency.
- Licensing: third-party datasets must be checked for redistribution rights.

## External Dependencies
Key libraries & tools:
- scikit-learn (modeling)
- pandas, numpy (data)
- joblib (save/load)
- fastapi, uvicorn (API)
- Jupyter (experimentation)
- pytest, black, flake8, isort

Optional tools:
- MLflow or Weights & Biases for experiment tracking (optional)
- Docker for containerizing the API

Notes:
- Pin versions in `requirements.txt` or `pyproject.toml` when creating reproducible environments.
