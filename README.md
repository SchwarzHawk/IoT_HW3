# IoT_HW3 — Spam SMS Classifier (Logistic Regression)

This repository contains a small, reproducible machine learning pipeline that trains a logistic regression model to detect spam messages (SMS). It includes training scripts, a serializable sklearn pipeline, and a Streamlit app for interactive prediction and visualization.

Contents
- `dataset/` — sample dataset CSV (`sms_spam_no_header.csv`)
- `src/` — library code: preprocessing, model, persistence
- `scripts/train.py` — training script that produces `models/pipeline.joblib` and `models/metrics.json`
- `models/` — model artifacts (gitignored by default)
- `streamlit_app.py` — Streamlit UI for predictions, dataset preview, and metrics
- `openspec/` — OpenSpec proposals, specs, and project metadata
- `requirements.txt` — Python dependencies

Quick start (local)

1. Install dependencies (use a virtual environment):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Train the model on the provided dataset:

```powershell
python scripts/train.py --dataset dataset/sms_spam_no_header.csv --model logreg
```

This will write `models/pipeline.joblib` and `models/metrics.json`.

3. Run the Streamlit app locally:

```powershell
streamlit run streamlit_app.py
```

Deployment to Streamlit Cloud

- Push your repository to GitHub (e.g., `SchwarzHawk/IoT_HW2`).
- On Streamlit Cloud, create a new app and point it to `streamlit_app.py` in the repo and branch (usually `main`).
- Provide the trained model to the app either by committing `models/pipeline.joblib` (not recommended if it contains PII) or by uploading the model as a release asset or to cloud storage and setting the `MODEL_URL` environment variable in Streamlit to a direct download URL.

Example: upload the model as a GitHub release asset and set MODEL_URL to the asset's download link in Streamlit settings.

Notes and best practices
- Do not commit raw datasets containing PII. Use the training script to download datasets at runtime or include a sanitized sample for tests.
- The project prefers logistic regression for explainability; the training script supports `--model svm` for experiments.
- Use `black`, `isort`, and `flake8` to format and lint code before opening PRs.

Project structure

```
.
├── dataset/
# IoT_HW3 — Spam SMS Classifier (Logistic Regression)

This repository contains a compact, reproducible machine learning pipeline that trains a logistic regression model to detect spam messages (SMS). It includes training scripts, a serializable scikit-learn pipeline, and a Streamlit app for interactive prediction, dataset preview, and evaluation visualizations.

Contents
- `dataset/` — sample dataset CSV (`sms_spam_no_header.csv`)
- `src/` — library code: `preprocessing.py`, `model.py`, `persistence.py`
- `scripts/train.py` — training script that produces `models/pipeline.joblib` and `models/metrics.json`
- `models/` — model artifacts (gitignored by default)
- `streamlit_app.py` — Streamlit UI for predictions, dataset preview, and metrics
- `openspec/` — OpenSpec proposals, specs, and project metadata
- `requirements.txt` — Python dependencies

Quick start (local)

1. Create and activate a virtual environment (PowerShell):

```powershell
cd "c:\Users\USER\Desktop\HW\IoT\IoT_HW3"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Train the model on the provided dataset:

```powershell
python scripts/train.py --dataset dataset/sms_spam_no_header.csv --model logreg
```

This will write `models/pipeline.joblib` and `models/metrics.json`.

3. Run the Streamlit app locally:

```powershell
streamlit run streamlit_app.py
```

Deployment to Streamlit Cloud

- Push your repository to GitHub (for example `SchwarzHawk/IoT_HW2`).
- On Streamlit Cloud (share.streamlit.io) create a new app and point it to `streamlit_app.py` in your repository and `main` branch.
- Provide the trained model to the app either by committing `models/pipeline.joblib` into the repo (not recommended if it contains PII) or by hosting the model artifact and setting the `MODEL_URL` environment variable in the Streamlit app settings to a direct download URL (GitHub release asset, S3 presigned URL, etc.).

Example: upload the model as a GitHub release asset and set `MODEL_URL` to the asset download URL in the Streamlit app settings.

Notes and best practices
- Do not commit raw datasets containing PII. Use the training script to download datasets at runtime or include a sanitized sample for tests.
- Logistic Regression is the primary model for explainability. The training script supports `--model svm` to experiment with SVMs.
- Use `black`, `isort`, and `flake8` to format and lint code before opening PRs.

Project structure

```
.
├── dataset/
│   └── sms_spam_no_header.csv
├── models/                # generated artifacts (gitignored)
├── openspec/              # proposals, specs, project metadata
├── scripts/
│   └── train.py
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── persistence.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

If you'd like, I can scaffold a GitHub Actions workflow to run linting and tests, and optionally create a release with the trained model artifact.

License

This project is provided as-is for educational purposes. Review dataset licensing before redistribution.

Contact

If you need help deploying to Streamlit Cloud, creating a release, or adding CI, tell me which option you prefer and I will prepare the necessary scripts and workflows.