## ADDED Requirements

### Requirement: Train Logistic Regression Spam Classifier
The system SHALL provide a reproducible training pipeline that trains a logistic regression model for spam classification and persists a serializable pipeline (preprocessing + estimator).

#### Scenario: Training completes successfully and produces a model artifact
- **GIVEN** a labeled training dataset (spam/ham) is available locally for training
- **WHEN** the training script is executed with default configuration
- **THEN** a model artifact (pipeline.joblib) SHALL be created in `models/` and evaluation metrics (precision, recall, f1) SHALL be output to the console and saved to `models/metrics.json`.

#### Scenario: Model meets minimum sanity thresholds on validation
- **GIVEN** a validation split held out during training
- **WHEN** the training finishes
- **THEN** the validation precision SHALL be reported and SHOULD be above a project-defined sanity threshold (e.g., 0.70) for the trained model to be considered acceptable for further review.

### Requirement: Expose Prediction API
The system SHALL provide a REST prediction API that accepts a plain text message and returns a classification and a confidence score.

#### Scenario: Predict endpoint returns label and probability
- **GIVEN** the model artifact is available and the API server is running
- **WHEN** a client POSTS a JSON payload `{ "text": "..." }` to `/predict`
- **THEN** the API SHALL return HTTP 200 with a JSON body `{ "label": "spam|ham", "probability": 0.0-1.0 }` within an operational latency budget (e.g., < 200ms for typical short messages).
