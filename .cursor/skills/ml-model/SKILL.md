---
name: ml-model
description: Trains, evaluates, and serves an algae/pond risk classification model (IoT sensor data). Use when working on ML training, prediction, feature engineering, API endpoints, or model artifacts in this project.
---

# ML Model (Algae / Pond Risk)

This project trains a classifier for **risk_level** (Low / Medium / High) from IoT pond sensor data, then serves predictions via a Flask API. The agent should follow existing patterns for features, training, and inference.

## Project layout

- **config.py** — `MODEL_PATH`, `SCALER_PATH`, `METADATA_PATH`, `BASE_DIR`, `RANDOM_STATE`. Use these; do not hardcode paths.
- **ML/features.py** — Re-exports `engineer` and `build_features` from `features-engineering.py`. All feature logic lives in `ML/features-engineering.py`.
- **ML/features-engineering.py** — `engineer(df)` and `build_features(df)` return `(df, feature_cols)`. Required raw columns: `created_at`, `Temperature(C)`, `Turbidity(NTU)`, `PH`, `Ammonia(g/ml)`, `Nitrate(g/ml)`.
- **ML/train.py** — Loads data, runs `engineer`, train/test split, `StandardScaler`, optional SMOTE, fits classifier, saves model + scaler + metadata (features, labels, metrics).
- **ML/predict.py** — `Predictor` loads model and metadata, uses `build_features` on input data, predicts last row; requires same feature set as training.
- **APIs/app.py** — Flask app: `/health`, `/status`, `/predict` (POST JSON with `data`), `/predict-sample`, `/feature-cols`, `/artifacts` (DELETE). Uses `engineer`, scaler, and metadata; returns `predicted_level`, `probabilities`, `recommendation`.

## Conventions

1. **Paths**: Import from `config` (e.g. `MODEL_PATH`, `METADATA_PATH`). Use `os.path.join` and project-relative paths; avoid absolute paths and `os.chdir` in shared code.
2. **Features**: Changing features must be done in `features-engineering.py`. Keep `features.py` as a thin re-export so `ML.features.engineer` / `build_features` stay the single source of truth for both training and API.
3. **Training**: Scale with the same `StandardScaler` used at inference; fit on train only, save scaler with the model. Persist `metadata.json` with `features`, `labels`, and optionally `metrics`/`best_model`.
4. **API**: Expect payload `{"data": [ {...row...}, ... ]}`. Apply `engineer`, then select `feature_cols` from metadata, scale, predict. Return class label and optionally probabilities and a text recommendation.
5. **Target**: `risk_level` is categorical: `Low`, `Medium`, `High`. Use consistent label order in metadata and API (e.g. `['Low','Medium','High']`).

## When adding or changing behavior

- **New feature column**: Add in `features-engineering.py` inside `engineer`; ensure it is included in the returned `feature_cols` and not in the exclude list.
- **New model or metric**: In `train.py`, add to the models dict and/or metrics, then update metadata and keep scaler/model paths from config.
- **New API route**: In `APIs/app.py`, use existing `model`, `scaler`, `feature_cols`, `labels` and `engineer`; do not duplicate feature logic.
- **Prediction from script**: Use `Predictor` from `ML.predict` with input that includes the required sensor columns and enough rows for lags/rolling; get the last row prediction.

## Quick reference

- Train: run `ML/train.py` (expects data path/format as in current script; ensure `config` paths point to desired `Models/` or `models/`).
- Predict (code): `from ML.predict import Predictor; p = Predictor(); p.predict(list_of_row_dicts)`.
- API: Start Flask app in `APIs/app.py`; POST to `/predict` with `{"data": [...]}`; ensure artifacts exist (run training first if needed).

For more detail on feature formulas and artifact layout, see [reference.md](reference.md).
