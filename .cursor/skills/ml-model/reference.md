# ML Model — Reference

## Config paths (config.py)

| Variable | Purpose |
|----------|---------|
| `BASE_DIR` | Project root |
| `MODEL_PATH` | Classifier joblib (e.g. `Models/classifier_v1.joblib`) |
| `SCALER_PATH` | StandardScaler joblib |
| `METADATA_PATH` | JSON: `features`, `labels`, optional `metrics`, `best_model` |
| `RANDOM_STATE` | Reproducibility (e.g. 42) |

## Feature engineering (features-engineering.py)

- **Input**: DataFrame with `created_at`, `Temperature(C)`, `Turbidity(NTU)`, `PH`, `Ammonia(g/ml)`, `Nitrate(g/ml)`.
- **Preprocessing**: Parse `created_at`, sort, interpolate/ffill/bfill, drop rows that remain NaN after rolling/lags.
- **Derived**: `algae_risk_score` (weighted combo), `risk_level` (binned); `hour`, `hour_sin`, `hour_cos`, `is_daytime`; per-sensor lags (1,3,6,12), diff(1), rolling mean/std/max (window 12); `nutrient_ratio`, `temp_ph_interact`, `un_ionized_ammonia`, `bloom_risk_proxy`.
- **Excluded from features**: `created_at`, `algae_risk_score`, `risk_level`, `Turbidity(NTU)`, `entry_id`. All other columns form `feature_cols`.

## Artifacts

- **classifier_v1.joblib**: sklearn-style classifier with `.predict()` and `.predict_proba()`.
- **scaler_v1.joblib**: StandardScaler fitted on training data only.
- **metadata.json**: `{"features": [...], "labels": ["Low","Medium","High"], "best_model": "...", "metrics": {...}}`.

## API response shape

- **POST /predict**: `{"predicted_level": "Low|Medium|High", "probabilities": {"Low": float, "Medium": float, "High": float}, "recommendation": "..."}`.
- **GET /predict-sample**: Example payload for `data` (sensor keys + `created_at`).
