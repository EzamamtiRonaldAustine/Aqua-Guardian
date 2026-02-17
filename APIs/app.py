import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
# Ensure project root and ML package are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ML_DIR = os.path.join(PROJECT_ROOT, "ML")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if ML_DIR not in sys.path:
    sys.path.append(ML_DIR)
try:
    from ML.features import engineer
except ModuleNotFoundError:
    # Fallback to direct module import if package name resolution fails
    import importlib.util
    spec = importlib.util.spec_from_file_location("features", os.path.join(ML_DIR, "features.py"))
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    engineer = features.engineer
from config import MODEL_PATH, SCALER_PATH, METADATA_PATH
app = Flask(__name__)
model = None
scaler = None
feature_cols = []
labels = ['Low', 'Medium', 'High']
def artifacts_exist():
    return all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, METADATA_PATH])
def load_artifacts():
    if not artifacts_exist():
        return False
    global model, scaler, feature_cols, labels
    model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    with open(METADATA_PATH, 'r') as f:
        md = json.load(f)
    feature_cols = md.get('features', [])
    labels = md.get('labels', ['Low','Medium','High'])
    return True
READY = load_artifacts()
@app.route('/health', methods=['GET'])
def health():
    status = {'model': os.path.exists(MODEL_PATH), 'scaler': os.path.exists(SCALER_PATH), 'metadata': os.path.exists(METADATA_PATH)}
    return jsonify({'status': 'ok', 'artifacts': status})
@app.route('/status', methods=['GET'])
def status():
    info = {'model_path': MODEL_PATH, 'scaler_path': SCALER_PATH, 'metadata_path': METADATA_PATH}
    if READY:
        info.update({'ready': True, 'classes': labels, 'feature_count': len(feature_cols)})
    else:
        info.update({'ready': False, 'classes': labels, 'feature_count': 0})
    return jsonify(info)
@app.route('/feature-cols', methods=['GET'])
def get_feature_cols():
    return jsonify({'feature_cols': feature_cols, 'ready': READY})
@app.route('/predict-sample', methods=['GET'])
def predict_sample():
    sample = {
        "data": [
            {
                "created_at": "2021-10-12T00:00:00Z",
                "Temperature(C)": 26.0,
                "Turbidity(NTU)": 15.0,
                "PH": 8.1,
                "Ammonia(g/ml)": 0.03,
                "Nitrate(g/ml)": 120.0
            }
        ]
    }
    return jsonify(sample)

@app.route('/artifacts', methods=['DELETE'])
def delete_artifacts():
    removed = []
    for p in [MODEL_PATH, SCALER_PATH, METADATA_PATH]:
        if os.path.exists(p):
            try:
                os.remove(p)
                removed.append(os.path.basename(p))
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'removed': removed})
def recommend(level):
    if level == 'Low':
        return 'Maintain routine operations'
    if level == 'Medium':
        return 'Increase monitoring and reduce nutrient inputs'
    if level == 'High':
        return 'Immediate alert: initiate mitigation actions'
    return 'Unknown'
@app.route('/predict', methods=['POST'])
def predict():
    if not READY:
        return jsonify({'error': 'artifacts missing; run /retrain or the training script'}), 503
    payload = request.get_json(force=True)
    if not payload or 'data' not in payload:
        return jsonify({'error': "Request JSON must include a 'data' field with a list of readings."}), 400

    data = payload.get('data', [])
    if not isinstance(data, list) or not data:
        return jsonify({'error': "'data' must be a non-empty list of sensor readings."}), 400

    raw_df = pd.DataFrame(data)

    # Validate required raw sensor columns before feature engineering
    required_raw_cols = {
        "created_at",
        "Temperature(C)",
        "Turbidity(NTU)",
        "PH",
        "Ammonia(g/ml)",
        "Nitrate(g/ml)",
    }
    missing_raw = sorted(required_raw_cols - set(raw_df.columns))
    if missing_raw:
        return jsonify(
            {
                'error': 'Missing required raw sensor columns.',
                'missing_columns': missing_raw,
            }
        ), 400

    df, _ = engineer(raw_df)
    if df.empty:
        return jsonify(
            {
                'error': 'Not enough historical data after feature engineering for a stable prediction.'
            }
        ), 400

    # Ensure all model feature columns are present
    missing_features = sorted(set(feature_cols) - set(df.columns))
    if missing_features:
        return jsonify(
            {
                'error': 'Engineered feature columns are missing; check training/prediction feature alignment.',
                'missing_features': missing_features,
            }
        ), 500

    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    x_last = X_scaled[-1:].copy()
    pred = model.predict(x_last)[0]
    proba = model.predict_proba(x_last)[0]
    # Map probabilities using the model's actual classes to avoid mismatch
    # if the trained model does not contain all expected labels.
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        model_classes = list(range(len(proba)))
    probs = {str(c): float(p) for c, p in zip(model_classes, proba)}
    return jsonify(
        {
            'predicted_level': str(pred),
            'probabilities': probs,
            'recommendation': recommend(str(pred)),
        }
    )
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
