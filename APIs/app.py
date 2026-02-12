import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ML.features import engineer
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
    for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, LABELS_PATH]:
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
    data = payload.get('data', [])
    df = pd.DataFrame(data)
    df, feature_cols_built = engineer(df)
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    x_last = X_scaled[-1:].copy()
    pred = model.predict(x_last)[0]
    proba = model.predict_proba(x_last)[0]
    probs = {labels[i]: float(proba[i]) for i in range(len(labels))}
    return jsonify({'predicted_level': str(pred), 'probabilities': probs, 'recommendation': recommend(str(pred))})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
