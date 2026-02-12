import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
from feature_pipeline import engineer
import time
app = Flask(__name__)
MODEL_PATH = os.path.join(r'C:\Users\dell\Desktop\ML Model', 'classifier.joblib')
SCALER_PATH = os.path.join(r'C:\Users\dell\Desktop\ML Model', 'scaler.joblib')
FEATURES_PATH = os.path.join(r'C:\Users\dell\Desktop\ML Model', 'feature_cols.json')
LABELS_PATH = os.path.join(r'C:\Users\dell\Desktop\ML Model', 'labels.json')
def ensure_artifacts():
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from imblearn.over_sampling import SMOTE
    from joblib import dump
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH)):
        df = pd.read_excel(os.path.join(r'C:\Users\dell\Desktop', 'IoTPond6.xlsx'), sheet_name='IoTPond6')
        df, feature_cols_local = engineer(df)
        X = df[feature_cols_local]
        y = df['risk_level']
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        scaler_local = StandardScaler()
        X_train_scaled = scaler_local.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        }
        results = {}
        for name, m in models.items():
            m.fit(X_train_resampled, y_train_resampled)
            X_test_scaled = scaler_local.transform(X_test)
            y_pred = m.predict(X_test_scaled)
            results[name] = {'f1': f1_score(y_test, y_pred, average='weighted')}
        best_name = max(results, key=lambda x: results[x]['f1'])
        dump(models[best_name], MODEL_PATH)
        dump(scaler_local, SCALER_PATH)
        with open(FEATURES_PATH, 'w') as f:
            json.dump(feature_cols_local, f)
        with open(LABELS_PATH, 'w') as f:
            json.dump(['Low','Medium','High'], f)
ensure_artifacts()
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
with open(FEATURES_PATH, 'r') as f:
    feature_cols = json.load(f)
with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)
@app.route('/health', methods=['GET'])
def health():
    status = {
        'model': os.path.exists(MODEL_PATH),
        'scaler': os.path.exists(SCALER_PATH),
        'feature_cols': os.path.exists(FEATURES_PATH),
        'labels': os.path.exists(LABELS_PATH)
    }
    return jsonify({'status': 'ok', 'artifacts': status})
@app.route('/status', methods=['GET'])
def status():
    info = {
        'classes': labels,
        'feature_count': len(feature_cols),
        'model_path': MODEL_PATH,
        'scaler_path': SCALER_PATH
    }
    return jsonify(info)
@app.route('/feature-cols', methods=['GET'])
def get_feature_cols():
    return jsonify({'feature_cols': feature_cols})
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
@app.route('/retrain', methods=['PUT'])
def retrain():
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from imblearn.over_sampling import SMOTE
    from joblib import dump
    df = pd.read_excel(os.path.join(r'C:\Users\dell\Desktop', 'IoTPond6.xlsx'), sheet_name='IoTPond6')
    df, feature_cols_local = engineer(df)
    X = df[feature_cols_local]
    y = df['risk_level']
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_train_resampled, y_train_resampled)
        X_test_scaled = scaler_local.transform(X_test)
        y_pred = m.predict(X_test_scaled)
        scores[name] = float(f1_score(y_test, y_pred, average='weighted'))
    best_name = max(scores, key=lambda x: scores[x])
    dump(models[best_name], MODEL_PATH)
    dump(scaler_local, SCALER_PATH)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(feature_cols_local, f)
    with open(LABELS_PATH, 'w') as f:
        json.dump(['Low','Medium','High'], f)
    return jsonify({'message': 'retrained', 'best_model': best_name, 'scores': scores})
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
