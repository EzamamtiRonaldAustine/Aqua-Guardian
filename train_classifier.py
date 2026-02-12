import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump
from feature_pipeline import engineer
os.chdir(r'C:\Users\dell\Desktop')
df = pd.read_excel('IoTPond6.xlsx', sheet_name='IoTPond6')
df, feature_cols = engineer(df)
X = df[feature_cols]
y = df['risk_level']
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_model_name]
dump(best_model, os.path.join(r'C:\Users\dell\Desktop\ML Model', 'classifier.joblib'))
dump(scaler, os.path.join(r'C:\Users\dell\Desktop\ML Model', 'scaler.joblib'))
with open(os.path.join(r'C:\Users\dell\Desktop\ML Model', 'feature_cols.json'), 'w') as f:
    json.dump(feature_cols, f)
with open(os.path.join(r'C:\Users\dell\Desktop\ML Model', 'labels.json'), 'w') as f:
    json.dump(['Low','Medium','High'], f)
print(best_model_name)
print(results[best_model_name])
