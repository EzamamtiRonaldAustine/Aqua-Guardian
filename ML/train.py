import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from joblib import dump

# Ensure project root is on the path when running as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import (
    MODEL_PATH,
    METADATA_PATH,
    SCALER_PATH,
    DATA_RAW_DIR,
    RANDOM_STATE,
)
from ML.features import engineer


DATA_FILE_NAME = "IoTPond6.xlsx"
DATA_SHEET_NAME = "IoTPond6"


def load_data():
    """
    Load the raw IoT pond data, run feature engineering, and
    return features, labels, and the feature column list.
    """
    data_path = os.path.join(DATA_RAW_DIR, DATA_FILE_NAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Update DATA_RAW_DIR or DATA_FILE_NAME if the location has changed."
        )

    df = pd.read_excel(data_path, sheet_name=DATA_SHEET_NAME)
    df, feature_cols = engineer(df)

    if "risk_level" not in df.columns:
        raise ValueError(
            "Target column 'risk_level' is missing after feature engineering. "
            "Ensure 'engineer' creates 'risk_level' for supervised training."
        )

    X = df[feature_cols]
    y = df["risk_level"]
    return X, y, feature_cols


def time_based_split(X, y, train_frac=0.7, val_frac=0.15):
    """
    Chronological train/validation/test split.
    Assumes rows are ordered by time (handled in feature engineering).
    """
    n_samples = len(X)
    if n_samples < 20:
        raise ValueError(
            f"Not enough samples ({n_samples}) to perform train/val/test split."
        )

    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))

    # Ensure indices are in a valid range
    train_end = max(1, min(train_end, n_samples - 2))
    val_end = max(train_end + 1, min(val_end, n_samples - 1))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_model_search_space():
    """
    Define candidate models and small hyperparameter grids.
    Grids are intentionally small to keep training time reasonable.
    """
    return {
        "RandomForest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "param_grid": {
                "n_estimators": [120, 200],
                "max_depth": [10, None],
                "min_samples_leaf": [1, 3],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        },
    }


def search_best_model(X_train_resampled, y_train_resampled, X_val_scaled, y_val):
    """
    Train multiple models/hyperparameters and select the best on validation F1.
    """
    search_space = get_model_search_space()
    best = {
        "model_name": None,
        "params": None,
        "model": None,
        "f1": -np.inf,
        "precision": 0.0,
        "recall": 0.0,
    }

    for name, cfg in search_space.items():
        base_estimator = cfg["estimator"]
        for params in ParameterGrid(cfg["param_grid"]):
            model = clone(base_estimator)
            model.set_params(**params)
            model.fit(X_train_resampled, y_train_resampled)

            y_val_pred = model.predict(X_val_scaled)
            f1 = f1_score(y_val, y_val_pred, average="weighted")
            precision = precision_score(
                y_val, y_val_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_val, y_val_pred, average="weighted", zero_division=0
            )

            if f1 > best["f1"]:
                best.update(
                    {
                        "model_name": name,
                        "params": params,
                        "model": model,
                        "f1": float(f1),
                        "precision": float(precision),
                        "recall": float(recall),
                    }
                )

    if best["model"] is None:
        raise RuntimeError("No model could be trained; check data and configuration.")

    return best


def main():
    # 1. Load data and engineer features
    X, y, feature_cols = load_data()

    # 2. Time-based train/validation/test split
    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(X, y)

    # 3. Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Address class imbalance on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    # 5. Hyperparameter search on validation set
    best = search_best_model(
        X_train_resampled, y_train_resampled, X_val_scaled, y_val
    )

    # 6. Evaluate the selected model on the held-out test set
    y_test_pred = best["model"].predict(X_test_scaled)
    f1_test = f1_score(y_test, y_test_pred, average="weighted")
    precision_test = precision_score(
        y_test, y_test_pred, average="weighted", zero_division=0
    )
    recall_test = recall_score(
        y_test, y_test_pred, average="weighted", zero_division=0
    )
    report_test = classification_report(
        y_test, y_test_pred, output_dict=True, zero_division=0
    )

    # 7. Persist model, scaler, and metadata
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(best["model"], MODEL_PATH)
    dump(scaler, SCALER_PATH)

    metrics = {
        "val": {
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
        },
        "test": {
            "f1": float(f1_test),
            "precision": float(precision_test),
            "recall": float(recall_test),
            "report": report_test,
        },
    }

    metadata = {
        "features": feature_cols,
        "labels": ["Low", "Medium", "High"],
        "best_model": best["model_name"],
        "best_params": best["params"],
        "metrics": metrics,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # 8. Print a concise summary
    print("Best model:", best["model_name"])
    print("Best params:", best["params"])
    print("Validation weighted F1:", best["f1"])
    print("Test weighted F1:", float(f1_test))


if __name__ == "__main__":
    main()

