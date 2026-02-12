# Inference logic only.

import joblib
import json
import pandas as pd
from config import MODEL_PATH, METADATA_PATH
from ml.features import build_features


class Predictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        self.feature_cols = metadata["features"]

    def predict(self, data):

        df = pd.DataFrame(data)

        df, _ = build_features(df)

        if df.empty:
            raise ValueError("Not enough historical data for prediction.")

        X = df[self.feature_cols]

        prediction = self.model.predict(X)

        return prediction[-1]
