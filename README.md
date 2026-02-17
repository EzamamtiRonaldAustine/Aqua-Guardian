# Algae Bloom Risk Prediction System

A machine learning-based early warning system that predicts pond algae bloom risk levels (Low/Medium/High) from IoT water quality sensor data.

## 🎯 Project Overview

This system monitors key water quality indicators (temperature, turbidity, pH, ammonia, nitrate) and uses advanced feature engineering and ensemble models to predict algae bloom risk, enabling proactive pond management.

### Key Features

- **Intelligent Feature Engineering**: Temporal features (lags, rolling statistics), domain-specific features (nutrient ratios, un-ionized ammonia), and time-of-day encodings
- **Robust ML Pipeline**: Time-based train/validation/test splits, hyperparameter tuning, SMOTE for class imbalance
- **Production-Ready API**: RESTful Flask API with comprehensive input validation and error handling
- **Model Persistence**: Saves trained models, scalers, and metadata for consistent inference

## 📁 Project Structure

```
ML Model/
├── ML/
│   ├── features-engineering.py  # Core feature engineering logic
│   ├── features.py              # Feature module interface
│   ├── train.py                 # Model training pipeline
│   ├── predict.py               # Prediction helper class
│   └── evaluate.py              # Model evaluation utilities
├── APIs/
│   ├── app.py                   # Flask REST API
│   └── schemas.py               # API validation schemas
├── Models/                      # Saved model artifacts
│   ├── classifier_v1.joblib
│   ├── scaler_v1.joblib
│   └── metadata.json
├── config.py                    # Configuration paths and constants
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your data file is placed in `data/raw/IoTPond6.xlsx` (or update `DATA_RAW_DIR` in `config.py`)

### Training the Model

```bash
python ML/train.py
```

This will:
- Load and engineer features from your data
- Perform time-based train/validation/test split
- Search hyperparameters for RandomForest and GradientBoosting
- Save the best model, scaler, and metadata to `Models/`

### Running the API

```bash
python APIs/app.py
```

The API will start on `http://0.0.0.0:5000`

### Making Predictions

**Via API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "created_at": "2024-01-15T10:00:00Z",
      "Temperature(C)": 26.5,
      "Turbidity(NTU)": 18.0,
      "PH": 8.2,
      "Ammonia(g/ml)": 0.035,
      "Nitrate(g/ml)": 125.0
    }]
  }'
```

**Via Python:**
```python
from ML.predict import Predictor

predictor = Predictor()
result = predictor.predict_with_proba(sensor_data_list)
print(result)
```

## 📊 Model Performance

The model achieves:
- **Weighted F1 Score**: ~0.87 (validation), ~0.87 (test)
- **Precision**: ~0.83
- **Recall**: ~0.91

Metrics are saved in `Models/metadata.json` after training.

## 🔧 Configuration

Key settings in `config.py`:
- `MODEL_PATH`: Where trained models are saved
- `DATA_RAW_DIR`: Location of input data files
- `RANDOM_STATE`: Random seed for reproducibility

## 📡 API Endpoints

- `GET /health` - Check if model artifacts exist
- `GET /status` - Get model info (classes, feature count)
- `GET /feature-cols` - List all feature columns used
- `GET /predict-sample` - Example request payload
- `POST /predict` - Make a prediction (requires JSON payload)
- `DELETE /artifacts` - Remove saved model artifacts

## 🧪 Model Details

### Feature Engineering

The pipeline creates 50+ features including:
- **Temporal features**: Lagged values (1, 3, 6, 12 steps), rolling statistics (mean, std, max over 12 steps)
- **Domain features**: Nutrient ratios, un-ionized ammonia calculation, bloom risk proxy
- **Time features**: Hour of day, daytime indicator, cyclic hour encoding

### Models

- **RandomForest**: Ensemble of decision trees with balanced class weights
- **GradientBoosting**: Sequential boosting with learning rate tuning

Hyperparameters are automatically tuned via grid search on validation set.

## ⚠️ Important Notes

- The model predicts **risk levels** based on environmental indicators, not confirmed biological blooms
- Requires sufficient historical data (12+ rows) for stable predictions due to rolling features
- Input data must include all required sensor columns: `created_at`, `Temperature(C)`, `Turbidity(NTU)`, `PH`, `Ammonia(g/ml)`, `Nitrate(g/ml)`

## 📝 License

[Add your license here]

## 👥 Authors

[Add author information]

## 🙏 Acknowledgments

[Add any acknowledgments]
