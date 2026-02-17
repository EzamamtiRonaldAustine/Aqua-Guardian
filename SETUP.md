# Setup Guide

## Environment Setup

### 1. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your Excel file (`IoTPond6.xlsx`) in the `data/raw/` directory:

```
ML Model/
└── data/
    └── raw/
        └── IoTPond6.xlsx
```

If your data is in a different location, update `DATA_RAW_DIR` in `config.py`.

### 4. Verify Data Format

Your Excel file should contain a sheet named `IoTPond6` with columns:
- `created_at` (datetime)
- `Temperature(C)` (numeric)
- `Turbidity(NTU)` (numeric)
- `PH` (numeric)
- `Ammonia(g/ml)` (numeric)
- `Nitrate(g/ml)` (numeric)
- `entry_id` (optional)

## Training Your First Model

1. Run the training script:
   ```bash
   python ML/train.py
   ```

2. Check output:
   - Model saved to `Models/classifier_v1.joblib`
   - Scaler saved to `Models/scaler_v1.joblib`
   - Metadata saved to `Models/metadata.json`

3. Review metrics printed to console and in `metadata.json`

## Testing the API

1. Start the Flask server:
   ```bash
   python APIs/app.py
   ```

2. In another terminal, test the health endpoint:
   ```bash
   curl http://localhost:5000/health
   ```

3. Test prediction (see README.md for example payload)

## Troubleshooting

### "Data file not found"
- Check that `data/raw/IoTPond6.xlsx` exists
- Verify `DATA_RAW_DIR` in `config.py` points to the correct location

### "Not enough samples"
- Ensure your dataset has at least 20 rows
- Check that feature engineering isn't dropping too many rows

### Import errors
- Ensure you're in the project root directory
- Verify all packages in `requirements.txt` are installed
- Check Python version (3.8+ required)

### API returns 503
- Run `ML/train.py` first to generate model artifacts
- Check that `Models/` directory contains all three files (model, scaler, metadata)
