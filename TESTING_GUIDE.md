# Testing Guide - Step by Step

## Prerequisites

Before running tests, you need to have:
1. ✅ Python 3.8+ installed
2. ✅ All dependencies installed (`pip install -r requirements.txt`)
3. ✅ A trained model (run training first!)

## Step-by-Step Testing Instructions

### Step 1: Train the Model First

**Why?** Tests need the model artifacts (model file, scaler, metadata) to exist.

```bash
# Navigate to your project directory
cd "c:\Users\dell\Desktop\ML Model"

# Run the training script
python ML/train.py
```

**Expected Output:**
- You should see: "Best model: RandomForest" (or GradientBoosting)
- Model files created in `Models/` folder:
  - `classifier_v1.joblib`
  - `scaler_v1.joblib`
  - `metadata.json`

**Troubleshooting:**
- If you get "Data file not found": Make sure `data/raw/IoTPond6.xlsx` exists
- If you get import errors: Run `pip install -r requirements.txt`

---

### Step 2: Run the Basic Tests

**Option A: Run all tests in the tests folder**

```bash
# IMPORTANT: run this from the PROJECT ROOT, not from venv\Scripts
# Good:
#   cd "C:\Users\dell\Desktop\ML Model"
#   python -m pytest tests -v
#
# Bad (will fail to find tests/):
#   cd "C:\Users\dell\Desktop\ML Model\venv\Scripts"
#   python -m pytest tests -v

cd "C:\Users\dell\Desktop\ML Model"
python -m pytest tests -v
```

**Option B: Run tests using unittest (built-in)**

```bash
# Run the predictor tests specifically
python tests/test_predictor.py
```

**Option C: Run with more verbose output**

```bash
python -m unittest tests.test_predictor -v
```

**Expected Output:**
```
test_predictor_initialization ... ok
test_predict_with_sample_data ... ok
test_predict_with_proba ... ok
test_predict_with_insufficient_data ... ok

----------------------------------------------------------------------
Ran 4 tests in X.XXXs

OK
```

---

### Step 3: Understand What Each Test Does

#### Test 1: `test_predictor_initialization`
- **What it checks:** That the Predictor class loads correctly
- **Checks:** Model exists, scaler exists, feature columns loaded, labels are correct
- **Why it matters:** Ensures your saved model can be loaded for predictions

#### Test 2: `test_predict_with_sample_data`
- **What it checks:** That predictions work with valid sensor data
- **Checks:** Returns a valid risk level (Low/Medium/High)
- **Why it matters:** Core functionality - making predictions

#### Test 3: `test_predict_with_proba`
- **What it checks:** That probability predictions work
- **Checks:** Returns probabilities that sum to 1.0
- **Why it matters:** Shows prediction confidence

#### Test 4: `test_predict_with_insufficient_data`
- **What it checks:** That errors are raised for bad input
- **Checks:** Raises ValueError when data is insufficient
- **Why it matters:** Ensures proper error handling

---

### Step 4: Run API Tests (Optional)

If you want to test the Flask API:

```bash
# Terminal 1: Start the API server
python APIs/app.py

# Terminal 2: Test the API endpoints
python tests/test_api.py
```

---

## Quick Test Commands Reference

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_predictor.py

# Run with coverage report (if pytest-cov installed)
python -m pytest tests/ --cov=ML --cov-report=html

# Run tests and see print statements
python -m pytest tests/ -v -s
```

---

## What to Do If Tests Fail

### Error: "Model artifacts not found"
**Solution:** Run `python ML/train.py` first to create model files

### Error: "ModuleNotFoundError"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Error: "AssertionError" in a test
**Solution:** 
- Check that your model was trained successfully
- Verify `Models/` folder contains all three files
- Re-run training: `python ML/train.py`

### Error: "ValueError" in test_predict_with_insufficient_data
**Solution:** This is actually GOOD - it means error handling works! The test expects this error.

---

## Testing Checklist

Before pushing to main, verify:

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Model trains successfully: `python ML/train.py`
- [ ] Predictor loads correctly: `python tests/test_predictor.py`
- [ ] API starts without errors: `python APIs/app.py`
- [ ] No import errors when running tests

---

## Next Steps: Adding More Tests

Want to add more tests? Here's what you could test:

1. **Feature Engineering Tests**
   - Test that features are created correctly
   - Test outlier clipping works
   - Test missing data handling

2. **Training Pipeline Tests**
   - Test data loading
   - Test train/val/test split
   - Test model saving

3. **API Tests**
   - Test all endpoints
   - Test error handling
   - Test input validation

See `tests/test_api.py` and `tests/test_features.py` for examples (if they exist).
