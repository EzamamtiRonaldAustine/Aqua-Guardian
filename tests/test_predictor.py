"""
Basic tests for the Predictor class.
"""
import os
import sys
import unittest
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ML.predict import Predictor
from config import MODEL_PATH, METADATA_PATH, SCALER_PATH


class TestPredictor(unittest.TestCase):
    """Test cases for Predictor class."""
    
    @classmethod
    def setUpClass(cls):
        """Check if model artifacts exist before running tests."""
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, METADATA_PATH]):
            raise unittest.SkipTest("Model artifacts not found. Run ML/train.py first.")
        
        cls.predictor = Predictor()
    
    def test_predictor_initialization(self):
        """Test that Predictor initializes correctly."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertIsInstance(self.predictor.feature_cols, list)
        self.assertGreater(len(self.predictor.feature_cols), 0)
        self.assertIn('Low', self.predictor.labels)
        self.assertIn('Medium', self.predictor.labels)
        self.assertIn('High', self.predictor.labels)
    
    def test_predict_with_sample_data(self):
        """Test prediction with minimal valid data."""
        # Create sample data with enough rows for rolling features
        sample_data = []
        base_time = pd.Timestamp('2024-01-01 10:00:00')
        
        for i in range(15):  # Enough rows for lag_12 and rolling_12
            sample_data.append({
                'created_at': (base_time + pd.Timedelta(hours=i)).isoformat(),
                'Temperature(C)': 25.0 + (i % 3),
                'Turbidity(NTU)': 15.0 + (i % 5),
                'PH': 8.0 + (i % 2) * 0.2,
                'Ammonia(g/ml)': 0.03 + (i % 2) * 0.01,
                'Nitrate(g/ml)': 100.0 + (i % 4) * 10,
            })
        
        prediction = self.predictor.predict(sample_data)
        self.assertIn(prediction, ['Low', 'Medium', 'High'])
    
    def test_predict_with_proba(self):
        """Test prediction with probabilities."""
        sample_data = []
        base_time = pd.Timestamp('2024-01-01 10:00:00')
        
        for i in range(15):
            sample_data.append({
                'created_at': (base_time + pd.Timedelta(hours=i)).isoformat(),
                'Temperature(C)': 26.0,
                'Turbidity(NTU)': 20.0,
                'PH': 8.1,
                'Ammonia(g/ml)': 0.035,
                'Nitrate(g/ml)': 120.0,
            })
        
        result = self.predictor.predict_with_proba(sample_data)
        self.assertIn('predicted_level', result)
        self.assertIn('probabilities', result)
        self.assertIn(result['predicted_level'], ['Low', 'Medium', 'High'])
        
        # Check probabilities sum to ~1.0
        prob_sum = sum(result['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
    
    def test_predict_with_insufficient_data(self):
        """Test that predictor raises error with insufficient data."""
        insufficient_data = [{
            'created_at': '2024-01-01T10:00:00Z',
            'Temperature(C)': 25.0,
            'Turbidity(NTU)': 15.0,
            'PH': 8.0,
            'Ammonia(g/ml)': 0.03,
            'Nitrate(g/ml)': 100.0,
        }]
        
        with self.assertRaises(ValueError):
            self.predictor.predict(insufficient_data)


if __name__ == '__main__':
    unittest.main()
