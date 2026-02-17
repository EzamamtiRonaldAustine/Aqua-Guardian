"""
Tests for the Flask API endpoints.
"""
import os
import sys
import unittest
import json
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import Flask test client
try:
    from APIs.app import app
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


@unittest.skipIf(not FLASK_AVAILABLE, "Flask app not available")
class TestAPI(unittest.TestCase):
    """Test cases for Flask API endpoints."""
    
    def setUp(self):
        """Set up test client for each test."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test the /health endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('artifacts', data)
    
    def test_status_endpoint(self):
        """Test the /status endpoint."""
        response = self.app.get('/status')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('ready', data)
        self.assertIn('classes', data)
        self.assertIn('feature_count', data)
    
    def test_feature_cols_endpoint(self):
        """Test the /feature-cols endpoint."""
        response = self.app.get('/feature-cols')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('feature_cols', data)
        self.assertIn('ready', data)
    
    def test_predict_sample_endpoint(self):
        """Test the /predict-sample endpoint."""
        response = self.app.get('/predict-sample')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)
        self.assertGreater(len(data['data']), 0)
    
    def test_predict_endpoint_missing_data(self):
        """Test /predict endpoint with missing data field."""
        response = self.app.post(
            '/predict',
            json={},  # Missing 'data' field
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_endpoint_empty_data(self):
        """Test /predict endpoint with empty data list."""
        response = self.app.post(
            '/predict',
            json={'data': []},  # Empty list
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_endpoint_missing_columns(self):
        """Test /predict endpoint with missing required columns."""
        response = self.app.post(
            '/predict',
            json={
                'data': [{
                    'created_at': '2024-01-01T10:00:00Z',
                    # Missing other required columns
                }]
            },
            content_type='application/json'
        )
        # Should return 400 for missing columns
        self.assertIn(response.status_code, [400, 500])
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_predict_endpoint_valid_data(self):
        """Test /predict endpoint with valid data (if model is ready)."""
        # Create sample data with enough rows for rolling features
        sample_data = []
        base_time = '2024-01-01T10:00:00Z'
        
        for i in range(15):
            sample_data.append({
                'created_at': f'2024-01-01T{10+i:02d}:00:00Z',
                'Temperature(C)': 26.0,
                'Turbidity(NTU)': 20.0,
                'PH': 8.1,
                'Ammonia(g/ml)': 0.035,
                'Nitrate(g/ml)': 120.0,
            })
        
        response = self.app.post(
            '/predict',
            json={'data': sample_data},
            content_type='application/json'
        )
        
        # If model is ready, should return 200
        # If model not ready, should return 503
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('predicted_level', data)
            self.assertIn('probabilities', data)
            self.assertIn('recommendation', data)
            self.assertIn(data['predicted_level'], ['Low', 'Medium', 'High'])
        else:
            # Model not ready - that's okay for testing
            self.assertEqual(response.status_code, 503)


if __name__ == '__main__':
    unittest.main()
