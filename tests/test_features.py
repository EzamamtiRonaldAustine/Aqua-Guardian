"""
Tests for feature engineering functions.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ML.features import engineer, build_features


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering."""
    
    def setUp(self):
        """Create sample data for testing."""
        self.sample_data = pd.DataFrame({
            # pandas expects lowercase frequency aliases (use 'h' not 'H')
            'created_at': pd.date_range('2024-01-01', periods=20, freq='h'),
            'Temperature(C)': np.random.uniform(20, 30, 20),
            'Turbidity(NTU)': np.random.uniform(10, 50, 20),
            'PH': np.random.uniform(7.0, 9.0, 20),
            'Ammonia(g/ml)': np.random.uniform(0.01, 0.05, 20),
            'Nitrate(g/ml)': np.random.uniform(50, 200, 20),
            'entry_id': range(1, 21)
        })
    
    def test_engineer_returns_dataframe_and_features(self):
        """Test that engineer returns both dataframe and feature columns."""
        df, feature_cols = engineer(self.sample_data.copy())
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(feature_cols, list)
        self.assertGreater(len(feature_cols), 0)
    
    def test_engineer_creates_risk_level(self):
        """Test that engineer creates risk_level column."""
        df, _ = engineer(self.sample_data.copy())
        
        self.assertIn('risk_level', df.columns)
        self.assertTrue(df['risk_level'].isin(['Low', 'Medium', 'High']).all())
    
    def test_engineer_creates_temporal_features(self):
        """Test that temporal features (lags, rolling) are created."""
        df, feature_cols = engineer(self.sample_data.copy())
        
        # Check for lag features
        lag_features = [col for col in feature_cols if '_lag_' in col]
        self.assertGreater(len(lag_features), 0)
        
        # Check for rolling features
        rolling_features = [col for col in feature_cols if '_roll_' in col]
        self.assertGreater(len(rolling_features), 0)
    
    def test_engineer_creates_time_features(self):
        """Test that time-based features are created."""
        df, feature_cols = engineer(self.sample_data.copy())
        
        # Check for time features
        self.assertIn('hour', feature_cols)
        self.assertIn('is_daytime', feature_cols)
        self.assertIn('hour_sin', feature_cols)
        self.assertIn('hour_cos', feature_cols)
    
    def test_engineer_handles_missing_values(self):
        """Test that engineer handles missing values."""
        data_with_nans = self.sample_data.copy()
        data_with_nans.loc[5:10, 'Temperature(C)'] = np.nan
        
        df, _ = engineer(data_with_nans)
        
        # Should not have NaN values in feature columns
        self.assertFalse(df[df.columns.intersection(df.columns)].isna().any().any())
    
    def test_engineer_clips_outliers(self):
        """Test that outlier clipping works."""
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, 'Temperature(C)'] = 100  # Extreme outlier
        data_with_outliers.loc[1, 'PH'] = 15  # Extreme outlier
        
        df, _ = engineer(data_with_outliers)
        
        # Temperature should be clipped to max 40
        self.assertLessEqual(df['Temperature(C)'].max(), 40)
        
        # PH should be clipped to max 10
        self.assertLessEqual(df['PH'].max(), 10)
    
    def test_build_features_alias(self):
        """Test that build_features is an alias for engineer."""
        df1, cols1 = engineer(self.sample_data.copy())
        df2, cols2 = build_features(self.sample_data.copy())
        
        # Should produce same results
        self.assertEqual(set(cols1), set(cols2))
        self.assertEqual(len(df1), len(df2))
    
    def test_engineer_requires_minimum_rows(self):
        """Test that engineer needs sufficient data."""
        # Very small dataset
        tiny_data = self.sample_data.head(5).copy()
        
        df, _ = engineer(tiny_data)
        
        # Should still work but may have fewer features due to lag requirements
        self.assertIsInstance(df, pd.DataFrame)
    
    def test_feature_columns_exclude_targets(self):
        """Test that feature columns don't include target variables."""
        df, feature_cols = engineer(self.sample_data.copy())
        
        # Risk level and algae score should not be in features
        self.assertNotIn('risk_level', feature_cols)
        self.assertNotIn('algae_risk_score', feature_cols)
        self.assertNotIn('created_at', feature_cols)


if __name__ == '__main__':
    unittest.main()
