import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.portfolio_optimizer import process_single_asset, convert_floats

class TestPortfolioOptimizerLogic(unittest.TestCase):

    def test_convert_floats_scalar(self):
        """Tests convert_floats with scalar values."""
        self.assertEqual(convert_floats(10.551), 10.551)
        self.assertEqual(convert_floats("test"), "test")
        self.assertEqual(convert_floats(10), 10)

    def test_convert_floats_nested(self):
        """Tests convert_floats with nested structures."""
        data = {
            "a": 1.231,
            "b": [1.231, {"c": 5.671}]
        }
        expected = {
            "a": 1.231,
            "b": [1.231, {"c": 5.671}]
        }
        self.assertEqual(convert_floats(data), expected)

    @patch('backend.portfolio_optimizer.fetch_historical_data')
    @patch('backend.portfolio_optimizer.calculate_indicators')
    @patch('backend.portfolio_optimizer.backtest_strategy')
    @patch('backend.portfolio_optimizer.analyze_sentiment')
    @patch('backend.portfolio_optimizer.xgb_model') 
    def test_process_single_asset_success(self, mock_xgb, mock_sentiment, mock_backtest, mock_indicators, mock_hist):
        """Tests process_single_asset with complete mock success."""
        # Provide 30 rows to satisfy rolling window requirements (14, 21, etc.)
        mock_data = pd.DataFrame({
            "Close": np.linspace(100, 110, 30), 
            "High": np.linspace(105, 115, 30), 
            "Low": np.linspace(95, 105, 30), 
            "Volume": np.linspace(1000, 1100, 30)
        }, index=pd.date_range("2023-01-01", periods=30))
        
        mock_hist.return_value = mock_data
        mock_indicators.return_value = {"RSI": 60, "MACD": 1}
        mock_backtest.return_value = 0.15
        mock_sentiment.return_value = {"score": 0.5, "trend": "positive"}
        mock_xgb.predict.return_value = ("Buy", 0.8)

        asset_data = {
            "Close": 110,
            "Currency": "USD"
        }

        result = process_single_asset(
            asset="AAPL",
            category="Stocks",
            timeframe=6,
            market_data_entry=asset_data,
            min_annual_return=-10.0,
            ignore_filters=True
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["asset"], "AAPL")
        self.assertAlmostEqual(result["backtest_annual_return"], 0.0723805, places=5)
        self.assertEqual(result["signal"], "Buy")

    @patch('backend.portfolio_optimizer.fetch_historical_data')
    @patch('backend.portfolio_optimizer.calculate_indicators')
    @patch('backend.portfolio_optimizer.backtest_strategy')
    @patch('backend.portfolio_optimizer.analyze_sentiment')
    @patch('backend.portfolio_optimizer.xgb_model')
    def test_process_single_asset_precomputed_sentiment(self, mock_xgb, mock_sentiment, mock_backtest, mock_indicators, mock_hist):
        """Tests process_single_asset using precomputed sentiment."""
        mock_data = pd.DataFrame({
            "Close": np.linspace(100, 110, 30), 
            "High": np.linspace(105, 115, 30), 
            "Low": np.linspace(95, 105, 30), 
            "Volume": np.linspace(1000, 1100, 30)
        }, index=pd.date_range("2023-01-01", periods=30))
        
        mock_hist.return_value = mock_data
        mock_indicators.return_value = {"RSI": 60, "MACD": 1}
        mock_backtest.return_value = 0.15
        mock_xgb.predict.return_value = ("Buy", 0.8)
        
        precomputed = {"AAPL": {"score": 0.9, "trend": "positive"}}
        
        # mock_sentiment should NOT be called
        result = process_single_asset(
            asset="AAPL",
            category="Stocks",
            timeframe=6,
            market_data_entry={"Close": 110},
            min_annual_return=-10.0,
            ignore_filters=True,
            precomputed_sentiment=precomputed
        )
        
        self.assertFalse(mock_sentiment.called)
        self.assertIsNotNone(result)
        self.assertEqual(result["asset"], "AAPL")

if __name__ == '__main__':
    unittest.main()
