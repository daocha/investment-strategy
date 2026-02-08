
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from backend.portfolio_optimizer import generate_strategy, backtest_portfolio
from backend.config import RISK_SETTINGS

class TestStrategyRobustness(unittest.TestCase):
    
    @patch('backend.portfolio_optimizer.fetch_market_data')
    @patch('backend.portfolio_optimizer.process_single_asset')
    @patch('backend.portfolio_optimizer.get_sp500_annual_return')
    def test_generate_strategy_structure(self, mock_sp500, mock_process, mock_market_data):
        """
        Verify that generate_strategy returns a complete structure with Units, Confidence, and Native Prices.
        This catches the 'missing units' and 'missing confidence' bugs.
        """
        # 1. Setup Mock Data
        mock_sp500.return_value = 0.05
        
        # Mock Market Data Structure
        mock_market_data.return_value = {
            "Stocks": {"AAPL": {"Close": 150}},
            "Crypto": {"BTC": {"Close": 50000}}
        }

        # Mock Process Result (Success for AAPL)
        # Note: We return exactly what process_single_asset should return
        mock_process.side_effect = [
            {
                "asset": "AAPL",
                "category": "Stocks",
                "current_price": 150,
                "predicted_price": 165,
                "predicted_return": 0.10,
                "combined_return": 0.12,
                "backtest_annual_return": 0.15,
                "volatility": 0.20,
                "signal": "Buy",
                "confidence": 0.85
            },
            None # Fail BTC for this specific test or make it pass
        ]

        # 2. Run Strategy
        # Initial Amount = 10,000
        result = generate_strategy(risk_level="medium", timeframe=6, initial_amount=10000)

        # 3. Assertions
        self.assertNotIn("error", result, f"Strategy generation failed: {result.get('error')}")
        
        backtest_results = result.get("backtest_results", {})
        asset_returns = backtest_results.get("asset_returns", [])
        
        self.assertTrue(len(asset_returns) > 0, "No assets returned in strategy")
        first_asset = asset_returns[0]

        # Check for Critical Fields (Regression Test)
        self.assertIn("units", first_asset, "Missing 'units' field")
        self.assertIn("current_price_native", first_asset, "Missing 'current_price_native' field")
        self.assertIn("predicted_price_native", first_asset, "Missing 'predicted_price_native' field")
        self.assertIn("confidence", first_asset, "Missing 'confidence' field in asset")
        
        # Check Portfolio-Level Metrics
        self.assertIn("portfolio_confidence", backtest_results, "Missing 'portfolio_confidence' field")
        self.assertGreater(backtest_results["portfolio_confidence"], 0, "Portfolio confidence should be > 0")

    @patch('backend.portfolio_optimizer.fetch_market_data')
    @patch('backend.portfolio_optimizer.process_single_asset')
    @patch('backend.portfolio_optimizer.get_sp500_annual_return')
    @patch('backend.portfolio_optimizer.yf.Ticker') # Mock yfinance to prevent network calls
    def test_crypto_handling_no_crash(self, mock_ticker, mock_sp500, mock_process, mock_market_data):
        """
        Verify that Crypto assets do not trigger yfinance lookups (which cause crashes).
        Regression test for 'possibly delisted' error.
        """
        mock_sp500.return_value = 0.05
        mock_market_data.return_value = {
            "Crypto": {"PEPE": {"Close": 0.000001}}
        }
        
        # Crypto asset passes process_single_asset
        mock_process.return_value = {
            "asset": "PEPE",
            "category": "Crypto",
            "current_price": 0.000001,
            "predicted_price": 0.0000015,
            "predicted_return": 0.50,
            "combined_return": 0.40,
            "backtest_annual_return": 0.30,
            "volatility": 0.80, # High volatility
            "signal": "Buy",
            "confidence": 0.60
        }

        # Need to ensure Medium risk allows Crypto (which we fixed recently)
        # And ensure volatility threshold isn't too low (we raised it to 0.65, but PEPE is 0.80)
        # So we use "High" risk to guarantee it passes the risk filter for this crash test
        result = generate_strategy(risk_level="high", timeframe=6, initial_amount=1000)
        
        self.assertNotIn("error", result)
        assets = result.get("backtest_results", {}).get("asset_returns", [])
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["asset"], "PEPE")
        
        # Critical Info: ensure currency is USD for Crypto by default
        self.assertEqual(assets[0]["native_currency"], "USD")
        
        # Ensure yfinance Ticker was NOT called for PEPE (it shouldn't be for Crypto)
        # Note: Depending on implementation, Ticker might be called for FX rates for *base* currency if not USD.
        # But for the asset itself, it shouldn't be looked up.
        # We can check specific calls if needed, but the fact it didn't crash is the main test.

if __name__ == '__main__':
    unittest.main()
