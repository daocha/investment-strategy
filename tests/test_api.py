import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.main import app

class TestInvestmentAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('backend.main.run_portfolio_analysis')
    def test_analyze_portfolio_endpoint(self, mock_analyze):
        """Tests the POST /analyze-portfolio endpoint."""
        # Mocking the analysis result
        mock_analyze.return_value = {
            "portfolio_allocation": [],
            "backtest_results": {"portfolio_predicted_return": 10.0}
        }
        
        test_payload = {
            "portfolio_data": "Ticker,Category,Units\nAAPL,Stocks,10",
            "currency": "USD",
            "timeframe": 6
        }
        
        response = self.app.post('/analyze-portfolio', 
                                data=json.dumps(test_payload),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("backtest_results", data)
        self.assertEqual(data["backtest_results"]["portfolio_predicted_return"], 10.0)

    @patch('backend.main.generate_strategy')
    def test_generate_strategy_endpoint(self, mock_generate):
        """Tests the POST /generate-strategy endpoint."""
        mock_generate.return_value = {
            "portfolio_allocation": [{"asset": "BTC", "allocation_pct": 100}]
        }
        
        test_payload = {
            "risk_level": "medium",
            "timeframe": 12,
            "initial_amount": 1000,
            "currency": "USD"
        }
        
        response = self.app.post('/generate-strategy', 
                                data=json.dumps(test_payload),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data["portfolio_allocation"]), 1)
        self.assertEqual(data["portfolio_allocation"][0]["asset"], "BTC")

if __name__ == '__main__':
    unittest.main()
