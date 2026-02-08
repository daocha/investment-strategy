import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import backend.market_data
from backend.market_data import get_fx_rate, MARKET_DATA_CACHE, fetch_yfinance_data, fetch_crypto_data
from backend.config import CRYPTO_ETF_MAPPING

class TestMarketData(unittest.TestCase):

    def setUp(self):
        """Clear the global market data cache before each test to ensure isolation."""
        MARKET_DATA_CACHE.clear()

    @patch('backend.market_data.yf.Ticker')
    def test_get_fx_rate_success(self, mock_ticker_class):
        """Tests get_fx_rate with a successful Yahoo Finance response."""
        # Setup mock ticker
        mock_ticker = MagicMock()
        # Mock history() to return a DataFrame with a "Close" price
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [7.8]}, 
            index=pd.to_datetime(["2023-01-01"])
        )
        mock_ticker_class.return_value = mock_ticker

        rate = get_fx_rate("USD", "HKD")
        
        self.assertEqual(rate, 7.8)
        self.assertTrue(mock_ticker.history.called)

    def test_get_fx_rate_same_currency(self):
        """Tests get_fx_rate when base and target currencies are the same."""
        with patch('backend.market_data.yf.Ticker') as mock_ticker_class:
            rate = get_fx_rate("USD", "USD")
            self.assertEqual(rate, 1.0)
            self.assertFalse(mock_ticker_class.called)

    def test_crypto_etf_mapping(self):
        """Verifies that common crypto ETFs are correctly mapped to their underlying."""
        self.assertEqual(CRYPTO_ETF_MAPPING.get("IBIT"), "BTC")
        self.assertEqual(CRYPTO_ETF_MAPPING.get("3439.HK"), "BTC")

    @patch('backend.market_data.yf.Ticker')
    def test_fetch_yfinance_data_mock(self, mock_ticker_class):
        """Verifies yfinance data fetching with deep mocks."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"longName": "Apple Inc.", "currency": "USD"}
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [150.0], "Volume": [1000]}, 
            index=pd.to_datetime(["2023-01-01"])
        )
        mock_ticker.fast_info.get.return_value = "USD"
        mock_ticker_class.return_value = mock_ticker

        result = fetch_yfinance_data("AAPL")
        self.assertEqual(result["Close"], 150.0)

    @patch('backend.market_data.requests.get')
    def test_fetch_crypto_data_mock(self, mock_get):
        """Verifies crypto data fetching with mocked Binance response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "lastPrice": "50000.0",
            "quoteVolume": "1000000.0",
            "priceChangePercent": "2.5"
        }
        mock_get.return_value = mock_response

        result = fetch_crypto_data("BTC", current_time=1234567)
        self.assertEqual(result["Close"], 50000.0)

if __name__ == '__main__':
    unittest.main()
