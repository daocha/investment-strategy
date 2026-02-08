import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.sentiment_analysis import analyze_sentiment, analyze_sentiment_batch

class TestSentimentAnalysis(unittest.TestCase):

    @patch('backend.sentiment_analysis.USE_DEEPSEEK_API', False)
    def test_analyze_sentiment_mock(self):
        """Tests single asset sentiment with mock logic."""
        result = analyze_sentiment("AAPL")
        self.assertIn("score", result)
        self.assertIn("trend", result)
        self.assertIsInstance(result["score"], float)
        self.assertIn(result["trend"], ["positive", "negative", "neutral"])

    @patch('backend.sentiment_analysis.USE_DEEPSEEK_API', False)
    def test_analyze_sentiment_batch_mock(self):
        """Tests batch sentiment with mock logic."""
        assets = ["AAPL", "BTC", "GOOGL"]
        results = analyze_sentiment_batch(assets)
        self.assertEqual(len(results), 3)
        for asset in assets:
            self.assertIn(asset, results)
            self.assertIn("score", results[asset])
            self.assertIn("trend", results[asset])

    @patch('backend.sentiment_analysis.USE_DEEPSEEK_API', True)
    @patch('backend.sentiment_analysis.DEEPSEEK_API_KEY', 'test_key')
    @patch('requests.post')
    def test_analyze_sentiment_batch_api_success(self, mock_post):
        """Tests batch sentiment with success API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"AAPL": {"score": 0.8, "trend": "positive"}, "BTC": {"score": -0.5, "trend": "negative"}}'
                }
            }]
        }
        mock_post.return_value = mock_response

        assets = ["AAPL", "BTC"]
        results = analyze_sentiment_batch(assets)
        
        self.assertEqual(results["AAPL"]["score"], 0.8)
        self.assertEqual(results["BTC"]["trend"], "negative")

    @patch('backend.sentiment_analysis.USE_DEEPSEEK_API', True)
    @patch('backend.sentiment_analysis.DEEPSEEK_API_KEY', 'test_key')
    @patch('requests.post')
    def test_analyze_sentiment_batch_api_failure(self, mock_post):
        """Tests batch sentiment with API failure."""
        mock_post.side_effect = Exception("API Error")
        
        assets = ["AAPL"]
        results = analyze_sentiment_batch(assets)
        
        # Should fallback to neutral
        self.assertEqual(results["AAPL"]["score"], 0)
        self.assertEqual(results["AAPL"]["trend"], "neutral")

if __name__ == '__main__':
    unittest.main()
