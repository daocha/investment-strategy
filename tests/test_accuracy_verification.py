
import unittest
import pandas as pd
import numpy as np
import logging
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pytz

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.portfolio_optimizer import generate_strategy

# Configure logging
LOG_FILE = "backtest_verification.log"
# logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
#                     format="%(asctime)s - %(message)s", filemode='w')

# Use basicConfig without filename to stream to stdout by default (or configure both)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class TestAccuracyVerification(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Prepare real market data for a few benchmark assets."""
        from backend.market_data import fetch_historical_data
        
        # Reduced Asset List for Speed
        cls.assets = {
            "Stocks": ["AAPL", "NVDA", "SPY"],
            "Crypto": ["BTC"]
        }
        
        cls.full_history = {}
        logging.info("Step 1: Fetching MAX history for benchmark assets...")
        
        for category, tickers in cls.assets.items():
            for ticker in tickers:
                try:
                    # Fetching 'max' to support deep historical backtests
                    print(f"Fetching {ticker} (max)...")
                    df = fetch_historical_data(ticker, category, period="5y") # Revert to 5y for speed
                    if df is not None and not df.empty:
                        # Ensure timezone-naive for comparison
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        cls.full_history[ticker] = df
                        logging.info(f"   Fetched {len(df)} rows for {ticker}")
                    else:
                        logging.warning(f"   Failed to fetch history for {ticker}")
                except Exception as e:
                    logging.error(f"   Error fetching {ticker}: {e}")

    def get_price_at_date(self, ticker, date):
        """Helper: Get closing price on or before a specific date."""
        df = self.full_history.get(ticker)
        if df is None: return None
        
        # Filter for data up to date
        past_data = df[df.index <= date]
        if past_data.empty: return None
        
        return past_data.iloc[-1]["Close"]

    @patch('backend.market_data.fetch_historical_data')
    @patch('backend.portfolio_optimizer.fetch_market_data')
    @patch('backend.portfolio_optimizer.analyze_sentiment')
    @patch('backend.portfolio_optimizer.xgb_model') # Mock model to isolate verify logic first
    @patch('backend.portfolio_optimizer.get_sp500_annual_return')
    def test_backtest_accuracy(self, mock_sp500, mock_xgb, mock_sentiment, mock_market_data, mock_hist_data):
        """
        Simulate portfolio generation at past points using REAL sliced history.
        """
        logging.info("\n=== STARTING TIME-TRAVEL ACCURACY VERIFICATION ===")
        
        timeframes = {
            "6M": 180,
            "1Y": 365, 
            "2Y": 365 * 2,
            "3Y": 365 * 3,
            "5Y": 365 * 5
        }
        
        today = datetime.now()
        mock_sp500.return_value = 0.08 
        
        # Mock Sentiment: Neutral (0.5) to test technicals only
        mock_sentiment.return_value = {"score": 0.5, "trend": "neutral"}
        
        # Mock XGBoost (Simple Logistic Regression Logic or Fixed Signal)
        # To test 'real' logic, we should ideally let it run, but without trained model file it might fail.
        # Let's mock it to be slightly bullish if trend is up.
        def mock_predict(df):
            # Simple Trend Following Mock
            if df.empty: return "Hold", 0.0
            last_price = df.iloc[-1]["Close"]
            ma50 = df["Close"].rolling(50).mean()
            # If 50-day average has NaN, use 'Buy' anyway to ensure test data flows
            if ma50.empty or pd.isna(ma50.iloc[-1]):
                 return "Buy", 0.5
            val_ma50 = ma50.iloc[-1]
            if last_price > val_ma50 * 0.9: # Relaxed condition (within 10% of MA)
                return "Buy", 0.8
            else:
                return "Buy", 0.4 # Even on sell signal, return weak buy to avoid empty portfolio in test
        mock_xgb.predict.side_effect = mock_predict

        results = []

        for label, days in timeframes.items():
            simulated_date = today - timedelta(days=days)
            
            # Check if simulated date is valid based on available data
            # Use AAPL as reference for last available date
            aapl_data = self.full_history.get("AAPL")
            if aapl_data is not None and not aapl_data.empty:
               last_date_avail = aapl_data.index[-1]
               # Check if simulated date is > last available date
               if simulated_date > last_date_avail:
                   logging.warning(f"   Simulated date {simulated_date.strftime('%Y-%m-%d')} is in future relative to data ({last_date_avail.strftime('%Y-%m-%d')}). Using last avail.")
                   simulated_date = last_date_avail

            logging.info(f"\n--- Testing Timeframe: {label} (Simulated Date: {simulated_date.strftime('%Y-%m-%d')}) ---")
            
            # 1. Mock Market Data (Current Prices at Simulated Date)
            mock_market_response = {"Stocks": {}, "Crypto": {}}
            valid_tickers = []
            
            for category, tickers in self.assets.items():
                for ticker in tickers:
                    price = self.get_price_at_date(ticker, simulated_date)
                    if price:
                        entry = {"Close": price}
                        if category == "Stocks":
                            mock_market_response["Stocks"][ticker] = entry
                        else:
                            mock_market_response["Crypto"][ticker] = entry
                        valid_tickers.append((ticker, category, price))
            
            if not valid_tickers:
                logging.warning(f"   Skipping {label}: No valid data.")
                continue

            mock_market_data.return_value = mock_market_response
            
            # 2. Mock fetch_historical_data (Time Travel!)
            def side_effect_hist_data(ticker, category, period="1y"):
                # Fetch full history
                if ticker not in self.full_history: return None
                full_df = self.full_history[ticker]
                
                # Slice: Data strictly BEFORE simulated_date
                # And satisfy 'period' length (roughly)
                # Parse period
                days_back = 365 # Default to 1 year minimum for indicators
                if period == "6m": days_back = 365 # Force 1Y for 6M strat to calc SMA200
                elif period == "2y": days_back = 730
                elif period == "5y": days_back = 1825
                
                # Fetch EXTRA buffer for indicators (SMA200 needs 200 bars + warmup)
                days_back += 100 
                
                start_date = simulated_date - timedelta(days=days_back)
                
                # Sliced DataFrame
                mask = (full_df.index >= start_date) & (full_df.index <= simulated_date)
                sliced_df = full_df.loc[mask].copy()
                
                if sliced_df.empty: return None
                return sliced_df

            mock_hist_data.side_effect = side_effect_hist_data
            
            # 3. Generate Strategy (Runs REAL logic with sliced data)
            strategy_timeframe_months = int(days / 30.4)
            if strategy_timeframe_months == 0: strategy_timeframe_months = 6
            
            logging.info(f"   Generating strategy for next {strategy_timeframe_months} months...")
            # Use 'high' risk to ensure we pick up volatile assets
            strategy = generate_strategy(risk_level="high", timeframe=strategy_timeframe_months, initial_amount=10000)
            
            if "error" in strategy:
                logging.error(f"   Strategy generation failed: {strategy['error']}")
                continue
                
            predicted_return_period = strategy["backtest_results"]["portfolio_predicted_return"]
            # Check if it's already percentage (e.g. 15.0) or decimal (0.15)
            # portfolio_optimizer.py returns 'round(total * 100, 2)' -> Percentage 15.0
            # So we divide by 100
            predicted_return_period /= 100.0
            
            # 4. Calculate ACTUAL Return
            end_date_period = simulated_date + timedelta(days=days)
            if end_date_period > today: end_date_period = today
            
            allocation = strategy["portfolio_allocation"]
            if not allocation:
                logging.warning("   Empty portfolio generated.")
                continue
                
            total_start_value = 0
            total_end_value = 0
            
            logging.info("   Portfolio Composition:")
            for item in allocation:
                asset = item["asset"]
                weight = item["allocation_pct"] / 100.0
                start_price = item["current_price_native"]
                
                end_price = self.get_price_at_date(asset, end_date_period)
                if not end_price: end_price = self.get_price_at_date(asset, today)
                if not end_price: end_price = start_price
                
                actual_asset_return = (end_price - start_price) / start_price
                
                total_start_value += weight * 10000
                total_end_value += (weight * 10000) * (1 + actual_asset_return)
                
                logging.info(f"      {asset}: {weight*100:.1f}% | Start=${start_price:.2f} -> End=${end_price:.2f} | Return={actual_asset_return*100:.1f}%")

            actual_total_return = (total_end_value - total_start_value) / total_start_value
            
            deviation = abs(predicted_return_period - actual_total_return)
            
            log_msg = (
                f"   [RESULT {label}] "
                f"Predicted (Period): {predicted_return_period*100:.2f}% | "
                f"Actual (Realized): {actual_total_return*100:.2f}% | "
                f"Deviation: {deviation*100:.2f}%"
            )
            logging.info(log_msg)
            print(log_msg)
            
            if deviation > 0.5:
                logging.warning(f"   ⚠️ LARGE DEVIATION for {label}.")
            
            results.append({
                "timeframe": label,
                "predicted": predicted_return_period,
                "actual": actual_total_return,
                "deviation": deviation
            })

if __name__ == '__main__':
    unittest.main()
