
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from backend.market_data import fetch_historical_data
from backend.portfolio_optimizer import generate_strategy, convert_floats, xgb_model

logging.basicConfig(level=logging.INFO)

class VerificationService:
    def __init__(self):
        self.assets = {
            "Stocks": ["AAPL", "NVDA", "SPY"],
            "Crypto": ["BTC"]
        }
        self.full_history = {}
        # Lazy loading in run_simulation

    def _ensure_history_loaded(self, additional_assets=None):
        """Pre-fetch history for simulation (5Y max)."""
        to_fetch = self.assets.copy()
        if additional_assets:
            for item in additional_assets:
                cat = item.get("category", "Stocks")
                ticker = item["asset"]
                if cat not in to_fetch: to_fetch[cat] = []
                if ticker not in to_fetch[cat]:
                    to_fetch[cat].append(ticker)

        logging.info(f"Checking historical data for {sum(len(v) for v in to_fetch.values())} assets...")
        for category, tickers in to_fetch.items():
            for ticker in tickers:
                if ticker in self.full_history: continue
                try:
                    df = fetch_historical_data(ticker, category, period="10y")
                    if df is not None and not df.empty:
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        self.full_history[ticker] = df
                    else:
                        logging.warning(f"No 5Y history found for {ticker}")
                except Exception as e:
                    logging.error(f"Error fetching {ticker}: {e}")

    def get_price_at_date(self, ticker, date):
        """Helper to get price on or before a date."""
        df = self.full_history.get(ticker)
        if df is None: return None
        past_data = df[df.index <= date]
        if past_data.empty: return None
        return past_data.iloc[-1]["Close"]

    def run_simulation(self, custom_holdings=None, current_timeframe_months=12):
        """
        Runs the time-travel simulation for 1Y, 2Y, 3Y, 5Y benchmarks.
        Also includes a benchmark matching the user's current selection.
        Returns a list of results.
        """
        results = []
        
        # Base benchmarks
        timeframes = {
            "1Y": 365,
            "2Y": 365 * 2,
            "3Y": 365 * 3,
            "5Y": 365 * 5
        }
        
        # Add current selection if unique
        selection_label = f"{current_timeframe_months}M" if current_timeframe_months < 12 else f"{current_timeframe_months//12}Y"
        selection_days = int(current_timeframe_months * 30.4)
        if selection_label not in timeframes:
            timeframes[selection_label] = selection_days

        # Sort timeframes by days
        timeframes = dict(sorted(timeframes.items(), key=lambda x: x[1]))
        
        today = datetime.now()
        
        # Load data for custom portfolio assets too
        self._ensure_history_loaded(additional_assets=custom_holdings)

        # Mock Context Managers
        with patch('backend.market_data.fetch_historical_data') as mock_hist_data, \
             patch('backend.portfolio_optimizer.fetch_market_data') as mock_market_data, \
             patch('backend.portfolio_optimizer.analyze_sentiment') as mock_sentiment, \
             patch('backend.portfolio_optimizer.get_sp500_annual_return') as mock_sp500:

            # Setup Common Mocks
            mock_sp500.return_value = 0.08
            mock_sentiment.return_value = {"score": 0.5, "trend": "neutral"}


            # Iterate Timeframes
            for label, days in timeframes.items():
                simulated_date = today - timedelta(days=days)
                
                # Check data availability
                aapl_data = self.full_history.get("AAPL")
                if aapl_data is not None:
                    last_avail = aapl_data.index[-1]
                    if simulated_date > last_avail:
                        simulated_date = last_avail

                # 1. Setup Mock Historical Data Fetching (Time Travel Logic)
                def side_effect_hist_data(ticker, category, period="1y"):
                    if ticker not in self.full_history: return None
                    full_df = self.full_history[ticker]
                    
                    # Determine slice depth
                    days_back = 365
                    if period == "2y": days_back = 730
                    elif period == "5y": days_back = 1825
                    days_back += 100 
                    
                    start_date = simulated_date - timedelta(days=days_back)
                    mask = (full_df.index >= start_date) & (full_df.index <= simulated_date)
                    sliced = full_df.loc[mask].copy()
                    if sliced.empty: return None
                    return sliced

                mock_hist_data.side_effect = side_effect_hist_data

                # 2. Determine Portfolio (Generated vs Custom)
                if custom_holdings:
                    allocation = custom_holdings
                else:
                    months = int(days / 30.4)
                    if months == 0: months = 6
                    try:
                        strategy = generate_strategy(risk_level="high", timeframe=months, initial_amount=10000)
                        if "error" in strategy:
                             results.append({"timeframe": label, "error": strategy["error"]})
                             continue
                        allocation = [{"asset": item["asset"], "category": item["category"], "weight": item["allocation_pct"]} for item in strategy["portfolio_allocation"]]
                    except Exception as e:
                        results.append({"timeframe": label, "error": str(e)})
                        continue

                # 3. Process Assets (Performance & Prediction)
                end_date = simulated_date + timedelta(days=days)
                if end_date > today: end_date = today

                total_start = 0
                total_end = 0
                total_pred_weighted = 0
                total_effective_weight = 0
                all_holdings_status = []
                
                timeframe_factor = days / 365.25

                for item in allocation:
                    asset = item["asset"]
                    category = item.get("category", "Stocks")
                    weight = item["weight"] / 100.0
                    
                    # A. Historical Prediction (What model *would* have thought THEN)
                    hist = side_effect_hist_data(asset, category)
                    prediction_possible = False
                    pred_period = 0
                    
                    if hist is not None and not hist.empty:
                        sig, conf = xgb_model.predict(hist)
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                        
                        if sig == "Buy":
                            pred_annual = volatility * conf
                        elif sig == "Sell":
                            pred_annual = -volatility * conf
                        else:
                            pred_annual = 0.05 * conf
                        
                        pred_period = (1 + pred_annual) ** timeframe_factor - 1
                        prediction_possible = True

                    # B. Actual Performance (What *actually* happened)
                    start_price = self.get_price_at_date(asset, simulated_date)
                    end_price = self.get_price_at_date(asset, end_date)
                    
                    performance_possible = bool(start_price and end_price)
                    
                    if prediction_possible and performance_possible:
                        # Calculation for Actual Return
                        ret = (end_price - start_price) / start_price
                        total_start += weight
                        total_end += weight * (1 + ret)
                        
                        # Calculation for Normalized Predicted Return
                        total_pred_weighted += pred_period * weight
                        total_effective_weight += weight
                        all_holdings_status.append({**item, "skipped": False})
                    else:
                        all_holdings_status.append({**item, "skipped": True})

                if total_effective_weight == 0:
                    results.append({
                        "timeframe": label, 
                        "error": "Historical data/prices missing for this period", 
                        "holdings": all_holdings_status
                    })
                    continue

                actual_return = (total_end - total_start) / total_start
                predicted_return = total_pred_weighted / total_effective_weight
                deviation = abs(predicted_return - actual_return)
                
                # Check for direction match (both positive, both negative, or both effectively zero)
                is_direction_correct = (predicted_return * actual_return) >= 0 or (abs(predicted_return) < 0.02 and abs(actual_return) < 0.02)

                results.append({
                    "timeframe": label,
                    "simulated_date": simulated_date.strftime("%Y-%m-%d"),
                    "predicted_return": round(predicted_return * 100, 2),
                    "actual_return": round(actual_return * 100, 2),
                    "deviation": round(deviation * 100, 2),
                    "is_large_deviation": bool(deviation > 0.5),
                    "is_direction_correct": bool(is_direction_correct),
                    "holdings": all_holdings_status
                })

        return convert_floats(results)

# Singleton instance
verification_service = VerificationService()
