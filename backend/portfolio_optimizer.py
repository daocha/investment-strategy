import numpy as np
import pandas as pd
from scipy.optimize import minimize
from backend.config import (
    RISK_SETTINGS, RSI_THRESHOLD, MACD_THRESHOLD, MAX_NUM_ASSETS, 
    MODEL_PATH, PORTFOLIO_FILE, BINANCE_API_URL, DEFAULT_BACKTEST_PERIOD,
    CRYPTO_ETF_MAPPING
)
from backend.market_data import fetch_market_data, fetch_historical_returns, fetch_historical_data, fetch_yfinance_data
from backend.sentiment_analysis import analyze_sentiment
from backend.technical_analysis import calculate_indicators
from backend.backtesting import run_backtest as backtest_strategy
from backend.price_prediction import CryptoXGBoost
from backend.features import FeaturesPipeline
import logging
import re
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global XGBoost Model Instance
xgb_model = CryptoXGBoost()
# Try to load existing model
if os.path.exists(MODEL_PATH):
    xgb_model.load_model()
else:
    logging.warning(f"⚠️ XGBoost model not found at {MODEL_PATH}. Predictions will default to 'Hold'.")

# Constants now imported from backend.config

def get_sp500_annual_return():
    """Fetches the S&P 500 annual return as a benchmark and converts it to decimal."""
    try:
        sp500 = yf.Ticker("^GSPC")
        history = sp500.history(period="1y")

        if history.empty:
            logging.warning("Could not fetch S&P 500 historical data. Using default threshold (5%).")
            return 0.05  # Default 5% return if unavailable

        # Compute S&P 500 annual return
        annual_return = ((history["Close"].iloc[-1] - history["Close"].iloc[0]) / history["Close"].iloc[0])
        # logging.info(f"✅ S&P 500 Annual Return: {annual_return * 100:.2f}%")
        return round(annual_return, 4)

    except Exception as e:
        logging.error(f"⚠️ Error fetching S&P 500 data: {e}. Using default 5% (0.05).")
        return 0.05  # Default fallback to 5%

def parse_annual_return(annual_return):
    """Converts annual return values to decimal."""
    if isinstance(annual_return, str):
        cleaned_value = re.sub(r"[^\d.]", "", annual_return)
        try:
            return float(cleaned_value) / 100
        except ValueError:
            return None
    elif isinstance(annual_return, (float, int)):
        return float(annual_return)
    return None

def filter_by_risk(asset_list, risk_level):
    """Filters assets based on risk level, asset type, and volatility."""
    allowed_assets = RISK_SETTINGS[risk_level]["allowed_assets"]
    max_volatility = RISK_SETTINGS[risk_level]["max_volatility"]

    filtered_assets = []
    for asset in asset_list:
        asset_volatility = asset.get("volatility", 0)

        if asset["category"] in allowed_assets:
            if max_volatility is None or asset_volatility <= max_volatility:
                filtered_assets.append(asset)
    return filtered_assets

def backtest_portfolio(portfolio_allocation):
    """Runs backtesting for the entire portfolio and calculates overall performance."""
    logging.info("Running portfolio-wide backtesting...")

    total_portfolio_backtest_return = 0
    total_portfolio_predicted_return = 0
    total_portfolio_predicted_profit = 0
    total_portfolio_confidence = 0
    asset_returns = []

    for asset_data in portfolio_allocation:
        asset = asset_data["asset"]
        category = asset_data["category"]
        allocation_pct = asset_data["allocation_pct"] / 100

        current_price = asset_data["current_price"]
        predicted_price = asset_data["predicted_price"]
        predicted_return = asset_data["predicted_return"]
        predicted_profit = asset_data.get("predicted_profit", 0)
        combined_return = asset_data["combined_return"]
        backtest_annual_return = asset_data["backtest_annual_return"]
        confidence = asset_data.get("confidence", 0)

        weighted_predicted_return = predicted_return * allocation_pct
        weighted_backtest_return = backtest_annual_return * allocation_pct
        weighted_confidence = confidence * allocation_pct

        total_portfolio_backtest_return += weighted_backtest_return
        total_portfolio_predicted_return += weighted_predicted_return
        total_portfolio_predicted_profit += predicted_profit
        total_portfolio_confidence += weighted_confidence

        asset_result = asset_data.copy()
        asset_result.update({
            "predicted_return": predicted_return,
            "weighted_predicted_return": weighted_predicted_return,
            "weighted_backtest_return": weighted_backtest_return,
            "backtest_annual_return": backtest_annual_return,
            "combined_return": combined_return,
            "confidence": confidence
        })
        asset_returns.append(asset_result)

    logging.info(f"Final portfolio backtest return: {total_portfolio_backtest_return:.2f}%")

    # DEBUG: Log asset names to verify propagation
    if asset_returns:
        logging.info("--- DEBUG ASSET NAMES START ---")
        for i, asset in enumerate(asset_returns[:3]):
             logging.info(f"Asset: {asset.get('asset')} | Name: {asset.get('asset_name')} | Category: {asset.get('category')}")
        logging.info("--- DEBUG ASSET NAMES END ---")

    return {
        "portfolio_predicted_return": round(total_portfolio_predicted_return, 2),
        "portfolio_backtest_return": round(total_portfolio_backtest_return, 2),
        "portfolio_predicted_profit": round(total_portfolio_predicted_profit, 2),
        "portfolio_confidence": round(total_portfolio_confidence, 2),
        "asset_returns": asset_returns
    }

def convert_floats(obj):
    """Recursively converts NumPy types to native Python types."""
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    return obj

def process_single_asset(asset, category, timeframe, market_data_entry, min_annual_return, ignore_filters=False, precomputed_sentiment=None):
    """
    Process a single asset: Sentiment -> Indicators -> Backtest -> XGBoost Prediction
    Returns asset data dict or None if filtered out.
    """
    try:
        # 0. Fetch Historical Data Centrally (Reuse for Indicators, Backtest, Prediction)
        # Handles Stocks, ETFs, and Crypto with local caching
        hist_data = fetch_historical_data(asset, category, period=DEFAULT_BACKTEST_PERIOD)
        
        if hist_data is None or hist_data.empty:
            logging.warning(f"⚠️ No historical data for {asset}. Skipping.")
            return None

        # 1. Sentiment Analysis (Route Crypto ETFs to underlying assets)
        sentiment_asset = asset
        is_crypto_etf = asset in CRYPTO_ETF_MAPPING
        if is_crypto_etf:
            sentiment_asset = CRYPTO_ETF_MAPPING[asset]
            logging.info(f"🔄 Routing sentiment for {asset} to underlying {sentiment_asset}")
            
        if precomputed_sentiment and sentiment_asset in precomputed_sentiment:
            sentiment = precomputed_sentiment[sentiment_asset]
        else:
            sentiment = analyze_sentiment(sentiment_asset)
            
        sentiment_score = sentiment["score"]
        sentiment_trend = sentiment["trend"]

        if not ignore_filters and sentiment_trend == "negative":
            return None

        # 2. Technical Indicators (Pass pre-fetched data)
        indicators = calculate_indicators(asset, category, data=hist_data)
        if not indicators: 
            return None

        # Filter by RSI/MACD if sentiment is just neutral
        if not ignore_filters and sentiment_trend == "neutral":
            if indicators.get("RSI", 0) <= RSI_THRESHOLD or indicators.get("MACD", 0) <= MACD_THRESHOLD:
                return None

        # 3. Backtesting (Pass pre-fetched data)
        backtest_annual_return = backtest_strategy(asset, category, period=DEFAULT_BACKTEST_PERIOD, data=hist_data)
        
        # Robust check for backtest result
        if backtest_annual_return is None:
            return None

        if backtest_annual_return is None:
            return None

        # 4. Generate All Features (Centralized Pipeline)
        df_features = FeaturesPipeline.generate_feature_set(hist_data)
        
        # XGBoost Prediction
        signal = "Hold"
        confidence = 0.0
        predicted_return = 0.0
        
        if not df_features.empty:
            signal, confidence = xgb_model.predict(hist_data) # predict() internaly calls Pipeline, so it's consistent
            
            # Extract volatility from the market module's output
            latest_row = df_features.iloc[-1]
            volatility = latest_row.get('Volatility', 0.20)
            
            if signal == "Buy":
                predicted_return = volatility * confidence
            elif signal == "Sell":
                predicted_return = -volatility * confidence
            else:
                predicted_return = 0.05 * confidence
        else:
            volatility = 0.20

        # 5. Combined Return Calculation (Scale by timeframe/12 using geometric compounding)
        timeframe_factor = timeframe / 12.0
        
        # Correct geometric scaling: (1 + r)^factor - 1
        predicted_return = (1 + predicted_return) ** timeframe_factor - 1
        backtest_return_for_period = (1 + backtest_annual_return) ** timeframe_factor - 1
        
        # Adjust weight for Crypto/Crypto-ETFs (80% Prediction vs 20% Backtest)
        if category == "Crypto" or is_crypto_etf:
            combined_return = (backtest_return_for_period * 0.2) + (predicted_return * 0.8)
        else:
            combined_return = (backtest_return_for_period * 0.4) + (predicted_return * 0.6)

        if not ignore_filters and combined_return < min_annual_return:
            return None

        current_price = market_data_entry["Close"]
        predicted_price_target = current_price * (1 + predicted_return)

        return {
            "asset": asset,
            "asset_name": market_data_entry.get("Name", asset),
            "category": category,
            "current_price": current_price,
            "combined_return": combined_return,
            "backtest_annual_return": backtest_return_for_period, # Returning period return for display
            "predicted_return": predicted_return,
            "predicted_price": predicted_price_target,
            "volatility": volatility,
            "signal": signal,
            "confidence": confidence
        }

    except Exception as e:
        logging.error(f"Error processing asset {asset}: {e}")
        return None

def generate_strategy(risk_level, timeframe, initial_amount, base_currency="HKD"):
    """
    Builds an optimized investment portfolio using Parallel Processing and XGBoost.
    Values are returned in the specified base_currency.
    """
    from backend.market_data import get_fx_rate
    
    # Currency symbol mapping
    currency_symbols = {"USD": "$", "HKD": "HK$", "TWD": "NT$", "EUR": "€", "GBP": "£"}
    base_symbol = currency_symbols.get(base_currency, base_currency)

    # Map integer risk level to category
    if isinstance(risk_level, (int, float)) or (isinstance(risk_level, str) and risk_level.isdigit()):
        risk_val = int(risk_level)
        if risk_val <= 3:
            risk_level = "low"
        elif risk_val <= 7:
            risk_level = "medium"
        else:
            risk_level = "high"
    elif isinstance(risk_level, str):
        risk_level = risk_level.lower()

    if risk_level not in RISK_SETTINGS:
        logging.warning(f"Invalid risk level '{risk_level}', defaulting to 'medium'")
        risk_level = "medium"

    MINIMUM_ANNUAL_RETURN = get_sp500_annual_return()
    if risk_level == "low":
        MINIMUM_ANNUAL_RETURN *= 0.2 # Low risk only needs to be positive/conservative (~2% threshold)
    elif risk_level == "medium":
        MINIMUM_ANNUAL_RETURN *= 0.8 # Medium risk should aim for benchmark vicinity
        
    logging.info(f"Using MINIMUM_ANNUAL_RETURN threshold: {MINIMUM_ANNUAL_RETURN * 100:.2f}% for {risk_level} risk")

    logging.info("Fetching market data...")
    market_data = fetch_market_data()
    
    # Flatten assets for processing
    assets_to_process = []
    for category, assets in market_data.items():
        for asset, data in assets.items():
            assets_to_process.append((asset, category, data))

    logging.info(f"Processing {len(assets_to_process)} assets in parallel...")
    
    asset_performance = []
    
    # Parallel Execution
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_single_asset, asset, category, timeframe, data, MINIMUM_ANNUAL_RETURN): asset 
            for asset, category, data in assets_to_process
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                asset_performance.append(result)

    logging.info(f"Total assets processed: {len(asset_performance)}")

    if not asset_performance:
        logging.warning("No assets survived performance/XGBoost filtering.")
        return {"error": "No assets match performance criteria"}

    # Apply Risk-Based Filtering
    allowed_assets = RISK_SETTINGS[risk_level]["allowed_assets"]
    max_vol = RISK_SETTINGS[risk_level]["max_volatility"]
    
    filtered_performance = []
    for p in asset_performance:
        if p["category"] in allowed_assets:
            if max_vol is None or p["volatility"] <= max_vol:
                filtered_performance.append(p)
            else:
                logging.info(f"Skipping {p['asset']}: Volatility {p['volatility']} > {max_vol}")
        else:
            logging.info(f"Skipping {p['asset']}: Category {p['category']} not in {allowed_assets}")
            
    asset_performance = filtered_performance
    logging.info(f"Assets matching risk {risk_level}: {len(asset_performance)}")

    # **Diversification Logic**
    # To prevent any single category (like Crypto) from dominating the portfolio, 
    # we enforce a maximum of 50% (5 assets) per category.
    MAX_PER_CATEGORY = 5
    sorted_assets = sorted(asset_performance, key=lambda x: x["combined_return"], reverse=True)
    
    selected_assets = []
    category_counts = {"Stocks": 0, "ETFs": 0, "Crypto": 0}
    
    for p in sorted_assets:
        cat = p["category"]
        if len(selected_assets) < MAX_NUM_ASSETS:
            if category_counts.get(cat, 0) < MAX_PER_CATEGORY:
                selected_assets.append(p)
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
    # Fallback: if we still don't have enough assets, fill with next best regardless of category
    if len(selected_assets) < MAX_NUM_ASSETS:
        for p in sorted_assets:
            if p not in selected_assets and len(selected_assets) < MAX_NUM_ASSETS:
                selected_assets.append(p)

    asset_performance = selected_assets
    logging.info(f"Top selected assets (diversified): {[a['asset'] for a in asset_performance]}")

    if not asset_performance:
         return {"error": "No assets match risk criteria"}

    # **Portfolio Optimization**
    num_assets = len(asset_performance)
    
    # 1. Fetch real historical returns for selected assets (Fetch again but only for top 10)
    tickers_with_categories = [(entry["asset"], entry["category"]) for entry in asset_performance]
    returns_df = fetch_historical_returns(tickers_with_categories)
    
    weights = np.array([max(0.1, a["confidence"]) for a in asset_performance])
    weights = weights / weights.sum() # Initialize with Confidence-Based weights as default

    if not returns_df.empty:
        try:
            # Match returns_df columns with our selected assets
            available_tickers = returns_df.columns.tolist()
            if len(available_tickers) >= 2: # Need at least 2 for cov/optimization
                sub_indices = [i for i, a in enumerate(asset_performance) if a["asset"] in available_tickers]
                sub_returns = returns_df[available_tickers]
                
                cov_matrix = sub_returns.cov().values
                mean_returns = sub_returns.mean().values
                
                # Robust Return Handling: If all returns are negative, adjust for relative optimization
                adj_returns = mean_returns
                if np.all(mean_returns <= 0):
                    logging.info("📉 All mean returns are negative. Adjusting for relative optimization.")
                    adj_returns = mean_returns - np.min(mean_returns) + 0.01 

                num_sub = len(available_tickers)
                def negative_sharpe(w):
                    p_ret = np.sum(adj_returns * w) * 252
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
                    return - (p_ret / p_vol) if p_vol > 0.0001 else 0

                initial_guess = [1 / num_sub] * num_sub
                bounds = [(0.05, 0.40) for _ in range(num_sub)] 
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

                logging.info(f"Running Sharpe Optimization for {num_sub} assets with data...")
                result = minimize(negative_sharpe, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)
                
                if result.success and not np.allclose(result.x, 1/num_sub, atol=0.01):
                    # Map optimized weights back to the full weights array
                    new_weights = weights.copy()
                    # First, calculate how much weight is 'available' after accounting for assets with NO data
                    # For simplicity, let's just overwrite the available ones and re-normalize
                    for idx, weight_val in zip(sub_indices, result.x):
                        new_weights[idx] = weight_val
                    
                    # Normalize to ensure sum is 1 (in case some assets didn't have data)
                    weights = new_weights / new_weights.sum()
                    logging.info(f"✅ Optimization Success. Dynamic Weights: {weights}")
                else:
                    reason = result.message if not result.success else "Result too close to equal weight"
                    logging.warning(f"⚠️ Optimization suboptimal ({reason}). Using Confidence Weighting.")
            else:
                logging.info(f"ℹ️ Insufficient data for MVO ({len(available_tickers)} assets). Using Confidence Weighting.")
        except Exception as e:
            logging.error(f"❌ Optimization error: {e}. Falling back to Confidence Weighting.")
    else:
        logging.warning("⚠️ No historical returns found. Using Confidence Weighting fallback.")

    portfolio_allocation = []
    for i, entry in enumerate(asset_performance):
        # Note: process_single_asset currently doesn't return native_currency, 
        # but we can infer it or fetch it if needed. For now, let's assume it's USD 
        # unless it's a known international ticker. 
        # Better: let's update process_single_asset to include currency.
        
        # Actually, let's just use the ticker suffix heuristic for now to avoid refactoring process_single_asset too much
        # or just fetch it again since it's cached.
        native_currency = "USD"
        if entry["asset"].endswith(".HK"): native_currency = "HKD"
        elif entry["asset"].endswith(".TW"): native_currency = "TWD"
        # ... but we already have fetch_yfinance_data which gets it.
        
        # Let's just fetch the currency from the cache/ticker info
        native_currency = "USD"
        if entry["category"] == "Crypto":
            native_currency = "USD"
        else:
            try:
                temp_stock = yf.Ticker(entry["asset"])
                native_currency = temp_stock.fast_info.get("currency", "USD")
            except Exception:
                # Heuristic fallback if fast_info fails
                if entry["asset"].endswith(".HK"): native_currency = "HKD"
                elif entry["asset"].endswith(".TW"): native_currency = "TWD"
                elif entry["asset"].endswith(".L"): native_currency = "GBP"
        
        fx_rate = get_fx_rate(native_currency, base_currency)
        native_symbol = currency_symbols.get(native_currency, native_currency)

        allocation = round(weights[i] * initial_amount, 2)
        current_price_native = entry["current_price"]
        current_price_converted = round(current_price_native * fx_rate, 2)
        units = round(allocation / current_price_converted, 4) if current_price_converted > 0 else 0

        portfolio_allocation.append({
            "asset": entry["asset"],
            "asset_name": entry.get("asset_name", entry["asset"]),
            "category": entry["category"],
            "units": units,
            "native_currency": native_currency,
            "native_symbol": native_symbol,
            "current_price_native": current_price_native,
            "predicted_price_native": entry["predicted_price"],
            "current_price": current_price_converted,
            "market_value": allocation,
            "combined_return": round(entry["combined_return"] * 100, 4),
            "backtest_annual_return": round(entry["backtest_annual_return"] * 100, 4),
            "weighted_backtest_return": round(entry["backtest_annual_return"] * weights[i] * 100, 4),
            "predicted_return": round(entry["predicted_return"] * 100, 4),
            "predicted_price": round(entry["predicted_price"] * fx_rate, 2),
            "predicted_profit": round(allocation * entry["predicted_return"], 2),
            "signal": entry["signal"],
            "confidence": round(entry["confidence"], 2),
            "allocation": allocation,
            "allocation_pct": round(weights[i] * 100, 2)
        })

    portfolio_backtest_results = backtest_portfolio(portfolio_allocation)

    # **Process Benchmark (S&P 500)**
    benchmark_results = None
    try:
        gspc_data = fetch_market_data()["Stocks"].get("^GSPC") or fetch_yfinance_data("^GSPC")
        if gspc_data:
            benchmark_asset = process_single_asset("^GSPC", "Stocks", timeframe, gspc_data, -10.0, ignore_filters=True)
            if benchmark_asset:
                benchmark_results = {
                    "asset": "S&P 500",
                    "predicted_return": benchmark_asset["predicted_return"] * 100,
                    "backtest_annual_return": benchmark_asset["backtest_annual_return"] * 100
                }
    except Exception as e:
        logging.error(f"Error processing benchmark: {e}")

    # Add benchmark to backtest_results for UI consistency
    if benchmark_results:
        portfolio_backtest_results["benchmark_results"] = benchmark_results
    
    portfolio_backtest_results["total_valuation"] = initial_amount 
    portfolio_backtest_results["base_currency"] = base_currency
    portfolio_backtest_results["base_symbol"] = base_symbol

    return convert_floats({
        "portfolio_allocation": portfolio_allocation,
        "base_currency": base_currency,
        "base_symbol": base_symbol,
        "backtest_results": portfolio_backtest_results
    })
