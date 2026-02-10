import os
import sys
import pandas as pd
import logging
from io import StringIO
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from backend.config import PORTFOLIO_FILE, MIN_RETURN_THRESHOLD, CURRENCY_SYMBOLS, DEFAULT_BACKTEST_PERIOD, CRYPTO_ETF_MAPPING
from backend.portfolio_optimizer import process_single_asset, convert_floats
from backend.market_data import fetch_stock_etf_snapshot, fetch_crypto_snapshot, CustomJSONEncoder, get_fx_rate, fetch_historical_data, get_historical_fx_rate
from backend.sentiment_analysis import analyze_sentiment_batch

def run_portfolio_analysis(df_holdings, base_currency="HKD", timeframe=6):
    """
    Core logic to analyze a given portfolio (DataFrame with Ticker, Category, Units).
    Returns a dictionary with valuation and performance metrics converted to base_currency.
    """
    total_value_now = 0
    total_value_then = 0
    analyzed_assets = []
    
    base_symbol = CURRENCY_SYMBOLS.get(base_currency, base_currency)
    
    # 0. Pre-fetch Batch Sentiment for all assets
    tickers = df_holdings["Ticker"].unique().tolist()
    # Also include the underlying for crypto ETFs to ensure we have them in the batch
    sentiment_tickers = set()
    for t in tickers:
        t_clean = str(t).strip()
        sentiment_tickers.add(t_clean)
        if t_clean in CRYPTO_ETF_MAPPING:
            sentiment_tickers.add(CRYPTO_ETF_MAPPING[t_clean])
    
    portfolio_sentiment = analyze_sentiment_batch(list(sentiment_tickers))
    
    # Use a mock min_return threshold to ensure all holdings are analyzed
    # 1. Parallel Fetching & Processing
    logging.info(f"🚀 Analyzing {len(df_holdings)} portfolio assets in parallel...")
    t_start = time.time()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_asset = {}
        
        for _, row in df_holdings.iterrows():
            asset = str(row["Ticker"]).strip()
            category = str(row["Category"]).strip()
            units = float(row["Units"])
            
            # Submit each asset analysis as a parallel task
            future = executor.submit(
                analyze_single_holding, 
                asset, category, units, timeframe, base_currency, portfolio_sentiment
            )
            future_to_asset[future] = asset

        for future in as_completed(future_to_asset):
            try:
                result_data = future.result()
                if result_data:
                    analyzed_assets.append(result_data)
                    total_value_now += result_data["market_value"]
                    total_value_then += result_data["market_value_then"] # We need to return this from helper
            except Exception as e:
                logging.error(f"❌ Error analyzing {future_to_asset[future]}: {e}")

    logging.info(f"Analysis loop took {time.time() - t_start:.2f}s")

    if not analyzed_assets or total_value_now == 0:
        return {"error": "No valid assets analyzed."}

    # 4. Final Portfolio Metrics Calculation
    portfolio_predicted_profit = 0
    portfolio_allocation = []
    
    for asset_data in analyzed_assets:
        weight = asset_data["market_value"] / total_value_now
        asset_data["allocation_pct"] = round(weight * 100, 2)
        
        # Predicted Profit in base currency (market_value * predicted_return_decimal)
        # Note: predicted_return is already scaled to the timeframe in process_single_asset
        asset_data["predicted_profit"] = asset_data["market_value"] * asset_data["predicted_return"]
        portfolio_predicted_profit += asset_data["predicted_profit"]
        
        # Weighted Backtest Return for individual display
        # backtest_annual_return is currently a decimal (e.g. 0.26)
        asset_data["weighted_backtest_return"] = asset_data["backtest_annual_return"] * weight * 100
        
        # Scaling for UI display (%)
        asset_data["predicted_return"] = asset_data["predicted_return"] * 100
        asset_data["backtest_annual_return"] = asset_data["backtest_annual_return"] * 100
        asset_data["combined_return"] = asset_data["combined_return"] * 100
        
        portfolio_allocation.append(asset_data)

    # Aggregate Portfolio Backtest Return (Geometric/Actual)
    if total_value_then > 0:
        portfolio_backtest_return_period = (total_value_now / total_value_then) - 1
        portfolio_backtest_ret_ui = portfolio_backtest_return_period * 100
    else:
        portfolio_backtest_ret_ui = 0

    # Aggregate Portfolio Predicted Return (Weighted Average of period returns)
    portfolio_predicted_return_ui = sum(a["predicted_return"] * (a["market_value"] / total_value_now) for a in portfolio_allocation)
    
    # Portfolio Confidence (Weighted Average)
    portfolio_confidence = sum(a["confidence"] * (a["market_value"] / total_value_now) for a in portfolio_allocation)

    # 5. Process Benchmark (S&P 500)
    benchmark_results = None
    try:
        gspc_data = fetch_stock_etf_snapshot("^GSPC")
        if gspc_data:
            benchmark_asset = process_single_asset("^GSPC", "Stocks", timeframe, gspc_data, MIN_RETURN_THRESHOLD, ignore_filters=True)
            if benchmark_asset:
                benchmark_results = {
                    "asset": "S&P 500",
                    "predicted_return": benchmark_asset["predicted_return"] * 100,
                    "backtest_annual_return": benchmark_asset["backtest_annual_return"] * 100
                }
    except Exception as e:
        logging.error(f"Error processing benchmark: {e}")

    return convert_floats({
        "portfolio_allocation": portfolio_allocation,
        "base_currency": base_currency,
        "base_symbol": base_symbol,
        "backtest_results": {
            "portfolio_predicted_return": portfolio_predicted_return_ui,
            "portfolio_backtest_return": portfolio_backtest_ret_ui,
            "portfolio_predicted_profit": portfolio_predicted_profit,
            "total_valuation": total_value_now,
            "portfolio_confidence": portfolio_confidence,
            "asset_returns": portfolio_allocation,
            "benchmark_results": benchmark_results,
            "base_currency": base_currency,
            "base_symbol": base_symbol
        }
    })

def analyze_single_holding(asset, category, units, timeframe, base_currency, portfolio_sentiment):
    """
    Helper function to process a single holding in a separate thread.
    Returns the enriched result dict or None.
    """
    try:
        # 1. Fetch Current Data
        data = None
        if category in ["Stocks", "ETFs"]:
            data = fetch_stock_etf_snapshot(asset)
        elif category == "Crypto":
            data = fetch_crypto_snapshot(asset)
        
        if not data:
            logging.warning(f"⚠️ No current data for {asset}. Skipping.")
            return None
            
        native_price = float(data["Close"])
        native_currency = data.get("Currency", "USD") if category != "Crypto" else "USD"
        native_symbol = CURRENCY_SYMBOLS.get(native_currency, native_currency)
        
        # Fetch Current FX rate to base_currency
        fx_rate = get_fx_rate(native_currency, base_currency)
        
        # 2. Process Analytics (Pass precomputed sentiment)
        result = process_single_asset(
            asset, category, timeframe, data, MIN_RETURN_THRESHOLD, 
            ignore_filters=True, 
            precomputed_sentiment=portfolio_sentiment
        )
        
        if not result:
            logging.warning(f"⚠️ Failed to process analytics for {asset}. Skipping.")
            return None

        base_symbol = CURRENCY_SYMBOLS.get(base_currency, base_currency)

        # 3. Calculate Historical Performance (Value Then)
        # Find the price approximately 'timeframe' months ago
        hist_df = fetch_historical_data(asset, category, period=DEFAULT_BACKTEST_PERIOD)
        price_then = native_price
        fx_rate_then = fx_rate
        
        if hist_df is not None and not hist_df.empty:
            # Target date: now - timeframe months
            target_date = pd.Timestamp.now() - pd.DateOffset(months=timeframe)
            closest_idx = hist_df.index.get_indexer([target_date], method='nearest')[0]
            if closest_idx != -1:
                price_then = float(hist_df["Close"].iloc[closest_idx])
                # Get historical FX rate for that date
                fx_rate_then = get_historical_fx_rate(native_currency, base_currency, date=hist_df.index[closest_idx])

        # Valuation
        market_value_now = units * native_price * fx_rate
        market_value_then = units * price_then * fx_rate_then
        
        # Asset Results Enhancement
        result.update({
            "units": units,
            "market_value": market_value_now,
            "allocation": market_value_now, # Synonym for compatibility
            "native_currency": native_currency,
            "native_symbol": native_symbol,
            "base_currency": base_currency,
            "base_symbol": base_symbol,
            "current_price_native": native_price,
            "predicted_price_native": result["predicted_price"], # Was calculated in native
            "predicted_price": result["predicted_price"] * fx_rate, # Convert to base
            "market_value_then": market_value_then
        })
        
        return result

    except Exception as e:
        logging.error(f"❌ Error inside thread for {asset}: {e}")
        return None

if __name__ == "__main__":
    # Configure logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    if not os.path.exists(PORTFOLIO_FILE):
        logging.error(f"❌ Portfolio file not found at {PORTFOLIO_FILE}")
        sys.exit(1) # Exit if portfolio file is not found

    df_portfolio = pd.read_csv(PORTFOLIO_FILE)
    
    analysis = run_portfolio_analysis(df_portfolio, base_currency="HKD")
    
    # Save to JSON for frontend
    output_file = os.path.join(os.path.dirname(__file__), "my_portfolio_analysis.json")
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=4, cls=CustomJSONEncoder)
        
    logging.info(f"✅ Analysis Complete. Saved to {output_file}")
