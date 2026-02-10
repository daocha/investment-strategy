import pandas as pd
import numpy as np
import logging
import time
from backend.config import CACHE_TTL, PRIMARY_CACHE_PERIOD
from backend.market_data import MARKET_DATA_CACHE, CACHE_LOCK, save_cache

def run_backtest(asset, category=None, period=None, data=None):
    """
    Performs a simplified backtest for an asset.
    If 'data' is provided (DataFrame), it uses it to avoid API calls.
    Functions:
    Returns the annualized return from the first to the last data point.
    Uses caching to avoid redundant calculations.
    """
    current_time = time.time()
    cache_key = f"backtest_{asset}"
    
    # 1. Check Cache
    with CACHE_LOCK:
        if cache_key in MARKET_DATA_CACHE:
            cached_result, backtest_ts = MARKET_DATA_CACHE[cache_key]
            
            # Check price history timestamp (dependency)
            hist_key = f"hist_{asset}_{PRIMARY_CACHE_PERIOD}"
            if hist_key in MARKET_DATA_CACHE:
                _, hist_ts = MARKET_DATA_CACHE[hist_key]
                # If backtest is older than latest price data, force recalculate
                if backtest_ts < hist_ts:
                    logging.info(f"🔄 Price history for {asset} updated. Recalculating backtest...")
                elif current_time - backtest_ts < CACHE_TTL:
                    logging.info(f"💾 Using cached backtest for {asset}")
                    return cached_result
            elif current_time - backtest_ts < CACHE_TTL:
                # No hist key in cache, fallback to standard TTL
                logging.info(f"💾 Using cached backtest for {asset} (no hist dependency found)")
                return cached_result

    try:
        if data is None or data.empty:
            logging.warning(f"⚠️ No data provided for {asset} backtest.")
            return 0.0

        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Simplified Annual Return Logic
        start_price = float(data['Close'].iloc[0])
        end_price = float(data['Close'].iloc[-1])
        
        # Calculate days in the dataset
        days = (data.index[-1] - data.index[0]).days
        if days <= 0:
            return 0.0
            
        total_return = (end_price - start_price) / start_price
        
        # Annualize the return: (1 + total_return) ^ (365 / days) - 1
        annualized_return = (1 + total_return) ** (365.0 / max(days, 1)) - 1
        
        # 2. Save to Cache
        with CACHE_LOCK:
            MARKET_DATA_CACHE[cache_key] = (annualized_return, current_time)
        save_cache()

        return annualized_return

    except Exception as e:
        logging.error(f"❌ Error in backtesting {asset}: {e}")
        return 0.0
