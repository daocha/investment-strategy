import yfinance as yf
import requests
import pandas as pd
import numpy as np
import logging
import time
from backend.config import CACHE_TTL
from backend.features.technical import calculate_technical_features
from backend.features import FeaturesPipeline
from backend.market_data import (
    MARKET_DATA_CACHE, CACHE_LOCK, save_cache, 
    fetch_stock_etf_timeseries, fetch_crypto_timeseries
)


def calculate_indicators(asset, category, data=None):
    """
    Calculates technical indicators (SMA, RSI, MACD) for the given asset.
    - Stocks & ETFs: Data from Yahoo Finance
    - Crypto: Data from Binance
    - Can optionally accept a pre-fetched DataFrame (`data`) to avoid API calls.
    - Uses caching to avoid redundant calculations.
    """
    current_time = time.time()
    cache_key = f"indicators_{asset}"
    
    # 1. Check Cache
    if data is None: # Only use cache if we are not passing custom/forced data
        with CACHE_LOCK:
            if cache_key in MARKET_DATA_CACHE:
                cached_data, timestamp = MARKET_DATA_CACHE[cache_key]
                if current_time - timestamp < CACHE_TTL:
                    logging.info(f"💾 Using cached indicators for {asset}")
                    return cached_data

    logging.info(f"📊 Calculating indicators for {asset} ({category})...")

    df = None
    if data is not None and not data.empty:
        df = data
    elif category in ["Stocks", "ETFs"]:
        df = fetch_stock_etf_timeseries(asset)
    elif category == "Crypto":
        df = fetch_crypto_timeseries(asset + "USDT")  # Binance uses USDT pairs
    else:
        logging.warning(f"⚠️ Unknown category {category} for {asset}. Skipping technical analysis.")
        return None

    if df is None or df.empty:
        logging.warning(f"⚠️ No data available for {asset}. Skipping technical analysis.")
        return None

    # Use the consolidated features module to calculate indicators
    df = FeaturesPipeline.generate_feature_set(df)

    latest_data = df[["SMA_50", "RSI", "MACD"]].iloc[-1].to_dict()

    # 2. Save to Cache
    with CACHE_LOCK:
        MARKET_DATA_CACHE[cache_key] = (latest_data, current_time)
    save_cache()

    logging.info(f"✅ Calculated Indicators for {asset}: {latest_data}")
    return latest_data

# Example Usage
if __name__ == "__main__":
    assets = [
        ("NVDA", "Stocks"),  # Yahoo Finance
        ("QQQ", "ETFs"),  # Yahoo Finance
        ("BTC", "Crypto"),  # Binance
        ("ETH", "Crypto")  # Binance
    ]

    for asset, category in assets:
        indicators = calculate_indicators(asset, category)
