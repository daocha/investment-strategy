import yfinance as yf
import requests
import pandas as pd
import numpy as np
import logging
from backend.config import BINANCE_API_URL
from backend.features.technical import calculate_technical_features

def fetch_stock_etf_data(symbol, period="6mo"):
    """
    Fetches historical data for stocks and ETFs from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            logging.warning(f"⚠️ No historical data found for {symbol}. It may be delisted.")
            return None

        # Force UTC then strip TZ to handle any mixed TZs reliably
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol} from Yahoo Finance: {e}")
        return None

def fetch_crypto_data(symbol, interval="1d", limit=180):
    """
    Fetches historical data for cryptocurrencies from Binance.
    Supports larger limits (>1000) by paginating calls.
    """
    all_data = []
    current_limit = limit
    end_time = None

    try:
        while current_limit > 0:
            fetch_count = min(current_limit, 1000)
            params = {"symbol": symbol, "interval": interval, "limit": fetch_count}
            if end_time:
                params["endTime"] = end_time - 1
            
            response = requests.get(BINANCE_API_URL, params=params)
            data = response.json()

            if not data or "code" in data or len(data) == 0:
                break
            
            # Binance returns rows ordered by time (oldest first in the result list)
            # To paginate BACKWARDS, we prepend new data to our list and use the first timestamp as our new endTime
            all_data = data + all_data
            end_time = data[0][0] # The timestamp of the oldest record in this batch
            
            current_limit -= len(data)
            if len(data) < fetch_count:
                break # No more data available
        
        if not all_data:
            return None

        # Binance returns: [timestamp, open, high, low, close, volume, ...]
        df = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume",
                                         "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])

        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='ms')
        df.set_index("Timestamp", inplace=True)

        # Remove duplicates if any (due to overlapping boundaries)
        df = df[~df.index.duplicated(keep='first')]
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol} from Binance: {e}")
        return None

def calculate_indicators(asset, category, data=None):
    """
    Calculates technical indicators (SMA, RSI, MACD) for the given asset.
    - Stocks & ETFs: Data from Yahoo Finance
    - Crypto: Data from Binance
    - Can optionally accept a pre-fetched DataFrame (`data`) to avoid API calls.
    """
    logging.info(f"📊 Calculating indicators for {asset} ({category})...")

    df = None
    if data is not None and not data.empty:
        df = data
    elif category in ["Stocks", "ETFs"]:
        df = fetch_stock_etf_data(asset)
    elif category == "Crypto":
        df = fetch_crypto_data(asset + "USDT")  # Binance uses USDT pairs
    else:
        logging.warning(f"⚠️ Unknown category {category} for {asset}. Skipping technical analysis.")
        return None

    if df is None or df.empty:
        logging.warning(f"⚠️ No data available for {asset}. Skipping technical analysis.")
        return None

    # Use the consolidated features module to calculate indicators
    from backend.features import FeaturesPipeline
    df = FeaturesPipeline.generate_feature_set(df)

    latest_data = df[["SMA_50", "RSI", "MACD"]].iloc[-1].to_dict()

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
