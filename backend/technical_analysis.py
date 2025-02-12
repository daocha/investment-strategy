import yfinance as yf
import requests
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"  # Binance API endpoint for historical data

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

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol} from Yahoo Finance: {e}")
        return None

def fetch_crypto_data(symbol, interval="1d", limit=180):
    """
    Fetches historical data for cryptocurrencies from Binance.
    Example symbol format: BTCUSDT, ETHUSDT
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(BINANCE_API_URL, params=params)
        data = response.json()

        if "code" in data:  # Binance returns an error code if request fails
            logging.warning(f"⚠️ Binance API Error for {symbol}: {data}")
            return None

        # Binance returns: [timestamp, open, high, low, close, volume, ...]
        df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume",
                                         "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])

        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol} from Binance: {e}")
        return None

def calculate_indicators(asset, category):
    """
    Calculates technical indicators (SMA, RSI, MACD) for the given asset.
    - Stocks & ETFs: Data from Yahoo Finance
    - Cryptocurrencies: Data from Binance
    """
    logging.info(f"📊 Fetching data for {asset} ({category})...")

    if category in ["Stocks", "ETFs"]:
        df = fetch_stock_etf_data(asset)
    elif category == "Cryptocurrencies":
        df = fetch_crypto_data(asset + "USDT")  # Binance uses USDT pairs
    else:
        logging.warning(f"⚠️ Unknown category {category} for {asset}. Skipping technical analysis.")
        return None

    if df is None or df.empty:
        logging.warning(f"⚠️ No data available for {asset}. Skipping technical analysis.")
        return None

    # Calculate SMA (Simple Moving Average)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Calculate RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Calculate MACD (Moving Average Convergence Divergence)
    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema

    latest_data = df[["SMA_50", "RSI", "MACD"]].iloc[-1].to_dict()

    logging.info(f"✅ Calculated Indicators for {asset}: {latest_data}")
    return latest_data

# Example Usage
if __name__ == "__main__":
    assets = [
        ("NVDA", "Stocks"),  # Yahoo Finance
        ("QQQ", "ETFs"),  # Yahoo Finance
        ("BTC", "Cryptocurrencies"),  # Binance
        ("ETH", "Cryptocurrencies")  # Binance
    ]

    for asset, category in assets:
        indicators = calculate_indicators(asset, category)
        print(f"{asset} ({category}): {indicators}")
