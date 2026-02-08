import requests
import re
import logging
import yfinance as yf
import time
import json # Added json
import os
import numpy as np # Added numpy
import pandas as pd # Added pandas
from dotenv import load_dotenv

import threading # Added threading for lock

load_dotenv()

from backend.config import ASSET_LIST, CACHE_FILE, CACHE_TTL, USE_DEEPSEEK_API, BINANCE_API_URL, DEEPSEEK_API_URL, DEFAULT_BACKTEST_PERIOD

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

MARKET_DATA_CACHE = {}
CACHE_LOCK = threading.Lock() # Added lock for thread-safety

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder for NumPy and Pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'): # Handle pd.Timestamp and datetime
            return obj.isoformat()
        return super().default(obj)

def load_cache():
    """Loads market data cache from local disk."""
    global MARKET_DATA_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                MARKET_DATA_CACHE = json.load(f)
            logging.info(f"📂 Loaded market data cache from {CACHE_FILE}")
        except Exception as e:
            logging.error(f"⚠️ Failed to load cache: {e}")
            MARKET_DATA_CACHE = {}

def save_cache():
    """Saves market data cache to local disk."""
    try:
        with CACHE_LOCK:
            # Create a copy to avoid "changed size during iteration" if json.dump
            # somehow triggers threading issues (though lock should handle it)
            cache_copy = MARKET_DATA_CACHE.copy()
            
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_copy, f, cls=CustomJSONEncoder)
        # logging.info("💾 Saved market data cache to disk.")
    except Exception as e:
        logging.error(f"⚠️ Failed to save cache: {e}")

# Load cache on module import
load_cache()

def get_trending_assets():
    """Uses DeepSeek AI to find trending assets, with a fallback to configured list, and caching."""
    # If API is disabled, always use fresh config from config.py
    if not USE_DEEPSEEK_API:
        logging.info("DeepSeek API disabled. Using fresh ASSET_LIST from config.")
        # Flatten ASSET_LIST if it contains nested dictionaries (e.g. Stocks -> Subsectors -> Tickers)
        flat_asset_list = {}
        for category, content in ASSET_LIST.items():
            if isinstance(content, dict):
                # Flatten subcategories
                flat_list = []
                for subcat, tickers in content.items():
                    flat_list.extend(tickers)
                # MUST return a dict of {subcategory: [tickers]} for fetch_market_data compatibility
                flat_asset_list[category] = {"Default": flat_list}
            else:
                flat_asset_list[category] = {"Default": content}
        return flat_asset_list

    current_time = time.time()
    cache_key = "trending_assets"
    
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < 3600: # Cache trending assets for 1 hour
            logging.info("Using cached trending assets list.")
            return data

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "I need a list of the most promising investment assets across stocks, ETFs, and cryptocurrencies.  Only include assets with strong BUY indicators based on current market sentiment, technical strength, capital inflows, and analyst ratings. Ensure the list is diverse (technology, financials, consumer, crypto, etc.), and large enough for proper portfolio selection (~100-200 assets).  For the technology & crypto sector i am expecting at least 50 options respectively. Format the response as a comma-separated list of tickers."
                )
            }
        ]
    }

    if USE_DEEPSEEK_API:
        try:
            logging.info("Calling DeepSeek API for trending assets...")
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            assets_response = response.json()["choices"][0]["message"]["content"]
            assets_json = extract_tickers(assets_response)
            
            # SUCCESS: Save AI result to cache
            with CACHE_LOCK:
                MARKET_DATA_CACHE[cache_key] = (assets_json, current_time)
            save_cache()
        except Exception as e:
            logging.error(f"❌ DeepSeek API failed: {e}. Using configured ASSET_LIST.")
            assets_json = ASSET_LIST
    else:
        logging.info("DeepSeek API disabled. Using configured ASSET_LIST.")
        assets_json = ASSET_LIST

    return assets_json

def extract_tickers(response_text):
    """Parses DeepSeek AI response and extracts structured asset categories with tickers."""
    sections = ["Stocks", "ETFs", "Crypto"]
    asset_data = {"Stocks": {}, "ETFs": {}, "Crypto": {}}

    current_section = None

    # Process each line
    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Identify section headers
        if line in sections:
            current_section = line
            continue

        # Extract category and tickers
        match = re.match(r"(.+?):\s*([\w,\s]+)", line)
        if match and current_section:
            category = match.group(1).strip()
            tickers = [ticker.strip() for ticker in match.group(2).split(",")]
            asset_data[current_section][category] = tickers

    return asset_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_fx_rate(from_currency, to_currency="USD"):
    """Fetches the current FX rate from Yahoo Finance."""
    if from_currency == to_currency:
        return 1.0
    
    pair = f"{from_currency}{to_currency}=X"
    cache_key = f"fx_{pair}"
    current_time = time.time()
    
    if cache_key in MARKET_DATA_CACHE:
        rate, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < 3600: # Cache FX for 1 hour
            return rate

    try:
        ticker = yf.Ticker(pair)
        data = ticker.history(period="1d")
        if not data.empty:
            rate = float(data["Close"].iloc[-1])
            with CACHE_LOCK:
                MARKET_DATA_CACHE[cache_key] = (rate, current_time)
            save_cache()
            return rate
    except Exception as e:
        logging.error(f"Error fetching FX rate for {pair}: {e}")
    
    return 1.0 # Default to 1.0 if fetch fails

def parse_period_to_days(period):
    """
    Dynamically converts a period string (e.g., '1y', '6m', '30d') to an approximate number of days.
    """
    if not period or not isinstance(period, str):
        return 365
    
    amount = int(''.join(filter(str.isdigit, period)) or 1)
    unit = period[-1].lower()
    
    if unit == 'y':
        return amount * 365
    elif unit == 'm':
        return amount * 30
    elif unit == 'd':
        return amount
    return 365

def get_historical_fx_rate(from_currency, to_currency="USD", date=None):
    """
    Fetches the FX rate from a specific historical date.
    If date is None, returns current rate.
    """
    if from_currency == to_currency:
        return 1.0
    
    pair = f"{from_currency}{to_currency}=X"
    try:
        ticker = yf.Ticker(pair)
        # Fetch a small window around the date to ensure we get a valid trading day
        if date:
            start_date = pd.to_datetime(date)
            # Use 5 days window to handle weekends/holidays
            end_date = start_date + pd.Timedelta(days=5)
            data = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if not data.empty:
                return float(data["Close"].iloc[0])
        else:
            return get_fx_rate(from_currency, to_currency)
    except Exception as e:
        logging.error(f"Error fetching historical FX rate for {pair} on {date}: {e}")
    
    return 1.0

def fetch_yfinance_data(ticker, current_time=None):
    if current_time is None:
        current_time = time.time()
    cache_key = f"yfinance_{ticker}"
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < CACHE_TTL:
            return data

    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            try:
                info = stock.info
            except Exception:
                info = {}
            
            if info is None:
                info = {}

            # Safe currency extraction
            currency = "USD"
            try:
                currency = stock.fast_info.get("currency", "USD")
            except Exception:
                # If fast_info fails, try simple info or heuristic
                currency = info.get("currency", "USD")

            data = {
                "Name": info.get("longName") or info.get("shortName") or ticker,
                "Close": history["Close"].iloc[-1],
                "Volume": history["Volume"].iloc[-1],
                "Market Cap": info.get("marketCap", "N/A"),
                "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
                "Currency": currency
            }
            with CACHE_LOCK:
                MARKET_DATA_CACHE[cache_key] = (data, current_time)
            save_cache()
            return data
    except Exception as e:
        logging.error(f"Error fetching {ticker}: {e}")
    return None

def fetch_crypto_data(ticker, current_time=None):
    if current_time is None:
        current_time = time.time()
    # Ensure ticker is clean (remove -USD if present for Binance)
    clean_ticker = ticker.replace("-USD", "")
    cache_key = f"binance_{clean_ticker}"
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < CACHE_TTL:
            return data

    try:
        # logging.info(f"[Binance] Fetching market data for {clean_ticker}...")
        response = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={clean_ticker}USDT").json()
        if "lastPrice" in response:
            name_mapping = {
                "BTC": "Bitcoin",
                "ETH": "Ethereum",
                "SOL": "Solana",
                "BNB": "Binance Coin",
                "XRP": "Ripple",
                "ADA": "Cardano",
                "AVAX": "Avalanche",
                "DOT": "Polkadot",
                "LINK": "Chainlink",
                "MATIC": "Polygon",
                "LTC": "Litecoin",
                "BCH": "Bitcoin Cash",
                "AAVE": "Aave",
                "UNI": "Uniswap"
            }
            data = {
                "Name": name_mapping.get(clean_ticker, f"{clean_ticker} Crypto"),
                "Close": float(response["lastPrice"]),
                "Volume": float(response["quoteVolume"]),
                "24h Change": float(response["priceChangePercent"]),
                "Market Cap": "N/A"
            }
            with CACHE_LOCK:
                MARKET_DATA_CACHE[cache_key] = (data, current_time)
            save_cache()
            return data
    except Exception as e:
        logging.error(f"Error fetching {clean_ticker}: {e}")
    return None

def fetch_market_data():
    """
    Fetches real-time market data for trending stocks, ETFs, and cryptos.
    Uses in-memory caching to avoid hitting APIs too frequently.
    """
    trending_assets = get_trending_assets()
    market_data = {"Stocks": {}, "ETFs": {}, "Crypto": {}}
    current_time = time.time()

    # Fetch Stocks & ETFs data
    for category, tickers in trending_assets["Stocks"].items():
        logging.info(f"Checking Stocks in category: {category}...")
        for ticker in tickers:
            logging.info(f"   - Fetching {ticker} (Stocks)...")
            data = fetch_yfinance_data(ticker, current_time)
            if data:
                market_data["Stocks"][ticker] = data

    for category, tickers in trending_assets["ETFs"].items():
        logging.info(f"Checking ETFs in category: {category}...")
        for ticker in tickers:
            logging.info(f"   - Fetching {ticker} (ETFs)...")
            data = fetch_yfinance_data(ticker, current_time)
            if data:
                market_data["ETFs"][ticker] = data

    # Fetch Crypto data
    for category, tickers in trending_assets["Crypto"].items():
        logging.info(f"Checking Crypto in category: {category}...")
        for ticker in tickers:
            logging.info(f"   - Fetching {ticker} (Crypto)...")
            data = fetch_crypto_data(ticker, current_time)
            if data:
                market_data["Crypto"][ticker] = data

    logging.info("Finished fetching all market data.")
    return market_data

def fetch_historical_data(ticker, category, period=None):
    """
    Fetches historical price data. 
    period can be "1y", "2y", "5y", etc. Defaults to DEFAULT_BACKTEST_PERIOD from config.
    """
    if period is None:
        period = DEFAULT_BACKTEST_PERIOD
        
    cache_key = f"hist_{ticker}_{period}"
    current_time = time.time()
    
    # 1. Try Cache First
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < 86400: # Cache historical data for 24 hours
             # Check if data is in 'split' format (dict with 'index', 'columns', 'data')
             if isinstance(data, dict) and "columns" in data:
                 df_cached = pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
                 df_cached.index = pd.to_datetime(df_cached.index, utc=True).tz_localize(None)
                 return df_cached
             else:
                 # Legacy format handling
                 df_cached = pd.DataFrame(data)
                 if not df_cached.empty:
                     df_cached.index = pd.to_datetime(df_cached.index, utc=True).tz_localize(None)
                 return df_cached

    logging.info(f"📅 Fetching historical data for {ticker} ({category}, {period})...")
    df = None
    try:
        if category in ["Stocks", "ETFs"]:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is not None and not df.empty:
                # Force UTC then strip TZ to handle any mixed TZs reliably
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        elif category == "Crypto":
            # Avoid circular import by importing inside function if needed, 
            # or just use the logic from technical_analysis
            from backend.technical_analysis import fetch_crypto_data as fetch_binance_hist
            # Calculate limit dynamically from period
            limit = parse_period_to_days(period)
            df = fetch_binance_hist(ticker + "USDT", limit=limit)
        
        if df is not None and not df.empty:
            # Convert to split format to avoid Timestamp-as-key JSON errors
            with CACHE_LOCK:
                MARKET_DATA_CACHE[cache_key] = (df.to_dict(orient='split'), current_time)
            save_cache()
            return df
    except Exception as e:
        logging.error(f"Error fetching historical data for {ticker}: {e}")
    
    logging.warning(f"❌ fetch_historical_data failed for {ticker} ({category}) - returning None")
    return None

def fetch_historical_returns(tickers_with_categories, period="1y"):
    """
    Fetches historical returns for a list of tickers.
    Uses fetch_historical_data for centralized caching.
    """
    returns_df = pd.DataFrame()
    
    for ticker, category in tickers_with_categories:
        df = fetch_historical_data(ticker, category, period=period)
        if df is not None and not df.empty:
            # Force UTC then strip TZ to handle any mixed TZs reliably
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            returns_df[ticker] = df["Close"].pct_change()
            
    return returns_df.dropna()
