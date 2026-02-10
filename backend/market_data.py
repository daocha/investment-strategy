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
import redis

import threading # Added threading for lock

load_dotenv()

from backend.config import (
    ASSET_LIST, CACHE_FILE, CACHE_TTL, USE_DEEPSEEK_API, 
    BINANCE_KLINES_URL, BINANCE_SNAPSHOT_URL, DEEPSEEK_API_URL, DEFAULT_BACKTEST_PERIOD,
    PRIMARY_CACHE_PERIOD, REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class MarketDataCache:
    """A proxy dictionary that interfaces with Redis or a local dict for granular caching."""
    def __init__(self):
        self._local_cache = {}
        self._redis_client = None
        self._prefix = "market_data:"

    def set_redis_client(self, client):
        self._redis_client = client

    def _get_redis_key(self, key):
        return f"{self._prefix}{key}"

    def __getitem__(self, key):
        if self._redis_client:
            try:
                # Use HGETALL to get all fields in the hash
                val_hash = self._redis_client.hgetall(self._get_redis_key(key))
                if val_hash:
                    data = json.loads(val_hash.get("data"))
                    timestamp = float(val_hash.get("timestamp", 0))
                    return (data, timestamp)
                raise KeyError(key)
            except Exception as e:
                logging.error(f"❌ Redis error in __getitem__: {e}")
        return self._local_cache[key]

    def __setitem__(self, key, value):
        if self._redis_client:
            try:
                # Expecting value to be (data, timestamp)
                data, timestamp = value
                redis_key = self._get_redis_key(key)
                
                # Store as hash fields
                self._redis_client.hset(redis_key, mapping={
                    "data": json.dumps(data, cls=CustomJSONEncoder),
                    "timestamp": str(timestamp)
                })
                # Set TTL
                self._redis_client.expire(redis_key, CACHE_TTL)
                return
            except Exception as e:
                logging.error(f"❌ Redis error in __setitem__: {e}")
        self._local_cache[key] = value

    def __contains__(self, key):
        if self._redis_client:
            try:
                return self._redis_client.exists(self._get_redis_key(key)) > 0
            except Exception as e:
                logging.error(f"❌ Redis error in __contains__: {e}")
        return key in self._local_cache

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key, default=None):
        if self._redis_client:
            try:
                val = self.get(key)
                self._redis_client.delete(self._get_redis_key(key))
                return val
            except Exception as e:
                logging.error(f"❌ Redis error in pop: {e}")
        return self._local_cache.pop(key, default)

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def copy(self):
        """Returns a snapshot of the cache as a dict. Use sparingly with Redis."""
        if self._redis_client:
            try:
                keys = []
                cursor = '0'
                while cursor != 0:
                    cursor, batch = self._redis_client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
                    keys.extend(batch)
                
                if not keys:
                    return {}

                result = {}
                for k in keys:
                    v_hash = self._redis_client.hgetall(k)
                    if v_hash:
                        internal_key = k.replace(self._prefix, "", 1)
                        data = json.loads(v_hash.get("data"))
                        timestamp = float(v_hash.get("timestamp", 0))
                        result[internal_key] = (data, timestamp)
                return result
            except Exception as e:
                logging.error(f"❌ Redis error in copy: {e}")
        return self._local_cache.copy()

    def __len__(self):
        """Returns the number of keys. Note: For Redis, this only returns the local fallback count to avoid expensive scans."""
        return len(self._local_cache)

MARKET_DATA_CACHE = MarketDataCache()
CACHE_LOCK = threading.Lock()
API_LOCK = threading.RLock() # Serializes external API calls. RLock prevents deadlocks if nested calls occur.
LAST_SAVE_TIME = 0
SAVE_INTERVAL = 30 # Only save to disk every 30 seconds to avoid bottleneck

# Initialize Redis Client if configured
redis_client = None
if REDIS_HOST:
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        # Test connection
        redis_client.ping()
        MARKET_DATA_CACHE.set_redis_client(redis_client)
        logging.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logging.warning(f"⚠️ Redis configured but failed to connect: {e}. Falling back to file cache.")
        redis_client = None

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
    """Loads market data cache from Redis or local disk."""
    with CACHE_LOCK:
        if redis_client:
            # Check if Redis has any data
            try:
                # Use SCAN to check for existence of granular keys
                cursor, keys = redis_client.scan(cursor=0, match="market_data:*", count=1)
                if keys:
                    logging.debug("📂 Redis granular cache active")
                    return # Already has data, lazy loading will handle the rest
                logging.info("ℹ️ Redis cache is empty.")
            except Exception as e:
                logging.error(f"⚠️ Failed to check Redis status: {e}")

        # Fallback/Seeding logic: load file and push to Redis if empty
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    file_data = json.load(f)
                
                if redis_client:
                    logging.info(f"🚀 Seeding Redis cache from {CACHE_FILE}...")
                    # Bulk set using pipeline to be efficient
                    pipe = redis_client.pipeline()
                    for k, v in file_data.items():
                        # Expecting file data to be tuple/list of (data, timestamp)
                        try:
                            data, timestamp = v
                            redis_key = f"market_data:{k}"
                            pipe.hset(redis_key, mapping={
                                "data": json.dumps(data, cls=CustomJSONEncoder),
                                "timestamp": str(timestamp)
                            })
                            pipe.expire(redis_key, CACHE_TTL)
                        except Exception as e:
                            logging.warning(f"⚠️ Skipping invalid cache entry {k}: {e}")
                    pipe.execute()
                    logging.info(f"✅ Seeding complete ({len(file_data)} keys)")
                else:
                    MARKET_DATA_CACHE._local_cache = file_data
                    logging.info(f"📂 Loaded market data cache from {CACHE_FILE}")
            except Exception as e:
                logging.error(f"⚠️ Failed to load/seed file cache: {e}")
        else:
            if not redis_client:
                logging.info("ℹ️ No cache file found and Redis not configured. Starting fresh.")

def save_cache(force=False):
    """Saves market data cache to local disk using an atomic write with throttling."""
    global LAST_SAVE_TIME
    current_time = time.time()
    
    if not force and (current_time - LAST_SAVE_TIME < SAVE_INTERVAL):
        return

    try:
        with CACHE_LOCK:
            if redis_client:
                # Redis writes are handled granularly via MarketDataCache.__setitem__
                # Just update the save time to prevent redundant logs/checks
                LAST_SAVE_TIME = current_time
                if force:
                    logging.info("💾 Redis granular storage active")
                return

            # Fallback to File Save (only if Redis is not active)
            cache_copy = MARKET_DATA_CACHE._local_cache.copy()
            tmp_file = f"{CACHE_FILE}.tmp"
            with open(tmp_file, "w") as f:
                json.dump(cache_copy, f, cls=CustomJSONEncoder)
            
            # Atomic overwrite
            os.replace(tmp_file, CACHE_FILE)
            LAST_SAVE_TIME = current_time
            if force:
                logging.info(f"💾 Cache saved to disk (FORCE, {len(cache_copy)} keys)")
            else:
                logging.debug(f"💾 Cache saved to disk ({len(cache_copy)} keys)")
    except Exception as e:
        logging.error(f"⚠️ Failed to save cache: {e}")
        # Clean up tmp file
        tmp_file = f"{CACHE_FILE}.tmp"
        if os.path.exists(tmp_file):
            try: os.remove(tmp_file)
            except: pass

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
            response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
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
        if current_time - timestamp < CACHE_TTL:
            return rate

    with API_LOCK:
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
    with API_LOCK:
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

def fetch_stock_etf_timeseries(symbol, period="6mo"):
    """
    Fetches historical OHLCV (Open, High, Low, Close, Volume) data for stocks and ETFs.

    Source: Yahoo Finance (^GSPC, NVDA, AAPL, etc.)
    
    Args:
        symbol (str): The ticker symbol to fetch (e.g., 'AAPL').
        period (str): The time range to fetch (e.g., '1d', '6mo', '1y').

    Returns:
        pd.DataFrame: A cleaned DataFrame with DatetimeIndex and OHLCV columns, 
                     or None if an error occurs or no data is found.
    """
    with API_LOCK:
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

def fetch_crypto_timeseries(symbol, interval="1d", limit=180):
    """
    Fetches historical OHLCV (candlestick) data for cryptocurrencies from Binance.
    Supports larger limits (>1000) by automatically paginating requests.

    Source: Binance API (/api/v3/klines)
    
    Args:
        symbol (str): The Binance symbol pair (e.g., 'BTCUSDT').
        interval (str): K-line interval (default '1d').
        limit (int): Total number of candlesticks to fetch.

    Returns:
        pd.DataFrame: A DataFrame with DatetimeIndex and OHLCV columns,
                     or None if an error occurs.
    """
    all_data = []
    current_limit = limit
    end_time = None

    with API_LOCK:
        try:
            while current_limit > 0:
                fetch_count = min(current_limit, 1000)
                params = {"symbol": symbol, "interval": interval, "limit": fetch_count}
                if end_time:
                    params["endTime"] = end_time - 1
                
                response = requests.get(BINANCE_KLINES_URL, params=params)
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

def fetch_stock_etf_snapshot(ticker, current_time=None):
    """
    Fetches the current real-time snapshot for a stock or ETF, including metadata.

    Source: Yahoo Finance (Ticker.fast_info and Ticker.info)
    
    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL').
        current_time (float, optional): Reference Unix timestamp for caching.

    Returns:
        dict: A dictionary containing 'Name', 'Close', 'Volume', 'Market Cap', 
              'Currency', etc., or None if fetch fails.
    """
    if current_time is None:
        current_time = time.time()
    cache_key = f"yfinance_{ticker}"
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < CACHE_TTL:
            return data

    # Serialize API calls to avoid 429s/blocking
    with API_LOCK:
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
            logging.error(f"Error fetching snapshot for {ticker}: {e}")
    return None

def fetch_crypto_snapshot(ticker, current_time=None):
    """
    Fetches the current real-time snapshot for a cryptocurrency from Binance.

    Source: Binance API (/api/v3/ticker/24hr)
    
    Args:
        ticker (str): The crypto symbol (e.g., 'BTC', 'ETH').
        current_time (float, optional): Reference Unix timestamp for caching.

    Returns:
        dict: A dictionary containing 'Name', 'Close' (price), 'Volume', 
              '24h Change', and 'Market Cap', or None if fetch fails.
    """
    if current_time is None:
        current_time = time.time()
    # Ensure ticker is clean (remove -USD if present for Binance)
    clean_ticker = ticker.replace("-USD", "")
    cache_key = f"binance_{clean_ticker}"
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < CACHE_TTL:
            return data

    with API_LOCK:
        try:
            # logging.info(f"[Binance] Fetching market data for {clean_ticker}...")
            response = requests.get(f"{BINANCE_SNAPSHOT_URL}?symbol={clean_ticker}USDT").json()
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
            logging.error(f"Error fetching crypto snapshot for {clean_ticker}: {e}")
    return None


from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_market_data():
    """
    Fetches real-time market data for trending assets in PARALLEL.
    """
    trending_assets = get_trending_assets()
    market_data = {"Stocks": {}, "ETFs": {}, "Crypto": {}}
    current_time = time.time()

    tasks = []
    # Identify all assets to fetch across categories
    flatten_start = time.time()
    for category, sub_cats in trending_assets.items():
        for sub_cat, tickers in sub_cats.items():
            for ticker in tickers:
                tasks.append((ticker, category))


    logging.info(f"🚀 Parallel fetching {len(tasks)} market snapshots...")
    fetch_start = time.time()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_asset = {}
        for ticker, category in tasks:
            if category in ["Stocks", "ETFs"]:
                future_to_asset[executor.submit(fetch_stock_etf_snapshot, ticker, current_time)] = (ticker, category)
            elif category == "Crypto":
                future_to_asset[executor.submit(fetch_crypto_snapshot, ticker, current_time)] = (ticker, category)

        for future in as_completed(future_to_asset):
            ticker, category = future_to_asset[future]
            try:
                data = future.result()
                if data:
                    market_data[category][ticker] = data
            except Exception as e:
                logging.error(f"❌ Parallel fetch error for {ticker}: {e}")

    # One final save after batch fetch
    save_start = time.time()
    save_cache(force=True)
    logging.info(f"Finished fetching all market data. Fetch took {save_start - fetch_start:.2f}s, Save took {time.time() - save_start:.2f}s")
    return market_data

def fetch_historical_data(ticker, category, period=PRIMARY_CACHE_PERIOD):
    """
    Fetches historical price data. 
    period can be "1y", "2y", "5y", etc. Defaults to PRIMARY_CACHE_PERIOD from config.
    Now optimized to always use 10y as the base dataset to minimize redundant fetches.
    """
        
    # Always use the primary cache period (e.g., 10y) as the master key to avoid redundancy
    primary_period = PRIMARY_CACHE_PERIOD
    cache_key = f"hist_{ticker}_{primary_period}"
    current_time = time.time()
    
    # 1. Try Cache First (Always look for the 10y version)
    df = None
    if cache_key in MARKET_DATA_CACHE:
        data, timestamp = MARKET_DATA_CACHE[cache_key]
        if current_time - timestamp < CACHE_TTL: # Cache historical data for period from config
             # Check if data is in 'split' format (dict with 'index', 'columns', 'data')
             if isinstance(data, dict) and "columns" in data:
                 df = pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
                 df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
             else:
                 # Legacy format handling
                 df = pd.DataFrame(data)
                 if not df.empty:
                     df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    # 2. Fetch if not in cache or if we need a fresh fetch
    if df is None:
        logging.info(f"📅 Fetching 10Y historical data for {ticker} ({category}) as base...")
        try:
            if category in ["Stocks", "ETFs"]:
                df = fetch_stock_etf_timeseries(ticker, period=primary_period)
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            elif category == "Crypto":
                limit = parse_period_to_days(primary_period)
                df = fetch_crypto_timeseries(ticker + "USDT", limit=limit)
            
            if df is not None and not df.empty:
                with CACHE_LOCK:
                    MARKET_DATA_CACHE[cache_key] = (df.to_dict(orient='split'), current_time)
                save_cache()
        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {e}")

    # 3. Dynamic Slicing for the requested period
    if df is not None and not df.empty:
        if period == primary_period:
            return df
            
        # Slice for smaller periods
        requested_days = parse_period_to_days(period)
        cutoff_date = df.index[-1] - pd.Timedelta(days=requested_days)
        sliced_df = df[df.index >= cutoff_date].copy()
        
        # Ensure we have enough data (if 10y fetch returned less than 1y, sliced might be empty)
        if not sliced_df.empty:
            return sliced_df
        return df # Fallback to whatever we have if slice is empty

    logging.warning(f"❌ fetch_historical_data failed for {ticker} ({category}) - returning None")
    return None

def fetch_historical_returns(tickers_with_categories, period="1y"):
    """
    Fetches historical returns in parallel to avoid sequential bottlenecks.
    """
    series_dict = {}
    
    logging.info(f"🚀 Batch fetching historical returns for {len(tickers_with_categories)} assets...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {
            executor.submit(fetch_historical_data, ticker, category, period=period): ticker
            for ticker, category in tickers_with_categories
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                    series_dict[ticker] = df["Close"].pct_change()
            except Exception as e:
                logging.error(f"❌ Error fetching historical returns for {ticker}: {e}")
    
    if not series_dict:
        return pd.DataFrame()

    returns_df = pd.concat(series_dict, axis=1).fillna(0.0)
    if not returns_df.empty:
        returns_df = returns_df.iloc[1:]
            
    return returns_df
