import os

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, "my_portfolio.csv")
CACHE_FILE = os.path.join(BASE_DIR, "market_data_cache.json")
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.json")

# Market Data & Time Settings
CACHE_TTL = 86400  # 1 day
USE_DEEPSEEK_API = False
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DAYS_IN_YEAR = 365.25
DEFAULT_BACKTEST_PERIOD = "10y"

# Portfolio Analysis Thresholds
RSI_THRESHOLD = 50
MACD_THRESHOLD = 0
MAX_NUM_ASSETS = 10
MIN_RETURN_THRESHOLD = -10.0  # Mock threshold to ensure all holdings are analyzed

# Risk-Based Rules
RISK_SETTINGS = {
    "low": {"allowed_assets": ["Stocks", "ETFs"], "max_volatility": 0.25},
    "medium": {"allowed_assets": ["Stocks", "ETFs", "Crypto"], "max_volatility": 0.60}, # Increased from 0.45
    "high": {"allowed_assets": ["Stocks", "ETFs", "Crypto"], "max_volatility": None},
}

# Currency Symbols Mapping
CURRENCY_SYMBOLS = {
    "USD": "$",
    "HKD": "HK$",
    "TWD": "NT$",
    "EUR": "€",
    "GBP": "£"
}

# Comprehensive Asset List
ASSET_LIST = {
    "Stocks": {
        "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "Technology": ["AMD", "AVGO", "ASML", "CRM", "SNPS", "NOW", "PANW", "NET", "PLTR", "SMCI", "SNOW", "DBRG", "AI", "TSM", "QCOM", "LRCX", "KLAC", "MRVL", "TER", "AMAT", "ADBE", "ORCL", "INTU", "ZS", "CRWD", "OKTA", "DDOG", "TEAM", "DOCU", "SHOP", "U", "ROKU", "TWLO", "ZM", "PYPL", "ABNB", "SPOT", "NFLX", "DIS", "MU", "QRVO", "SWKS", "TXN", "ON"],
        "Crypto-Proxy": ["MSTR", "COIN", "MARA", "RIOT", "CLSK"],
        "Financials": ["JPM", "BAC", "BLK", "GS", "MS", "SPGI", "ICE", "PNC", "C", "WFC", "SCHW", "AXP", "COF", "USB", "MET", "PRU", "AON", "MSCI", "RJF"],
        "Healthcare": ["LLY", "UNH", "JNJ", "MRK", "PFE", "ABBV", "AMGN", "VRTX", "REGN", "ISRG", "DXCM", "MRNA", "TDOC", "BIIB", "GILD", "DHR", "BDX", "ZTS", "ILMN", "HUM"],
        "Consumer": ["COST", "PG", "KO", "PEP", "SBUX", "MCD", "NKE", "TGT", "HD", "LOW", "EL", "LULU", "TSN", "KR", "DG", "DLTR", "CMG", "YUM", "DPZ", "HAS"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "LNG", "FANG", "DVN", "OXY", "PSX", "VLO", "HAL", "EQT"],
        "Industrials": ["CAT", "DE", "UNP", "HON", "RTX", "LMT", "GE", "BA", "GD", "NOC", "WM", "RSG", "FAST", "CMI", "EMR"],
        "Materials": ["LIN", "DOW", "NEM", "FCX", "APD", "SHW", "ECL", "PPG", "AA", "CF"]
    },
    "Indices": {
        "US Major": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
        "Global/Other": ["^FTSE", "^N225", "^GDAXI", "^FCHI", "^HSI"]
    },
    "ETFs": {
        "Index Trackers": ["SPY", "DIA", "QQQ", "IWM", "VOO"],
        "Tech": ["XLK", "VGT", "SMH", "SOXX", "BOTZ", "AIQ", "CIBR", "HACK", "SKYY", "CLOU", "WCLD", "ARKK", "IGV", "FINX"],
        "Sector": ["XLF", "XLV", "XLE", "XLI", "XLY", "XLB", "XLU", "XLRE", "XBI", "IBB"],
        "Dividends": ["SCHD", "VIG", "DVY", "SPHD", "NOBL", "SDY"],
        "International": ["EEM", "VWO", "IEMG", "FLKR", "EWY", "EWT"],
        "Thematic": ["ICLN", "TAN", "PBW", "LIT", "BLOK", "BUG", "DRIV", "BETZ"]
    },
    "Crypto": {
        "Major": ["BTC", "ETH", "SOL", "ADA"],
        "DeFi/L1": ["LINK", "AAVE", "MKR"]
    }
}

# Crypto ETF to Underlying Mapping
CRYPTO_ETF_MAPPING = {
    # US Spot ETFs
    "IBIT": "BTC",
    "FBTC": "BTC",
    "ARKB": "BTC",
    "BITB": "BTC",
    "HODL": "BTC",
    "BRRR": "BTC",
    "BTCO": "BTC",
    "EZBC": "BTC",
    "BTCW": "BTC",
    "GBTC": "BTC",
    "BITO": "BTC", # Futures
    "ETHV": "ETH",
    "ETHE": "ETH",
    "FETH": "ETH",
    "ETHA": "ETH",
    
    # Hong Kong Spot ETFs
    "3439.HK": "BTC", # Bosera BTC
    "3042.HK": "BTC", # Harvest BTC
    "3046.HK": "BTC", # ChinaAMC BTC
    "3168.HK": "ETH", # Bosera ETH
    "3179.HK": "ETH", # Harvest ETH
    "3008.HK": "ETH", # ChinaAMC ETH
}

