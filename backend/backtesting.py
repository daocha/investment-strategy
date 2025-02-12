import yfinance as yf
import requests
import pandas as pd
import numpy as np
import backtrader as bt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"  # Binance API endpoint for historical data

def fetch_stock_etf_data(symbol, period="1y"):
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

def fetch_crypto_data(symbol, interval="1d", limit=365):
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

        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("Date", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        logging.error(f"❌ Error fetching data for {symbol} from Binance: {e}")
        return None

def get_historical_data(asset, category):
    """
    Determines the correct source (Yahoo Finance or Binance) for fetching historical data.
    """
    if category in ["Stocks", "ETFs"]:
        return fetch_stock_etf_data(asset)
    elif category == "Cryptocurrencies":
        return fetch_crypto_data(asset + "USDT")  # Binance uses USDT pairs
    else:
        logging.warning(f"⚠️ Unknown category {category} for {asset}. Cannot fetch data.")
        return None

class BacktestStrategy(bt.Strategy):
    """
    A simple moving average crossover strategy for backtesting.
    """
    params = (("sma_short", 10), ("sma_long", 50),)

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_short)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_long)

    def next(self):
        if self.sma_short[0] > self.sma_long[0]:  # Buy Signal
            if not self.position:
                self.buy()
        elif self.sma_short[0] < self.sma_long[0]:  # Sell Signal
            if self.position:
                self.sell()

def backtest_strategy(asset, category):
    """
    Backtests a simple moving average crossover strategy using the appropriate data source.
    """
    logging.info(f"📊 Running backtest for {asset} ({category})...")

    df = get_historical_data(asset, category)
    if df is None or df.empty:
        logging.warning(f"⚠️ No historical data available for {asset}. Skipping backtest.")
        return {"annual_return": "N/A", "error": "No data available"}

    # Convert DataFrame to Backtrader feed format
    data = bt.feeds.PandasData(dataname=df)

    # Run Backtest
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BacktestStrategy)
    cerebro.adddata(data)
    cerebro.run()

    # Calculate Portfolio Performance (Assuming initial capital = 100)
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    annual_return = ((end_price - start_price) / start_price) * 100  # Convert to %

    logging.info(f"✅ {asset} Backtest Result: Annual Return = {annual_return:.2f}%")
    return {"annual_return": f"{annual_return:.2f}%"}

# Example Usage
if __name__ == "__main__":
    assets = [
        ("NVDA", "Stocks"),  # Yahoo Finance
        ("QQQ", "ETFs"),  # Yahoo Finance
        ("BTC", "Cryptocurrencies"),  # Binance
        ("ETH", "Cryptocurrencies")  # Binance
    ]

    for asset, category in assets:
        result = backtest_strategy(asset, category)
        print(f"{asset} ({category}) Backtest: {result}")
