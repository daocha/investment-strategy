import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_volatility(prices):
    """Calculates the asset's volatility as the standard deviation of returns."""
    returns = prices.pct_change().dropna()
    return np.std(returns)

def predict_stock_price(asset, months_ahead=1):
    """Predicts the price of a stock in x months using a Hybrid Model (LSTM + Random Forest) based on volatility."""
    stock_data = yf.Ticker(asset).history(period="1y")
    if stock_data.empty:
        return None

    volatility = calculate_volatility(stock_data["Close"])
    logging.info(f"Detected volatility for {asset}: {volatility:.4f}")

    stock_data["Day"] = np.arange(len(stock_data))
    close_prices = stock_data["Close"].values.reshape(-1, 1)

    # Predict price x months ahead (approx. 30 * months_ahead days)
    future_days = 30 * months_ahead

    # Random Forest Model
    X_rf = stock_data["Day"].values.reshape(-1, 1)
    y_rf = stock_data["Close"].values
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_rf, y_rf)

    future_day_rf = np.array([[len(stock_data) + (months_ahead * 30)]])
    predicted_price_rf = rf_model.predict(future_day_rf)[0]

    # Hybrid Prediction (Weighted Average)
    if volatility > 0.05:
        logging.info(f"Using Hybrid model (LSTM + Random Forest) for {asset} due to high volatility.")
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        input_seq = np.array([scaled_data[-10:]])

        # Prepare LSTM training data
        X_train, y_train = [], []
        for i in range(10, len(scaled_data)):
            X_train.append(scaled_data[i-10:i])
            y_train.append(scaled_data[i])

        X_train, y_train = np.array(X_train), np.array(y_train)

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(10, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
        for _ in range(future_days):
            predicted_value = model.predict(input_seq)[0][0]
            input_seq = np.append(input_seq[:, 1:, :], [[[predicted_value]]], axis=1)
        predicted_price_lstm = scaler.inverse_transform([[predicted_value]])[0][0]
        predicted_price = (predicted_price_lstm * 0.6) + (predicted_price_rf * 0.4)
    else:
        logging.info(f"Using Random Forest model alone for {asset} due to low volatility.")
        predicted_price = predicted_price_rf

    return predicted_price

def predict_etf_price(asset, months_ahead=1):
    """Predicts the price of an ETF in x months using ARIMA model."""
    stock_data = yf.Ticker(asset).history(period="1y")
    if stock_data.empty:
        return None

    model = ARIMA(stock_data["Close"], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=months_ahead * 30)  # Predict next x months

    return forecast.iloc[-1]

def predict_crypto_price(asset, months_ahead=1):
    """Fetches the last 1 year of daily price data of a cryptocurrency from Binance and predicts the price in x months from now using LSTM."""
    BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
    params = {"symbol": asset + "USDT", "interval": "1d", "limit": 365}

    try:
        response = requests.get(BINANCE_API_URL, params=params)
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"])
            df["Close"] = df["Close"].astype(float)

            # Normalize data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

            # Prepare training data
            X_train, y_train = [], []
            for i in range(10, len(scaled_data)):
                X_train.append(scaled_data[i-10:i])
                y_train.append(scaled_data[i])

            X_train, y_train = np.array(X_train), np.array(y_train)

            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(10, 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

            # Predict price x months ahead (approx. 30 * months_ahead days)
            future_days = 30 * months_ahead
            input_seq = np.array([scaled_data[-10:]])

            for _ in range(future_days):
                predicted_value = model.predict(input_seq)[0][0]
                input_seq = np.append(input_seq[:, 1:, :], [[[predicted_value]]], axis=1)

            predicted_price = scaler.inverse_transform([[predicted_value]])[0][0]

            return predicted_price
        else:
            logging.warning(f"⚠️ Binance API response did not contain data for {asset}")
            return None
    except Exception as e:
        logging.error(f"❌ Error fetching price data for {asset} from Binance: {e}")
        return None

def predict_price(asset, category, timeframe):
    """Determines the appropriate prediction model based on asset category."""
    if category == "Stocks":
        return predict_stock_price(asset, timeframe)
    elif category == "ETFs":
        return predict_etf_price(asset, timeframe)
    elif category == "Cryptocurrencies":
        return predict_crypto_price(asset, timeframe)
    else:
        logging.warning(f"Unknown category {category} for {asset}. Skipping prediction.")
        return None
