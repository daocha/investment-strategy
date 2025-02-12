import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_price(asset):
    """Predicts future price of an asset using linear regression."""
    stock_data = yf.Ticker(asset).history(period="6mo")

    if stock_data.empty:
        return None

    stock_data["Day"] = np.arange(len(stock_data))  # Add a numeric index
    X = stock_data["Day"].values.reshape(-1, 1)
    y = stock_data["Close"].values

    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(stock_data)]])
    return model.predict(next_day)[0]
