# Investment Strategy Optimization System

## Overview

This project is a comprehensive **Investment Strategy Platform** that integrates machine learning models, market data, technical analysis, and backtesting to provide optimized investment strategies. It dynamically selects the best prediction model based on asset type and volatility.

## Features

- **Stock, ETF, and Crypto Price Prediction**
  - Uses **Hybrid Models (XGBoost + ARIMA)** for stock price prediction.
  - Uses **ARIMA** for ETFs.
  - Uses **LSTM** for crypto predictions.
  - Dynamically selects models based on asset volatility.
- **Market Data Integration**
  - Fetches historical and real-time data from **Binance** and **Yahoo Finance**.
- **Sentiment Analysis**
  - Uses **DeepSeek AI** to analyze financial sentiment for better investment insights.
- **Technical Analysis**
  - Computes indicators like **SMA, RSI, MACD** for enhanced decision-making.
- **Backtesting**
  - Runs historical performance tests on investment strategies.
- **Portfolio Optimization**
  - Builds optimized portfolios based on **risk level, sentiment, technical analysis, and backtesting results**.

## Model Selection

| Asset Type               | Model Used               |
| ------------------------ | ------------------------ |
| Stocks (Low Volatility)  | XGBoost                  |
| Stocks (High Volatility) | Hybrid (XGBoost + ARIMA) |
| ETFs                     | ARIMA                    |
| Crypto                   | LSTM                     |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/daocha/investment-strategy.git
   cd investment-strategy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```bash
   python main.py
   ```
4. Start the frontend:
   ```bash
   npm install
   npm start
   ```

## Usage

### Predict Price for an Asset

```python
from price_prediction import predict_stock_price, predict_etf_price, predict_crypto_price

# Predict stock price 6 months ahead
predicted_price = predict_stock_price("MSTR", months_ahead=6)
print("Predicted Price:", predicted_price)
```

### Generate an Optimized Investment Strategy

```python
from portfolio_optimizer import generate_strategy

strategy = generate_strategy(risk_level="medium", timeframe=6, initial_amount=10000)
print("Optimized Portfolio:", strategy)
```

## API Endpoints

| Method | Endpoint             | Description                                |
| ------ | -------------------- | ------------------------------------------ |
| POST   | `/generate-strategy` | Generates an optimized investment strategy |

## Future Improvements

- Implement **Reinforcement Learning** for dynamic trading strategies.
- Enhance **Sentiment Analysis** by integrating social media and news sentiment.
- Develop **Automated Trading** features for executing strategies in real-time.
- Expand **Multi-Asset Portfolio Optimization** with additional asset classes.

## LAN Access
To access the application from other devices on your local network:
1. Ensure the backend is running (it binds to `0.0.0.0` by default).
2. Access the frontend via `http://YOUR_LOCAL_IP:3000`. The API URL is dynamically detected.

## Troubleshooting

### Frontend Launch Failed
If you encounter a blank page or "Launch Failed" error in the frontend:
- Ensure the backend is running on port 8848.
- Check the browser console -> if you see errors related to `pieChartData`, pull the latest code as this was fixed in a recent update.
- Verify `node_modules` are installed (`npm install`).

### XGBoost Model Warning
If you see `XGBoost model not found`, this is normal on a fresh install. The system will function using fallback predictions or default to "Hold" signals until the model is trained. The `xgboost_model.json` file is ignored in git to prevent large file uploads.

## Contributors

- Ray LI (@daocha)

## License

MIT License

