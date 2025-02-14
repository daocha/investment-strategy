# Investment Strategy Optimization System

## Overview

This project is a comprehensive **Investment Strategy Platform** that integrates machine learning models, market data, technical analysis, and backtesting to provide optimized investment strategies. It dynamically selects the best prediction model based on asset type and volatility.

## Features

- **Stock, ETF, and Cryptocurrency Price Prediction**
  - Uses **Hybrid Models (XGBoost + ARIMA)** for stock price prediction.
  - Uses **ARIMA** for ETFs.
  - Uses **LSTM** for cryptocurrency predictions.
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
| Cryptocurrencies         | LSTM                     |

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

## Contributors

- Ray LI (@daocha)

## License

MIT License

