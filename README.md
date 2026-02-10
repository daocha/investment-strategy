# Investment Strategy Optimization System

## Overview

This project is a comprehensive **Investment Strategy Platform** that integrates machine learning models, market data, technical analysis, and backtesting to provide optimized investment strategies. It dynamically selects the best prediction model based on asset type and volatility.

## Features

- **Stock, ETF, and Crypto Price Prediction**
  - Uses a **unified XGBoost Classifier** for all asset types.
  - Dynamically computes confidence-weighted returns based on asset volatility.
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
- **Automated Maintenance**
  - Background worker handles **12-hour market data refreshes** and **weekly AI retraining** (HKT schedule).
- **High-Performance Caching**
  - Uses **Redis** for sub-millisecond data retrieval of technical indicators and market snapshots.
  - **Automatic Fallback**: Seamlessly switches to local JSON file caching if Redis is unavailable.
  - Periodic persistence to disk ensures cache stability across restarts.
 
## Preview
![Screenshot 2026-02-10 at 12 02 53 AM](https://github.com/user-attachments/assets/56458fbd-7066-4090-8bab-2cc667a81650)


## Model Selection

| Asset Type               | Model Used               |
| ------------------------ | ------------------------ |
| Stocks                   | XGBoost                  |
| ETFs                     | XGBoost                  |
| Crypto                   | XGBoost                  |

## Installation

### Prerequisites
- **Python 3.9+**
- **Node.js 16+** & **npm**
- **Redis Server** (Running on port 6379 by default)
- **Git**

### Quick Start (Recommended)
We provide a unified startup script that installs dependencies, fixes common environment issues, and launches both the backend and frontend.

```bash
# 1. Clone the repository
git clone https://github.com/daocha/investment-strategy.git
cd investment-strategy

# 2. Run the startup script
chmod +x startup.sh
./startup.sh
```

- **Backend**: Runs on `http://localhost:8848`
- **Frontend**: Runs on `http://localhost:3848`
- **Maintenance**: Background worker refreshes data at 05:30/16:30 HKT.

### Manual Installation
If you prefer identifying issues yourself:

```bash
# Backend
pip install -r backend/requirements.txt
python backend/main.py

# Frontend
cd investment-ui
npm install
PORT=3848 npm start
```

## AI Model Training
The system uses an **XGBoost Classifier** to predict Buy/Sell/Hold signals.

- **Auto-Generation**: If `backend/xgboost_model.json` is missing, `startup.sh` will automatically trigger training using 10 years of historical data.
- **Scheduled Retraining**: The maintenance worker automatically retrains the model every Sunday at 00:00 HKT using refreshed data.
- **Manual Retraining**: You can force a retrain at any time: `python3 backend/train_model.py`

## Usage

### Generate an Optimized Investment Strategy
1. Open the dashboard at `http://localhost:3848`.
2. Select your **Risk Level** (Low, Medium, High).
3. Enter your **Initial Capital**.
4. Click **"Generate Optimal"**.

### Analyze Custom Portfolio
1. Click **"Analyse Custom"**.
2. Enter your portfolio CSV data in the format:
   ```
   Ticker,Units,Category
   BTC,1.5,Crypto
   AAPL,10,Stocks
   ```
3. Your input is **automatically saved in browser cache** and restored on page refresh.
4. View the **Weighted Forecast** to see how your assets are predicted to perform.

### Redis Configuration

The system automatically detects and connects to Redis. You can override defaults using environment variables:

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `REDIS_HOST` | Redis server hostname | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_DB` | Redis database index | `0` |
| `REDIS_PASSWORD` | Optional connection password | `None` |

### Port Conflicts
The script tries to clear ports `8848` and `3848` before starting. If you still see "Address already in use", you can manually kill processes:
```bash
lsof -ti :8848,3848 | xargs kill -9
```

## Contributors

- @daocha

## License

MIT License

