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
 
## Preview
![Screenshot 2026-02-10 at 12 02 53 AM](https://github.com/user-attachments/assets/56458fbd-7066-4090-8bab-2cc667a81650)


## Model Selection

| Asset Type               | Model Used               |
| ------------------------ | ------------------------ |
| Stocks                   | XGBoost                  |
| ETFs                     | XGBoost                  |
| Crypto                   | XGBoost                  |

## Installation

### Prerequisities
- **Python 3.9+**
- **Node.js 16+** & **npm**
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
- **Auto-Healing**: The script automatically detects missing dependencies (like `scikit-learn` or Node modules) and attempts to fix them.

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

- **Auto-Generation**: If `backend/xgboost_model.json` is missing, `startup.sh` will automatically trigger `backend/train_model.py` to train a new model using 10 years of historical data.
- **Manual Retraining**: You can force a retrain at any time:
  ```bash
  python3 backend/train_model.py
  ```

## Usage

### Generate an Optimized Investment Strategy
1. Open the dashboard at `http://localhost:3848`.
2. Select your **Risk Level** (Low, Medium, High).
3. Enter your **Initial Capital**.
4. Click **"Generate Optimal"**.

### Analyze Custom Portfolio
1. Click **"Analyse Custom"**.
2. Enter your CSV data or type `myself` to load your saved portfolio.
3. View the **Weighted Forecast** to see how your assets are predicted to perform.

## Troubleshooting

### Ubuntu / Linux Issues
- **`ModuleNotFoundError: No module named 'sklearn'`**: This is required for XGBoost. Run `pip install scikit-learn` or use `./startup.sh` which fixes this automatically.
- **`Error: Cannot find module ...` (Node.js)**: If you see this during `npm start`, run:
  ```bash
  cd investment-ui
  npx browserslist@latest --update-db
  rm -rf node_modules package-lock.json && npm install
  ```

### Port Conflicts
- The script tries to clear ports `8848` and `3848` before starting.
- If you still see "Address already in use", you can manually kill "zombie" processes:
  ```bash
  lsof -ti :8848 | xargs kill -9
  lsof -ti :3848 | xargs kill -9
  ```

## Contributors

- @daocha

## License

MIT License

