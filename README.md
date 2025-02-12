# Investment Strategy Optimization System

## Overview
This project is an **automated investment strategy optimizer** that:
- Fetches **real-time market data** from Yahoo Finance (stocks/ETFs) and Binance (cryptos).
- Analyzes **market sentiment** using DeepSeek AI.
- Computes **technical indicators** (RSI, MACD, SMA) to filter high-potential assets.
- Performs **backtesting** on selected assets.
- **Optimizes portfolio allocation** using **risk-based constraints** and **return maximization**.

## Key Features
### **1️⃣ Market Data Fetching**
- **Stocks & ETFs** → Yahoo Finance (`yfinance` API).
- **Cryptocurrencies** → Binance API.
- Ensures **live market data integration**.

### **2️⃣ Sentiment Analysis**
- Calls DeepSeek API to analyze **financial news & trends**.
- Assets with **strong positive sentiment** get higher allocation boosts.

### **3️⃣ Technical Indicator Analysis**
- Uses **RSI (Relative Strength Index)** to detect momentum.
- Uses **MACD (Moving Average Convergence Divergence)** for trend confirmation.
- Uses **SMA (50-day Simple Moving Average)** for stability check.

### **4️⃣ Backtesting for Performance Validation**
- Runs **historical performance tests** on selected assets.
- Compares asset returns **against S&P 500 benchmark**.
- Rejects assets with underperformance.

### **5️⃣ Portfolio Optimization**
- Uses **Scipy’s `minimize()` function** to find the best asset allocation.
- **Diversifies assets** to reduce risk while maximizing returns.
- Ensures **total portfolio allocation = 100%**.

### **6️⃣ Risk-Based Filtering**
- Allows users to select **Low, Medium, or High risk profiles**.
- Filters assets based on **volatility & asset class constraints**.

## How It Works
1. **User Inputs** → Risk level, investment timeframe, initial capital.
2. **Market Analysis** → Fetches data, calculates indicators, and sentiment scores.
3. **Filtering & Backtesting** → Removes weak assets, retains high-performing ones.
4. **Portfolio Optimization** → Allocates funds based on **return vs. risk trade-offs**.
5. **Final Output** → Recommends the **top 10 assets** with an optimized allocation.
6. **Portfolio Backtesting** → Validates performance before execution.

## Installation
```bash
# Clone repository
git clone https://github.com/daocha/investment-strategy.git
cd investment-strategy

# Install dependencies
pip install -r requirements.txt

# Run the backend
python main.py

# Run the frontend
cd investment-ui
npm install
npm start
```

## API Endpoints
- `POST /generate-strategy` → Generates an optimized portfolio.
- `GET /market-data` → Fetches current market data.
- `GET /backtest-results` → Runs historical performance tests.

## Frontend (React UI)
- Displays **investment recommendations in a pie chart**.
- Shows **expected portfolio return vs. S&P 500 benchmark**.
- Allows users to **input risk level, timeframe, and capital**.

## Future Enhancements
- Integrate **Machine Learning for better trend prediction**.
- Add **real-time portfolio tracking with API integrations**.
- Implement **automated execution via brokerage API**.

---
📈 **Designed for investors who want an automated, data-driven portfolio strategy.** 🚀

