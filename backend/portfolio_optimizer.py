import numpy as np
import pandas as pd
from scipy.optimize import minimize
from market_data import fetch_market_data
from sentiment_analysis import analyze_sentiment
from technical_analysis import calculate_indicators
from backtesting import backtest_strategy
from price_prediction import predict_price  # Importing the updated price prediction model
import logging
import re
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define risk-based rules
RISK_SETTINGS = {
    "low": {"allowed_assets": ["ETFs", "Dividend Stocks", "Large-Cap Stocks"], "max_volatility": 0.15},
    "medium": {"allowed_assets": ["Stocks", "ETFs", "Large-Cap Cryptocurrencies"], "max_volatility": 0.25},
    "high": {"allowed_assets": ["Stocks", "ETFs", "Cryptocurrencies"], "max_volatility": None},  # No limit
}
RSI_THRESHOLD = 50
MACD_THRESHOLD = 0
MAX_NUM_ASSETS = 10

def get_sp500_annual_return():
    """Fetches the S&P 500 annual return as a benchmark and converts it to decimal."""
    try:
        sp500 = yf.Ticker("^GSPC")
        history = sp500.history(period="1y")

        if history.empty:
            logging.warning("Could not fetch S&P 500 historical data. Using default threshold (5%).")
            return 0.05  # Default 5% return if unavailable

        # Compute S&P 500 annual return
        annual_return = ((history["Close"].iloc[-1] - history["Close"].iloc[0]) / history["Close"].iloc[0])
        logging.info(f"✅ S&P 500 Annual Return: {annual_return * 100:.2f}%")

        return round(annual_return, 4)

    except Exception as e:
        logging.error(f"⚠️ Error fetching S&P 500 data: {e}. Using default 5% (0.05).")
        return 0.05  # Default fallback to 5%

def parse_annual_return(annual_return):
    """
    Converts annual return values from percentage format (e.g., "5.2%") to decimal (e.g., 0.052).

    Handles:
    - "5.2%" → 0.052
    - "5.2" → 0.052
    - 0.052 (already in decimal) → 0.052
    - Invalid values default to None

    Args:
        annual_return (str, float, or int): The annual return value to parse.

    Returns:
        float: Converted annual return in decimal format, or None if invalid.
    """
    if isinstance(annual_return, str):
        # Remove non-numeric characters (like "%")
        cleaned_value = re.sub(r"[^\d.]", "", annual_return)
        try:
            return float(cleaned_value) / 100  # Convert percentage to decimal
        except ValueError:
            logging.error(f"❌ Invalid annual return format: {annual_return}")
            return None

    elif isinstance(annual_return, (float, int)):
        # If already a decimal (e.g., 0.052), return as is
        return float(annual_return)

    logging.error(f"❌ Unsupported annual return type: {type(annual_return)}")
    return None  # Return None for invalid input

def filter_by_risk(asset_list, risk_level):
    """Filters assets based on risk level, asset type, and volatility."""
    allowed_assets = RISK_SETTINGS[risk_level]["allowed_assets"]
    max_volatility = RISK_SETTINGS[risk_level]["max_volatility"]

    filtered_assets = []
    for asset in asset_list:
        asset_volatility = asset.get("volatility", 0)  # Assume calculated volatility is provided

        if asset["category"] in allowed_assets:
            if max_volatility is None or asset_volatility <= max_volatility:
                filtered_assets.append(asset)
            else:
                logging.warning(f"Skipping {asset['asset']} due to high volatility ({asset_volatility:.2%})")
        else:
            logging.warning(f"Skipping {asset['asset']} due to asset type restriction ({asset['category']})")

    return filtered_assets

def backtest_portfolio(portfolio_allocation):
    """Runs backtesting for the entire portfolio and calculates overall performance."""
    logging.info("Running portfolio-wide backtesting...")

    total_portfolio_backtest_return = 0
    total_portfolio_predicted_return = 0
    asset_returns = []

    for asset_data in portfolio_allocation:
        asset = asset_data["asset"]
        category = asset_data["category"]
        allocation_pct = asset_data["allocation_pct"] / 100  # Convert to decimal

        logging.info(f"Backtesting asset: {asset} (Allocation: {allocation_pct:.2%})")

        current_price = asset_data["current_price"]
        predicted_price = asset_data["predicted_price"]
        predicted_return = asset_data["predicted_return"]
        combined_return = asset_data["combined_return"]
        backtest_annual_return = asset_data["backtest_annual_return"]
        weighted_predicted_return = predicted_return * allocation_pct
        weighted_backtest_return = backtest_annual_return * allocation_pct
        total_portfolio_backtest_return += weighted_backtest_return
        total_portfolio_predicted_return += weighted_predicted_return
        asset_returns.append({"asset": asset, "category": category, "current_price": current_price, "predicted_price": predicted_price, "predicted_return": predicted_return * 100, "weighted_predicted_return": weighted_predicted_return * 100, "weighted_backtest_return": weighted_backtest_return * 100, "backtest_annual_return": backtest_annual_return * 100, "combined_return": combined_return * 100})

    logging.info(f"Final portfolio backtest return: {total_portfolio_backtest_return * 100:.2f}%")

    return {
        "portfolio_predicted_return": round(total_portfolio_predicted_return * 100, 2),
        "portfolio_backtest_return": round(total_portfolio_backtest_return * 100, 2),
        "asset_returns": asset_returns
    }

def convert_floats(obj):
    """Recursively converts NumPy float32 to native Python float."""
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    return obj

def generate_strategy(risk_level, timeframe, initial_amount):
    """
    Builds an optimized investment portfolio considering sentiment, RSI, MACD, backtesting, and risk level.
    - Uses S&P 500 annual return as a minimum benchmark.
    - Filters assets based on RSI, MACD, sentiment analysis, and risk tolerance.
    - Suggests only the top 10 best-performing assets.
    - Gives higher allocation to assets with highly positive sentiment.
    - Runs portfolio-wide backtesting after optimization.
    """
    MINIMUM_ANNUAL_RETURN = get_sp500_annual_return()

    logging.info(f"Using MINIMUM_ANNUAL_RETURN threshold: {MINIMUM_ANNUAL_RETURN * 100:.2f}%")

    logging.info("Fetching market data for strategy generation...")
    market_data = fetch_market_data()
    asset_category_map = {}

    for category in ["Stocks", "ETFs", "Cryptocurrencies"]:
        if category in market_data:
            for asset in market_data[category].keys():
                asset_category_map[asset] = category

    asset_performance = []

    for asset in asset_category_map.keys():
        category = asset_category_map[asset]

        logging.info(f"Analyzing sentiment for {asset}...")
        sentiment = analyze_sentiment(asset)
        print(sentiment)
        sentiment_score = sentiment["score"]  # Score between -1 and +1
        sentiment_trend = sentiment["trend"]  # 'positive', 'neutral', 'negative'

        logging.info(f"Sentiment for {asset}: Score {sentiment_score}, Trend: {sentiment_trend}")

        logging.info(f"Calculating technical indicators for {asset}...")

        indicators = calculate_indicators(asset, category)
        logging.info(f"Technical indicators for {asset}: {indicators}")

        # Apply sentiment-based filtering
        if sentiment_trend == "negative":
            logging.warning(f"Skipping {asset} due to negative sentiment.")
            continue

        if (
            sentiment_trend == "positive"
            or (sentiment_trend == "neutral" and indicators.get("RSI") > RSI_THRESHOLD and indicators.get("MACD") > MACD_THRESHOLD)
        ):
            logging.info(f"Running backtesting for {asset} ({category})...")
            backtest_result = backtest_strategy(asset, category)
            backtest_annual_return = parse_annual_return(backtest_result.get("annual_return"))

            logging.info(f"Running price prediction for {asset} ({category})...")
            predicted_price = predict_price(asset, category, timeframe)
            if predicted_price is None:
                logging.warning(f"Skipping {asset} due to failure in price prediction.")
                continue

            current_price = market_data[category][asset]["Close"]
            predicted_return = (predicted_price - current_price) / current_price
            combined_return = (backtest_annual_return * 0.6) + (predicted_return * 0.4)  # Weighted avg of past & future returns
            logging.info(f"Predicted return for {asset}: {predicted_return * 100:.2f}%, backtest return: {backtest_annual_return * 100:.2f}%, combined return: {combined_return * 100:.2f}%")

            if combined_return < MINIMUM_ANNUAL_RETURN:
                logging.warning(f"Skipping {asset} due to combined return {combined_return * 100:.2f}% lower return than S&P 500 benchmark.")
                continue

            asset_performance.append({
                "asset": asset,
                "category": category,
                "current_price": current_price,
                "combined_return": combined_return,
                "backtest_annual_return": backtest_annual_return,
                "predicted_return": predicted_return,
                "predicted_price": predicted_price,
                "volatility": np.random.uniform(0.1, 0.5)  # Replace with actual calculation
            })

    logging.info(f"Total assets meeting sentiment, RSI, MACD, and return criteria: {len(asset_performance)}")

    if not asset_performance:
        logging.warning("No suitable assets found after backtesting.")
        return {"error": "No suitable assets found"}

    # Apply Risk-Based Filtering
    asset_performance = filter_by_risk(asset_performance, risk_level)

    # Select the Top 10(MAX_NUM_ASSETS) Best-Performing Assets
    asset_performance = sorted(asset_performance, key=lambda x: x["combined_return"], reverse=True)[:MAX_NUM_ASSETS]

    logging.info(f"Top 10 best-performing assets selected: {asset_performance}")

    # **Portfolio Optimization**
    num_assets = len(asset_performance)
    np.random.seed(42)
    returns = np.array([entry["combined_return"] for entry in asset_performance])
    cov_matrix = np.random.rand(num_assets, num_assets)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    initial_guess = [1 / num_assets] * num_assets
    bounds = [(0.05, 0.50) for _ in range(num_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    logging.info("Running portfolio optimization...")

    # optimize the portfolio
    result = minimize(portfolio_volatility, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)

    portfolio_allocation = [
        {
            "asset": entry["asset"],
            "category": entry["category"],
            "current_price": round(entry["current_price"], 2),
            "combined_return": round(entry["combined_return"], 4),
            "backtest_annual_return": round(entry["backtest_annual_return"], 4),
            "predicted_return": round(entry["predicted_return"], 4),
            "predicted_price": round(entry["predicted_price"], 2),
            "allocation": round(alloc * initial_amount, 2),
            "allocation_pct": round(alloc * 100, 2)
        }
        for entry, alloc in zip(asset_performance, result.x)
    ]

    logging.info(f"Final optimized portfolio allocation: {portfolio_allocation}")

    # **Run Portfolio Backtesting**
    portfolio_backtest_results = backtest_portfolio(portfolio_allocation)

    return convert_floats({
        "portfolio_allocation": portfolio_allocation,
        "backtest_results": portfolio_backtest_results
    })
