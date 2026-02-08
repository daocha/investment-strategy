import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging
from backend.features import FeaturesPipeline
from backend.market_data import fetch_historical_data
from backend.config import MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Representative assets for training
TRAINING_ASSETS = {
    "Stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V"],
    "ETFs": ["SPY", "QQQ", "VTI", "VOO", "IWM"],
    "Crypto": ["BTC", "ETH", "SOL", "BNB"]
}

def create_labels(df, window=7, threshold=0.03):
    """
    Creates labels based on future N-day returns.
    0: Hold, 1: Buy, 2: Sell
    """
    future_return = df['Close'].shift(-window) / df['Close'] - 1
    
    labels = pd.Series(0, index=df.index)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = 2
    
    # Drop rows where we don't have future data
    return labels.iloc[:-window]

def train_model():
    all_features = []
    all_labels = []

    logging.info("🚀 Starting Model Training with 10Y historical data...")

    for category, tickers in TRAINING_ASSETS.items():
        for ticker in tickers:
            logging.info(f"📊 Processing {ticker} ({category})...")
            try:
                # Fetch 10 years of data as requested by the user
                df = fetch_historical_data(ticker, category, period="10y")
                
                if df is None or len(df) < 200:
                    logging.warning(f"⚠️ Insufficient data for {ticker}. Skipping.")
                    continue

                # Generate features using unified pipeline
                df_features = FeaturesPipeline.generate_feature_set(df)
                
                # Define feature columns (parity with price_prediction.py)
                feature_cols = [
                    'RSI', 'MACD', 'Signal_Line', 'EMA_12', 'EMA_26', 'EMA_50', 
                    'EMA_200', '%K', '%D', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV', 'Volume'
                ]
                
                # Ensure all features exist
                if not all(col in df_features.columns for col in feature_cols):
                    logging.warning(f"⚠️ Missing columns for {ticker}. Skipping.")
                    continue

                # Create labels
                labels = create_labels(df_features)
                
                # Align features with labels
                features = df_features[feature_cols].loc[labels.index]
                
                all_features.append(features)
                all_labels.append(labels)
                
            except Exception as e:
                logging.error(f"❌ Error processing {ticker}: {e}")

    if not all_features:
        logging.error("❌ No data collected for training. Aborting.")
        return

    # Combine all assets
    X = pd.concat(all_features)
    y = pd.concat(all_labels)

    logging.info(f"📈 Training on {len(X)} samples...")

    # Initialize model with production params
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )

    # Fit model
    model.fit(X, y)

    # Save model
    model.save_model(MODEL_PATH)
    logging.info(f"✅ Model successfully trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
