import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import logging # Added import logging
# Configure logging - MUST be before backend imports to take precedence
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🚀 Script received start command. Initializing...")

from backend.features import FeaturesPipeline
from backend.market_data import fetch_historical_data
from backend.config import MODEL_PATH, ASSET_LIST

def get_all_assets():
    """Flatten ASSET_LIST from config to get all trainable assets."""
    assets = {"Stocks": [], "ETFs": [], "Crypto": []}
    
    # Stocks
    for group, tickers in ASSET_LIST.get("Stocks", {}).items():
        assets["Stocks"].extend(tickers)
        
    # ETFs
    for group, tickers in ASSET_LIST.get("ETFs", {}).items():
        assets["ETFs"].extend(tickers)
        
    # Crypto
    for group, tickers in ASSET_LIST.get("Crypto", {}).items():
        assets["Crypto"].extend(tickers)
        
    # Indices (Treat as ETFs/Market proxies for training context if needed, but for now stick to tradeable assets)
    # Market data fetching might differ for indices, so excluding for this specific classifier model 
    # which is intended for asset prediction.
    
    return assets

# Dynamically load assets
TRAINING_ASSETS = get_all_assets()

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

def train_model(output_path=MODEL_PATH):
    all_features = []
    all_labels = []

    logging.info(f"🚀 Starting Model Training with 10Y historical data... (Target: {output_path})")

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

    # Atomic Save Logic
    if output_path == MODEL_PATH:
        # Use a temporary file for atomic overwrite
        temp_path = f"{output_path}.tmp"
        model.save_model(temp_path)
        os.replace(temp_path, output_path)
        logging.info(f"✅ Model successfully trained and saved atomically to {output_path}")
    else:
        # Direct save to the requested location
        model.save_model(output_path)
        logging.info(f"✅ Model successfully trained and saved to {output_path}")

if __name__ == "__main__":
    train_model()
