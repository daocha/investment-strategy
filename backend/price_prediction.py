import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import os
from backend.features import FeaturesPipeline
from backend.config import MODEL_PATH

class CryptoXGBoost:
    def __init__(self):
        # Configuration matches the previously trained model
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,  # 0: Hold, 1: Buy, 2: Sell
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        self.features = [
            'RSI', 'MACD', 'Signal_Line', 'EMA_12', 'EMA_26', 'EMA_50', 
            'EMA_200', '%K', '%D', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV', 'Volume'
        ]

    def load_model(self):
        """Loads the pre-trained XGBoost model from MODEL_PATH."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_model(MODEL_PATH)
                logging.info(f"📂 XGBoost model loaded from {MODEL_PATH}")
                return True
            except Exception as e:
                logging.error(f"❌ Failed to load XGBoost model: {e}")
                return False
        return False

    def predict(self, df):
        """
        Generates a prediction (Hold, Buy, Sell) and confidence for the latest data point.
        """
        try:
            # Ensure technical indicators are present
            df_features = FeaturesPipeline.generate_feature_set(df)
            
            # Use only the latest row for prediction
            latest_row = df_features[self.features].iloc[-1:]
            
            # XGBoost prediction
            preds = self.model.predict(latest_row)
            probs = self.model.predict_proba(latest_row)
            
            prediction_map = {0: "Hold", 1: "Buy", 2: "Sell"}
            signal = prediction_map.get(int(preds[0]), "Hold")
            confidence = float(np.max(probs))
            
            return signal, confidence
        except Exception as e:
            logging.error(f"❌ Error during XGBoost prediction: {e}")
            return "Hold", 0.0
