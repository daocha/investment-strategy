import pandas as pd
import numpy as np

def calculate_market_features(df):
    """
    Adds market-related features (Volume, OBV, Volatility) to the DataFrame.
    """
    df = df.copy()
    
    # OBV (On-Balance Volume)
    price_diff = df['Close'].diff()
    df['OBV'] = (np.sign(price_diff) * df['Volume']).fillna(0).cumsum()
    
    # Volatility (Annualized)
    df['Volatility'] = df['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
    
    return df
