import pandas as pd
import numpy as np
import logging

def run_backtest(asset, category=None, period=None, data=None):
    """
    Performs a simplified backtest for an asset.
    If 'data' is provided (DataFrame), it uses it to avoid API calls.
    Returns the annualized return from the first to the last data point.
    """
    try:
        if data is None or data.empty:
            logging.warning(f"⚠️ No data provided for {asset} backtest.")
            return 0.0

        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Simplified Annual Return Logic
        start_price = float(data['Close'].iloc[0])
        end_price = float(data['Close'].iloc[-1])
        
        # Calculate days in the dataset
        days = (data.index[-1] - data.index[0]).days
        if days <= 0:
            return 0.0
            
        total_return = (end_price - start_price) / start_price
        
        # Annualize the return: (1 + total_return) ^ (365 / days) - 1
        annualized_return = (1 + total_return) ** (365.0 / max(days, 1)) - 1
        
        return annualized_return

    except Exception as e:
        logging.error(f"❌ Error in backtesting {asset}: {e}")
        return 0.0
