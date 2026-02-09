import time
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.market_data import fetch_historical_data
from backend.train_model import train_model, TRAINING_ASSETS
from backend.config import MODEL_PATH

# Define HKT (UTC+8) timezone
HKT = timezone(timedelta(hours=8))

def refresh_all_market_data():
    """Iterates through all assets to refresh the 10Y cache."""
    logging.info("🔋 Starting scheduled market data refresh (all assets)...")
    count = 0
    for category, tickers in TRAINING_ASSETS.items():
        for ticker in tickers:
            try:
                # This automatically updates the 10y cache in market_data.py
                fetch_historical_data(ticker, category, period="10y")
                count += 1
            except Exception as e:
                logging.error(f"Error refreshing {ticker}: {e}")
    logging.info(f"✅ Refresh complete. Updated {count} assets.")

def run_maintenance_cycle(is_training_day=False):
    """Refreshes data and optionally retrains the model."""
    refresh_all_market_data()
    
    if is_training_day:
        logging.info("🧠 Scheduled WEEKLY training starting...")
        train_model()
        logging.info("⭐ Weekly training completed successfully.")

def main_loop():
    logging.info("🛰️ Maintenance Worker started. Watching for 05:30/16:30 Refreshes and Sunday 00:00 Training.")
    
    last_run_date = None
    last_run_type = None # 'morning', 'afternoon', 'weekly'

    while True:
        now = datetime.now(HKT)
        today = now.date()
        current_hour = now.hour
        current_minute = now.minute
        current_weekday = now.weekday() # 6 is Sunday

        # Check for Sunday 00:00 (Weekly Training)
        if current_weekday == 6 and current_hour == 0 and current_minute < 5:
            if last_run_date != today or last_run_type != 'weekly':
                logging.info(f"📅 It's Sunday 00:00 HKT. Triggering weekly maintenance.")
                run_maintenance_cycle(is_training_day=True)
                last_run_date = today
                last_run_type = 'weekly'

        # Check for 05:30 (Morning Refresh)
        elif current_hour == 5 and current_minute >= 30 and current_minute < 35:
            if last_run_date != today or last_run_type != 'morning':
                logging.info(f"☀️ It's 05:30 HKT. Triggering morning refresh.")
                run_maintenance_cycle(is_training_day=False)
                last_run_date = today
                last_run_type = 'morning'

        # Check for 16:30 (Afternoon Refresh)
        elif current_hour == 16 and current_minute >= 30 and current_minute < 35:
            if last_run_date != today or last_run_type != 'afternoon':
                logging.info(f"🌅 It's 16:30 HKT. Triggering afternoon refresh.")
                run_maintenance_cycle(is_training_day=False)
                last_run_date = today
                last_run_type = 'afternoon'

        # Sleep for a bit to avoid CPU spike
        time.sleep(60)

if __name__ == "__main__":
    # If run directly with --now, run a cycle immediately for testing
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        run_maintenance_cycle(is_training_day=True)
    else:
        main_loop()
