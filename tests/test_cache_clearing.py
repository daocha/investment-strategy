import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.market_data import MARKET_DATA_CACHE

def test_cache_clearing():
    print("🧪 Testing Cache Clearing Logic...")
    
    # 1. Seed some dummy data
    MARKET_DATA_CACHE["prediction_TEST1"] = ({"signal": "Buy"}, time.time())
    MARKET_DATA_CACHE["prediction_TEST2"] = ({"signal": "Sell"}, time.time())
    MARKET_DATA_CACHE["yfinance_TEST"] = ({"Close": 100}, time.time())
    MARKET_DATA_CACHE["hist_TEST_10y"] = ({"data": []}, time.time())
    MARKET_DATA_CACHE["other_key"] = ("data", time.time())
    
    # 2. Test clear_predictions
    print("Testing clear_predictions()...")
    MARKET_DATA_CACHE.clear_predictions()
    
    if "prediction_TEST1" in MARKET_DATA_CACHE or "prediction_TEST2" in MARKET_DATA_CACHE:
        print("❌ Failed: prediction keys still exist!")
    else:
        print("✅ Success: prediction keys cleared.")
        
    # 3. Test clear_market_data
    print("Testing clear_market_data()...")
    MARKET_DATA_CACHE.clear_market_data()
    
    market_keys = ["yfinance_TEST", "hist_TEST_10y"]
    if any(k in MARKET_DATA_CACHE for k in market_keys):
        print("❌ Failed: market keys still exist!")
    else:
        print("✅ Success: market keys cleared.")
        
    # 4. Ensure other keys remain
    if "other_key" not in MARKET_DATA_CACHE:
        print("❌ Failed: independent keys were accidentally deleted!")
    else:
        print("✅ Success: independent keys preserved.")

    print("🏁 Verification Complete.")

if __name__ == "__main__":
    test_cache_clearing()
