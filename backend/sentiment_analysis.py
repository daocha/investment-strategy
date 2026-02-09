import requests
import logging
import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

from backend.config import DEEPSEEK_API_URL, USE_DEEPSEEK_API, CACHE_TTL
from backend.market_data import MARKET_DATA_CACHE, CACHE_LOCK, save_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def analyze_sentiment(asset_name):
    """
    Analyzes the sentiment of a single asset (Legacy wrapper for batch function).
    """
    results = analyze_sentiment_batch([asset_name])
    return results.get(asset_name, {"score": 0, "trend": "neutral"})

def analyze_sentiment_batch(asset_names):
    """
    Analyzes the sentiment of multiple assets in a single batch.
    Returns a dictionary mapping asset names to results.
    Uses caching to avoid redundant API calls.
    """
    if not asset_names:
        return {}

    current_time = time.time()
    results = {}
    remaining_assets = []

    # 1. Check Cache
    with CACHE_LOCK:
        for asset in asset_names:
            cache_key = f"sentiment_{asset}"
            if cache_key in MARKET_DATA_CACHE:
                cached_data, timestamp = MARKET_DATA_CACHE[cache_key]
                if current_time - timestamp < CACHE_TTL:
                    results[asset] = cached_data
                    continue
            remaining_assets.append(asset)

    if not remaining_assets:
        logging.info(f"💾 Using cached sentiment for all {len(asset_names)} assets")
        return results

    if not USE_DEEPSEEK_API:
        logging.info(f"DeepSeek API disabled. Skipping sentiment analysis for {len(remaining_assets)} assets (returning neutral).")
        for asset in remaining_assets:
            results[asset] = {"score": 0, "trend": "neutral"}
        return results

    if not DEEPSEEK_API_KEY:
        logging.error("DEEPSEEK_API_KEY not found. Returning neutral for all remaining.")
        for asset in remaining_assets:
            results[asset] = {"score": 0, "trend": "neutral"}
        return results

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        asset_list_str = ", ".join(remaining_assets)
        prompt = f"Analyze market sentiment for these assets: {asset_list_str}. Respond ONLY in JSON: {{\"Asset\": {{\"score\": float, \"trend\": \"positive\"|\"negative\"|\"neutral\"}}}}"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a financial sentiment analyzer."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }

        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        sentiment_data = json.loads(content)
        
        # 2. Save new results to Cache
        with CACHE_LOCK:
            for asset, data in sentiment_data.items():
                cache_key = f"sentiment_{asset}"
                MARKET_DATA_CACHE[cache_key] = (data, current_time)
                results[asset] = data
        save_cache()
        
        logging.info(f"Sentiment Analysis for {len(remaining_assets)} assets (API BATCH)")
        return results

    except Exception as e:
        logging.error(f"Error in batch sentiment analysis: {e}")
        for asset in remaining_assets:
            if asset not in results:
                results[asset] = {"score": 0, "trend": "neutral"}
        return results
