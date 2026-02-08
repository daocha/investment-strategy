import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import os
from dotenv import load_dotenv

load_dotenv()

from backend.config import DEEPSEEK_API_URL, USE_DEEPSEEK_API

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
    """
    if not asset_names:
        return {}

    if not USE_DEEPSEEK_API:
        # Mock Logic for multiple assets
        import random
        results = {}
        for asset in asset_names:
            score = random.uniform(0.1, 0.9)
            trend = "positive" if score > 0.3 else ("negative" if score < -0.3 else "neutral")
            results[asset] = {"score": score, "trend": trend}
        logging.info(f"Sentiment Analysis for {len(asset_names)} assets (MOCKED BATCH)")
        return results

    if not DEEPSEEK_API_KEY:
        logging.error("DEEPSEEK_API_KEY not found. Returning neutral for all.")
        return {asset: {"score": 0, "trend": "neutral"} for asset in asset_names}

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        asset_list_str = ", ".join(asset_names)
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
        
        import json
        sentiment_data = json.loads(content)
        
        logging.info(f"Sentiment Analysis for {len(asset_names)} assets (API BATCH)")
        return sentiment_data

    except Exception as e:
        logging.error(f"Error in batch sentiment analysis: {e}")
        return {asset: {"score": 0, "trend": "neutral"} for asset in asset_names}
