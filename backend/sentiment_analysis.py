import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = "sk-50993c19c1854c13a050b994b7fdfbd7"  # Replace with your actual API key

def analyze_sentiment(text):

    # mock sentiment result
    if True:
        return {"score": 0.5, "trend": "positive"}

    """
    Analyzes the sentiment of the provided text using DeepSeek's API.
    Returns a dictionary with 'score' and 'trend'.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Analyze the sentiment of the following text and provide a sentiment score between -1 (very negative) and 1 (very positive):\n\n{text}"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a financial sentiment analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Extract the sentiment score from the response
        sentiment_analysis = result['choices'][0]['message']['content'].strip()
        score = float(sentiment_analysis)  # Assuming the model returns a numeric score

        # Determine the sentiment trend based on the score
        if score > 0.3:
            trend = "positive"
        elif score < -0.3:
            trend = "negative"
        else:
            trend = "neutral"

        logging.info(f"Sentiment Analysis: Score = {score}, Trend = {trend}")
        return {"score": score, "trend": trend}

    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return {"score": 0, "trend": "neutral"}  # Default to neutral in case of error
