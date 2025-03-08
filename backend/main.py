from flask import Flask, request, jsonify
from flask_cors import CORS
from portfolio_optimizer import generate_strategy

app = Flask(__name__)
CORS(app)  # Enables CORS for frontend requests

@app.post("/generate-strategy")
def generate_investment_strategy():
    """Generates an optimized investment strategy based on user input."""
    data = request.json
    risk_level = data.get("risk_level")
    timeframe = data.get("timeframe")
    initial_amount = data.get("initial_amount")
    return generate_strategy(
        risk_level,
        timeframe,
        initial_amount
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8848, debug=True)
