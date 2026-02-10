import logging
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging early to capture all backend logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add project root to sys.path to allow imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import PORTFOLIO_FILE
from backend.portfolio_optimizer import generate_strategy
from backend.analyze_my_portfolio import run_portfolio_analysis
from backend.verification_service import verification_service
import pandas as pd
from io import StringIO

app = Flask(__name__)
CORS(app)  # Enables CORS for frontend requests

@app.post("/generate-strategy")
def generate_investment_strategy():
    """Generates an optimized investment strategy based on user input."""
    data = request.json
    risk_level = data.get("risk_level")
    timeframe = data.get("timeframe")
    initial_amount = data.get("initial_amount")
    base_currency = data.get("currency") or data.get("base_currency") or "HKD"
    return generate_strategy(
        risk_level,
        timeframe,
        initial_amount,
        base_currency=base_currency
    )

@app.route("/analyze-portfolio", methods=["POST"])
def analyze_user_portfolio():
    """Analyzes the user's custom portfolio provided in the request body (JSON or CSV)."""
    try:
        data = request.json
        base_currency = data.get("currency", "HKD")
        timeframe = int(data.get("timeframe", 6))
        portfolio_csv = data.get("portfolio_data") # Expecting a CSV-formatted string
        
        if not portfolio_csv or not portfolio_csv.strip():
            return jsonify({"error": "Portfolio data is required"}), 400

        # Handle potential header issues or empty trailing lines
        portfolio_csv = portfolio_csv.strip()
        df_holdings = pd.read_csv(StringIO(portfolio_csv))

        # Validate required columns
        required_cols = {"Ticker", "Units", "Category"}
        missing = required_cols - set(df_holdings.columns)
        if missing:
             # Try case-insensitive fallback for convenience
             col_map = {c.lower(): c for c in df_holdings.columns}
             new_df = pd.DataFrame()
             found_all = True
             for col in required_cols:
                 if col.lower() in col_map:
                     new_df[col] = df_holdings[col_map[col.lower()]]
                 else:
                     found_all = False
                     break
             if found_all:
                 df_holdings = new_df
             else:
                 return jsonify({"error": f"Malformed CSV. Missing columns: {', '.join(missing)}"}), 400

        result = run_portfolio_analysis(df_holdings, base_currency=base_currency, timeframe=timeframe)
        return jsonify(result)
    except Exception as e:
        logging.error(f"❌ Error during portfolio analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/run-verification", methods=["POST"])
def run_accuracy_verification_simulation():
    """Triggers the backend accuracy verification simulation."""
    try:
        data = request.json or {}
        holdings = data.get("holdings")
        timeframe_months = int(data.get("timeframe", 12))
        logging.info(f"Starting verification simulation with timeframe {timeframe_months}M and custom holdings..." if holdings else f"Starting verification simulation with timeframe {timeframe_months}M...")
        results = verification_service.run_simulation(custom_holdings=holdings, current_timeframe_months=timeframe_months)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error during verification: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8848, debug=True)
