import unittest
import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestInvestmentMath(unittest.TestCase):
    def test_geometric_compounding(self):
        """Tests the geometric return scaling formula."""
        annual_return = 0.20
        timeframe_months = 6
        timeframe_factor = timeframe_months / 12.0
        
        # Current formula used in portfolio_optimizer.py
        geometric_return = (1 + annual_return) ** timeframe_factor - 1
        
        # Expected: sqrt(1.2) - 1 approx 0.095445
        self.assertAlmostEqual(geometric_return, 0.0954451, places=6)

    def test_value_then_vs_now_logic(self):
        """Tests the aggregate portfolio return logic (Value Then vs Now)."""
        # Scenario: 
        # Asset A: Weight 50%, Return 10%
        # Asset B: Weight 50%, Return 20%
        # Total Value Then = 2000
        # Asset A then = 1000, now = 1100
        # Asset B then = 1000, now = 1200
        # Total Value Now = 2300
        # Portfolio Return = (2300 / 2000) - 1 = 15%
        
        total_value_then = 2000
        total_value_now = 2300
        
        portfolio_return = (total_value_now / total_value_then) - 1
        self.assertAlmostEqual(portfolio_return, 0.15, places=7)

    def test_fx_impact_logic(self):
        """Tests how FX rate changes impact the base currency valuation."""
        units = 100
        native_price_then = 10.0
        fx_rate_then = 7.8  # USD to HKD
        value_base_then = units * native_price_then * fx_rate_then # 7800
        
        native_price_now = 11.0 # 10% gains in native
        fx_rate_now = 7.7 # HKD strengthens (USD drops)
        value_base_now = units * native_price_now * fx_rate_now # 110 * 77 = 8470
        
        total_return_base = (value_base_now / value_base_then) - 1
        # Expected: 8470 / 7800 - 1 approx 0.08589... (approx 8.59%)
        # Note: even though native gained 10%, FX loss reduced final return
        self.assertAlmostEqual(total_return_base, 0.085897, places=6)

if __name__ == '__main__':
    unittest.main()
