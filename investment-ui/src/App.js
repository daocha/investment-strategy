import React, { useState } from "react";
import axios from "axios";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const API_URL = "http://localhost:5000/generate-strategy"; // Backend API URL

const App = () => {
  const [riskLevel, setRiskLevel] = useState("medium");
  const [timeframe, setTimeframe] = useState(6); // Default to 6 months
  const [initialAmount, setInitialAmount] = useState(10000); // Default to $10,000
  const [portfolioData, setPortfolioData] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPortfolio = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.post(API_URL, {
        risk_level: riskLevel,
        timeframe: parseInt(timeframe, 10),
        initial_amount: parseFloat(initialAmount),
      });

      setPortfolioData(response.data.portfolio_allocation);
      setBacktestResults(response.data.backtest_results);
    } catch (err) {
      setError("Error fetching portfolio. Please try again.");
    }
    setLoading(false);
  };

  const pieChartData = portfolioData
    ? {
        labels: portfolioData.map(
          (asset) => `${asset.asset} (${asset.category}) - ${asset.allocation_pct}%`
        ), // Show asset name, category, and percentage
        datasets: [
          {
            data: portfolioData.map((asset) => asset.allocation_pct),
            backgroundColor: [
              "#FF6384", "#36A2EB", "#FFCE56", "#4CAF50", "#9966FF",
              "#FF5733", "#33FFCE", "#F44336", "#2196F3", "#FFC107"
            ],
            borderWidth: 2,
            hoverOffset: 10,
          },
        ],
      }
    : null;

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Investment Strategy Builder</h1>

      <div style={{ marginBottom: "10px" }}>
        <label>Risk Level:</label>
        <select value={riskLevel} onChange={(e) => setRiskLevel(e.target.value)}>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>
      </div>

      <div style={{ marginBottom: "10px" }}>
        <label>Timeframe (months):</label>
        <input
          type="number"
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
        />
      </div>

      <div style={{ marginBottom: "10px" }}>
        <label>Initial Investment ($):</label>
        <input
          type="number"
          value={initialAmount}
          onChange={(e) => setInitialAmount(e.target.value)}
        />
      </div>

      <button onClick={fetchPortfolio} disabled={loading}>
        {loading ? "Generating..." : "Generate Portfolio"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {backtestResults && (
        <div style={{ marginTop: "20px", fontSize: "18px", fontWeight: "bold" }}>
          📈 <span style={{ color: "green" }}>Portfolio Backtested Return: {backtestResults.portfolio_annual_return.toFixed(2)}%</span>
        </div>
      )}

      {portfolioData && (
        <div style={{ width: "500px", margin: "20px auto" }}>
          <h2>Portfolio Allocation</h2>
          <Pie
            data={pieChartData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: "right",
                  labels: {
                    usePointStyle: true,
                    font: { size: 14 },
                  },
                },
                tooltip: {
                  callbacks: {
                    label: function (tooltipItem) {
                      let dataIndex = tooltipItem.dataIndex;
                      let asset = portfolioData[dataIndex];
                      return `${asset.asset} (${asset.category}): ${asset.allocation_pct}%`;
                    },
                  },
                },
              },
            }}
          />
        </div>
      )}

      {backtestResults && backtestResults.asset_returns && (
        <div style={{ marginTop: "30px" }}>
          <h2>Backtest Results</h2>
          <table style={{ width: "80%", margin: "auto", borderCollapse: "collapse", textAlign: "center" }}>
            <thead>
              <tr style={{ backgroundColor: "#f2f2f2" }}>
                <th style={{ padding: "10px", borderBottom: "2px solid #ddd" }}>Asset</th>
                <th style={{ padding: "10px", borderBottom: "2px solid #ddd" }}>Annual Return (%)</th>
                <th style={{ padding: "10px", borderBottom: "2px solid #ddd" }}>Weighted Return (%)</th>
              </tr>
            </thead>
            <tbody>
              {backtestResults.asset_returns.map((result, index) => (
                <tr key={index}>
                  <td style={{ padding: "10px", borderBottom: "1px solid #ddd" }}>{result.asset}</td>
                  <td style={{ padding: "10px", borderBottom: "1px solid #ddd" }}>{result.annual_return.toFixed(2)}%</td>
                  <td style={{ padding: "10px", borderBottom: "1px solid #ddd" }}>{result.weighted_return.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default App;
