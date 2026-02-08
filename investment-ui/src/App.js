import React, { useState } from "react";
import axios from "axios";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const API_BASE = `http://${window.location.hostname}:8848`;

const App = () => {

  const [riskLevel, setRiskLevel] = useState("medium");
  const [timeframe, setTimeframe] = useState(6); // Default to 6 months
  const timeframeLabel = timeframe < 12 ? `${timeframe}M` : `${timeframe / 12}Y`;
  const [initialAmount, setInitialAmount] = useState(1000000); // Default to $1,000,000
  const [isEditingAmount, setIsEditingAmount] = useState(false);
  const [selectedCurrency, setSelectedCurrency] = useState("HKD");
  const [portfolioData, setPortfolioData] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [verificationResults, setVerificationResults] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [showVerification, setShowVerification] = useState(false);
  const [errorV, setErrorV] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [portfolioInput, setPortfolioInput] = useState("");
  const [tooltip, setTooltip] = useState({ show: false, text: "", x: 0, y: 0 });

  const handleTooltip = (e, text) => {
    if (text) {
      setTooltip({ show: true, text, x: e.clientX, y: e.clientY });
    } else {
      setTooltip({ ...tooltip, show: false });
    }
  };

  const validatePortfolioCSV = (input) => {
    // Check for empty input (whitespace-only)
    if (!input || input.trim() === "") {
      return "Portfolio data cannot be empty.";
    }

    // Treat "myself" as valid (load default portfolio from file)
    if (input.trim().toLowerCase() === "myself") {
      return null;
    }

    const lines = input.trim().split("\n");
    const header = lines[0].trim().toLowerCase();

    // Check header
    if (header !== "ticker,units,category") {
      return "Invalid header format. Expected: Ticker,Units,Category";
    }

    if (lines.length < 2) {
      return "Portfolio must contain at least one data row.";
    }

    // Validate each row
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].trim();
      if (!row) continue; // skip empty lines

      const parts = row.split(",");
      if (parts.length !== 3) {
        return `Row ${i + 1} is invalid. Expected 3 columns (Ticker, Units, Category).`;
      }

      const [ticker, units, category] = parts.map(p => p.trim());

      if (!ticker) return `Row ${i + 1}: Ticker is missing.`;
      if (!category) return `Row ${i + 1}: Category is missing.`;

      const unitValue = parseFloat(units);
      if (isNaN(unitValue) || unitValue <= 0) {
        return `Row ${i + 1}: Units must be a positive number for ${ticker}.`;
      }
    }

    return null; // All good
  };

  const fetchPortfolio = async () => {
    // Basic validation for amount
    if (isNaN(initialAmount) || initialAmount <= 0) {
      setError("Initial amount must be a positive number.");
      return;
    }

    setLoading(true);
    setError("");
    setPortfolioData(null);
    setBacktestResults(null);
    setVerificationResults(null); // Clear verification results on new portfolio generation
    setShowVerification(false); // Hide verification section
    try {
      const response = await axios.post(`${API_BASE}/generate-strategy`, {
        risk_level: riskLevel,
        timeframe: parseInt(timeframe, 10),
        initial_amount: parseFloat(initialAmount),
        base_currency: selectedCurrency,
      });

      // Sort by allocation percentage descending
      const sortedAllocation = response.data.portfolio_allocation.sort((a, b) => (b.allocation_pct || 0) - (a.allocation_pct || 0));
      const sortedAssetReturns = response.data.backtest_results.asset_returns.sort((a, b) => (b.allocation_pct || 0) - (a.allocation_pct || 0));

      setPortfolioData(sortedAllocation);
      const results = {
        ...response.data.backtest_results,
        asset_returns: sortedAssetReturns
      };
      setBacktestResults(results);

      // Auto-show and auto-run verification
      setShowVerification(true);
      runVerification(sortedAssetReturns, parseInt(timeframe, 10));
    } catch (err) {
      console.error(err);
      setError("Error fetching portfolio. Please try again.");
    }
    setLoading(false);
  };

  const fetchPortfolioAnalysis = async () => {
    // Validate custom portfolio input
    const csvError = validatePortfolioCSV(portfolioInput);
    if (csvError) {
      setError(csvError);
      return;
    }

    setLoading(true);
    setError("");
    setPortfolioData(null);
    setBacktestResults(null);
    setVerificationResults(null); // Clear verification results on new portfolio analysis
    setShowVerification(false); // Hide verification section
    try {
      const response = await axios.post(`${API_BASE}/analyze-portfolio`, {
        currency: selectedCurrency,
        timeframe: parseInt(timeframe, 10),
        portfolio_data: portfolioInput.trim()
      });

      // Sort by allocation percentage descending
      const sortedAllocation = response.data.portfolio_allocation.sort((a, b) => (b.allocation_pct || 0) - (a.allocation_pct || 0));
      const sortedAssetReturns = response.data.backtest_results.asset_returns.sort((a, b) => (b.allocation_pct || 0) - (a.allocation_pct || 0));

      setPortfolioData(sortedAllocation);
      const results = {
        ...response.data.backtest_results,
        asset_returns: sortedAssetReturns
      };
      setBacktestResults(results);

      // Auto-show and auto-run verification
      setShowVerification(true);
      runVerification(sortedAssetReturns, parseInt(timeframe, 10));
    } catch (err) {
      console.error(err);
      setError("Error analyzing portfolio. Please check your input format and try again.");
    }
    setLoading(false);
  };

  const runVerification = async (holdingsData = null, tfOverride = null) => {
    const currentResults = holdingsData || (backtestResults ? backtestResults.asset_returns : null);
    const currentTf = tfOverride || timeframe;

    if (!currentResults) {
      setErrorV("No portfolio data available to verify.");
      return;
    }

    setIsVerifying(true);
    setErrorV(null);
    try {
      const holdings = currentResults.map(a => ({
        asset: a.asset,
        category: a.category,
        weight: a.allocation_pct // Assuming allocation_pct is the weight
      }));
      const res = await axios.post(`${API_BASE}/run-verification`, {
        holdings,
        timeframe: currentTf
      });
      if (res.data.error) setErrorV(res.data.error);
      else setVerificationResults(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      console.error(err);
      setErrorV("Failed to run verification simulation.");
    } finally {
      setIsVerifying(false);
    }
  };

  const pieChartData = portfolioData
    ? {
      labels: portfolioData.map(
        (asset) => `${asset.asset} (${asset.category}) - ${asset.allocation_pct || 0}%`
      ),
      datasets: [
        {
          data: portfolioData.map((asset) => asset.allocation_pct || 0),
          backgroundColor: [
            "#36A2EB", "#4CAF50", "#FFC107", "#9966FF", "#FF6384",
            "#00BCD4", "#8BC34A", "#FF9800", "#673AB7", "#F44336"
          ],
          borderWidth: 0,
          hoverOffset: 15,
        },
      ],
    }
    : null;

  const glassStyle = {
    backgroundColor: "var(--glass-bg)",
    backdropFilter: "blur(10px)",
    borderRadius: "16px",
    border: "1px solid var(--border-color)",
    padding: "24px",
    boxShadow: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
    marginBottom: "24px"
  };

  const inputStyle = {
    padding: "12px 16px",
    borderRadius: "8px",
    border: "1px solid var(--border-color)",
    backgroundColor: "var(--surface-color-light)",
    color: "var(--text-primary)",
    fontSize: "14px",
    outline: "none",
    width: "100%",
    maxWidth: "400px"
  };

  const buttonStyle = (type, disabled) => ({
    padding: "12px 24px",
    backgroundColor: disabled ? "var(--surface-color-light)" : (type === "primary" ? "var(--accent-blue)" : "var(--accent-green)"),
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: disabled ? "not-allowed" : "pointer",
    fontSize: "16px",
    fontWeight: "600",
    transition: "all 0.3s ease",
    opacity: disabled ? 0.5 : 1,
    boxShadow: disabled ? "none" : "0 4px 14px 0 rgba(0,0,0,0.39)"
  });

  return (
    <div style={{ maxWidth: "1200px", margin: "auto", padding: "40px 20px" }}>
      <h1 style={{ fontSize: "32px", fontWeight: "800", marginBottom: "40px", textAlign: "center", background: "linear-gradient(90deg, #36A2EB, #4CAF50)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
        Investment Strategy Builder
      </h1>

      <div style={glassStyle}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "24px", marginBottom: "32px" }}>
          <div>
            <label style={{ display: "block", marginBottom: "8px", color: "var(--text-secondary)", fontSize: "14px" }}>Risk Level</label>
            <select value={riskLevel} onChange={(e) => setRiskLevel(e.target.value)} style={inputStyle}>
              <option value="low">Low Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="high">High Risk</option>
            </select>
          </div>

          <div>
            <label style={{ display: "block", marginBottom: "8px", color: "var(--text-secondary)", fontSize: "14px" }}>Timeframe (Months)</label>
            <input
              type="number"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              style={inputStyle}
            />
          </div>

          <div>
            <label style={{ display: "block", marginBottom: "8px", color: "var(--text-secondary)", fontSize: "14px" }}>
              Initial Investment ({selectedCurrency})
            </label>
            <input
              type="text"
              value={isEditingAmount ? initialAmount : parseFloat(initialAmount || 0).toLocaleString()}
              onFocus={() => setIsEditingAmount(true)}
              onBlur={() => setIsEditingAmount(false)}
              onChange={(e) => {
                const val = e.target.value.replace(/[^0-9.]/g, "");
                setInitialAmount(val);
              }}
              style={inputStyle}
            />
          </div>

          <div>
            <label style={{ display: "block", marginBottom: "8px", color: "var(--text-secondary)", fontSize: "14px" }}>Base Currency</label>
            <select value={selectedCurrency} onChange={(e) => setSelectedCurrency(e.target.value)} style={inputStyle}>
              <option value="HKD">HKD (HK$)</option>
              <option value="USD">USD ($)</option>
              <option value="TWD">TWD (NT$)</option>
              <option value="EUR">EUR (€)</option>
            </select>
          </div>
        </div>

        <div style={{ marginBottom: "32px" }}>
          <label style={{ display: "block", marginBottom: "8px", color: "var(--text-secondary)", fontSize: "14px" }}>Portfolio Data (Ticker, Units, Category)</label>
          <textarea
            value={portfolioInput}
            onChange={(e) => setPortfolioInput(e.target.value)}
            rows="6"
            style={{ ...inputStyle, maxWidth: "100%", width: "100%", fontFamily: "monospace", resize: "vertical" }}
            placeholder={`Type "myself" to load your saved portfolio, or enter CSV:
Ticker,Units,Category
BTC,1.5,Crypto
AAPL,10,Stocks`}
          />
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "8px" }}>
            💡 <strong>Tip:</strong> Type <code>myself</code> above to automatically load from <code>my_portfolio.csv</code>.
          </p>
        </div>

        <div style={{ display: "flex", gap: "16px", justifyContent: "center" }}>
          <button onClick={fetchPortfolio} disabled={loading} style={buttonStyle("primary", loading)}>
            {loading ? "Generating..." : "Generate Optimal"}
          </button>
          <button onClick={fetchPortfolioAnalysis} disabled={loading} style={buttonStyle("secondary", loading)}>
            {loading ? "Analysing..." : "Analyse Custom"}
          </button>
        </div>
      </div>

      {error && <div style={{ padding: "16px", borderRadius: "8px", backgroundColor: "rgba(244, 67, 54, 0.1)", border: "1px solid var(--accent-red)", color: "var(--accent-red)", marginBottom: "24px", textAlign: "center" }}>{error}</div>}

      {loading && (
        <div className="loading-container">
          <svg className="btc-icon" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="32" cy="32" r="32" fill="#F7931A" />
            <path d="M44.662 25.602c.515-3.444-2.108-5.297-5.694-6.533l1.162-4.662-2.838-.707-1.13 4.534c-.745-.185-1.511-.36-2.274-.531l1.137-4.563-2.839-.707-1.162 4.661c-.618-.141-1.22-.278-1.802-.423l.002-.007-3.916-.977-.756 3.033s2.107.483 2.063.513c1.15.287 1.357 1.047 1.323 1.65l-1.325 5.31c.079.02.182.049.295.093l-.297-.074-1.856 7.443c-.14.349-.496.872-1.3.673.029.042-2.063-.514-2.063-.514l-1.41 3.252 3.696.921c.683.172 1.353.349 2.012.518l-1.171 4.699 2.837.707 1.162-4.66c.775.21 1.527.41 2.261.6l-1.158 4.646 2.84.708 1.17-4.697c4.84.918 8.483.548 10.015-3.83 1.235-3.526-.06-5.558-2.598-6.88 1.849-.427 3.24-1.643 3.61-4.14zm-6.46 9.055c-.878 3.522-6.812 1.618-8.735 1.14l1.558-6.248c1.923.479 8.086 1.428 7.177 5.108zm.882-9.103c-.8 3.212-5.742 1.58-7.34 1.182l1.412-5.666c1.598.399 6.749 1.144 5.928 4.484z" fill="white" />
          </svg>
          <div style={{ color: "var(--text-primary)", fontSize: "18px", fontWeight: "600" }}>
            Processing Your Strategy...
          </div>
          <div style={{ color: "var(--text-secondary)", fontSize: "14px", marginTop: "8px" }}>
            Analyzing historical patterns & AI signals
          </div>
        </div>
      )}

      {backtestResults && (
        <>

          {/* Summary & Chart Section - Split Layout */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "24px", alignItems: "stretch", marginBottom: "40px" }}>

            {/* Left Column: Metrics */}
            <div style={{ flex: "1", minWidth: "300px", display: "flex", flexDirection: "column", gap: "24px" }}>
              <div style={glassStyle}>
                <p style={{ margin: "0 0 8px 0", color: "var(--text-secondary)", fontSize: "14px" }}>Total Portfolio Valuation</p>
                <h2 style={{ margin: 0, fontSize: "32px", fontWeight: "700" }}>
                  {backtestResults.base_symbol}{backtestResults.total_valuation?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || "0.00"}
                </h2>
                {backtestResults.portfolio_confidence && (
                  <div style={{ marginTop: "12px", height: "4px", backgroundColor: "var(--surface-color-light)", borderRadius: "2px" }}>
                    <div style={{ height: "100%", width: `${(backtestResults.portfolio_confidence * 100).toFixed(0)}%`, backgroundColor: backtestResults.portfolio_confidence > 0.7 ? "var(--accent-green)" : "orange", borderRadius: "2px" }} />
                    <span style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px", display: "block" }}>Confidence: {(backtestResults.portfolio_confidence * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>

              <div style={glassStyle}>
                <p style={{ margin: "0 0 12px 0", color: "var(--text-secondary)", fontSize: "14px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>Performance Summary</p>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
                  <span
                    style={{ cursor: "help", borderBottom: "1px dashed var(--text-secondary)" }}
                    onMouseEnter={(e) => handleTooltip(e, `Weighted average of all assets' predicted returns for the selected ${timeframeLabel} timeframe.`)}
                    onMouseLeave={() => handleTooltip(null, null)}
                    onMouseMove={(e) => handleTooltip(e, `Weighted average of all assets' predicted returns for the selected ${timeframeLabel} timeframe.`)}
                  >
                    Predicted Return ({timeframeLabel}):
                  </span>
                  <span style={{ fontWeight: "700", color: (backtestResults.portfolio_predicted_return || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                    {(backtestResults.portfolio_predicted_return || 0).toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px" }}>
                  <span
                    style={{ cursor: "help", borderBottom: "1px dashed var(--text-secondary)" }}
                    onMouseEnter={(e) => handleTooltip(e, `Historical performance of this portfolio over the selected ${timeframeLabel} timeframe.`)}
                    onMouseLeave={() => handleTooltip(null, null)}
                    onMouseMove={(e) => handleTooltip(e, `Historical performance of this portfolio over the selected ${timeframeLabel} timeframe.`)}
                  >
                    Backtest Return ({timeframeLabel}):
                  </span>
                  <span style={{ fontWeight: "700", color: (backtestResults.portfolio_backtest_return || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                    {(backtestResults.portfolio_backtest_return || 0).toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", borderTop: "1px solid var(--border-color)", paddingTop: "10px" }}>
                  <span
                    style={{ cursor: "help", borderBottom: "1px dashed var(--text-secondary)" }}
                    onMouseEnter={(e) => handleTooltip(e, "Projected financial gain in your base currency, calculated as: Total Valuation × Predicted Return.")}
                    onMouseLeave={() => handleTooltip(null, null)}
                    onMouseMove={(e) => handleTooltip(e, "Projected financial gain in your base currency, calculated as: Total Valuation × Predicted Return.")}
                  >
                    Expected Profit:
                  </span>
                  <span style={{ fontWeight: "700", color: (backtestResults.portfolio_predicted_profit || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                    {backtestResults.base_symbol}{backtestResults.portfolio_predicted_profit?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || "0.00"}
                  </span>
                </div>
              </div>

              {backtestResults.benchmark_results && (
                <div style={glassStyle}>
                  <p style={{ margin: "0 0 8px 0", color: "var(--text-secondary)", fontSize: "14px" }}>Benchmark (S&P 500)</p>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                    <span>Predicted:</span>
                    <span style={{ fontWeight: "700" }}>{(backtestResults.benchmark_results.predicted_return || 0).toFixed(2)}%</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Backtest:</span>
                    <span style={{ fontWeight: "700" }}>{(backtestResults.benchmark_results.backtest_annual_return || 0).toFixed(2)}%</span>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Chart */}
            <div style={{ flex: "1", minWidth: "300px", ...glassStyle, display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <h2 style={{ fontSize: "20px", marginBottom: "20px", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px", textAlign: "center" }}>Allocation Strategy</h2>
              <div style={{ height: "300px", position: "relative" }}>
                {portfolioData && <Pie data={pieChartData} options={{ maintainAspectRatio: false }} />}
              </div>
            </div>

          </div>






          {/* Detailed Allocation Table */}
          {portfolioData && (
            <div style={{ ...glassStyle, overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-color)", color: "var(--text-secondary)", fontSize: "12px", textTransform: "uppercase", letterSpacing: "1px" }}>
                    <th style={{ padding: "16px 8px" }}>Asset</th>
                    <th style={{ padding: "16px 8px" }}>Category</th>
                    <th style={{ padding: "16px 8px" }}>Units</th>
                    <th style={{ padding: "16px 8px" }}>Price</th>
                    <th style={{ padding: "16px 8px" }}>Value ({selectedCurrency})</th>
                    <th style={{ padding: "16px 8px" }}>Alloc. %</th>
                    <th
                      style={{ padding: "16px 8px", cursor: "help" }}
                      onMouseEnter={(e) => handleTooltip(e, "Predicted monetary gain based on capital allocated.")}
                      onMouseLeave={() => handleTooltip(null, null)}
                    >Pred. Profit</th>
                    <th
                      style={{ padding: "16px 8px", cursor: "help" }}
                      onMouseEnter={(e) => handleTooltip(e, "**The Driver**. This is the AI's raw forecast for this specific asset. The Total Portfolio Predicted Return is the weighted sum of these values.")}
                      onMouseLeave={() => handleTooltip(null, null)}
                    >Est. Return</th>
                    <th
                      style={{ padding: "16px 8px", cursor: "help" }}
                      onMouseEnter={(e) => handleTooltip(e, "**Reality Check**. How this asset actually performed over the past timeframe. Calculated as: (Current Price / Price N Months Ago) - 1.")}
                      onMouseLeave={() => handleTooltip(null, null)}
                    >Backtest Return</th>
                    <th
                      style={{ padding: "16px 8px", cursor: "help" }}
                      onMouseEnter={(e) => handleTooltip(e, "**Hybrid**. A secondary metric aligning historical data with future predictions. Not used for the main 'Predicted Profit' calculation.")}
                      onMouseLeave={() => handleTooltip(null, null)}
                    >Combined</th>
                    <th
                      style={{ padding: "16px 8px", cursor: "help", borderLeft: "1px solid var(--border-color)" }}
                      onMouseEnter={(e) => handleTooltip(e, "**Contribution (Future)**. The actual percentage this asset adds to your Total Portfolio Predicted Return. (Est. Return × Allocation %).")}
                      onMouseLeave={() => handleTooltip(null, null)}
                    >Weighted Forecast</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolioData.map((asset, index) => {
                    const assetBacktest = backtestResults?.asset_returns?.find(r => r.asset === asset.asset) || {};
                    const weightedForecast = (assetBacktest.predicted_return || 0) * (asset.allocation_pct || 0) / 100;

                    return (
                      <tr key={index} style={{ borderBottom: "1px solid var(--border-color)" }}>
                        <td style={{ padding: "16px 8px", fontWeight: "600" }}>
                          <span
                            style={{ cursor: "help", borderBottom: "1px dashed var(--text-secondary)" }}
                            onMouseEnter={(e) => handleTooltip(e, asset.asset_name || asset.asset)}
                            onMouseLeave={() => handleTooltip(null, null)}
                            onMouseMove={(e) => handleTooltip(e, asset.asset_name || asset.asset)}
                          >
                            {asset.asset}
                          </span>
                        </td>
                        <td style={{ padding: "16px 8px" }}>
                          <span style={{
                            padding: "4px 8px",
                            borderRadius: "12px",
                            backgroundColor: asset.category === "Crypto" ? "rgba(247, 147, 26, 0.2)" : "rgba(33, 150, 243, 0.2)",
                            color: asset.category === "Crypto" ? "#F7931A" : "#2196F3",
                            fontSize: "12px",
                            fontWeight: "500"
                          }}>
                            {asset.category}
                          </span>
                        </td>
                        <td style={{ padding: "16px 8px" }}>{asset.units?.toFixed(4) || "0.0000"}</td>
                        <td style={{ padding: "16px 8px" }}>{asset.native_symbol}{asset.current_price_native?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || "0.00"}</td>
                        <td style={{ padding: "16px 8px" }}>{backtestResults?.base_symbol}{(asset.allocation || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                        <td style={{ padding: "16px 8px" }}>{asset.allocation_pct?.toFixed(2)}%</td>
                        <td style={{ padding: "16px 8px", color: (assetBacktest.predicted_profit || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)", fontWeight: "600" }}>
                          {backtestResults?.base_symbol}{assetBacktest.predicted_profit?.toLocaleString(undefined, { minimumFractionDigits: 2 }) || "0.00"}
                        </td>
                        <td style={{ padding: "16px 8px", color: (assetBacktest.predicted_return || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {(assetBacktest.predicted_return || 0).toFixed(2)}%
                        </td>
                        <td style={{ padding: "16px 8px", color: (assetBacktest.backtest_annual_return || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {(assetBacktest.backtest_annual_return || 0).toFixed(2)}%
                        </td>
                        <td style={{ padding: "16px 8px", color: (assetBacktest.combined_return || 0) >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {(assetBacktest.combined_return || 0).toFixed(2)}%
                        </td>
                        <td style={{ padding: "16px 8px", fontWeight: "bold", borderLeft: "1px solid var(--border-color)", color: weightedForecast >= 0 ? "var(--accent-green)" : "var(--accent-red)" }}>
                          {weightedForecast.toFixed(2)}%
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {
        tooltip.show && (
          <div style={{
            position: "fixed",
            top: tooltip.y + 15,
            left: tooltip.x + 15,
            backgroundColor: "#1e2229",
            color: "white",
            padding: "8px 12px",
            borderRadius: "6px",
            fontSize: "12px",
            zIndex: 10000,
            pointerEvents: "none",
            maxWidth: "250px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            border: "1px solid rgba(255,255,255,0.1)",
            lineHeight: "1.4"
          }}>
            {tooltip.text}
          </div>
        )}

      {/* Accuracy Verification Section */}
      {showVerification && (
        <div style={{ ...glassStyle, marginTop: "40px", marginBottom: "40px" }}>
          <h2 style={{ fontSize: "20px", marginBottom: "12px", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px" }}>
            Validation: Accuracy of Current Selection (Time-Travel)
          </h2>
          <p style={{ color: "var(--text-secondary)", marginBottom: "20px", fontSize: "14px", lineHeight: "1.6" }}>
            Test how the <strong>currently displayed portfolio</strong> would have performed in the past.
            This simulates running the strategy with these specific assets at past dates to verify the model's reliability for this specific selection.
          </p>

          {isVerifying && (
            <div style={{ marginBottom: "20px", color: "var(--accent-blue)", fontSize: "14px", display: "flex", alignItems: "center", gap: "10px", padding: "12px", background: "rgba(0,122,255,0.05)", borderRadius: "8px", border: "1px solid rgba(0,122,255,0.1)" }}>
              <div style={{ width: "12px", height: "12px", border: "2px solid var(--accent-blue)", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
              Simulating historical scenarios for this specific portfolio...
            </div>
          )}

          {errorV && <div style={{ color: "var(--accent-red)", marginTop: "16px", padding: "12px", background: "rgba(255, 59, 48, 0.1)", borderRadius: "8px" }}>{errorV}</div>}

          {Array.isArray(verificationResults) && verificationResults.length > 0 && (
            <div style={{ marginTop: "24px", overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "14px" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-color)", color: "var(--text-secondary)", textTransform: "uppercase", fontSize: "12px" }}>
                    <th style={{ padding: "12px 8px" }}>Timeframe</th>
                    <th style={{ padding: "12px 8px" }}>Simulated Date</th>
                    <th style={{ padding: "12px 8px" }}>Holdings</th>
                    <th style={{ padding: "12px 8px" }}>Predicted Return</th>
                    <th style={{ padding: "12px 8px" }}>Actual (Realized)</th>
                    <th style={{ padding: "12px 8px" }}>Deviation</th>
                    <th style={{ padding: "12px 8px" }}>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {verificationResults.map((res, idx) => {
                    const isSelectedTimeframe = res.timeframe === (timeframe < 12 ? `${timeframe}M` : `${timeframe / 12}Y`);
                    return (
                      <tr key={idx} style={{
                        borderBottom: "1px solid var(--border-color)",
                        background: isSelectedTimeframe ? "rgba(0, 122, 255, 0.08)" : "transparent",
                        boxShadow: isSelectedTimeframe ? "inset 4px 0 0 var(--accent-blue)" : "none"
                      }}>
                        <td style={{ padding: "12px 8px", fontWeight: "600", color: isSelectedTimeframe ? "var(--accent-blue)" : "inherit" }}>
                          {res.timeframe} Ago {isSelectedTimeframe && "🎯"}
                        </td>
                        <td style={{ padding: "12px 8px", color: "var(--text-secondary)" }}>{res.simulated_date || "-"}</td>
                        <td style={{ padding: "12px 8px" }}>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                            {res.holdings?.map((h, i) => (
                              <span key={i} style={{
                                fontSize: "10px",
                                background: h.skipped ? "rgba(255,59,48,0.05)" : "rgba(255,255,255,0.1)",
                                padding: "2px 6px",
                                borderRadius: "4px",
                                border: h.skipped ? "1px dashed var(--accent-red)" : "1px solid var(--border-color)",
                                textDecoration: h.skipped ? "line-through" : "none",
                                color: h.skipped ? "var(--text-secondary)" : "inherit",
                                opacity: h.skipped ? 0.7 : 1
                              }}>
                                {h.skipped ? "❌ " : ""}{h.asset} ({h.weight}%)
                              </span>
                            ))}
                          </div>
                        </td>
                        <td style={{ padding: "12px 8px" }}>{res.predicted_return}%</td>
                        <td style={{ padding: "12px 8px" }}>{res.actual_return}%</td>
                        <td style={{ padding: "12px 8px" }}>{res.deviation}%</td>
                        <td style={{ padding: "12px 8px" }}>
                          {res.error ? (
                            <span style={{ color: "var(--accent-red)" }}>Error: {res.error}</span>
                          ) : res.deviation < 5 ? (
                            <span
                              style={{ color: "var(--accent-green)", fontWeight: "bold", cursor: "help" }}
                              onMouseEnter={(e) => handleTooltip(e, "Accuracy in both Direction AND Magnitude (within 5% gap).")}
                              onMouseLeave={() => handleTooltip(null)}
                            >✅ Precise Hit</span>
                          ) : res.is_direction_correct ? (
                            <span
                              onMouseEnter={(e) => handleTooltip(e, "The AI correctly predicted the market direction (Logic). Use this to verify if the 'Buy/Sell' decision was correct.")}
                              onMouseLeave={() => handleTooltip(null)}
                              style={{ color: "var(--accent-blue)", fontWeight: "500", cursor: "help" }}
                            >
                              🧠 Correct Trend
                            </span>
                          ) : res.predicted_return < res.actual_return ? (
                            <span
                              style={{ color: "var(--accent-yellow)", cursor: "help" }}
                              onMouseEnter={(e) => handleTooltip(e, "The model was conservative; actual returns were higher than predicted.")}
                              onMouseLeave={() => handleTooltip(null)}
                            >📈 Trend Miss (Under)</span>
                          ) : (
                            <span
                              style={{ color: "var(--accent-red)", cursor: "help" }}
                              onMouseEnter={(e) => handleTooltip(e, "The model was optimistic; actual returns were lower than predicted.")}
                              onMouseLeave={() => handleTooltip(null)}
                            >❌ Trend Miss (Over)</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default App;
