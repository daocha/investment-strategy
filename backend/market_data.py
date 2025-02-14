import requests
import re
import logging
import yfinance as yf

DEEPSEEK_API_KEY = "sk-50993c19c1854c13a050b994b7fdfbd7"

def get_trending_assets():
    """Uses DeepSeek AI to find trending assets with strong BUY indicators."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "I need a list of the most promising investment assets across stocks, ETFs, and cryptocurrencies.  Only include assets with strong BUY indicators based on current market sentiment, technical strength, capital inflows, and analyst ratings. Ensure the list is diverse (technology, financials, consumer, crypto, etc.), and large enough for proper portfolio selection (~100-200 assets).  For the technology & crypto sector i am expecting at least 50 options respectively. Format the response as a comma-separated list of tickers."
                )
            }
        ]
    }

    # temporarily disable deepseek api, use manual way
    #response = requests.post(url, json=data, headers=headers)

    #assets_response = response.json()["choices"][0]["message"]["content"]
    assets_response = "Stocks\nTechnology: NVDA, TSLA, MSFT, AMD, AVGO, ASML, CRM, SNPS, NOW, PANW, NET, GOOGL, AMZN, META, PLTR, SMCI, SNOW, DBRG, AI, TSM, QCOM, LRCX, KLAC, MRVL, TER, AMAT, ADBE, ORCL, INTU, ZS, CRWD, OKTA, DDOG, TEAM, DOCU, SQ, SHOP, U, ROKU, TWLO, ZM, PYPL, ABNB, SPOT, NFLX, DIS, MU, QRVO, SWKS, TXN, ON\nFinancials: JPM, BAC, BLK, GS, MS, SPGI, ICE, PNC, C, WFC, SCHW, AXP, COF, USB, DFS, MET, PRU, AON, MSCI, RJF\nCrypto: COIN, MSTR, BITO, GBTC, BLOK, RIOT, MARA, HIVE, BITF, HUT, BTBT, SI, BKCH, LEGR, DAPP, BLCN, KOIN, CRPT, SATO, DEFI, BITQ, BIDS, SPBC, BTF, XBTF, BITI, BTCR, BTCC, EBIT, BTCX, BTCQ, BTCY, BTCZ, BTCB, BTCE, GLXY, HVBTF, WGMI, SDIG, CLSK, CIFR, IREN, GRIID, BTDR, ARBK, CORZ, GREE, DGHI, CIFR\nHealthcare: LLY, UNH, JNJ, MRK, PFE, ABBV, AMGN, VRTX, REGN, ISRG, DXCM, MRNA, TDOC, BIIB, GILD, DHR, BDX, ZTS, ILMN, HUM\nConsumer: COST, PG, KO, PEP, SBUX, MCD, NKE, TGT, HD, LOW, EL, LULU, TSN, KR, DG, DLTR, CMG, YUM, DPZ, HAS\nEnergy: XOM, CVX, COP, SLB, EOG, MPC, LNG, FANG, DVN, OXY, PSX, VLO, HAL, EQT, PXD\nIndustrials: CAT, DE, UNP, HON, RTX, LMT, GE, BA, GD, NOC, WM, RSG, FAST, CMI, EMR\nMaterials: LIN, DOW, NEM, FCX, APD, SHW, ECL, PPG, AA, CF\n\nETFs\nTech: QQQ, XLK, VGT, SMH, SOXX, BOTZ, AIQ, CIBR, HACK, SKYY, CLOU, WCLD, ARKK, IGV, FINX\nSector: XLF, XLV, XLE, XLI, XLY, XLB, XLU, XLRE, XBI, IBB\nDividends: SCHD, VIG, DVY, SPHD, NOBL, SDY\nInternational: EEM, VWO, IEMG, FLKR, EWY, EWT\nThematic: ICLN, TAN, PBW, LIT, BLOK, BUG, DRIV, BETZ\n\nCryptocurrencies\nLarge-Cap: BTC, ETH, SOL, ADA\nMeme/Gaming: PEPE, BONK"
    #assets_response = "Stocks\nTechnology: NVDA, MSFT, TSLA, PLTR, MSTR, GOOG, META, AAPL, BABA, ASML, TSM\n\nETFs\nTech: QQQ, VOO, SQQQ, TQQQ\n\nCryptocurrencies\nLarge-Cap: BTC, ETH, SOL, PEPE, SUI"

    assets_json = extract_tickers(assets_response)
    print(assets_json)

    # Extract tickers from DeepSeek AI response
    return assets_json

def extract_tickers(response_text):
    """Parses DeepSeek AI response and extracts structured asset categories with tickers."""
    sections = ["Stocks", "ETFs", "Cryptocurrencies"]
    asset_data = {"Stocks": {}, "ETFs": {}, "Cryptocurrencies": {}}

    current_section = None

    # Process each line
    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Identify section headers
        if line in sections:
            current_section = line
            continue

        # Extract category and tickers
        match = re.match(r"(.+?):\s*([\w,\s]+)", line)
        if match and current_section:
            category = match.group(1).strip()
            tickers = [ticker.strip() for ticker in match.group(2).split(",")]
            asset_data[current_section][category] = tickers

    return asset_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_market_data():
    """
    Fetches real-time market data for trending stocks, ETFs, and cryptos recommended by DeepSeek AI.
    """
    logging.info("Fetching trending assets from DeepSeek AI...")
    trending_assets = get_trending_assets()  # Get structured asset list from DeepSeek AI
    market_data = {"Stocks": {}, "ETFs": {}, "Cryptocurrencies": {}}

    def fetch_yfinance_data(ticker):
        """Fetch latest price & key data for a given ticker using Yahoo Finance."""
        try:
            logging.info(f"[Yahoo Finance] Fetching market data for {ticker} (Stock/ETF)...")
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            if not history.empty:
                logging.info(f"Successfully fetched data for {ticker}.")
                return {
                    "Close": history["Close"].iloc[-1],
                    "Volume": history["Volume"].iloc[-1],
                    "Market Cap": stock.info.get("marketCap", "N/A"),
                    "52-Week High": stock.info.get("fiftyTwoWeekHigh", "N/A"),
                    "52-Week Low": stock.info.get("fiftyTwoWeekLow", "N/A")
                }
            else:
                logging.warning(f"No historical data found for {ticker}. Skipping...")
        except Exception as e:
            logging.error(f"Error fetching {ticker}: {e}")
        return None

    def fetch_crypto_data(ticker):
        """Fetch crypto market data from Binance API."""
        try:
            logging.info(f"[Binance] Fetching market data for {ticker} (Cryptocurrency)...")
            response = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={ticker}USDT").json()
            if "lastPrice" in response:
                logging.info(f"Successfully fetched data for {ticker}.")
                return {
                    "Close": float(response["lastPrice"]),
                    "Volume": float(response["quoteVolume"]),
                    "24h Change": float(response["priceChangePercent"]),
                    "Market Cap": "N/A"
                }
            else:
                logging.warning(f"Unexpected response format for {ticker}: {response}")
        except Exception as e:
            logging.error(f"Error fetching {ticker}: {e}")
        return None

    # Fetch Stocks & ETFs data
    for category, tickers in trending_assets["Stocks"].items():
        logging.info(f"Fetching Stocks in category: {category}...")
        for ticker in tickers:
            data = fetch_yfinance_data(ticker)
            if data:
                market_data["Stocks"][ticker] = data

    for category, tickers in trending_assets["ETFs"].items():
        logging.info(f"Fetching ETFs in category: {category}...")
        for ticker in tickers:
            data = fetch_yfinance_data(ticker)
            if data:
                market_data["ETFs"][ticker] = data

    # Fetch Cryptocurrencies data
    for category, tickers in trending_assets["Cryptocurrencies"].items():
        logging.info(f"Fetching Cryptocurrencies in category: {category}...")
        for ticker in tickers:
            data = fetch_crypto_data(ticker)
            if data:
                market_data["Cryptocurrencies"][ticker] = data

    logging.info("Finished fetching all market data.")
    #print(market_data)
    return market_data
