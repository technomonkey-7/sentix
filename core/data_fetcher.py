import ccxt
import os
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime, timedelta
import random
import time
from core.db import log_event, get_config

def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    """
    Fetches OHLCV candlestick data for a symbol and timeframe using ccxt.
    In LIVE_MODE (configurable), API failures raise exceptions instead of falling back to simulated data.
    In simulation mode, provides automatic fallback to synthetic data if network fails.
    """
    live_mode = get_config("live_mode", "false").lower() == "true"
    
    try:
        log_event("INFO", "DATA_FETCHER", f"Fetching {limit} candles for {symbol} ({timeframe}) from CCXT...")
        # Load configured exchange dynamically to bypass geographic/provider restrictions
        exchange_name = get_config("exchange_name") or os.getenv("EXCHANGE_NAME", "binance")
        exchange_class = getattr(ccxt, exchange_name.lower(), ccxt.binance)
        
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        # Add proxy configuration if defined in env
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy or https_proxy:
            exchange_config['proxies'] = {}
            if http_proxy:
                exchange_config['proxies']['http'] = http_proxy
            if https_proxy:
                exchange_config['proxies']['https'] = https_proxy
                
        exchange = exchange_class(exchange_config)
        
        # Call fetch_ohlcv
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) == 0:
            raise ValueError("Empty candle data returned from exchange.")
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Convert timestamp (ms) to ISO string or formatted string
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        df['_is_simulated'] = False
        log_event("SUCCESS", "DATA_FETCHER", f"Successfully fetched {len(df)} candles for {symbol}.")
        return df

    except Exception as e:
        if live_mode:
            log_event("ERROR", "DATA_FETCHER", f"CCXT fetch FAILED for {symbol} in LIVE MODE: {e}. Refusing to fall back to simulated data.")
            raise RuntimeError(f"LIVE MODE: Cannot fetch real market data for {symbol}. API error: {e}")
        
        log_event("WARNING", "DATA_FETCHER", f"CCXT fetch failed for {symbol}: {e}. Generating simulated fallback data (NOT REAL PRICES).")
        return generate_simulated_ohlcv(symbol, timeframe, limit)

def generate_simulated_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    """
    Generates synthetic but highly realistic OHLCV data using a random walk with drift.
    Ensures that the trading engine can always run and display charts, even offline.
    """
    # Base prices for assets to keep simulations realistic
    base_prices = {
        "BTC/USDT": 65000.0,
        "ETH/USDT": 3400.0,
        "SOL/USDT": 150.0,
        "BNB/USDT": 580.0,
        "XRP/USDT": 0.50,
        "ADA/USDT": 0.45,
        "DOGE/USDT": 0.15,
        "AVAX/USDT": 35.0,
        "LINK/USDT": 15.0,
        "DOT/USDT": 6.50,
        "NEAR/USDT": 6.0,
        "MATIC/USDT": 0.70,
        "SUI/USDT": 1.50,
        "APT/USDT": 9.00,
        "OP/USDT": 2.50,
        "ARB/USDT": 1.00,
        "LTC/USDT": 85.0,
        "TRX/USDT": 0.12,
        "XLM/USDT": 0.14,
        "UNI/USDT": 7.50,
        "ATOM/USDT": 8.0,
        "INJ/USDT": 25.0,
        "TIA/USDT": 5.0,
        "GRT/USDT": 0.25,
        "FET/USDT": 1.80,
        "RNDR/USDT": 8.0,
        "SHIB/USDT": 0.00002,
        "ETC/USDT": 28.0,
        "FIL/USDT": 5.50,
        "ICP/USDT": 12.0
    }
    base_price = base_prices.get(symbol, 100.0)
    
    # Time delta based on timeframe
    now = datetime.utcnow()
    delta = timedelta(hours=1) if timeframe == "1h" else timedelta(hours=4)
    
    timestamps = []
    for i in range(limit):
        t = now - (limit - i) * delta
        timestamps.append(t.strftime('%Y-%m-%d %H:%M:%S'))
        
    opens, highs, lows, closes, volumes = [], [], [], [], []
    current_price = base_price * (1 + random.uniform(-0.05, 0.05)) # Random start price
    
    for _ in range(limit):
        pct_change = random.normalvariate(0.0005, 0.008) # Small positive drift, volatility
        open_price = current_price
        close_price = current_price * (1 + pct_change)
        
        # Volatility noise for high and low
        high_price = max(open_price, close_price) * (1 + abs(random.normalvariate(0, 0.004)))
        low_price = min(open_price, close_price) * (1 - abs(random.normalvariate(0, 0.004)))
        
        volume = random.uniform(10, 500) if symbol != "BTC/USDT" else random.uniform(100, 2000)
        
        opens.append(round(open_price, 2))
        highs.append(round(high_price, 2))
        lows.append(round(low_price, 2))
        closes.append(round(close_price, 2))
        volumes.append(round(volume, 2))
        
        current_price = close_price
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    df['_is_simulated'] = True
    return df

def fetch_asset_news(symbol="BTC/USDT", limit=10):
    """
    Scrapes the latest news articles for a symbol using the free Google News RSS Feed.
    Extracts article title, snippet, and link, returning them as a clean list of dicts.
    """
    asset_name = symbol.split('/')[0]
    # Standard names for better search relevance
    search_keywords = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana",
        "BNB": "Binance Coin BNB",
        "XRP": "Ripple XRP",
        "ADA": "Cardano ADA",
        "DOGE": "Dogecoin DOGE",
        "AVAX": "Avalanche AVAX",
        "LINK": "Chainlink LINK",
        "DOT": "Polkadot DOT",
        "NEAR": "Near Protocol NEAR",
        "MATIC": "Polygon MATIC",
        "SUI": "Sui Network SUI",
        "APT": "Aptos APT",
        "OP": "Optimism OP",
        "ARB": "Arbitrum ARB",
        "LTC": "Litecoin LTC",
        "TRX": "Tron TRX",
        "XLM": "Stellar Lumens XLM",
        "UNI": "Uniswap UNI",
        "ATOM": "Cosmos ATOM",
        "INJ": "Injective INJ",
        "TIA": "Celestia TIA",
        "GRT": "The Graph GRT",
        "FET": "Fetch.ai FET artificial superintelligence",
        "RNDR": "Render Token RNDR",
        "SHIB": "Shiba Inu SHIB",
        "ETC": "Ethereum Classic ETC",
        "FIL": "Filecoin FIL",
        "ICP": "Internet Computer ICP"
    }
    query = search_keywords.get(asset_name, asset_name) + " crypto cryptocurrency market"
    encoded_query = urllib.parse.quote(query)
    
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    log_event("INFO", "DATA_FETCHER", f"Scraping live RSS news for {symbol} using Google News feed...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            raise requests.RequestException(f"Invalid HTTP status: {response.status_code}")
            
        root = ET.fromstring(response.content)
        import email.utils
        from datetime import datetime, timezone
        
        all_items = []
        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            description = item.find('description').text if item.find('description') is not None else ""
            
            # Clean up the RSS-injected source information from the title if present
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
                
            # Parse publication date for sorting
            parsed_dt = None
            if pub_date:
                try:
                    parsed_dt = email.utils.parsedate_to_datetime(pub_date)
                except Exception:
                    pass
            
            # If parsing failed, default to a safe old date to push to bottom
            # (datetime.min can cause OverflowError on some platforms with timezone ops)
            if not parsed_dt:
                parsed_dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
                    
            all_items.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "parsed_dt": parsed_dt,
                "description": description
            })
            
        # Sort news_items by parsed_dt in descending order (newest first)
        all_items.sort(key=lambda x: x["parsed_dt"], reverse=True)
        
        # Format the final limited list
        news_items = []
        for item in all_items[:limit]:
            news_items.append({
                "title": item["title"],
                "link": item["link"],
                "pub_date": item["pub_date"],
                "description": item["description"]
            })
            
        log_event("SUCCESS", "DATA_FETCHER", f"Scraped {len(news_items)} chronologically sorted news articles for {symbol}.")
        return news_items
        
    except Exception as e:
        log_event("WARNING", "DATA_FETCHER", f"Failed to scrape RSS news for {symbol}: {e}. Returning mock news database.")
        return generate_mock_news(symbol, limit)

def generate_mock_news(symbol="BTC/USDT", limit=5):
    """Generates synthetic news items in case of RSS feed request errors or network issues."""
    asset_name = symbol.split('/')[0]
    templates = [
        {"title": f"Why {asset_name} is Poised for a Major Breakout in the Coming Weeks", "description": "Analyst indicators suggest a historical support level has been verified, setting up potential positive momentum."},
        {"title": f"Institutional Inflows into {asset_name} Reach Multi-Month Highs", "description": "Regulatory filings show institutional funds increasing their spot exposure, citing strong utility and long term value."},
        {"title": f"Is {asset_name} Overvalued? Short-Term Correction Concerns Grow", "description": "Some derivative indicators indicate localized overleveraged long positions, which could trigger a brief liquidations flush."},
        {"title": f"New Technical Upgrade Announced for {asset_name} Ecosystem", "description": "Developers roll out a highly anticipated scalability update that significantly decreases transaction latency."},
        {"title": f"Macroeconomic Headwinds Inject Volatility into Crypto Markets", "description": "Global economic indicators and rate discussions keep retail traders cautious, resulting in consolidation across majors."}
    ]
    
    random.shuffle(templates)
    mock_items = []
    now = datetime.utcnow()
    
    for i in range(min(limit, len(templates))):
        pub_time = now - timedelta(hours=i * 2)
        mock_items.append({
            "title": templates[i]["title"],
            "link": "https://example.com/crypto-news",
            "pub_date": pub_time.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "description": templates[i]["description"]
        })
        
    return mock_items

if __name__ == "__main__":
    # Mini test block
    df = fetch_ohlcv("BTC/USDT", "1h", limit=5)
    print(df)
    news = fetch_asset_news("BTC/USDT", limit=3)
    print(news)
