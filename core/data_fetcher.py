import yfinance as yf
import os
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime, timedelta, timezone
import random
import time
from core.db import log_event, get_config

def fetch_ohlcv(symbol="AAPL/USD", timeframe="1h", limit=100):
    """
    Fetches OHLCV candlestick data for a stock symbol and timeframe using yfinance.
    In LIVE_MODE, failures raise exceptions. In simulation mode, provides fallback to synthetic data.
    """
    live_mode = (get_config("live_mode") or os.getenv("LIVE_MODE", "false")).lower() == "true"
    ticker_symbol = symbol.split('/')[0] if '/' in symbol else symbol
    
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            log_event("INFO", "DATA_FETCHER", f"Fetching candles for {ticker_symbol} ({timeframe}) from YFinance (Attempt {attempt+1}/{max_retries})...")
            
            # Use appropriate period for fetching (1h allows max 730 days)
            # Fetch 30 days for 1h, 90 days for 4h resample
            period = "90d" if timeframe == "4h" else "30d"
            
            ticker = yf.Ticker(ticker_symbol)
            # Fetch hourly data
            df_raw = ticker.history(period=period, interval="1h")
            
            if df_raw.empty:
                raise ValueError(f"Empty data returned from Yahoo Finance for {ticker_symbol}.")
            
            # Reset index and rename columns
            df_raw = df_raw.reset_index()
            df_raw.rename(columns={
                'Datetime': 'timestamp',
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Select required columns and convert index
            df = df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # If 4h timeframe is requested, resample 1h data to 4h
            if timeframe == "4h":
                df.set_index('timestamp', inplace=True)
                df_resampled = df.resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()
                df = df_resampled
            
            # Format timestamp back to string
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Take the last N candles according to limit
            df = df.tail(limit).reset_index(drop=True)
            df['_is_simulated'] = False
            
            log_event("SUCCESS", "DATA_FETCHER", f"Successfully fetched {len(df)} candles for {symbol} on attempt {attempt+1}.")
            return df

        except Exception as e:
            log_event("WARNING", "DATA_FETCHER", f"Attempt {attempt+1} failed to fetch candles for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                if live_mode:
                    log_event("ERROR", "DATA_FETCHER", f"YFinance fetch FAILED for {symbol} in LIVE MODE after {max_retries} attempts: {e}.")
                    raise RuntimeError(f"LIVE MODE: Cannot fetch real market data for {symbol}. API error: {e}")
                
                log_event("WARNING", "DATA_FETCHER", f"YFinance fetch failed for {symbol} after {max_retries} attempts. Generating simulated fallback data.")
                return generate_simulated_ohlcv(symbol, timeframe, limit)

def generate_simulated_ohlcv(symbol="AAPL/USD", timeframe="1h", limit=100):
    """
    Generates synthetic but highly realistic OHLCV data using a random walk with drift.
    Ensures that the trading engine can always run and display charts, even offline.
    """
    base_prices = {
        "AAPL/USD": 180.0,
        "MSFT/USD": 420.0,
        "NVDA/USD": 120.0,
        "AMD/USD": 170.0,
        "TSLA/USD": 180.0,
        "AMZN/USD": 180.0,
        "GOOGL/USD": 170.0,
        "META/USD": 480.0,
        "ARM/USD": 120.0,
        "TSM/USD": 140.0,
        "AVGO/USD": 1400.0,
        "ASML/USD": 950.0,
        "QQQ/USD": 440.0,
        "SPY/USD": 510.0
    }
    base_price = base_prices.get(symbol, 150.0)
    
    # Time delta based on timeframe
    now = datetime.now(timezone.utc)
    delta = timedelta(hours=1) if timeframe == "1h" else timedelta(hours=4)
    
    timestamps = []
    for i in range(limit):
        t = now - (limit - i) * delta
        timestamps.append(t.strftime('%Y-%m-%d %H:%M:%S'))
        
    opens, highs, lows, closes, volumes = [], [], [], [], []
    current_price = base_price * (1 + random.uniform(-0.05, 0.05)) # Random start price
    
    for _ in range(limit):
        pct_change = random.normalvariate(0.0002, 0.005) # Small positive drift, stock-like volatility
        open_price = current_price
        close_price = current_price * (1 + pct_change)
        
        # Volatility noise for high and low
        high_price = max(open_price, close_price) * (1 + abs(random.normalvariate(0, 0.003)))
        low_price = min(open_price, close_price) * (1 - abs(random.normalvariate(0, 0.003)))
        
        volume = random.uniform(1000, 50000)
        
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

def fetch_asset_news(symbol="AAPL/USD", limit=10):
    """
    Scrapes the latest news articles for a symbol using the free Google News RSS Feed.
    Filters out any articles older than the configured news_freshness_hours limit.
    Extracts article title, snippet, and link, returning them as a clean list of dicts.
    """
    try:
        hours = int(get_config("news_freshness_hours", "24"))
    except Exception:
        hours = 24

    asset_name = symbol.split('/')[0] if '/' in symbol else symbol
    
    # Stock-focused search queries
    search_keywords = {
        "AAPL": "Apple Inc AAPL",
        "MSFT": "Microsoft MSFT",
        "NVDA": "NVIDIA NVDA AI",
        "AMD": "Advanced Micro Devices AMD",
        "TSLA": "Tesla TSLA EV",
        "AMZN": "Amazon AMZN",
        "GOOGL": "Alphabet Google GOOGL",
        "META": "Meta Platforms META Mark Zuckerberg",
        "ARM": "ARM Holdings ARM chip design",
        "TSM": "TSMC Taiwan Semiconductor TSM",
        "AVGO": "Broadcom AVGO chip semiconductor",
        "ASML": "ASML photolithography chip semiconductor",
        "QQQ": "Nasdaq 100 QQQ ETF",
        "SPY": "S&P 500 SPY ETF"
    }
    
    query = search_keywords.get(asset_name, asset_name) + f" stock market finance when:{hours}h"
    encoded_query = urllib.parse.quote(query)
    
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        log_event("INFO", "DATA_FETCHER", f"Scraping live RSS stock news (last {hours}h) for {symbol} (Attempt {attempt+1}/{max_retries})...")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                raise requests.RequestException(f"Invalid HTTP status: {response.status_code}")
                
            root = ET.fromstring(response.content)
            import email.utils
            
            all_items = []
            for item in root.findall('.//item'):
                title = item.find('title').text if item.find('title') is not None else ""
                link = item.find('link').text if item.find('link') is not None else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                description = item.find('description').text if item.find('description') is not None else ""
                
                # Clean up source from title
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0]
                    
                parsed_dt = None
                if pub_date:
                    try:
                        parsed_dt = email.utils.parsedate_to_datetime(pub_date)
                    except Exception:
                        pass
                
                if parsed_dt:
                    if parsed_dt.tzinfo is None:
                        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    age = now_utc - parsed_dt
                    if age > timedelta(hours=hours):
                        continue
                else:
                    continue
                        
                all_items.append({
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "parsed_dt": parsed_dt,
                    "description": description
                })
                
            all_items.sort(key=lambda x: x["parsed_dt"], reverse=True)
            
            news_items = []
            for item in all_items[:limit]:
                news_items.append({
                    "title": item["title"],
                    "link": item["link"],
                    "pub_date": item["pub_date"],
                    "description": item["description"],
                    "_is_simulated": False
                })
                
            log_event("SUCCESS", "DATA_FETCHER", f"Scraped {len(news_items)} sorted news articles for {symbol}.")
            return news_items
            
        except Exception as e:
            log_event("WARNING", "DATA_FETCHER", f"Attempt {attempt+1} failed to scrape news: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                log_event("ERROR", "DATA_FETCHER", f"Failed to scrape RSS news after {max_retries} attempts: {e}.")
                return generate_mock_news(symbol, limit)

def generate_mock_news(symbol="AAPL/USD", limit=5):
    """Generates synthetic news items in case of RSS feed errors or network issues."""
    asset_name = symbol.split('/')[0] if '/' in symbol else symbol
    templates = [
        {"title": f"Why {asset_name} is Poised for a Major Breakout in the Stock Market", "description": f"Analysts suggest that the strong institutional demand and positive earnings estimates are backing a technical breakout for {asset_name}."},
        {"title": f"New Technological Breakthrough Announced by {asset_name}", "description": f"The development team has unveiled an advanced product line that is expected to expand the margins and market share of {asset_name}."},
        {"title": f"Is {asset_name} Stock Overvalued? Potential Short-Term Distribution Underway", "description": f"Several market indicators suggest that {asset_name} might be slightly overextended on daily timeframes, warning swing traders of profit taking."},
        {"title": f"Semiconductor Sector Analysis: Demand Remains Resilient", "description": f"Industry experts report that structural trends in computing power and AI models keep semiconductor leaders well positioned for the future."},
        {"title": f"Macroeconomic Outlook: Stock Markets Adjust to Global Interest Rate Decisions", "description": "Trading volumes consolidate as market participants wait for official macroeconomic indicators and rate updates."}
    ]
    
    random.shuffle(templates)
    mock_items = []
    now = datetime.now(timezone.utc)
    
    for i in range(min(limit, len(templates))):
        pub_time = now - timedelta(hours=i * 2)
        mock_items.append({
            "title": templates[i]["title"],
            "link": "https://example.com/stock-news",
            "pub_date": pub_time.strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "description": templates[i]["description"],
            "_is_simulated": True
        })
        
    return mock_items

def check_vpn_connection():
    """
    Checks if the VPN / Network connection is active by making requests using the configured proxy.
    Returns True if connected, False otherwise.
    """
    vpn_check_enabled = (get_config("vpn_check_enabled", "false")).lower() == "true"
    if not vpn_check_enabled:
        return True

    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy

    ip_str = "Unknown"
    try:
        resp_ip = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=5)
        if resp_ip.status_code == 200:
            ip_str = resp_ip.json().get("origin", "Unknown")
    except Exception:
        pass

    try:
        resp = requests.get("https://finance.yahoo.com", proxies=proxies, timeout=5)
        if resp.status_code == 200:
            log_event("SUCCESS", "VPN_CHECK", f"VPN / Network Connection is ACTIVE. Public IP: {ip_str}")
            return True
        else:
            log_event("WARNING", "VPN_CHECK", f"Yahoo Finance ping returned non-200 status: {resp.status_code}. Public IP: {ip_str}")
            return False
    except Exception as e:
        log_event("ERROR", "VPN_CHECK", f"VPN / Connection check FAILED: Yahoo Finance is unreachable (IP: {ip_str}). Error: {e}")
        return False

if __name__ == "__main__":
    df = fetch_ohlcv("AAPL/USD", "1h", limit=5)
    print(df)
    news = fetch_asset_news("AAPL/USD", limit=3)
    print(news)
