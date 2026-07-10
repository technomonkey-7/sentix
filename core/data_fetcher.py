"""Market data + news fetching via yfinance / Google News RSS.

v2 principles:
- All timestamps are timezone-aware UTC pandas datetimes.
- NO synthetic fallback data. When a feed is down we return None/[] and the
  strategy simply does not trade. Fake candles and fake news must never
  reach a trading decision (v1 bug class).
"""
import email.utils
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

from core.db import log_event, get_config

_INTERVAL_SETTINGS = {
    "1h": {"yf_interval": "1h", "period": "60d"},
    "4h": {"yf_interval": "1h", "period": "120d"},   # resampled from 1h
    "1d": {"yf_interval": "1d", "period": "5y"},
}


def _normalize_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.reset_index()
    df.rename(columns={
        "Datetime": "timestamp", "Date": "timestamp",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }, inplace=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts
    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.set_index("timestamp")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return out


def fetch_ohlcv(symbol: str = "AAPL", timeframe: str = "1h", limit: int = 300,
                period: str | None = None, max_retries: int = 3):
    """Fetches OHLCV candles as a DataFrame with UTC 'timestamp' column.

    Returns None when real data cannot be fetched — callers must treat that
    as "do not trade", never substitute fake data.
    """
    settings = _INTERVAL_SETTINGS.get(timeframe)
    if settings is None:
        log_event("ERROR", "DATA_FETCHER", f"Unsupported timeframe '{timeframe}'")
        return None

    use_period = period or settings["period"]
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            df_raw = ticker.history(period=use_period, interval=settings["yf_interval"],
                                    auto_adjust=True)
            if df_raw is None or df_raw.empty:
                raise ValueError(f"Empty data returned for {symbol}")

            df = _normalize_frame(df_raw)
            if timeframe == "4h":
                df = resample_4h(df)
            if limit:
                df = df.tail(limit).reset_index(drop=True)
            return df
        except Exception as e:
            log_event("WARNING", "DATA_FETCHER",
                      f"Attempt {attempt + 1}/{max_retries} failed fetching {symbol} {timeframe}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    log_event("ERROR", "DATA_FETCHER",
              f"Could not fetch {timeframe} candles for {symbol} after {max_retries} attempts. "
              "No data returned (trading for this symbol is paused this cycle).")
    return None


def fetch_realtime_price(symbol: str, max_retries: int = 2):
    """Best-effort real-time price. Returns float or None (never a stale
    database value — the caller decides how to degrade)."""
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info["lastPrice"])
            if price > 0:
                return price
        except Exception:
            try:
                hist = yf.Ticker(symbol).history(period="1d", interval="1m")
                if hist is not None and not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(1.0)
    log_event("WARNING", "DATA_FETCHER", f"Real-time price unavailable for {symbol}")
    return None


# ----------------- News (Google News RSS) -----------------

_SEARCH_KEYWORDS = {
    "AAPL": "Apple Inc AAPL",
    "MSFT": "Microsoft MSFT",
    "NVDA": "NVIDIA NVDA AI",
    "AMD": "Advanced Micro Devices AMD",
    "TSLA": "Tesla TSLA",
    "AMZN": "Amazon AMZN",
    "GOOGL": "Alphabet Google GOOGL",
    "META": "Meta Platforms META",
    "ARM": "ARM Holdings chip design",
    "TSM": "TSMC Taiwan Semiconductor TSM",
    "AVGO": "Broadcom AVGO semiconductor",
    "ASML": "ASML lithography semiconductor",
    "QQQ": "Nasdaq 100 QQQ ETF",
    "SPY": "S&P 500 SPY ETF",
    "JPM": "JPMorgan Chase JPM bank",
    "V": "Visa V payments",
    "BRK-B": "Berkshire Hathaway Warren Buffett",
    "LLY": "Eli Lilly LLY pharma",
    "JNJ": "Johnson & Johnson JNJ",
    "UNH": "UnitedHealth UNH insurance",
    "XOM": "Exxon Mobil XOM oil",
    "CVX": "Chevron CVX oil",
    "PG": "Procter & Gamble PG consumer",
    "WMT": "Walmart WMT retail",
    "KO": "Coca-Cola KO",
    "COST": "Costco COST retail",
    "CAT": "Caterpillar CAT industrial",
    "GLD": "gold price GLD ETF",
    "BTC-USD": "Bitcoin BTC price",
    "ETH-USD": "Ethereum ETH price",
    "SOL-USD": "Solana SOL price",
}


def fetch_asset_news(symbol: str = "AAPL", limit: int = 10, max_retries: int = 3):
    """Latest news for a symbol from the free Google News RSS feed, filtered
    to the configured freshness window. Returns [] on failure (no mock news)."""
    try:
        hours = int(get_config("news_freshness_hours", "24"))
    except (TypeError, ValueError):
        hours = 24

    base = symbol.split("/")[0] if "/" in symbol else symbol
    query = _SEARCH_KEYWORDS.get(base, base) + f" stock market finance when:{hours}h"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                raise requests.RequestException(f"HTTP {response.status_code}")

            root = ET.fromstring(response.content)
            now_utc = datetime.now(timezone.utc)
            items = []
            for item in root.findall(".//item"):
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub_date = item.findtext("pubDate") or ""
                description = item.findtext("description") or ""
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0]
                try:
                    parsed_dt = email.utils.parsedate_to_datetime(pub_date)
                    if parsed_dt.tzinfo is None:
                        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if now_utc - parsed_dt > timedelta(hours=hours):
                    continue
                items.append({"title": title, "link": link, "pub_date": pub_date,
                              "parsed_dt": parsed_dt, "description": description})

            items.sort(key=lambda x: x["parsed_dt"], reverse=True)
            news = [{"title": i["title"], "link": i["link"], "pub_date": i["pub_date"],
                     "description": i["description"], "_is_simulated": False}
                    for i in items[:limit]]
            log_event("INFO", "DATA_FETCHER", f"Fetched {len(news)} news items for {symbol}")
            return news
        except Exception as e:
            log_event("WARNING", "DATA_FETCHER",
                      f"News fetch attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2.0)

    log_event("ERROR", "DATA_FETCHER", f"News unavailable for {symbol}; proceeding without news")
    return []


def check_vpn_connection():
    """Checks network/VPN reachability via the configured proxy."""
    vpn_check_enabled = (get_config("vpn_check_enabled", "false")).lower() == "true"
    if not vpn_check_enabled:
        return True

    proxies = {}
    for scheme, envs in (("http", ("HTTP_PROXY", "http_proxy")), ("https", ("HTTPS_PROXY", "https_proxy"))):
        for env in envs:
            if os.getenv(env):
                proxies[scheme] = os.getenv(env)
                break

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
            log_event("SUCCESS", "VPN_CHECK", f"Network connection ACTIVE. Public IP: {ip_str}")
            return True
        log_event("WARNING", "VPN_CHECK", f"Yahoo Finance returned HTTP {resp.status_code} (IP: {ip_str})")
        return False
    except Exception as e:
        log_event("ERROR", "VPN_CHECK", f"Connection check FAILED (IP: {ip_str}): {e}")
        return False


if __name__ == "__main__":
    print(fetch_ohlcv("AAPL", "1h", limit=5))
    print(fetch_realtime_price("AAPL"))
    print(fetch_asset_news("AAPL", limit=3)[:1])
