"""Pure indicator math. No database access, no network, no side effects.

Shared by the live worker (core.strategy) and the backtester (core.backtest)
so both always see identical numbers.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 (straight rally) RSI is 100 by definition
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's Average True Range. Expects columns high/low/close."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the standard indicator set used across the app.

    Columns added: ema (EMA20), ema50, ema200, rsi, macd, macd_signal,
    macd_hist, atr, vol_sma. Returns a copy.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    close = df["close"].astype(float)
    df["ema"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr"] = atr(df, 14)
    if "volume" in df.columns:
        df["vol_sma"] = sma(df["volume"].astype(float), 20)
    return df


def swing_levels(df: pd.DataFrame, window: int = 5, max_levels: int = 5):
    """Swing-point support/resistance levels from local extremes.

    Returns (supports, resistances), each sorted ascending, at most
    max_levels of the most recent levels.
    """
    supports, resistances = [], []
    if df is None or len(df) < 2 * window + 1:
        return supports, resistances
    lows = df["low"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    for i in range(window, len(df) - window):
        lo_win = lows[i - window : i + window + 1]
        hi_win = highs[i - window : i + window + 1]
        if lows[i] == lo_win.min():
            supports.append(float(lows[i]))
        if highs[i] == hi_win.max():
            resistances.append(float(highs[i]))
    supports = sorted(set(supports))[-max_levels:]
    resistances = sorted(set(resistances))[-max_levels:]
    return supports, resistances


def bullish_patterns(df: pd.DataFrame, completed_idx: int = -2):
    """Bullish reversal patterns on the completed candle at ``completed_idx``.

    Returns a list of pattern names (possibly empty).
    """
    patterns = []
    if df is None or len(df) < abs(completed_idx) + 1:
        return patterns
    c_prev = df.iloc[completed_idx - 1]
    c_curr = df.iloc[completed_idx]

    body = abs(float(c_curr["close"]) - float(c_curr["open"]))
    rng = float(c_curr["high"]) - float(c_curr["low"])

    prev_red = float(c_prev["close"]) < float(c_prev["open"])
    curr_green = float(c_curr["close"]) > float(c_curr["open"])
    if prev_red and curr_green:
        if float(c_curr["open"]) <= float(c_prev["close"]) and float(c_curr["close"]) >= float(c_prev["open"]):
            patterns.append("Bullish Engulfing")

    if rng > 0:
        lower_shadow = min(float(c_curr["open"]), float(c_curr["close"])) - float(c_curr["low"])
        upper_shadow = float(c_curr["high"]) - max(float(c_curr["open"]), float(c_curr["close"]))
        if body / rng < 0.35 and lower_shadow >= 2.0 * body and upper_shadow < 0.15 * rng:
            patterns.append("Hammer")

    return patterns


def crossed_above(prev_a, prev_b, curr_a, curr_b) -> bool:
    """True when series A crossed above series B between two consecutive bars."""
    try:
        return float(prev_a) <= float(prev_b) and float(curr_a) > float(curr_b)
    except (TypeError, ValueError):
        return False


def crossed_below(prev_a, prev_b, curr_a, curr_b) -> bool:
    try:
        return float(prev_a) >= float(prev_b) and float(curr_a) < float(curr_b)
    except (TypeError, ValueError):
        return False
