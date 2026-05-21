import pandas as pd
import pandas_ta as ta
from core.db import log_event

def calculate_indicators(df):
    """
    Computes technical indicators (RSI, MACD, EMA) using pandas-ta.
    Ensures safe operations and returns a clean DataFrame with computed columns.
    """
    if df is None or len(df) < 30:
        log_event("WARNING", "MATH_ENGINE", "Not enough data points to compute indicators (min 30 required).")
        return df

    # Make a copy to avoid SettingWithCopy warning
    df = df.copy()

    try:
        # 1. EMA 20
        df['ema'] = df.ta.ema(length=20)
        
        # 2. RSI 14
        df['rsi'] = df.ta.rsi(length=14)
        
        # 3. MACD (12, 26, 9)
        macd_df = df.ta.macd(fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            # Dynamically identify columns to tolerate different pandas-ta version schemas
            # Use startswith for robust matching (old filter with 's'/'h' character checks was fragile)
            macd_col = [c for c in macd_df.columns if c.startswith('MACD_') and not c.startswith('MACDs_') and not c.startswith('MACDh_')]
            sig_col = [c for c in macd_df.columns if c.startswith('MACDs_')]
            hist_col = [c for c in macd_df.columns if c.startswith('MACDh_')]
            
            if macd_col:
                df['macd'] = macd_df[macd_col[0]]
            if sig_col:
                df['macd_signal'] = macd_df[sig_col[0]]
            if hist_col:
                df['macd_hist'] = macd_df[hist_col[0]]
                
    except Exception as e:
        log_event("ERROR", "MATH_ENGINE", f"Failed to calculate indicators: {e}")
        
    return df

def check_triggers(df):
    """
    Checks for deterministic indicator crossover buy/sell triggers.
    Signals are calculated on the latest completed candle (index -2) and 
    the prior candle (index -3) to prevent active candle repainting.
    
    Returns:
        trigger_type: "BUY", "SELL", or None
        reason: Single string description of what triggered it (e.g., "MACD Crossover")
    """
    if df is None or len(df) < 5 or 'rsi' not in df.columns or 'macd' not in df.columns:
        return None, "Insufficient indicator data"

    # We evaluate on fully completed candles
    # index -1 = active forming candle
    # index -2 = latest completed candle
    # index -3 = previous completed candle
    
    c_prev = df.iloc[-3]
    c_curr = df.iloc[-2]

    # Extract current and previous values safely
    try:
        close_p, close_c = float(c_prev['close']), float(c_curr['close'])
        ema_p, ema_c = float(c_prev['ema']), float(c_curr['ema'])
        rsi_p, rsi_c = float(c_prev['rsi']), float(c_curr['rsi'])
        macd_p, macd_c = float(c_prev['macd']), float(c_curr['macd'])
        sig_p, sig_c = float(c_prev['macd_signal']), float(c_curr['macd_signal'])
    except (ValueError, KeyError, TypeError) as e:
        return None, f"Indicator computation incomplete: {e}"

    # Check Bullish (BUY) triggers
    # 1. MACD Bullish Cross: MACD line crosses above Signal line
    macd_bullish_cross = (macd_p <= sig_p) and (macd_c > sig_c)
    
    # 2. RSI Oversold recovery: RSI crosses back above 30, or enters oversold zone
    rsi_oversold_cross = (rsi_p >= 30) and (rsi_c < 30)
    rsi_oversold_recovery = (rsi_p < 30) and (rsi_c >= 30)
    
    # 3. EMA Bullish Breakout: Price crosses above EMA 20
    ema_bullish_cross = (close_p <= ema_p) and (close_c > ema_c)

    # Check Bearish (SELL) triggers
    # 1. MACD Bearish Cross: MACD line crosses below Signal line
    macd_bearish_cross = (macd_p >= sig_p) and (macd_c < sig_c)
    
    # 2. RSI Overbought exit: RSI crosses above 70, or falls back below 70
    rsi_overbought_cross = (rsi_p <= 70) and (rsi_c > 70)
    rsi_overbought_reentry = (rsi_p > 70) and (rsi_c <= 70)
    
    # 3. EMA Bearish Breakdown: Price crosses below EMA 20
    ema_bearish_cross = (close_p >= ema_p) and (close_c < ema_c)

    # Prioritize triggers and generate descriptive text
    if macd_bullish_cross:
        return "BUY", "MACD Bullish Cross (MACD crossed above Signal)"
    elif rsi_oversold_cross:
        return "BUY", "RSI Oversold Entry (RSI crossed below 30)"
    elif rsi_oversold_recovery:
        return "BUY", "RSI Oversold Recovery (RSI crossed back above 30)"
    elif ema_bullish_cross:
        return "BUY", "EMA Crossover (Price broke above 20 EMA)"
        
    elif macd_bearish_cross:
        return "SELL", "MACD Bearish Cross (MACD crossed below Signal)"
    elif rsi_overbought_cross:
        return "SELL", "RSI Overbought Entry (RSI crossed above 70)"
    elif rsi_overbought_reentry:
        return "SELL", "RSI Overbought Re-entry (RSI crossed back below 70)"
    elif ema_bearish_cross:
        return "SELL", "EMA Crossover (Price fell below 20 EMA)"

    return None, "No indicator crossovers detected."
