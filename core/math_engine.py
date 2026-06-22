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

def calculate_support_resistance(df, window=5):
    """
    Identifies key support and resistance levels based on local price swing extremes.
    Returns:
        supports: list of float values representing support price levels (sorted ascending)
        resistances: list of float values representing resistance price levels (sorted ascending)
    """
    supports = []
    resistances = []
    
    if df is None or len(df) < 2 * window + 1:
        return supports, resistances
        
    try:
        # Loop over historical candles excluding recent margin window to avoid future leaks
        for i in range(window, len(df) - window):
            low_val = float(df.iloc[i]['low'])
            high_val = float(df.iloc[i]['high'])
            
            # Local Minima (Support Level)
            if low_val == df['low'].iloc[i-window : i+window+1].min():
                supports.append(low_val)
                
            # Local Maxima (Resistance Level)
            if high_val == df['high'].iloc[i-window : i+window+1].max():
                resistances.append(high_val)
                
        # Deduplicate, sort, and return the latest key historical levels
        supports = sorted(list(set(supports)))[-5:]
        resistances = sorted(list(set(resistances)))[-5:]
    except Exception as e:
        log_event("ERROR", "MATH_ENGINE", f"Failed to calculate support/resistance levels: {e}")
        
    return supports, resistances

def check_candlestick_patterns(df):
    """
    Checks the latest completed candle (index -2) for high-probability reversal structures.
    Returns:
        patterns: list of detected patterns (e.g., ["Bullish Engulfing"])
    """
    patterns = []
    if df is None or len(df) < 3:
        return patterns
        
    try:
        c_prev = df.iloc[-3]
        c_curr = df.iloc[-2]
        
        body_curr = abs(c_curr['close'] - c_curr['open'])
        range_curr = c_curr['high'] - c_curr['low']
        
        # 1. Bullish Engulfing
        prev_red = c_prev['close'] < c_prev['open']
        curr_green = c_curr['close'] > c_curr['open']
        if prev_red and curr_green:
            # Engulfing: open must be <= previous close and close must be >= previous open
            if c_curr['open'] <= c_prev['close'] and c_curr['close'] >= c_prev['open']:
                patterns.append("Bullish Engulfing")
                
        # 2. Hammer
        if range_curr > 0:
            body_pct = body_curr / range_curr
            lower_shadow = min(c_curr['open'], c_curr['close']) - c_curr['low']
            upper_shadow = c_curr['high'] - max(c_curr['open'], c_curr['close'])
            
            # Hammer criteria: body occupies < 35% of range, lower shadow is >= 2x body, upper shadow is minimal
            if body_pct < 0.35 and lower_shadow >= 2.0 * body_curr and upper_shadow < 0.1 * range_curr:
                # Must occur in a pullback (close is below 20 EMA)
                if 'ema' in c_curr and c_curr['close'] < c_curr['ema']:
                    patterns.append("Hammer")
                    
    except Exception as e:
        log_event("ERROR", "MATH_ENGINE", f"Error checking candlestick patterns: {e}")
        
    return patterns

def check_triggers(df):
    """
    Checks for deterministic indicator crossover buy/sell triggers.
    Signals are calculated on the latest completed candle (index -2) and 
    the prior candle (index -3) to prevent active candle repainting.
    """
    if df is None or len(df) < 5 or 'rsi' not in df.columns or 'macd' not in df.columns:
        return None, "Insufficient indicator data"
    
    c_prev = df.iloc[-3]
    c_curr = df.iloc[-2]

    try:
        close_p, close_c = float(c_prev['close']), float(c_curr['close'])
        ema_p, ema_c = float(c_prev['ema']), float(c_curr['ema'])
        rsi_p, rsi_c = float(c_prev['rsi']), float(c_curr['rsi'])
        macd_p, macd_c = float(c_prev['macd']), float(c_curr['macd'])
        sig_p, sig_c = float(c_prev['macd_signal']), float(c_curr['macd_signal'])
    except (ValueError, KeyError, TypeError) as e:
        return None, f"Indicator computation incomplete: {e}"

    # MACD Bullish Cross: MACD line crosses above Signal line
    macd_diff_pct = ((macd_c - sig_c) / close_c) * 100 if close_c > 0 else 0.0
    macd_bullish_cross = (macd_p <= sig_p) and (macd_c > sig_c) and (macd_diff_pct >= 0.005)
    
    # RSI Oversold recovery: RSI crosses back above 30, or enters oversold zone
    rsi_oversold_cross = (rsi_p >= 30) and (rsi_c < 30)
    rsi_oversold_recovery = (rsi_p < 30) and (rsi_c >= 30)
    
    # EMA Bullish Breakout: Price crosses above EMA 20 by at least 0.05%
    ema_bullish_cross = (close_p <= ema_p) and (close_c >= ema_c * 1.0005)

    # MACD Bearish Cross: MACD line crosses below Signal line
    macd_bearish_diff_pct = ((sig_c - macd_c) / close_c) * 100 if close_c > 0 else 0.0
    macd_bearish_cross = (macd_p >= sig_p) and (macd_c < sig_c) and (macd_bearish_diff_pct >= 0.005)
    
    # RSI Overbought exit: RSI crosses above 70, or falls back below 70
    rsi_overbought_cross = (rsi_p <= 70) and (rsi_c > 70)
    rsi_overbought_reentry = (rsi_p > 70) and (rsi_c <= 70)
    
    # EMA Bearish Breakdown: Price falls below EMA 20 by at least 0.05%
    ema_bearish_cross = (close_p >= ema_p) and (close_c <= ema_c * 0.9995)

    # Prioritize triggers
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

def confirm_with_higher_tf(df_1h, df_4h, trigger_side):
    """
    Performs double-check confirmation using 4h timeframe trend and 1h volume analysis.
    """
    if df_4h is None or len(df_4h) < 20 or 'ema' not in df_4h.columns:
        return False, "Insufficient 4h candle data to verify trend"
        
    c_4h = df_4h.iloc[-2] # Latest completed 4h candle
    
    try:
        close_4h = float(c_4h['close'])
        ema_4h = float(c_4h['ema'])
        
        macd_4h = float(c_4h['macd']) if 'macd' in c_4h and c_4h['macd'] is not None else None
        sig_4h = float(c_4h['macd_signal']) if 'macd_signal' in c_4h and c_4h['macd_signal'] is not None else None
        
        is_4h_bullish = close_4h > ema_4h
        if macd_4h is not None and sig_4h is not None:
            is_4h_bullish = is_4h_bullish or (macd_4h > sig_4h)
            
        is_4h_bearish = close_4h < ema_4h
        if macd_4h is not None and sig_4h is not None:
            is_4h_bearish = is_4h_bearish or (macd_4h < sig_4h)
            
    except (ValueError, KeyError, TypeError) as e:
        return False, f"Failed to parse 4h trend metrics: {e}"
        
    # Check volume on 1h candles
    if df_1h is None or len(df_1h) < 20:
        return False, "Insufficient 1h candle data to verify volume"
        
    try:
        latest_vol_1h = float(df_1h.iloc[-2]['volume'])
        vol_ma_1h = float(df_1h['volume'].tail(20).mean())
        volume_confirmed = latest_vol_1h >= vol_ma_1h
    except (ValueError, KeyError, TypeError) as e:
        return False, f"Failed to parse volume metrics: {e}"
        
    if trigger_side == "BUY":
        if not is_4h_bullish:
            return False, f"4h trend is not bullish (Close: {close_4h:.2f}, EMA 20: {ema_4h:.2f})"
        if not volume_confirmed:
            return False, f"1h volume ({latest_vol_1h:.1f}) is below 20-period average ({vol_ma_1h:.1f})"
        return True, f"Confirmed: 4h trend is bullish and volume ({latest_vol_1h:.1f} >= MA {vol_ma_1h:.1f}) is expanding"
        
    elif trigger_side == "SELL":
        if not is_4h_bearish:
            return False, f"4h trend is not bearish (Close: {close_4h:.2f}, EMA 20: {ema_4h:.2f})"
        return True, "Confirmed: 4h trend is bearish"
        
    return False, f"Unknown trigger side: {trigger_side}"
