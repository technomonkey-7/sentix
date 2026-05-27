import time
import os
import math
import ccxt
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from core.db import (
    init_db, log_event, save_candles, get_portfolio, 
    update_portfolio, record_trade, get_config, save_config,
    get_active_position, get_all_active_positions, close_active_position
)
from core.data_fetcher import fetch_ohlcv, fetch_asset_news
from core.math_engine import calculate_indicators, check_triggers
from ai.sentiment_analyzer import analyze_sentiment

# Load environment configurations
load_dotenv()

# Shared exchange instance for real-time price lookups
_exchange = None
_last_analyzed_timestamps = {}
_last_ai_check_times = {}
def _get_exchange():
    """Returns a shared configured ccxt exchange instance."""
    global _exchange
    if _exchange is None:
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
                
        _exchange = exchange_class(exchange_config)
    return _exchange

def fetch_realtime_price(symbol):
    """
    Fetches the current real-time price for a symbol using ccxt ticker.
    Falls back to the latest DB candle close price if the ticker call fails.
    This is critical for accurate SL/TP monitoring.
    """
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            exchange = _get_exchange()
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
            return price
        except Exception as e:
            log_event("WARNING", "WORKER", f"Attempt {attempt+1} failed to fetch real-time ticker for {symbol}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                live_mode = (get_config("live_mode") or os.getenv("LIVE_MODE", "false")).lower() == "true"
                if live_mode:
                    log_event("WARNING", "WORKER", f"Real-time ticker fetch failed for {symbol} in LIVE MODE. Refusing database price fallback.")
                    return None

                log_event("WARNING", "WORKER", f"Real-time ticker fetch failed for {symbol} after {max_retries} attempts. Falling back to DB candle price.")
                # Fallback to latest DB candle close
                from core.db import get_connection
                try:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT close FROM candles WHERE asset = ? ORDER BY timestamp DESC LIMIT 1", (symbol,))
                    row = cursor.fetchone()
                    conn.close()
                    if row:
                        return float(row[0])
                except Exception:
                    pass
                return None

def calculate_total_nav(portfolio):
    """
    Calculates the Net Asset Value (NAV) of the portfolio.
    USD balance + value of all crypto holdings using current market prices.
    """
    total_nav = portfolio.get("USD", {}).get("balance", 0.0)
    for asset_ticker, data in portfolio.items():
        if asset_ticker == "USD":
            continue
        balance = data.get("balance", 0.0)
        avg_entry = data.get("avg_entry_price", 0.0)
        if balance > 0:
            # We will use the latest database candle price as the current price estimate for the valuation
            # If the database candles are not found, fallback to avg_entry_price
            from core.db import get_connection
            current_price = avg_entry
            try:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT close FROM candles 
                    WHERE asset LIKE ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (f"{asset_ticker}/%",))
                row = cursor.fetchone()
                if row:
                    current_price = float(row[0])
                conn.close()
            except Exception:
                pass
            total_nav += balance * current_price
    return total_nav

def run_worker_cycle(force=False):
    """
    Wrapper to acquire database lock and prevent multiple concurrent analysis cycles.
    """
    lock_time_str = get_config("worker_lock_time")
    now_utc = datetime.now(timezone.utc)
    if lock_time_str:
        try:
            lock_time = datetime.fromisoformat(lock_time_str)
            if now_utc - lock_time < timedelta(minutes=3):
                log_event("WARNING", "WORKER", "Another analysis cycle is already in progress. Skipping execution to prevent conflicts.")
                if force:
                    raise RuntimeError("Başka bir analiz döngüsü şu anda aktif olarak çalışıyor. Lütfen biraz bekleyin.")
                return
        except RuntimeError:
            raise
        except Exception:
            pass
            
    try:
        save_config("worker_lock_time", now_utc.isoformat())
    except Exception:
        pass
        
    try:
        _execute_cycle_logic(force=force)
    finally:
        try:
            save_config("worker_lock_time", "")
        except Exception:
            pass

def _execute_cycle_logic(force=False):
    """
    Executes a single cycle of the background worker:
    1. Reads latest configurations.
    2. Fetches prices and computes indicators for all active assets.
    3. Runs real-time SL/TP monitor to liquidate open positions immediately when hit.
    4. Runs deterministic technical analysis to check for crossover triggers.
    5. Double-checks triggers with 4-hour timeframe trend and volume moving average.
    6. Batches news scraping and structured Gemini API sentiment checks for all candidates in one call.
    7. Handles paper trading buy/sell executions and state saves.
    """
    log_event("INFO", "WORKER", "Starting algorithmic market analysis tick...")
    
    # 1. Fetch config settings
    assets_str = get_config("selected_assets") or os.getenv("SELECTED_ASSETS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT,XRP/USDT,DOT/USDT,AVAX/USDT,LINK/USDT,NEAR/USDT,MATIC/USDT")
    active_assets = [a.strip() for a in assets_str.split(",") if a.strip()]
    
    sentiment_threshold = int(get_config("min_ai_sentiment_threshold") or os.getenv("MIN_AI_SENTIMENT_THRESHOLD", "3"))
    
    # Default risk settings
    risk_pct = float(get_config("risk_percentage") or os.getenv("RISK_PERCENTAGE", "2.0"))
    sl_pct = float(get_config("stop_loss_pct") or os.getenv("STOP_LOSS_PCT", "3.0"))
    tp_pct = float(get_config("take_profit_pct") or os.getenv("TAKE_PROFIT_PCT", "6.0"))
    
    portfolio = get_portfolio()
    total_nav = calculate_total_nav(portfolio)
    
    log_event("INFO", "WORKER", f"Total NAV: ${total_nav:.2f} | Risk Size: {risk_pct}% (${total_nav * (risk_pct/100.0):.2f})")
    
    candidates_to_analyze = []
    
    # Process each asset independently for technical crossover checking
    for asset in active_assets:
        try:
            # Update heartbeat inside the loop so UI knows bot is alive even during a long analysis cycle
            try:
                save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())
            except Exception:
                pass
                
            log_event("INFO", "WORKER", f"Analyzing asset: {asset}")
            
            # Fetch 1h historical candle data
            candles_df = fetch_ohlcv(symbol=asset, timeframe="1h", limit=100)
            
            if candles_df is None or candles_df.empty:
                log_event("WARNING", "WORKER", f"No candles fetched for {asset}. Skipping.")
                continue
                
            # Compute indicators (RSI, EMA, MACD)
            candles_df = calculate_indicators(candles_df)
            
            # Persist candles to database
            candles_list = []
            for _, row in candles_df.iterrows():
                candles_list.append({
                    "timestamp": row["timestamp"],
                    "asset": asset,
                    "interval": "1h",
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "rsi": row["rsi"] if not pd_isna(row["rsi"]) else None,
                    "macd": row["macd"] if not pd_isna(row["macd"]) else None,
                    "macd_signal": row["macd_signal"] if not pd_isna(row["macd_signal"]) else None,
                    "macd_hist": row["macd_hist"] if not pd_isna(row["macd_hist"]) else None,
                    "ema": row["ema"] if not pd_isna(row["ema"]) else None
                })
            
            save_candles(candles_list)
            
            # Skip trading decisions on simulated (fake) data
            is_simulated = False
            if '_is_simulated' in candles_df.columns:
                is_simulated = bool(candles_df['_is_simulated'].iloc[0])
            if is_simulated:
                log_event("WARNING", "WORKER", f"⚠️ {asset} is using SIMULATED data (exchange unreachable). Skipping trade decisions for safety.")
                continue
            
            # Extract latest close price AND fetch real-time price for SL/TP
            completed_candle = candles_df.iloc[-2]
            candle_close_price = float(completed_candle["close"]) # For indicator analysis
            completed_timestamp = completed_candle["timestamp"]
            
            # Use real-time ticker price for SL/TP monitoring
            realtime_price = fetch_realtime_price(asset)
            live_mode = (get_config("live_mode") or os.getenv("LIVE_MODE", "false")).lower() == "true"
            if realtime_price is None:
                if live_mode:
                    log_event("WARNING", "WORKER", f"Skipping SL/TP check for {asset} in cycle because real-time price fetch failed.")
                    current_price = None
                else:
                    current_price = candle_close_price
            else:
                current_price = realtime_price
            
            # Log verbose mathematical descriptions
            rsi_val = completed_candle.get("rsi")
            macd_val = completed_candle.get("macd")
            sig_val = completed_candle.get("macd_signal")
            ema_val = completed_candle.get("ema")
            
            if rsi_val is not None:
                rsi_status = "Aşırı Satım (Boğa Fırsatı)" if rsi_val <= 30 else ("Aşırı Alım (Ayı Riski)" if rsi_val >= 70 else "Nötr Bölge")
                log_event("INFO", "MATH", f"{asset} - RSI: {rsi_val:.2f} ({rsi_status}, Eşik değer: 30)")
            else:
                log_event("INFO", "MATH", f"{asset} - RSI: Veri yetersiz")
                
            if macd_val is not None and sig_val is not None:
                macd_diff = macd_val - sig_val
                macd_status = "Pozitif Bölge (Yükseliş Trendi)" if macd_diff > 0 else "Negatif Bölge (Düşüş Trendi)"
                log_event("INFO", "MATH", f"{asset} - MACD: {macd_val:.4f}, Sinyal: {sig_val:.4f} (Fark: {macd_diff:.4f} - {macd_status})")
            else:
                log_event("INFO", "MATH", f"{asset} - MACD: Veri yetersiz")
                
            if ema_val is not None and current_price is not None:
                ema_diff_pct = ((current_price - ema_val) / ema_val) * 100
                log_event("INFO", "MATH", f"{asset} - EMA 20: ${ema_val:.2f}, Fiyat: ${current_price:.2f} (Eşik farkı: %{ema_diff_pct:+.2f})")
            else:
                log_event("INFO", "MATH", f"{asset} - EMA 20: Veri veya fiyat yetersiz")
            
            # ---------------- RISK CHECK: STOP LOSS & TAKE PROFIT MONITORING ----------------
            active_pos = get_active_position(asset)
            if active_pos and current_price is not None:
                sl_val = active_pos.get("stop_loss")
                tp_val = active_pos.get("take_profit")
                buy_price = active_pos.get("price")
                log_event("INFO", "WORKER", f"Active position found for {asset} at entry price ${buy_price:.2f}. SL: ${sl_val:.2f}, TP: ${tp_val:.2f}. Real-time Price: ${current_price:.2f}")
                
                # Re-read portfolio from DB before any liquidation to avoid race conditions
                portfolio = get_portfolio()
                fee_pct = float(get_config("fee_pct", "0.001"))
                
                # Check Stop-Loss
                if sl_val and current_price <= sl_val:
                    log_event("WARNING", "WORKER", f"🚨 STOP LOSS BREACHED! Real-time price ${current_price:.2f} <= SL ${sl_val:.2f} for {asset}. Triggering liquidation...")
                    
                    asset_ticker = asset.split("/")[0]
                    asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
                    trade_value = asset_balance * current_price
                    trade_net_value = trade_value * (1 - fee_pct)
                    
                    portfolio["USD"]["balance"] = portfolio.get("USD", {}).get("balance", 0.0) + trade_net_value
                    portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
                    update_portfolio(portfolio)
                    
                    pnl_pct = close_active_position(asset, current_price, "STOP_LOSS", f"Stop Loss breached at ${sl_val:.2f}")
                    log_event("SUCCESS", "WORKER", f"💀 Stop Loss liquidation completed for {asset} with PnL: {pnl_pct:+.2f}%")
                    portfolio = get_portfolio()
                    total_nav = calculate_total_nav(portfolio)
                    continue
                
                # Check Take-Profit
                elif tp_val and current_price >= tp_val:
                    log_event("SUCCESS", "WORKER", f"🎯 TAKE PROFIT BREACHED! Real-time price ${current_price:.2f} >= TP ${tp_val:.2f} for {asset}. Triggering liquidation...")
                    
                    asset_ticker = asset.split("/")[0]
                    asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
                    trade_value = asset_balance * current_price
                    trade_net_value = trade_value * (1 - fee_pct)
                    
                    portfolio["USD"]["balance"] = portfolio.get("USD", {}).get("balance", 0.0) + trade_net_value
                    portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
                    update_portfolio(portfolio)
                    
                    pnl_pct = close_active_position(asset, current_price, "TAKE_PROFIT", f"Take Profit breached at ${tp_val:.2f}")
                    log_event("SUCCESS", "WORKER", f"🎉 Take Profit liquidation completed for {asset} with PnL: {pnl_pct:+.2f}%")
                    portfolio = get_portfolio()
                    total_nav = calculate_total_nav(portfolio)
                    continue
            
            # Check for deterministic crossover triggers
            trigger_side, trigger_reason = check_triggers(candles_df)
            
            global _last_analyzed_timestamps
            already_analyzed = (not force and _last_analyzed_timestamps.get(asset) == completed_timestamp)
            
            # Fresh portfolio state (we will read it here for checks)
            portfolio = get_portfolio()
            usd_balance = portfolio.get("USD", {}).get("balance", 0.0)
            asset_ticker = asset.split("/")[0]
            asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
            
            if already_analyzed:
                log_event("INFO", "WORKER", f"{asset}: Candle at {completed_timestamp} already analyzed. Skipping trade decisions.")
                continue
                
            if trigger_side:
                log_event("WARNING", "WORKER", f"💥 Technical [{trigger_side}] Trigger detected for {asset}! Reason: {trigger_reason}")
                
                # Check portfolio constraints first
                if trigger_side == "BUY" and usd_balance < 10.0:
                    log_event("INFO", "WORKER", f"Insufficient USD balance (${usd_balance:.2f}) to perform BUY on {asset}. Skipping.")
                    continue
                elif trigger_side == "SELL" and asset_balance <= 0.0001:
                    log_event("INFO", "WORKER", f"No holding of {asset_ticker} ({asset_balance:.4f}) to perform SELL. Skipping.")
                    continue
                
                # If we are BUY and we already have an active position, skip to avoid double exposure
                if trigger_side == "BUY" and active_pos:
                    log_event("INFO", "WORKER", f"Active position already exists for {asset}. Skipping double-exposure BUY order.")
                    continue
                
                # ---------------- DOUBLE-CHECK MATH FILTER (4h Trend & Volume) ----------------
                log_event("INFO", "WORKER", f"Performing Double-Check Math Filter for {asset} on 4h timeframe...")
                candles_df_4h = fetch_ohlcv(symbol=asset, timeframe="4h", limit=100)
                if candles_df_4h is not None and not candles_df_4h.empty:
                    # Check if 4h candles are simulated
                    is_4h_simulated = False
                    if '_is_simulated' in candles_df_4h.columns:
                        is_4h_simulated = bool(candles_df_4h['_is_simulated'].iloc[0])
                    if is_4h_simulated:
                        log_event("WARNING", "WORKER", f"⚠️ 4h timeframe for {asset} is using SIMULATED data. Skipping trade decisions for safety.")
                        _last_analyzed_timestamps[asset] = completed_timestamp
                        continue
                        
                    # Calculate indicators for 4h candles
                    candles_df_4h = calculate_indicators(candles_df_4h)
                    
                    from core.math_engine import confirm_with_higher_tf
                    confirmed, double_check_reason = confirm_with_higher_tf(candles_df, candles_df_4h, trigger_side)
                    
                    if not confirmed:
                        log_event("WARNING", "WORKER", f"❌ {asset} - {trigger_side} trigger REJECTED by 4h double-check filter. Reason: {double_check_reason}")
                        # Mark this completed candle timestamp as analyzed to prevent spamming
                        _last_analyzed_timestamps[asset] = completed_timestamp
                        continue
                    else:
                        log_event("SUCCESS", "WORKER", f"✅ {asset} - {trigger_side} trigger CONFIRMED by 4h double-check. Reason: {double_check_reason}")
                else:
                    log_event("WARNING", "WORKER", f"Could not fetch 4h candles for {asset}. Skipping double-check for safety.")
                    continue
                
                # Add to batch candidates
                candidates_to_analyze.append({
                    "asset": asset,
                    "trigger_side": trigger_side,
                    "trigger_reason": trigger_reason,
                    "completed_timestamp": completed_timestamp,
                    "current_price": current_price,
                    "asset_ticker": asset_ticker,
                    "active_pos": active_pos
                })
                
        except Exception as asset_err:
            log_event("ERROR", "WORKER", f"Error evaluating {asset} inside tick cycle: {asset_err}")
            # Reset analyzed timestamp so we can retry on next tick
            if asset in _last_analyzed_timestamps:
                _last_analyzed_timestamps[asset] = None
            import traceback
            traceback.print_exc()

    # Process batch sentiment for all candidates
    if candidates_to_analyze:
        log_event("INFO", "WORKER", f"Processing news sentiment for {len(candidates_to_analyze)} candidate assets in a single batch...")
        
        # Scrape news for each candidate
        batch_input = []
        successful_candidates = []
        for c in candidates_to_analyze:
            news = fetch_asset_news(c["asset"], limit=10)
            if not news:
                log_event("WARNING", "WORKER", f"Could not fetch live news for {c['asset']}. Skipping sentiment analysis for this cycle to avoid token waste.")
                # We do NOT mark the completed candle timestamp as analyzed, so it will retry next time
                continue
                
            batch_input.append({
                "symbol": c["asset"],
                "news_items": news
            })
            successful_candidates.append(c)
            
        candidates_to_analyze = successful_candidates
            
        # Call batch sentiment analyzer
        from ai.sentiment_analyzer import analyze_sentiment_batch
        ai_results = analyze_sentiment_batch(batch_input)
        
        # Map results back and execute trades
        ai_map = {res["symbol"]: res for res in ai_results}
        
        # Refresh portfolio/nav
        portfolio = get_portfolio()
        total_nav = calculate_total_nav(portfolio)
        
        for c in candidates_to_analyze:
            asset = c["asset"]
            trigger_side = c["trigger_side"]
            trigger_reason = c["trigger_reason"]
            completed_timestamp = c["completed_timestamp"]
            current_price = c["current_price"]
            asset_ticker = c["asset_ticker"]
            active_pos = c["active_pos"]
            
            # Get latest values for usd_balance / asset_balance to account for trades inside the loop
            usd_balance = portfolio.get("USD", {}).get("balance", 0.0)
            asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
            
            ai_res = ai_map.get(asset)
            if not ai_res:
                log_event("WARNING", "WORKER", f"No AI sentiment results returned for {asset}. Skipping.")
                _last_analyzed_timestamps[asset] = completed_timestamp
                continue
                
            sentiment_score = ai_res["sentiment_score"]
            ai_reason = ai_res["reason"]
            is_ai_simulated = ai_res.get("is_simulated", False)
            
            if is_ai_simulated:
                log_event("WARNING", "WORKER", f"⚠️ AI sentiment or news for {asset} is SIMULATED/MOCK. Skipping trade decisions for safety.")
                _last_analyzed_timestamps[asset] = completed_timestamp
                continue
            
            log_event("INFO", "WORKER", f"Gemini Sentiment outcome for {asset}: {sentiment_score} (Reason: '{ai_reason}')")
            
            # Evaluate trade thresholds
            if trigger_side == "BUY":
                if active_pos or usd_balance < 10.0:
                    log_event("INFO", "WORKER", f"Skipping trade execution for {asset} (usd_balance: ${usd_balance:.2f}, active_pos: {active_pos})")
                    _last_analyzed_timestamps[asset] = completed_timestamp
                    continue
                    
                if sentiment_score >= sentiment_threshold:
                    # Confirm paper buy execution using Cash Sizing Rule: risk_pct of Total NAV
                    trade_value = total_nav * (risk_pct / 100.0)
                    trade_value = min(trade_value, usd_balance)
                    
                    if trade_value < 10.0:
                        log_event("WARNING", "WORKER", f"Trade size ${trade_value:.2f} is below $10 minimum. Skipping BUY on {asset}.")
                        _last_analyzed_timestamps[asset] = completed_timestamp
                        continue
                        
                    fee_pct = float(get_config("fee_pct", "0.001"))
                    trade_net_value = trade_value * (1 - fee_pct)
                    amount_to_buy = trade_net_value / current_price
                    
                    # Update portfolio memory
                    portfolio["USD"]["balance"] = usd_balance - trade_value
                    
                    existing_asset = portfolio.get(asset_ticker, {"balance": 0.0, "avg_entry_price": 0.0})
                    old_balance = existing_asset["balance"]
                    old_avg = existing_asset["avg_entry_price"]
                    
                    new_balance = old_balance + amount_to_buy
                    new_avg = ((old_balance * old_avg) + (amount_to_buy * current_price)) / new_balance if new_balance > 0 else current_price
                    
                    portfolio[asset_ticker] = {
                        "balance": new_balance,
                        "avg_entry_price": new_avg
                    }
                    
                    sl_price = current_price * (1 - sl_pct / 100.0)
                    tp_price = current_price * (1 + tp_pct / 100.0)
                    
                    update_portfolio(portfolio)
                    record_trade(
                        asset=asset,
                        side="BUY",
                        price=current_price,
                        amount=amount_to_buy,
                        trade_type="AI_CONFIRMED",
                        sentiment_score=sentiment_score,
                        reason=f"Technical: {trigger_reason} | AI confirmed: {ai_reason}",
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        is_active=1
                    )
                    log_event("SUCCESS", "WORKER", f"🛒 PAPER BUY EXECUTED: Bought {amount_to_buy:.4f} {asset_ticker} at ${current_price:.2f} | SL: ${sl_price:.2f}, TP: ${tp_price:.2f} (NAV size: ${trade_value:.2f})")
                    
                    portfolio = get_portfolio()
                    total_nav = calculate_total_nav(portfolio)
                else:
                    log_event("WARNING", "WORKER", f"❌ Technical BUY filtered out by Gemini AI. Sentiment Score ({sentiment_score}) did not meet threshold (>= {sentiment_threshold}).")
                    
            elif trigger_side == "SELL":
                if asset_balance <= 0.0001:
                    log_event("INFO", "WORKER", f"Skipping sell trade execution for {asset} (asset_balance: {asset_balance:.4f})")
                    _last_analyzed_timestamps[asset] = completed_timestamp
                    continue
                    
                if sentiment_score <= -sentiment_threshold:
                    # Confirm paper sell execution (liquidate full asset holdings)
                    amount_to_sell = asset_balance
                    trade_value = amount_to_sell * current_price
                    fee_pct = float(get_config("fee_pct", "0.001"))
                    trade_net_value = trade_value * (1 - fee_pct)
                    
                    # Update portfolio memory
                    portfolio["USD"]["balance"] = usd_balance + trade_net_value
                    portfolio[asset_ticker] = {
                        "balance": 0.0,
                        "avg_entry_price": 0.0
                    }
                    
                    update_portfolio(portfolio)
                    
                    if active_pos:
                        pnl_pct = close_active_position(asset, current_price, "AI_CONFIRMED", f"Bearish crossover confirmed by AI: {ai_reason}")
                        log_event("SUCCESS", "WORKER", f"💰 PAPER SELL (POSITION CLOSED): Sold {amount_to_sell:.4f} {asset_ticker} at ${current_price:.2f} (Net: ${trade_net_value:.2f}, PnL: {pnl_pct:+.2f}%)")
                    else:
                        record_trade(
                            asset=asset,
                            side="SELL",
                            price=current_price,
                            amount=amount_to_sell,
                            trade_type="AI_CONFIRMED",
                            sentiment_score=sentiment_score,
                            reason=f"Technical: {trigger_reason} | AI confirmed: {ai_reason}",
                            is_active=0
                        )
                        log_event("SUCCESS", "WORKER", f"💰 PAPER SELL EXECUTED: Sold {amount_to_sell:.4f} {asset_ticker} at ${current_price:.2f} (Net: ${trade_net_value:.2f})")
                        
                    portfolio = get_portfolio()
                    total_nav = calculate_total_nav(portfolio)
                else:
                    log_event("WARNING", "WORKER", f"❌ Technical SELL filtered out by Gemini AI. Sentiment Score ({sentiment_score}) did not meet bearish threshold (<= -{sentiment_threshold}).")
            
            # Set this completed candle timestamp to analyzed
            _last_analyzed_timestamps[asset] = completed_timestamp
            
    log_event("INFO", "WORKER", "Algorithmic market analysis tick complete.")

def pd_isna(val):
    """Helper to safely check for pandas nan without importing pandas inside db/schema checks."""
    try:
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        return False
    except Exception:
        return False

def run_sltp_guardian():
    """
    Fast SL/TP guardian loop. Checks real-time prices against
    stop-loss and take-profit levels for ALL active positions.
    This runs much more frequently than the full analysis cycle
    to protect positions during flash crashes.
    """
    active_positions = get_all_active_positions()
    if not active_positions:
        return
    
    fee_pct = float(get_config("fee_pct", "0.001"))
    
    for pos in active_positions:
        asset = pos['asset']
        sl_val = pos.get('stop_loss')
        tp_val = pos.get('take_profit')
        buy_price = pos.get('price')
        
        if not sl_val and not tp_val:
            continue
        
        # Get real-time price
        current_price = fetch_realtime_price(asset)
        if current_price is None:
            log_event("WARNING", "SL_TP_GUARD", f"Cannot fetch price for {asset}. Skipping SL/TP check.")
            continue
        
        asset_ticker = asset.split("/")[0]
        
        # Check Stop-Loss
        if sl_val and current_price <= sl_val:
            log_event("WARNING", "SL_TP_GUARD", f"🚨 STOP LOSS BREACHED! Real-time ${current_price:.2f} <= SL ${sl_val:.2f} for {asset}")
            
            portfolio = get_portfolio()  # Fresh read
            asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
            trade_value = asset_balance * current_price
            trade_net_value = trade_value * (1 - fee_pct)
            
            portfolio["USD"]["balance"] = portfolio.get("USD", {}).get("balance", 0.0) + trade_net_value
            portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
            update_portfolio(portfolio)
            
            pnl_pct = close_active_position(asset, current_price, "STOP_LOSS", f"SL Guardian: breached at ${sl_val:.2f}")
            log_event("SUCCESS", "SL_TP_GUARD", f"💀 Stop Loss liquidation for {asset}: PnL {pnl_pct:+.2f}%")
        
        # Check Take-Profit
        elif tp_val and current_price >= tp_val:
            log_event("SUCCESS", "SL_TP_GUARD", f"🎯 TAKE PROFIT BREACHED! Real-time ${current_price:.2f} >= TP ${tp_val:.2f} for {asset}")
            
            portfolio = get_portfolio()  # Fresh read
            asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
            trade_value = asset_balance * current_price
            trade_net_value = trade_value * (1 - fee_pct)
            
            portfolio["USD"]["balance"] = portfolio.get("USD", {}).get("balance", 0.0) + trade_net_value
            portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
            update_portfolio(portfolio)
            
            pnl_pct = close_active_position(asset, current_price, "TAKE_PROFIT", f"TP Guardian: breached at ${tp_val:.2f}")
            log_event("SUCCESS", "SL_TP_GUARD", f"🎉 Take Profit liquidation for {asset}: PnL {pnl_pct:+.2f}%")

def main():
    """Main worker initialization and execution loop with dual-speed architecture."""
    print("====================================================")
    print("       SENTIX ALGORITHMIC TRADING BG WORKER         ")
    print("    Dual-Loop: SL/TP Guardian (30s) + Analysis (300s)")
    print("====================================================")
    
    # Initialize the database
    init_db()
    
    # Save default configurations if not existing in SQLite
    if not get_config("selected_assets"):
        save_config("selected_assets", os.getenv("SELECTED_ASSETS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT,XRP/USDT,DOT/USDT,AVAX/USDT,LINK/USDT,NEAR/USDT,MATIC/USDT"))
    if not get_config("min_ai_sentiment_threshold"):
        save_config("min_ai_sentiment_threshold", os.getenv("MIN_AI_SENTIMENT_THRESHOLD", "3"))
    if not get_config("summarizer_model"):
        save_config("summarizer_model", os.getenv("SUMMARIZER_MODEL", "gemini-3.1-flash-lite"))
    if not get_config("sentiment_model"):
        save_config("sentiment_model", os.getenv("SENTIMENT_MODEL", "gemini-3.5-flash"))
    if not get_config("gemini_api_key"):
        save_config("gemini_api_key", os.getenv("GEMINI_API_KEY", ""))
    if not get_config("risk_percentage"):
        save_config("risk_percentage", "2.0")
    if not get_config("stop_loss_pct"):
        save_config("stop_loss_pct", "3.0")
    if not get_config("take_profit_pct"):
        save_config("take_profit_pct", "6.0")
    if not get_config("fee_pct"):
        save_config("fee_pct", "0.001")
    if not get_config("news_freshness_hours"):
        save_config("news_freshness_hours", "24")
    if not get_config("live_mode"):
        save_config("live_mode", os.getenv("LIVE_MODE", "false"))

    log_event("INFO", "WORKER", "Background worker initialized with dual-loop architecture.")
    
    # Configurable intervals
    analysis_interval = int(os.getenv("SIMULATION_INTERVAL_SECONDS", "300"))  # Full analysis every 5 min (was 60s)
    guardian_interval = int(os.getenv("GUARDIAN_INTERVAL_SECONDS", "30"))      # SL/TP check every 30 seconds
    
    log_event("INFO", "WORKER", f"Analysis interval: {analysis_interval}s | SL/TP Guardian interval: {guardian_interval}s")
    
    # Fast first tick
    try:
        save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())
    except Exception:
        pass
    
    from core.data_fetcher import check_vpn_connection
    while not check_vpn_connection():
        log_event("WARNING", "WORKER", "⚠️ VPN connection is down on startup! Pausing all operations. Retrying VPN check in 15 seconds...")
        save_config("vpn_status", "disconnected")
        time.sleep(15)
        
    save_config("vpn_status", "connected")
    run_worker_cycle()
    
    # Dual-speed continuous loop
    last_analysis_time = time.time()
    last_vpn_check_time = time.time()
    vpn_connected = True
    
    while True:
        try:
            # Update heartbeat to notify UI worker is alive
            try:
                save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())
            except Exception:
                pass
                
            # Determine if we should check VPN
            now = time.time()
            elapsed_analysis = now - last_analysis_time
            should_run_analysis = (elapsed_analysis >= analysis_interval)
            
            # Check VPN if disconnected, if 5 minutes elapsed, or if about to run analysis
            should_check_vpn = (not vpn_connected) or (now - last_vpn_check_time >= 300) or should_run_analysis
            
            if should_check_vpn:
                vpn_connected = check_vpn_connection()
                last_vpn_check_time = now
                
                if not vpn_connected:
                    log_event("WARNING", "WORKER", "⚠️ VPN connection is down! Pausing all trading operations and SL/TP guardian checks. Retrying VPN check in 15 seconds...")
                    save_config("vpn_status", "disconnected")
                    time.sleep(15)
                    continue
                    
                save_config("vpn_status", "connected")
            
            # Always run fast SL/TP guardian check
            run_sltp_guardian()
            
            # Run full analysis cycle at the configured (slower) interval
            if should_run_analysis:
                log_event("INFO", "WORKER", f"Running full analysis cycle (elapsed: {elapsed_analysis:.0f}s)...")
                run_worker_cycle()
                last_analysis_time = time.time()
            
            time.sleep(guardian_interval)
            
        except KeyboardInterrupt:
            log_event("INFO", "WORKER", "Worker shutdown requested by user.")
            break
        except Exception as e:
            log_event("ERROR", "WORKER", f"Unhandled error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(guardian_interval)  # Wait and retry

if __name__ == "__main__":
    main()
