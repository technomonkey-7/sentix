import time
import os
import math
import ccxt
from datetime import datetime, timezone
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
def _get_exchange():
    """Returns a shared configured ccxt exchange instance."""
    global _exchange
    if _exchange is None:
        from core.config import get_config
        exchange_name = get_config("exchange_name") or os.getenv("EXCHANGE_NAME", "binance")
        exchange_class = getattr(ccxt, exchange_name.lower(), ccxt.binance)
        _exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    return _exchange

def fetch_realtime_price(symbol):
    """
    Fetches the current real-time price for a symbol using ccxt ticker.
    Falls back to the latest DB candle close price if the ticker call fails.
    This is critical for accurate SL/TP monitoring.
    """
    try:
        exchange = _get_exchange()
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker['last'])
        return price
    except Exception as e:
        log_event("WARNING", "WORKER", f"Real-time ticker fetch failed for {symbol}: {e}. Falling back to DB candle price.")
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

def run_worker_cycle():
    """
    Executes a single cycle of the background worker:
    1. Reads latest configurations.
    2. Fetches prices and computes indicators for all active assets.
    3. Runs real-time SL/TP monitor to liquidate open positions immediately when hit.
    4. Runs deterministic technical analysis to check for crossover triggers.
    5. If triggered, initiates live Google News scraping and hybrid Gemini API sentiment check.
    6. Handles paper trading buy/sell executions and state saves.
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
    
    # Process each asset independently
    for asset in active_assets:
        try:
            # Update heartbeat inside the loop so UI knows bot is alive even during a long analysis cycle
            try:
                save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())
            except Exception:
                pass
                
            log_event("INFO", "WORKER", f"Analyzing asset: {asset}")
            
            # Fetch 1h and 4h historical candle data (defaulting to 1h for signals)
            candles_df = fetch_ohlcv(symbol=asset, timeframe="1h", limit=100)
            
            if candles_df is None or candles_df.empty:
                log_event("WARNING", "WORKER", f"No candles fetched for {asset}. Skipping.")
                continue
                
            # Compute indicators (RSI, EMA, MACD)
            candles_df = calculate_indicators(candles_df)
            
            # Persist candles to database
            # Convert DF to list of dicts for the DB helper
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
            
            # Use real-time ticker price for SL/TP monitoring (more accurate than stale candle)
            realtime_price = fetch_realtime_price(asset)
            current_price = realtime_price if realtime_price else candle_close_price
            
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
                
            if ema_val is not None:
                ema_diff_pct = ((current_price - ema_val) / ema_val) * 100
                log_event("INFO", "MATH", f"{asset} - EMA 20: ${ema_val:.2f}, Fiyat: ${current_price:.2f} (Eşik farkı: %{ema_diff_pct:+.2f})")
            else:
                log_event("INFO", "MATH", f"{asset} - EMA 20: Veri yetersiz")
            
            # ---------------- RISK CHECK: STOP LOSS & TAKE PROFIT MONITORING ----------------
            active_pos = get_active_position(asset)
            if active_pos:
                sl_val = active_pos.get("stop_loss")
                tp_val = active_pos.get("take_profit")
                buy_price = active_pos.get("price")
                log_event("INFO", "WORKER", f"Active position found for {asset} at entry price ${buy_price:.2f}. SL: ${sl_val:.2f}, TP: ${tp_val:.2f}. Real-time Price: ${current_price:.2f}")
                
                # RACE CONDITION FIX: Re-read portfolio from DB before any liquidation
                portfolio = get_portfolio()
                fee_pct = float(get_config("fee_pct", "0.001"))  # Configurable fee
                
                # Check Stop-Loss
                if sl_val and current_price <= sl_val:
                    log_event("WARNING", "WORKER", f"🚨 STOP LOSS BREACHED! Real-time price ${current_price:.2f} <= SL ${sl_val:.2f} for {asset}. Triggering liquidation...")
                    
                    asset_ticker = asset.split("/")[0]
                    asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
                    trade_value = asset_balance * current_price
                    trade_net_value = trade_value * (1 - fee_pct)
                    
                    # Process database updates (re-read portfolio to avoid race condition)
                    portfolio["USD"]["balance"] = portfolio.get("USD", {}).get("balance", 0.0) + trade_net_value
                    portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
                    update_portfolio(portfolio)
                    
                    pnl_pct = close_active_position(asset, current_price, "STOP_LOSS", f"Stop Loss breached at ${sl_val:.2f}")
                    log_event("SUCCESS", "WORKER", f"💀 Stop Loss liquidation completed for {asset} with PnL: {pnl_pct:+.2f}%")
                    # Re-read portfolio after trade to prevent race conditions with next asset
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
                    
                    # Process database updates
                    portfolio["USD"]["balance"] = portfolio.get("USD", {}).get("balance", 0.0) + trade_net_value
                    portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
                    update_portfolio(portfolio)
                    
                    pnl_pct = close_active_position(asset, current_price, "TAKE_PROFIT", f"Take Profit breached at ${tp_val:.2f}")
                    log_event("SUCCESS", "WORKER", f"🎉 Take Profit liquidation completed for {asset} with PnL: {pnl_pct:+.2f}%")
                    # Re-read portfolio after trade to prevent race conditions with next asset
                    portfolio = get_portfolio()
                    total_nav = calculate_total_nav(portfolio)
                    continue
            
            # Check for deterministic crossover triggers
            trigger_side, trigger_reason = check_triggers(candles_df)
            
            if trigger_side:
                log_event("WARNING", "WORKER", f"💥 Technical [{trigger_side}] Trigger detected for {asset}! Reason: {trigger_reason}")
                
                # RACE CONDITION FIX: Re-read fresh portfolio state before trade decisions
                portfolio = get_portfolio()
                total_nav = calculate_total_nav(portfolio)
                usd_balance = portfolio.get("USD", {}).get("balance", 0.0)
                asset_ticker = asset.split("/")[0]
                asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
                
                if trigger_side == "BUY" and usd_balance < 10.0:
                    log_event("INFO", "WORKER", f"Insufficient USD balance (${usd_balance:.2f}) to perform BUY on {asset}. Skipping Gemini call.")
                    continue
                elif trigger_side == "SELL" and asset_balance <= 0.0001:
                    log_event("INFO", "WORKER", f"No holding of {asset_ticker} ({asset_balance:.4f}) to perform SELL. Skipping Gemini call.")
                    continue
                
                # If we are BUY and we already have an active position, skip to avoid double exposure
                if trigger_side == "BUY" and active_pos:
                    log_event("INFO", "WORKER", f"Active position already exists for {asset}. Skipping double-exposure BUY order.")
                    continue
                
                # STEP 2 & 3: Conditional AI news sentiment verification
                log_event("INFO", "WORKER", f"Triggering Conditional AI Sentiment check for {asset}...")
                news = fetch_asset_news(asset, limit=10)
                ai_result = analyze_sentiment(asset, news)
                
                sentiment_score = ai_result["sentiment_score"]
                ai_reason = ai_result["reason"]
                
                log_event("INFO", "WORKER", f"Gemini Sentiment outcome: {sentiment_score} (Reason: '{ai_reason}')")
                
                # Evaluate final trade decision using technical triggers and AI scores
                if trigger_side == "BUY":
                    if sentiment_score >= sentiment_threshold:
                        # Confirm paper buy execution using Cash Sizing Rule: risk_pct of Total NAV
                        trade_value = total_nav * (risk_pct / 100.0)
                        trade_value = min(trade_value, usd_balance)
                        
                        if trade_value < 10.0:
                            log_event("WARNING", "WORKER", f"Trade size ${trade_value:.2f} is below $10 minimum. Skipping BUY on {asset}.")
                            continue
                            
                        fee_pct = float(get_config("fee_pct", "0.001"))  # Configurable exchange fee
                        trade_net_value = trade_value * (1 - fee_pct)
                        amount_to_buy = trade_net_value / current_price
                        
                        # Update portfolio memory
                        portfolio["USD"]["balance"] = usd_balance - trade_value
                        
                        existing_asset = portfolio.get(asset_ticker, {"balance": 0.0, "avg_entry_price": 0.0})
                        old_balance = existing_asset["balance"]
                        old_avg = existing_asset["avg_entry_price"]
                        
                        new_balance = old_balance + amount_to_buy
                        # Weighted average entry price
                        new_avg = ((old_balance * old_avg) + (amount_to_buy * current_price)) / new_balance if new_balance > 0 else current_price
                        
                        portfolio[asset_ticker] = {
                            "balance": new_balance,
                            "avg_entry_price": new_avg
                        }
                        
                        # Calculate Stop Loss and Take Profit
                        sl_price = current_price * (1 - sl_pct / 100.0)
                        tp_price = current_price * (1 + tp_pct / 100.0)
                        
                        # Persist to database
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
                        # RACE CONDITION FIX: Re-read portfolio after successful trade
                        portfolio = get_portfolio()
                        total_nav = calculate_total_nav(portfolio)
                    else:
                        log_event("WARNING", "WORKER", f"❌ Technical BUY filtered out by Gemini AI. Sentiment Score ({sentiment_score}) did not meet threshold (>= {sentiment_threshold}).")
                        
                elif trigger_side == "SELL":
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
                        
                        # Persist to database
                        update_portfolio(portfolio)
                        
                        # Close active position in database if exists
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
                        # RACE CONDITION FIX: Re-read portfolio after trade
                        portfolio = get_portfolio()
                        total_nav = calculate_total_nav(portfolio)
                    else:
                        log_event("WARNING", "WORKER", f"❌ Technical SELL filtered out by Gemini AI. Sentiment Score ({sentiment_score}) did not meet bearish threshold (<= -{sentiment_threshold}).")
            else:
                log_event("INFO", "WORKER", f"No technical crossovers for {asset}. Bypassing Gemini API to conserve tokens.")
                
        except Exception as asset_err:
            log_event("ERROR", "WORKER", f"Error evaluating {asset} inside tick cycle: {asset_err}")
            import traceback
            traceback.print_exc()
            
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
        save_config("summarizer_model", os.getenv("SUMMARIZER_MODEL", "gemini-2.5-flash"))
    if not get_config("sentiment_model"):
        save_config("sentiment_model", os.getenv("SENTIMENT_MODEL", "gemini-2.5-pro"))
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
    run_worker_cycle()
    
    # Dual-speed continuous loop
    last_analysis_time = time.time()
    
    while True:
        try:
            # Update heartbeat to notify UI worker is alive
            try:
                save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())
            except Exception:
                pass
                
            # Always run fast SL/TP guardian check
            run_sltp_guardian()
            
            # Run full analysis cycle at the configured (slower) interval
            elapsed = time.time() - last_analysis_time
            if elapsed >= analysis_interval:
                log_event("INFO", "WORKER", f"Running full analysis cycle (elapsed: {elapsed:.0f}s)...")
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
