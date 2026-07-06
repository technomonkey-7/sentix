"""Sentix v2 background worker.

Dual-loop architecture:
- Analysis loop (default 300s): evaluates the strategy on completed candles,
  handles entries and structural exits.
- Guardian loop (default 60s): real-time stop-loss / take-profit / trailing
  stop protection for open positions.

All trading logic lives in core.strategy / core.risk / core.accounting —
this file only orchestrates.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from core.db import (
    init_db, log_event, save_candles, get_config, save_config,
    get_active_position, get_all_active_positions, get_cooldown,
    record_equity_snapshot, save_signal, update_trade_fields,
)
from core.config import StrategyConfig
from core.data_fetcher import fetch_ohlcv, fetch_asset_news, fetch_realtime_price, resample_4h
from core.indicators import add_indicators
from core import strategy, risk as riskmod, accounting

load_dotenv()

# In-process cache for slow-moving daily frames: {symbol: (fetched_at, df)}
_daily_cache = {}
_DAILY_TTL_SECONDS = 6 * 3600


def calculate_total_nav(portfolio=None):
    """Compatibility wrapper used by the Telegram bot and UI."""
    nav, _, _ = accounting.calculate_nav()
    return nav


def validate_api_key_for_start():
    """True when at least one Gemini API key is configured."""
    from ai.sentiment_analyzer import load_api_keys
    if load_api_keys():
        return True
    key = get_config("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    return bool(key and key.strip())


def _get_daily_frame(symbol):
    """Daily candles with indicators, cached for 6h (they only change once a day)."""
    now = time.time()
    cached = _daily_cache.get(symbol)
    if cached and now - cached[0] < _DAILY_TTL_SECONDS:
        return cached[1]
    df = fetch_ohlcv(symbol, "1d", limit=400)
    if df is not None and not df.empty:
        df = add_indicators(df)
        _daily_cache[symbol] = (now, df)
    return df


def _persist_candles(symbol, df_1h):
    candles = []
    for _, row in df_1h.iterrows():
        def _f(key):
            val = row.get(key)
            try:
                import pandas as pd
                return None if val is None or pd.isna(val) else float(val)
            except (TypeError, ValueError):
                return None
        candles.append({
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "asset": symbol, "interval": "1h",
            "open": _f("open"), "high": _f("high"), "low": _f("low"),
            "close": _f("close"), "volume": _f("volume"),
            "rsi": _f("rsi"), "macd": _f("macd"), "macd_signal": _f("macd_signal"),
            "macd_hist": _f("macd_hist"), "ema": _f("ema"),
        })
    save_candles(candles)


def _load_analyzed():
    try:
        return json.loads(get_config("analyzed_candles") or "{}")
    except (TypeError, ValueError):
        return {}


def _save_analyzed(analyzed):
    try:
        save_config("analyzed_candles", json.dumps(analyzed))
    except Exception:
        pass


def run_worker_cycle(force=False):
    """Acquires the cycle lock and runs one analysis cycle."""
    lock_time_str = get_config("worker_lock_time")
    now_utc = datetime.now(timezone.utc)
    if lock_time_str:
        try:
            lock_time = datetime.fromisoformat(lock_time_str)
            if now_utc - lock_time < timedelta(minutes=3):
                log_event("WARNING", "WORKER", "Another analysis cycle is already running; skipping.")
                if force:
                    raise RuntimeError("Başka bir analiz döngüsü şu anda çalışıyor. Lütfen biraz bekleyin.")
                return
        except RuntimeError:
            raise
        except Exception:
            pass

    save_config("worker_lock_time", now_utc.isoformat())
    try:
        _execute_cycle_logic(force=force)
    finally:
        save_config("worker_lock_time", "")


def _execute_cycle_logic(force=False):
    if not force and get_config("bot_running", "false") == "false":
        return

    cfg = StrategyConfig.from_db()
    log_event("INFO", "WORKER", f"Analysis cycle started for {len(cfg.watchlist)} symbols.")

    nav, cash, exposure = accounting.calculate_nav()
    record_equity_snapshot(nav, cash)
    cb_active, cb_detail = accounting.check_circuit_breaker(nav, cfg)
    if cb_active:
        log_event("WARNING", "RISK", f"Circuit breaker: {cb_detail}")

    ai_available = cfg.ai_enabled and validate_api_key_for_start()
    if cfg.ai_enabled and not ai_available:
        log_event("WARNING", "WORKER", "AI enabled but no Gemini key found — running technical-only.")

    # Benchmark daily frames (SPY for equities, BTC-USD when crypto is watched)
    benchmarks = {}
    if cfg.market_filter_enabled:
        benchmarks["equity"] = _get_daily_frame(cfg.benchmark_equity)
        if any(riskmod.is_crypto(s) for s in cfg.watchlist):
            benchmarks["crypto"] = _get_daily_frame(cfg.benchmark_crypto)

    analyzed = _load_analyzed()
    now_utc = datetime.now(timezone.utc)
    candidates = []

    for symbol in cfg.watchlist:
        try:
            save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())

            df_1h = fetch_ohlcv(symbol, "1h", limit=300)
            if df_1h is None or len(df_1h) < 60:
                log_event("WARNING", "WORKER", f"{symbol}: no/insufficient 1h data — skipping this cycle.")
                continue
            df_1h = add_indicators(df_1h)
            _persist_candles(symbol, df_1h)

            completed = df_1h.iloc[-2]
            completed_ts = completed["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            last_ts = df_1h.iloc[-1]["timestamp"].to_pydatetime()

            fresh = riskmod.data_is_fresh(symbol, last_ts, 60, now_utc)
            market_open = riskmod.is_market_open(symbol, now_utc)

            position = get_active_position(symbol)
            df_daily = _get_daily_frame(symbol)

            # ---------- Open position: structural exit checks ----------
            if position:
                exit_sig = strategy.evaluate_exit(position, df_daily, cfg, now_utc)
                if exit_sig.should_exit and market_open and fresh:
                    price = fetch_realtime_price(symbol) or float(completed["close"])
                    result = accounting.execute_sell(symbol, price, cfg,
                                                     trade_type=exit_sig.code, reason=exit_sig.detail)
                    if result:
                        log_event("SUCCESS", "WORKER",
                                  f"📤 {symbol} closed ({exit_sig.code}): {exit_sig.detail} "
                                  f"PnL {result['pnl_pct']:+.2f}%")
                    save_signal(symbol, "EXIT", None, price, json.dumps(
                        {"reason": exit_sig.detail, "code": exit_sig.code}))
                else:
                    save_signal(symbol, "HOLD", None, float(completed["close"]), json.dumps(
                        {"reason": "Position open; SL/TP guardian active"}))
                continue

            # ---------- No position: entry evaluation ----------
            op_gates = []
            op_gates.append(strategy.GateCheck("market_open", market_open,
                                               "Market open" if market_open else "Market closed"))
            op_gates.append(strategy.GateCheck("data_fresh", fresh,
                                               f"Last candle {last_ts.isoformat()}"))
            cooldown_until = get_cooldown(symbol)
            op_gates.append(strategy.GateCheck("cooldown", cooldown_until is None,
                                               f"Cooling down until {cooldown_until}" if cooldown_until else "No cooldown"))
            op_gates.append(strategy.GateCheck("circuit_breaker", not cb_active, cb_detail))

            already_analyzed = (not force and analyzed.get(symbol) == completed_ts)

            # drop the in-progress 4h bucket: strategy expects completed bars only
            df_4h = add_indicators(resample_4h(df_1h).iloc[:-1])
            bench = benchmarks.get("crypto" if riskmod.is_crypto(symbol) else "equity")
            sig = strategy.evaluate_entry(df_1h, df_4h, df_daily, bench, cfg, sentiment=None)

            all_gates = op_gates + sig.gates
            details = {
                "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in all_gates],
                "factors": sig.factors, "score": sig.score, "reason": sig.reason,
            }

            if not all(g.passed for g in op_gates) or not sig.should_enter or already_analyzed:
                decision = "SKIP"
                if already_analyzed and sig.should_enter:
                    details["reason"] = "Candle already analyzed (duplicate suppression)"
                save_signal(symbol, decision, sig.score, float(completed["close"]), json.dumps(details))
                continue

            log_event("INFO", "WORKER",
                      f"💡 {symbol}: entry candidate ({sig.trigger}) score {sig.score}. {sig.reason}")
            candidates.append({
                "symbol": symbol, "df_1h": df_1h, "df_4h": df_4h, "df_daily": df_daily,
                "bench": bench, "completed_ts": completed_ts, "details": details,
            })
        except Exception as err:
            log_event("ERROR", "WORKER", f"Error evaluating {symbol}: {err}")
            import traceback
            traceback.print_exc()

    # ---------- AI sentiment for candidates (batched, optional) ----------
    sentiment_map = {}
    if candidates and ai_available:
        batch_input = []
        for c in candidates:
            news = []
            try:
                news = fetch_asset_news(c["symbol"], limit=10)
            except Exception as e:
                log_event("WARNING", "WORKER", f"News fetch failed for {c['symbol']}: {e}")
            if news:
                batch_input.append({"symbol": c["symbol"], "news_items": news,
                                    "technical_summary": c["details"].get("reason", "")})
        if batch_input:
            try:
                from ai.sentiment_analyzer import analyze_sentiment_batch
                for res in analyze_sentiment_batch(batch_input):
                    if not res.get("is_simulated"):
                        sentiment_map[res["symbol"]] = int(res["sentiment_score"])
            except Exception as e:
                log_event("WARNING", "WORKER", f"AI sentiment batch failed ({e}); continuing technical-only.")

    # ---------- Execute entries ----------
    for c in candidates:
        symbol = c["symbol"]
        try:
            sentiment = sentiment_map.get(symbol)
            entry_price = fetch_realtime_price(symbol)
            if entry_price is None:
                entry_price = float(c["df_1h"].iloc[-2]["close"])

            sig = strategy.evaluate_entry(c["df_1h"], c["df_4h"], c["df_daily"], c["bench"],
                                          cfg, sentiment=sentiment, entry_price=entry_price)
            details = {
                "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in sig.gates],
                "factors": sig.factors, "score": sig.score, "reason": sig.reason,
                "sentiment": sentiment,
            }
            analyzed[symbol] = c["completed_ts"]

            if not sig.should_enter:
                save_signal(symbol, "SKIP", sig.score, entry_price, json.dumps(details))
                log_event("INFO", "WORKER", f"{symbol}: entry dropped after AI check. {sig.reason}")
                continue

            nav, cash, exposure = accounting.calculate_nav()
            open_count = accounting.count_open_positions()
            sizing = riskmod.compute_position_size(nav, cash, entry_price, sig.stop,
                                                   sig.conf_mult, exposure, open_count, cfg)
            if not sizing.ok:
                details["sizing"] = sizing.reason
                save_signal(symbol, "SKIP", sig.score, entry_price, json.dumps(details))
                log_event("INFO", "WORKER", f"{symbol}: sizing rejected — {sizing.reason}")
                continue

            trade = accounting.execute_buy(
                symbol, entry_price, sizing.quantity, cfg,
                trade_type="STRATEGY", sentiment_score=sentiment,
                reason=f"{sig.reason} | {sizing.reason}",
                stop_loss=sig.stop, take_profit=sig.take_profit,
                trail_dist=sig.trail_dist, entry_atr=sig.entry_atr,
                confluence_score=sig.score)
            if trade:
                details["sizing"] = sizing.reason
                save_signal(symbol, "ENTER", sig.score, trade["price"], json.dumps(details))
                log_event("SUCCESS", "WORKER",
                          f"🛒 BUY {symbol}: {trade['amount']:.4f} @ ${trade['price']:.2f} "
                          f"(score {sig.score}, risk ${sizing.risk_amount:.2f}) "
                          f"SL ${sig.stop:.2f} TP ${sig.take_profit:.2f}")
        except Exception as err:
            log_event("ERROR", "WORKER", f"Entry execution failed for {symbol}: {err}")

    _save_analyzed(analyzed)
    nav, cash, _ = accounting.calculate_nav()
    record_equity_snapshot(nav, cash, min_gap_seconds=0)
    log_event("INFO", "WORKER", f"Analysis cycle complete. NAV ${nav:.2f}, cash ${cash:.2f}.")


def run_sltp_guardian():
    """Fast loop: trailing-stop updates and SL/TP enforcement on live prices."""
    if get_config("bot_running", "false") != "true":
        return
    positions = get_all_active_positions()
    if not positions:
        return

    cfg = StrategyConfig.from_db()
    now_utc = datetime.now(timezone.utc)

    for pos in positions:
        symbol = pos["asset"]
        try:
            if not riskmod.is_market_open(symbol, now_utc):
                continue  # never act on stale off-hours prices

            price = fetch_realtime_price(symbol)
            if price is None:
                log_event("WARNING", "GUARDIAN", f"{symbol}: no live price; skipping check.")
                continue

            new_stop, new_hwm = strategy.update_trailing_stop(pos, price, cfg)
            old_stop = pos.get("stop_loss")
            if new_stop is not None and (old_stop is None or new_stop > float(old_stop) + 1e-9):
                update_trade_fields(pos["id"], stop_loss=new_stop, high_watermark=new_hwm)
                log_event("INFO", "GUARDIAN",
                          f"📈 {symbol}: trailing stop {float(old_stop or 0):.2f} → {new_stop:.2f} "
                          f"(price ${price:.2f})")
                pos["stop_loss"] = new_stop
            elif new_hwm > float(pos.get("high_watermark") or 0):
                update_trade_fields(pos["id"], high_watermark=new_hwm)

            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            if sl is not None and price <= float(sl):
                log_event("WARNING", "GUARDIAN", f"🚨 {symbol}: STOP hit (${price:.2f} <= ${float(sl):.2f})")
                accounting.execute_sell(symbol, price, cfg, trade_type="STOP_LOSS",
                                        reason=f"Stop-loss hit at ${float(sl):.2f}")
            elif tp is not None and price >= float(tp):
                log_event("SUCCESS", "GUARDIAN", f"🎯 {symbol}: TAKE PROFIT hit (${price:.2f} >= ${float(tp):.2f})")
                accounting.execute_sell(symbol, price, cfg, trade_type="TAKE_PROFIT",
                                        reason=f"Take-profit hit at ${float(tp):.2f}")
        except Exception as e:
            log_event("ERROR", "GUARDIAN", f"Guardian error for {symbol}: {e}")


def run_analysis_scheduler_loop():
    log_event("INFO", "WORKER", "Analysis scheduler thread started.")
    last_run = 0.0
    while True:
        try:
            if get_config("bot_running", "false") != "true":
                time.sleep(10)
                continue
            if get_config("vpn_check_enabled", "false") == "true" and \
                    get_config("vpn_status", "disconnected") != "connected":
                time.sleep(15)
                continue
            try:
                interval = int(get_config("simulation_interval_seconds")
                               or os.getenv("SIMULATION_INTERVAL_SECONDS", "300"))
            except (TypeError, ValueError):
                interval = 300
            if last_run == 0.0 or time.time() - last_run >= interval:
                try:
                    run_worker_cycle()
                except Exception as cycle_err:
                    log_event("ERROR", "WORKER", f"Cycle error: {cycle_err}")
                last_run = time.time()
            time.sleep(10)
        except Exception as e:
            log_event("ERROR", "WORKER", f"Scheduler loop error: {e}")
            time.sleep(10)


def _init_default_config():
    defaults = {
        "selected_assets": ",".join(StrategyConfig().watchlist),
        "min_ai_sentiment_threshold": "3",
        "summarizer_model": "gemini-3.1-flash-lite",
        "sentiment_model": "gemini-3.5-flash",
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "risk_per_trade_pct": "1.0",
        "max_open_positions": "5",
        "max_position_pct": "20.0",
        "max_total_exposure_pct": "80.0",
        "daily_loss_limit_pct": "3.0",
        "cooldown_hours": "24",
        "atr_mult_sl": "2.5",
        "rr_ratio": "2.0",
        "trail_atr_mult": "2.5",
        "fee_pct": "0.0005",
        "slippage_pct": "0.0005",
        "market_filter_enabled": "true",
        "ai_enabled": "true",
        "news_freshness_hours": "24",
        "vpn_check_enabled": "false",
        "bot_running": "true",
        "live_mode": os.getenv("LIVE_MODE", "false"),
    }
    for key, value in defaults.items():
        if not get_config(key):
            save_config(key, value)


def main():
    print("=" * 56)
    print("        SENTIX v2 TRADING WORKER (paper trading)")
    print("   Analysis loop + Guardian loop | ATR risk engine")
    print("=" * 56)

    init_db()
    _init_default_config()
    log_event("INFO", "WORKER", "Worker initialized (Sentix v2 engine).")

    guardian_interval = int(os.getenv("GUARDIAN_INTERVAL_SECONDS", "60"))
    save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())

    if get_config("vpn_check_enabled", "false") == "true":
        from core.data_fetcher import check_vpn_connection
        while not check_vpn_connection():
            log_event("WARNING", "WORKER", "VPN down on startup; retrying in 15s...")
            save_config("vpn_status", "disconnected")
            time.sleep(15)
    save_config("vpn_status", "connected")

    try:
        from core.telegram_bot import start_telegram_bot, send_telegram_message
        start_telegram_bot()
        send_telegram_message("🟢 *Sentix v2 başlatıldı!* ATR tabanlı risk motoru aktif.")
    except Exception as tg_err:
        log_event("WARNING", "WORKER", f"Telegram bot not started: {tg_err}")

    threading.Thread(target=run_analysis_scheduler_loop, daemon=True,
                     name="AnalysisSchedulerThread").start()

    last_vpn_check, vpn_connected = time.time(), True
    while True:
        try:
            save_config("worker_heartbeat", datetime.now(timezone.utc).isoformat())
            if get_config("bot_running", "false") != "true":
                time.sleep(5)
                continue

            if get_config("vpn_check_enabled", "false") == "true":
                now = time.time()
                if (not vpn_connected) or (now - last_vpn_check >= 300):
                    from core.data_fetcher import check_vpn_connection
                    state = check_vpn_connection()
                    last_vpn_check = now
                    if state != vpn_connected:
                        try:
                            from core.telegram_bot import send_telegram_message, get_vpn_logs
                            if state:
                                send_telegram_message("🔒 *VPN bağlantısı geri geldi.* Koruma devam ediyor.")
                            else:
                                send_telegram_message("🚨 *VPN koptu!* Tüm işlemler durduruldu.\n\n"
                                                      f"```\n{get_vpn_logs(tail=5)}\n```")
                        except Exception:
                            pass
                    vpn_connected = state
                    save_config("vpn_status", "connected" if state else "disconnected")
                    if not state:
                        time.sleep(15)
                        continue

            run_sltp_guardian()
            time.sleep(guardian_interval)
        except KeyboardInterrupt:
            log_event("INFO", "WORKER", "Worker shutdown requested.")
            break
        except Exception as e:
            log_event("ERROR", "WORKER", f"Main loop error: {e}")
            time.sleep(guardian_interval)


if __name__ == "__main__":
    main()
