"""Atomic paper-trade accounting.

Every balance mutation happens inside a single SQLite ``BEGIN IMMEDIATE``
transaction so the guardian thread and the analysis thread can never lose or
duplicate cash (the v1 read-modify-write race). PnL is fee- and
slippage-inclusive.
"""
from datetime import datetime, timedelta, timezone

from core.config import StrategyConfig
from core.db import get_connection, log_event, set_cooldown
from core import risk as riskmod


def _latest_close(cursor, asset: str):
    """Latest stored close for an asset; tolerates legacy 'AAPL/USD' rows."""
    cursor.execute("SELECT close FROM candles WHERE asset = ? ORDER BY timestamp DESC LIMIT 1", (asset,))
    row = cursor.fetchone()
    if row:
        return float(row["close"])
    cursor.execute("SELECT close FROM candles WHERE asset LIKE ? ORDER BY timestamp DESC LIMIT 1",
                   (f"{asset}/%",))
    row = cursor.fetchone()
    return float(row["close"]) if row else None


def calculate_nav(price_overrides: dict | None = None):
    """Returns (nav, cash, exposure). Holdings are valued at the freshest
    known price: an override (real-time) if given, else the latest candle."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM portfolio")
        rows = cursor.fetchall()
        cash, exposure = 0.0, 0.0
        for r in rows:
            if r["asset"] == "USD":
                cash = float(r["balance"] or 0.0)
                continue
            qty = float(r["balance"] or 0.0)
            if qty <= 0:
                continue
            price = None
            if price_overrides and r["asset"] in price_overrides:
                price = price_overrides[r["asset"]]
            if price is None:
                price = _latest_close(cursor, r["asset"])
            if price is None:
                price = float(r["avg_entry_price"] or 0.0)
            exposure += qty * price
        return cash + exposure, cash, exposure
    finally:
        conn.close()


def count_open_positions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS n FROM trades WHERE is_active = 1")
    n = cursor.fetchone()["n"]
    conn.close()
    return int(n)


def execute_buy(asset: str, price: float, quantity: float, cfg: StrategyConfig, *,
                trade_type="STRATEGY", sentiment_score=None, reason=None,
                stop_loss=None, take_profit=None, trail_dist=None, entry_atr=None,
                confluence_score=None):
    """Atomically fills a paper BUY: slippage-adjusted fill, fee deduction,
    cash check, portfolio update and active-trade row in one transaction.

    Returns the trade dict on success, None when rejected."""
    fill = price * (1 + cfg.slippage_pct)
    notional = quantity * fill
    fee = notional * cfg.fee_pct
    total_cost = notional + fee
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()

        cursor.execute("SELECT balance FROM portfolio WHERE asset = 'USD'")
        row = cursor.fetchone()
        cash = float(row["balance"]) if row else 0.0
        if total_cost > cash:
            # shrink to what cash allows rather than failing outright
            quantity = (cash * 0.995) / (fill * (1 + cfg.fee_pct))
            notional = quantity * fill
            fee = notional * cfg.fee_pct
            total_cost = notional + fee
            if notional < cfg.min_notional_usd:
                conn.rollback()
                log_event("WARNING", "ACCOUNTING",
                          f"BUY {asset} rejected: insufficient cash (${cash:.2f})")
                return None

        cursor.execute("SELECT is_active FROM trades WHERE asset = ? AND is_active = 1", (asset,))
        if cursor.fetchone():
            conn.rollback()
            log_event("WARNING", "ACCOUNTING", f"BUY {asset} rejected: position already open")
            return None

        cursor.execute("UPDATE portfolio SET balance = balance - ? WHERE asset = 'USD'", (total_cost,))
        cursor.execute("SELECT balance, avg_entry_price FROM portfolio WHERE asset = ?", (asset,))
        row = cursor.fetchone()
        if row:
            old_qty = float(row["balance"] or 0.0)
            old_avg = float(row["avg_entry_price"] or 0.0)
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * fill)) / new_qty if new_qty > 0 else fill
            cursor.execute("UPDATE portfolio SET balance = ?, avg_entry_price = ? WHERE asset = ?",
                           (new_qty, new_avg, asset))
        else:
            cursor.execute("INSERT INTO portfolio (asset, balance, avg_entry_price) VALUES (?, ?, ?)",
                           (asset, quantity, fill))

        cursor.execute("""
            INSERT INTO trades (timestamp, asset, side, price, amount, trade_type, sentiment_score,
                                reason, stop_loss, take_profit, fees, high_watermark, trail_dist,
                                entry_atr, confluence_score, is_active)
            VALUES (?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (timestamp, asset, fill, quantity, trade_type, sentiment_score, reason,
              stop_loss, take_profit, fee, fill, trail_dist, entry_atr, confluence_score))
        trade_id = cursor.lastrowid
        conn.commit()
    except Exception as e:
        conn.rollback()
        log_event("ERROR", "ACCOUNTING", f"BUY {asset} failed and was rolled back: {e}")
        return None
    finally:
        conn.close()

    log_event("SUCCESS", "ACCOUNTING",
              f"BUY {asset}: {quantity:.4f} @ ${fill:.2f} (notional ${notional:.2f}, fee ${fee:.2f}) "
              f"SL ${stop_loss:.2f} TP ${take_profit:.2f}" if stop_loss and take_profit else
              f"BUY {asset}: {quantity:.4f} @ ${fill:.2f}")

    _notify(asset, "BUY", fill, quantity, trade_type, sentiment_score, reason, stop_loss, take_profit, None)
    return {"id": trade_id, "asset": asset, "price": fill, "amount": quantity,
            "stop_loss": stop_loss, "take_profit": take_profit, "fees": fee}


def execute_sell(asset: str, price: float, cfg: StrategyConfig, *,
                 trade_type="STRATEGY", reason=None, sentiment_score=None):
    """Atomically closes the open position in ``asset`` at ``price``.

    Marks the active BUY row closed with fee-inclusive PnL, credits cash and
    records a SELL row. Sets the re-entry cooldown on stop-loss exits.
    Returns a result dict or None when there is nothing to sell."""
    fill = price * (1 - cfg.slippage_pct)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()

        cursor.execute("SELECT balance FROM portfolio WHERE asset = ?", (asset,))
        row = cursor.fetchone()
        qty = float(row["balance"]) if row else 0.0
        if qty <= 1e-9:
            conn.rollback()
            log_event("WARNING", "ACCOUNTING", f"SELL {asset} rejected: no holdings")
            return None

        proceeds = qty * fill
        fee = proceeds * cfg.fee_pct
        net = proceeds - fee

        cursor.execute("UPDATE portfolio SET balance = balance + ? WHERE asset = 'USD'", (net,))
        cursor.execute("UPDATE portfolio SET balance = 0.0, avg_entry_price = 0.0 WHERE asset = ?", (asset,))

        cursor.execute("SELECT * FROM trades WHERE asset = ? AND is_active = 1 ORDER BY id DESC LIMIT 1",
                       (asset,))
        buy_row = cursor.fetchone()
        pnl_usd = pnl_pct = r_multiple = None
        if buy_row:
            buy_cost = float(buy_row["price"]) * float(buy_row["amount"]) + float(buy_row["fees"] or 0.0)
            pnl_usd = net - buy_cost
            pnl_pct = (pnl_usd / buy_cost) * 100 if buy_cost > 0 else 0.0
            tp, entry = buy_row["take_profit"], float(buy_row["price"])
            if tp and cfg.rr_ratio > 0 and float(tp) > entry:
                r_dist = (float(tp) - entry) / cfg.rr_ratio
                risk_usd = r_dist * float(buy_row["amount"])
                r_multiple = pnl_usd / risk_usd if risk_usd > 0 else None
            cursor.execute("UPDATE trades SET is_active = 0, pnl = ?, pnl_usd = ?, r_multiple = ? WHERE id = ?",
                           (pnl_pct, pnl_usd, r_multiple, buy_row["id"]))

        cursor.execute("""
            INSERT INTO trades (timestamp, asset, side, price, amount, trade_type, sentiment_score,
                                reason, fees, pnl, pnl_usd, r_multiple, is_active)
            VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (timestamp, asset, fill, qty, trade_type, sentiment_score, reason, fee,
              pnl_pct, pnl_usd, r_multiple))
        conn.commit()
    except Exception as e:
        conn.rollback()
        log_event("ERROR", "ACCOUNTING", f"SELL {asset} failed and was rolled back: {e}")
        return None
    finally:
        conn.close()

    if trade_type == "STOP_LOSS" and cfg.cooldown_hours > 0:
        until = (datetime.now(timezone.utc) + timedelta(hours=cfg.cooldown_hours)).isoformat()
        set_cooldown(asset, until, f"Stopped out at ${fill:.2f}")

    pnl_str = f" PnL ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)" if pnl_usd is not None else ""
    log_event("SUCCESS", "ACCOUNTING", f"SELL {asset}: {qty:.4f} @ ${fill:.2f} net ${net:.2f}.{pnl_str}")

    _notify(asset, "SELL", fill, qty, trade_type, sentiment_score, reason, None, None, pnl_pct)
    return {"asset": asset, "price": fill, "amount": qty, "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct, "r_multiple": r_multiple, "net": net}


def _notify(asset, side, price, amount, trade_type, sentiment_score, reason, stop_loss, take_profit, pnl):
    try:
        from core.telegram_bot import send_trade_notification
        send_trade_notification(asset=asset, side=side, price=price, amount=amount,
                                trade_type=trade_type, sentiment_score=sentiment_score,
                                reason=reason, stop_loss=stop_loss, take_profit=take_profit, pnl=pnl)
    except Exception as tg_err:
        log_event("WARNING", "ACCOUNTING", f"Telegram notification failed: {tg_err}")


# ----------------- Daily circuit breaker -----------------

def check_circuit_breaker(nav: float, cfg: StrategyConfig):
    """Trips when NAV falls daily_loss_limit_pct below the day-start NAV.
    While tripped, no new entries are allowed (exits keep working).
    Returns (active: bool, detail: str)."""
    from core.db import get_config, save_config

    now = datetime.now(timezone.utc)

    tripped_until = get_config("cb_tripped_until")
    if tripped_until:
        try:
            until = datetime.fromisoformat(tripped_until)
            if now < until:
                return True, f"Circuit breaker active until {until.isoformat()}"
            save_config("cb_tripped_until", "")
        except ValueError:
            save_config("cb_tripped_until", "")

    today = now.date().isoformat()
    if get_config("cb_day") != today:
        save_config("cb_day", today)
        save_config("cb_day_start_nav", f"{nav:.6f}")
        return False, "New day baseline set"

    try:
        day_start = float(get_config("cb_day_start_nav") or nav)
    except (TypeError, ValueError):
        day_start = nav

    if day_start > 0 and nav <= day_start * (1 - cfg.daily_loss_limit_pct / 100.0):
        until = riskmod.next_session_start(now)
        save_config("cb_tripped_until", until.isoformat())
        detail = (f"Daily loss limit hit: NAV ${nav:.2f} is "
                  f"{(1 - nav / day_start) * 100:.2f}% below day start ${day_start:.2f}. "
                  f"New entries blocked until {until.isoformat()}")
        log_event("ERROR", "RISK", f"🛑 {detail}")
        try:
            from core.telegram_bot import send_telegram_message
            send_telegram_message(f"🛑 *Günlük zarar limiti aşıldı!* Yeni işlemler durduruldu.\n{detail}")
        except Exception:
            pass
        return True, detail

    return False, f"OK (day start ${day_start:.2f}, now ${nav:.2f})"
