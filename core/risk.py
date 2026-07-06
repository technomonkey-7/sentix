"""Risk management: market calendar, data freshness, position sizing, caps.

Pure functions — no database access. Stateful pieces (circuit breaker,
cooldowns) live in the worker and use core.db helpers.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.config import StrategyConfig, CRYPTO_SUFFIX

NY = ZoneInfo("America/New_York")
SESSION_OPEN = (9, 30)
SESSION_CLOSE = (16, 0)


def is_crypto(symbol: str) -> bool:
    return symbol.upper().endswith(CRYPTO_SUFFIX)


def is_market_open(symbol: str, now_utc: datetime | None = None) -> bool:
    """Regular-hours check. Crypto is always open. US equities: Mon-Fri
    09:30-16:00 New York. Exchange holidays are not modeled here — the
    data-freshness guard blocks trading on stale feeds instead."""
    if is_crypto(symbol):
        return True
    now = (now_utc or datetime.now(timezone.utc)).astimezone(NY)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1], second=0, microsecond=0)
    close_t = now.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1], second=0, microsecond=0)
    return open_t <= now < close_t


def market_minutes_elapsed(since_utc: datetime, now_utc: datetime | None = None) -> float:
    """Minutes of regular US session time between two instants (max 10 days back)."""
    now = now_utc or datetime.now(timezone.utc)
    since = since_utc.astimezone(NY)
    now_ny = now.astimezone(NY)
    if since >= now_ny:
        return 0.0
    if now_ny - since > timedelta(days=10):
        since = now_ny - timedelta(days=10)

    total = 0.0
    day = since.date()
    while day <= now_ny.date():
        if datetime(day.year, day.month, day.day, tzinfo=NY).weekday() < 5:
            sess_open = datetime(day.year, day.month, day.day, *SESSION_OPEN, tzinfo=NY)
            sess_close = datetime(day.year, day.month, day.day, *SESSION_CLOSE, tzinfo=NY)
            lo = max(sess_open, since)
            hi = min(sess_close, now_ny)
            if hi > lo:
                total += (hi - lo).total_seconds() / 60.0
        day += timedelta(days=1)
    return total


def data_is_fresh(symbol: str, last_candle_utc: datetime, interval_minutes: int = 60,
                  now_utc: datetime | None = None) -> bool:
    """True when the latest candle is recent enough to act on.

    Crypto: wall-clock age < 2 intervals. Equities: less than 2 intervals of
    *market-open time* have passed since the candle — so nights/weekends don't
    count, but a dead feed or an unmodeled holiday makes data stale and
    blocks trading (failsafe)."""
    now = now_utc or datetime.now(timezone.utc)
    if last_candle_utc.tzinfo is None:
        last_candle_utc = last_candle_utc.replace(tzinfo=timezone.utc)
    if is_crypto(symbol):
        return (now - last_candle_utc).total_seconds() / 60.0 < 2 * interval_minutes
    return market_minutes_elapsed(last_candle_utc, now) < 2 * interval_minutes


def clamp_stop(entry: float, raw_stop: float, cfg: StrategyConfig) -> float:
    """Clamps the stop distance into [sl_min_pct, sl_max_pct] of entry."""
    dist = entry - raw_stop
    min_dist = entry * cfg.sl_min_pct / 100.0
    max_dist = entry * cfg.sl_max_pct / 100.0
    dist = max(min_dist, min(max_dist, dist))
    return entry - dist


@dataclass
class SizingResult:
    quantity: float
    notional: float
    risk_amount: float
    reason: str

    @property
    def ok(self) -> bool:
        return self.quantity > 0


def compute_position_size(nav: float, cash: float, entry: float, stop: float,
                          conf_mult: float, current_exposure: float,
                          open_positions: int, cfg: StrategyConfig) -> SizingResult:
    """True risk-based sizing: quantity = (NAV * risk%) / (entry - stop),
    capped by per-position, total-exposure and cash limits."""
    if open_positions >= cfg.max_open_positions:
        return SizingResult(0, 0, 0, f"Max open positions reached ({cfg.max_open_positions})")

    per_share_risk = entry - stop
    if per_share_risk <= 0 or entry <= 0:
        return SizingResult(0, 0, 0, "Invalid stop distance")

    risk_amount = nav * (cfg.risk_per_trade_pct / 100.0) * conf_mult
    qty = risk_amount / per_share_risk
    notional = qty * entry

    caps = []
    max_pos_notional = nav * (cfg.max_position_pct / 100.0)
    if notional > max_pos_notional:
        notional = max_pos_notional
        caps.append(f"per-position cap {cfg.max_position_pct:.0f}% NAV")

    exposure_room = nav * (cfg.max_total_exposure_pct / 100.0) - current_exposure
    if notional > exposure_room:
        notional = max(0.0, exposure_room)
        caps.append(f"total exposure cap {cfg.max_total_exposure_pct:.0f}% NAV")

    cash_room = cash * 0.98  # leave headroom for fees/slippage
    if notional > cash_room:
        notional = max(0.0, cash_room)
        caps.append("available cash")

    if notional < cfg.min_notional_usd:
        return SizingResult(0, 0, risk_amount,
                            f"Position too small (${notional:.2f} < ${cfg.min_notional_usd:.0f} min)"
                            + (f" after caps: {', '.join(caps)}" if caps else ""))

    qty = notional / entry
    reason = f"Risk ${risk_amount:.2f} ({cfg.risk_per_trade_pct:.2f}% NAV x {conf_mult:.2f} confidence)"
    if caps:
        reason += f" | capped by: {', '.join(caps)}"
    return SizingResult(qty, notional, risk_amount, reason)


def next_session_start(now_utc: datetime | None = None) -> datetime:
    """UTC instant of the next regular US session open (used by the circuit breaker)."""
    now = (now_utc or datetime.now(timezone.utc)).astimezone(NY)
    candidate = now.replace(hour=SESSION_OPEN[0], minute=SESSION_OPEN[1], second=0, microsecond=0)
    if now >= candidate:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.astimezone(timezone.utc)
