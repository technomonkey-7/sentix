"""Sentix v2 strategy: long-only swing trading with regime gates, confluence
scoring and ATR-based risk.

Pure functions over prepared DataFrames — no database, no network — so the
live worker and the backtester execute the *same* code.

Frames are expected to carry the columns produced by
core.indicators.add_indicators (ema, ema50, ema200, rsi, macd, macd_signal,
macd_hist, atr, vol_sma).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from core.config import StrategyConfig
from core.indicators import crossed_above, swing_levels, bullish_patterns
from core.risk import clamp_stop


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class EntrySignal:
    should_enter: bool
    score: int = 0
    conf_mult: float = 0.0
    trigger: str | None = None
    stop: float = 0.0
    take_profit: float = 0.0
    trail_dist: float = 0.0
    entry_atr: float = 0.0
    gates: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    reason: str = ""


@dataclass
class ExitSignal:
    should_exit: bool
    code: str = ""       # REGIME_BREAK / TIME_STOP
    detail: str = ""


def _row(df: pd.DataFrame, idx: int):
    try:
        return df.iloc[idx]
    except (IndexError, TypeError):
        return None


def _valid(*vals) -> bool:
    for v in vals:
        if v is None or pd.isna(v):
            return False
    return True


def detect_trigger(df_1h: pd.DataFrame, completed_idx: int = -2) -> tuple[str | None, str]:
    """Buy trigger on the last completed 1h candle.

    Deliberately does NOT treat 'RSI crossed below 30' as a buy — that was the
    v1 falling-knife bug. Only momentum turning UP counts.
    """
    if df_1h is None or len(df_1h) < abs(completed_idx) + 1:
        return None, "Insufficient 1h data"
    curr = _row(df_1h, completed_idx)
    prev = _row(df_1h, completed_idx - 1)
    if curr is None or prev is None:
        return None, "Insufficient 1h data"

    if not _valid(curr.get("ema"), prev.get("ema"), curr.get("rsi"), prev.get("rsi"),
                  curr.get("macd"), prev.get("macd"), curr.get("macd_signal"), prev.get("macd_signal")):
        return None, "Indicators not ready"

    close_c, close_p = float(curr["close"]), float(prev["close"])

    if crossed_above(prev["macd"], prev["macd_signal"], curr["macd"], curr["macd_signal"]):
        # require a minimally meaningful separation to filter flat noise
        if (float(curr["macd"]) - float(curr["macd_signal"])) / close_c * 100 >= 0.005:
            return "MACD_CROSS_UP", "MACD line crossed above signal line"

    if crossed_above(close_p, prev["ema"], close_c, curr["ema"]) and close_c >= float(curr["ema"]) * 1.0005:
        return "EMA20_CROSS_UP", "Price closed above the 20 EMA"

    if float(prev["rsi"]) < 35.0 <= float(curr["rsi"]):
        return "RSI_RECOVERY", "RSI recovered up through 35"

    return None, "No entry trigger on the completed candle"


def trigger_mask(df_1h: pd.DataFrame) -> pd.Series:
    """Vectorized version of detect_trigger for backtest pre-filtering.

    True at bar i when a buy trigger fires on bar i (bar i treated as the
    completed candle). Must stay formula-identical to detect_trigger.
    """
    close = df_1h["close"].astype(float)
    ema20 = df_1h["ema"]
    rsi_s = df_1h["rsi"]
    macd_l = df_1h["macd"]
    macd_s = df_1h["macd_signal"]

    macd_cross = (macd_l.shift(1) <= macd_s.shift(1)) & (macd_l > macd_s) & \
                 ((macd_l - macd_s) / close * 100 >= 0.005)
    ema_cross = (close.shift(1) <= ema20.shift(1)) & (close > ema20) & (close >= ema20 * 1.0005)
    rsi_recovery = (rsi_s.shift(1) < 35.0) & (rsi_s >= 35.0)

    mask = (macd_cross | ema_cross | rsi_recovery)
    ready = ema20.notna() & rsi_s.notna() & macd_l.notna() & macd_s.notna() & \
            ema20.shift(1).notna() & rsi_s.shift(1).notna() & macd_l.shift(1).notna() & macd_s.shift(1).notna()
    return mask & ready


def evaluate_entry(df_1h: pd.DataFrame,
                   df_4h: pd.DataFrame | None,
                   df_daily: pd.DataFrame | None,
                   benchmark_daily: pd.DataFrame | None,
                   cfg: StrategyConfig,
                   sentiment: int | None = None,
                   entry_price: float | None = None,
                   completed_idx: int = -2) -> EntrySignal:
    """Full technical entry evaluation. Operational gates (market hours,
    freshness, cooldown, portfolio caps) belong to the caller."""
    sig = EntrySignal(should_enter=False)
    gates = sig.gates

    curr = _row(df_1h, completed_idx)
    if curr is None or not _valid(curr.get("close"), curr.get("atr"), curr.get("rsi")):
        gates.append(GateCheck("data", False, "1h indicators not ready"))
        sig.reason = "1h indicators not ready"
        return sig

    price = float(entry_price) if entry_price else float(curr["close"])

    # ---- Gate 1: daily regime (trade only uptrends) ----
    d = _row(df_daily, -1) if df_daily is not None else None
    if d is None or not _valid(d.get("close"), d.get("ema50"), d.get("ema200")):
        gates.append(GateCheck("daily_regime", False, "Insufficient daily history (need 200+ bars)"))
    else:
        ok = float(d["close"]) > float(d["ema200"]) and float(d["ema50"]) > float(d["ema200"])
        gates.append(GateCheck(
            "daily_regime", ok,
            f"close {float(d['close']):.2f} vs EMA200 {float(d['ema200']):.2f}, "
            f"EMA50 {float(d['ema50']):.2f}"))

    # ---- Gate 2: benchmark regime (don't fight the market) ----
    if cfg.market_filter_enabled:
        b = _row(benchmark_daily, -1) if benchmark_daily is not None else None
        if b is None or not _valid(b.get("close"), b.get("ema200")):
            gates.append(GateCheck("market_regime", False, "Benchmark daily data unavailable"))
        else:
            ok = float(b["close"]) > float(b["ema200"])
            gates.append(GateCheck(
                "market_regime", ok,
                f"benchmark close {float(b['close']):.2f} vs EMA200 {float(b['ema200']):.2f}"))

    # ---- Gate 3: entry trigger on completed 1h candle ----
    trigger, trig_detail = detect_trigger(df_1h, completed_idx)
    gates.append(GateCheck("trigger", trigger is not None, trig_detail))
    sig.trigger = trigger

    # ---- Gate 4: RSI sanity band ----
    rsi_val = float(curr["rsi"])
    rsi_ok = cfg.rsi_entry_min <= rsi_val <= cfg.rsi_entry_max
    gates.append(GateCheck("rsi_band", rsi_ok,
                           f"RSI {rsi_val:.1f} (must be {cfg.rsi_entry_min:.0f}-{cfg.rsi_entry_max:.0f})"))

    # ---- Gate 5: AI veto ----
    if cfg.ai_enabled and sentiment is not None and sentiment <= cfg.ai_veto_score:
        gates.append(GateCheck("ai_veto", False, f"AI sentiment {sentiment} <= veto {cfg.ai_veto_score}"))
    else:
        gates.append(GateCheck("ai_veto", True,
                               "No AI veto" if sentiment is None else f"AI sentiment {sentiment}"))

    if not all(g.passed for g in gates):
        failed = [g.name for g in gates if not g.passed]
        sig.reason = "Gates failed: " + ", ".join(failed)
        return sig

    # ---- Confluence scoring ----
    score = 50
    factors = sig.factors

    # df_4h must contain COMPLETED 4h bars only (callers slice off partials)
    h4 = _row(df_4h, -1) if df_4h is not None and len(df_4h) >= 1 else None
    if h4 is not None and _valid(h4.get("close"), h4.get("ema50")):
        if float(h4["close"]) > float(h4["ema50"]):
            score += 15
            factors.append("4h trend up (+15)")
        elif _valid(h4.get("macd_hist")) and float(h4["macd_hist"]) > 0:
            score += 8
            factors.append("4h MACD momentum (+8)")

    if _valid(curr.get("volume"), curr.get("vol_sma")) and float(curr["vol_sma"]) > 0:
        if float(curr["volume"]) >= cfg.volume_expansion_ratio * float(curr["vol_sma"]):
            score += 10
            factors.append("Volume expansion (+10)")

    supports, _ = swing_levels(df_1h.iloc[:len(df_1h) + completed_idx + 1])
    near_support = any(0 <= (price - s) / s * 100 <= cfg.support_proximity_pct for s in supports if s > 0)
    patterns = bullish_patterns(df_1h, completed_idx)
    if near_support or patterns:
        score += 10
        factors.append(("Near support" if near_support else f"Pattern: {', '.join(patterns)}") + " (+10)")

    if cfg.ai_enabled and sentiment is not None and sentiment >= cfg.ai_confirm_score:
        score += 15
        factors.append(f"AI sentiment {sentiment} (+15)")

    sig.score = score
    if score < cfg.min_confluence_score:
        gates.append(GateCheck("confluence", False,
                               f"Score {score} < minimum {cfg.min_confluence_score}"))
        sig.reason = (f"Confluence too weak: score {score}/100 "
                      f"({', '.join(factors) if factors else 'no extra factors'}), "
                      f"minimum is {cfg.min_confluence_score}")
        return sig
    gates.append(GateCheck("confluence", True, f"Score {score} >= {cfg.min_confluence_score}"))
    sig.conf_mult = 1.0 if score >= 80 else (0.75 if score >= 65 else 0.5)

    # ---- ATR-based stop / target ----
    atr_val = float(curr["atr"])
    sig.entry_atr = atr_val
    raw_stop = price - cfg.atr_mult_sl * atr_val
    sig.stop = clamp_stop(price, raw_stop, cfg)
    r_dist = price - sig.stop
    sig.take_profit = price + cfg.rr_ratio * r_dist
    sig.trail_dist = cfg.trail_atr_mult * atr_val

    sig.should_enter = True
    sig.reason = (f"{trig_detail}. Score {score}/100 ({', '.join(factors) if factors else 'base'}). "
                  f"Stop {sig.stop:.2f} ({(r_dist / price * 100):.2f}%), target {sig.take_profit:.2f}")
    return sig


def evaluate_exit(position: dict,
                  df_daily: pd.DataFrame | None,
                  cfg: StrategyConfig,
                  now: datetime | None = None) -> ExitSignal:
    """Structural (candle-based) exits. SL/TP/trailing are price-based and
    handled by the guardian via update_trailing_stop + stop checks."""
    d = _row(df_daily, -1) if df_daily is not None else None
    if d is not None and _valid(d.get("close"), d.get("ema200")):
        if float(d["close"]) < float(d["ema200"]):
            return ExitSignal(True, "REGIME_BREAK",
                              f"Daily close {float(d['close']):.2f} fell below EMA200 "
                              f"{float(d['ema200']):.2f} — uptrend broken")

    if cfg.max_holding_days > 0:
        opened = position.get("timestamp")
        if opened:
            try:
                opened_dt = datetime.fromisoformat(str(opened))
                if opened_dt.tzinfo is None:
                    opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                age_days = ((now or datetime.now(timezone.utc)) - opened_dt).days
                if age_days >= cfg.max_holding_days:
                    return ExitSignal(True, "TIME_STOP",
                                      f"Position age {age_days}d >= max {cfg.max_holding_days}d")
            except (ValueError, TypeError):
                pass

    return ExitSignal(False)


def update_trailing_stop(position: dict, current_price: float, cfg: StrategyConfig):
    """Breakeven + chandelier trailing logic.

    Returns (new_stop, new_high_watermark) — either may equal the stored
    value when nothing changed. R distance is derived from the recorded
    take-profit so it survives restarts.
    """
    entry = float(position.get("price") or 0)
    stop = position.get("stop_loss")
    stop = float(stop) if stop is not None else None
    tp = float(position.get("take_profit") or 0)
    trail_dist = float(position.get("trail_dist") or 0)
    hwm = float(position.get("high_watermark") or entry)

    new_hwm = max(hwm, float(current_price))
    if entry <= 0 or tp <= entry or cfg.rr_ratio <= 0:
        return stop, new_hwm

    r_dist = (tp - entry) / cfg.rr_ratio
    candidates = [stop] if stop is not None else []

    if new_hwm >= entry + cfg.breakeven_at_r * r_dist:
        candidates.append(entry * (1 + 2 * cfg.fee_pct))  # breakeven incl. round-trip fees

    if trail_dist > 0 and new_hwm >= entry + cfg.trail_at_r * r_dist:
        candidates.append(new_hwm - trail_dist)

    new_stop = max(c for c in candidates if c is not None) if candidates else stop
    return new_stop, new_hwm
