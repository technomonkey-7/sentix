"""Event-driven backtester running the exact live strategy code.

- Signals are evaluated on completed 1h bars; fills happen at the NEXT bar's
  open with slippage and fees (no lookahead).
- Daily/benchmark regime uses only daily bars strictly BEFORE the current
  bar's date (no intraday lookahead).
- Same-bar SL/TP conflicts resolve to the stop first (conservative).
- AI sentiment is unavailable historically, so backtests run technical-only
  (sentiment=None → no veto, no bonus), matching live behavior when AI is off.
"""
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from core.config import StrategyConfig
from core.data_fetcher import fetch_ohlcv, resample_4h
from core.indicators import add_indicators
from core import risk as riskmod
from core import strategy


@dataclass
class BtTrade:
    symbol: str
    entry_time: object
    exit_time: object
    entry: float
    exit: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float | None
    exit_reason: str
    score: int


@dataclass
class BtPosition:
    symbol: str
    qty: float
    entry: float
    entry_fees: float
    stop: float
    tp: float
    trail_dist: float
    hwm: float
    entry_time: object
    score: int
    risk_usd: float


@dataclass
class BacktestResult:
    equity: pd.Series = None
    benchmark: pd.Series = None
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def _completed_daily(df_daily: pd.DataFrame, ts) -> pd.DataFrame:
    """Daily bars strictly before the bar's calendar date (no lookahead)."""
    if df_daily is None:
        return None
    cutoff = pd.Timestamp(ts).normalize()
    return df_daily[df_daily["timestamp"] < cutoff]


def _completed_4h(df_4h: pd.DataFrame, ts) -> pd.DataFrame:
    if df_4h is None:
        return None
    end_times = df_4h["timestamp"] + pd.Timedelta(hours=4)
    return df_4h[end_times <= pd.Timestamp(ts)]


def run_backtest(symbols: list[str], cfg: StrategyConfig, initial_cash: float = 10000.0,
                 months: int = 12, progress_cb=None) -> BacktestResult:
    result = BacktestResult(config=cfg.to_dict())

    days = min(max(30, int(months * 30.4)), 700)
    period = f"{days + 30}d"

    # ---------- Load data ----------
    data = {}
    n_syms = len(symbols)
    for i, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(i / max(1, n_syms) * 0.3, f"Loading data: {sym}")
        df_1h = fetch_ohlcv(sym, "1h", limit=None, period=period)
        df_d = fetch_ohlcv(sym, "1d", limit=None, period="5y")
        if df_1h is None or df_d is None or len(df_1h) < 100 or len(df_d) < 210:
            result.errors.append(f"{sym}: insufficient history, excluded from backtest")
            continue
        df_1h = add_indicators(df_1h)
        df_d = add_indicators(df_d)
        df_4h = add_indicators(resample_4h(df_1h))
        data[sym] = {"1h": df_1h, "4h": df_4h, "1d": df_d,
                     "trigger": strategy.trigger_mask(df_1h)}

    if not data:
        result.errors.append("No symbols had sufficient data")
        return result

    bench_sym = cfg.benchmark_crypto if all(riskmod.is_crypto(s) for s in data) else cfg.benchmark_equity
    bench_daily = fetch_ohlcv(bench_sym, "1d", limit=None, period="5y")
    bench_daily = add_indicators(bench_daily) if bench_daily is not None else None
    if cfg.market_filter_enabled and bench_daily is None:
        result.errors.append(f"Benchmark {bench_sym} data unavailable — market filter disabled for this run")
        cfg.market_filter_enabled = False

    # ---------- Build unified timeline ----------
    start_cut = None
    for sym, d in data.items():
        last = d["1h"]["timestamp"].iloc[-1]
        first_allowed = last - pd.Timedelta(days=days)
        start_cut = first_allowed if start_cut is None else min(start_cut, first_allowed)

    all_ts = sorted(set().union(*[
        set(d["1h"]["timestamp"][d["1h"]["timestamp"] >= start_cut]) for d in data.values()
    ]))
    if not all_ts:
        result.errors.append("Empty timeline")
        return result

    # per-symbol row lookup: timestamp -> integer index
    for d in data.values():
        d["index_of"] = {ts: i for i, ts in enumerate(d["1h"]["timestamp"])}

    # ---------- Simulation state ----------
    cash = float(initial_cash)
    positions: dict[str, BtPosition] = {}
    pending: dict[str, dict] = {}    # symbol -> signal to fill at next bar open
    cooldown_until: dict[str, object] = {}
    trades: list[BtTrade] = []
    equity_times, equity_vals = [], []
    last_price: dict[str, float] = {}
    day_start_nav, current_day, cb_tripped_day = None, None, None
    fees_total = 0.0

    def nav_now():
        val = cash
        for sym, p in positions.items():
            val += p.qty * last_price.get(sym, p.entry)
        return val

    def close_position(p: BtPosition, price: float, ts, reason: str):
        nonlocal cash, fees_total
        fill = price * (1 - cfg.slippage_pct)
        proceeds = p.qty * fill
        fee = proceeds * cfg.fee_pct
        fees_total += fee
        cash += proceeds - fee
        buy_cost = p.qty * p.entry + p.entry_fees
        pnl_usd = (proceeds - fee) - buy_cost
        pnl_pct = pnl_usd / buy_cost * 100 if buy_cost > 0 else 0.0
        r_mult = pnl_usd / p.risk_usd if p.risk_usd > 0 else None
        trades.append(BtTrade(p.symbol, p.entry_time, ts, p.entry, fill, p.qty,
                              pnl_usd, pnl_pct, r_mult, reason, p.score))
        del positions[p.symbol]
        if reason == "STOP_LOSS" and cfg.cooldown_hours > 0:
            cooldown_until[p.symbol] = pd.Timestamp(ts) + timedelta(hours=cfg.cooldown_hours)

    total_steps = len(all_ts)
    for step, ts in enumerate(all_ts):
        if progress_cb and step % 200 == 0:
            progress_cb(0.3 + step / total_steps * 0.65, f"Simulating {pd.Timestamp(ts).date()}")

        # -- daily circuit breaker bookkeeping --
        bar_day = pd.Timestamp(ts).date()
        if bar_day != current_day:
            current_day = bar_day
            day_start_nav = nav_now()
        cb_active = (cb_tripped_day == current_day)

        for sym, d in data.items():
            idx = d["index_of"].get(ts)
            if idx is None:
                continue
            bar = d["1h"].iloc[idx]
            o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
            last_price[sym] = c

            # ---- 1) fill pending entry at this bar's open ----
            if sym in pending:
                sig = pending.pop(sym)
                if not cb_active and sym not in positions:
                    fill = o * (1 + cfg.slippage_pct)
                    if fill > sig["stop"]:
                        nav = nav_now()
                        exposure = sum(p.qty * last_price.get(s, p.entry) for s, p in positions.items())
                        sizing = riskmod.compute_position_size(
                            nav, cash, fill, sig["stop"], sig["conf_mult"], exposure,
                            len(positions), cfg)
                        if sizing.ok:
                            notional = sizing.quantity * fill
                            fee = notional * cfg.fee_pct
                            if notional + fee <= cash:
                                cash -= notional + fee
                                fees_total += fee
                                positions[sym] = BtPosition(
                                    symbol=sym, qty=sizing.quantity, entry=fill, entry_fees=fee,
                                    stop=sig["stop"], tp=sig["tp"], trail_dist=sig["trail_dist"],
                                    hwm=fill, entry_time=ts, score=sig["score"],
                                    risk_usd=sizing.quantity * (fill - sig["stop"]))

            # ---- 2) manage open position ----
            if sym in positions:
                p = positions[sym]
                # conservative same-bar rule: stop before target
                if o <= p.stop:
                    close_position(p, o, ts, "STOP_LOSS")
                elif l <= p.stop:
                    close_position(p, p.stop, ts, "STOP_LOSS")
                elif o >= p.tp:
                    close_position(p, o, ts, "TAKE_PROFIT")
                elif h >= p.tp:
                    close_position(p, p.tp, ts, "TAKE_PROFIT")
                else:
                    # trailing update on bar close (guardian analog)
                    pos_dict = {"price": p.entry, "stop_loss": p.stop, "take_profit": p.tp,
                                "trail_dist": p.trail_dist, "high_watermark": p.hwm}
                    new_stop, new_hwm = strategy.update_trailing_stop(pos_dict, c, cfg)
                    p.hwm = new_hwm
                    if new_stop is not None and new_stop > p.stop:
                        p.stop = new_stop
                    # structural exits on completed daily data
                    exit_sig = strategy.evaluate_exit(
                        {"timestamp": str(pd.Timestamp(p.entry_time))},
                        _completed_daily(d["1d"], ts), cfg,
                        pd.Timestamp(ts).to_pydatetime())
                    if exit_sig.should_exit:
                        close_position(p, c, ts, exit_sig.code)
                continue

            # ---- 3) look for new entry signal on this completed bar ----
            if cb_active or sym in pending:
                continue
            if not bool(d["trigger"].iloc[idx]):
                continue
            cd = cooldown_until.get(sym)
            if cd is not None and pd.Timestamp(ts) < cd:
                continue
            if len(positions) + len(pending) >= cfg.max_open_positions:
                continue

            window_start = max(0, idx - 250)
            df_slice = d["1h"].iloc[window_start: idx + 1]
            sig = strategy.evaluate_entry(
                df_slice,
                _completed_4h(d["4h"], ts),
                _completed_daily(d["1d"], ts),
                _completed_daily(bench_daily, ts) if cfg.market_filter_enabled else None,
                cfg, sentiment=None, completed_idx=-1)
            if sig.should_enter:
                pending[sym] = {"stop": sig.stop, "tp": sig.take_profit,
                                "trail_dist": sig.trail_dist, "conf_mult": sig.conf_mult,
                                "score": sig.score}

        # -- circuit breaker check after processing all symbols this hour --
        if day_start_nav and day_start_nav > 0:
            if nav_now() <= day_start_nav * (1 - cfg.daily_loss_limit_pct / 100.0):
                cb_tripped_day = current_day
                pending.clear()

        equity_times.append(ts)
        equity_vals.append(nav_now())

    # close remaining positions at last known price for final accounting
    for sym in list(positions.keys()):
        close_position(positions[sym], last_price.get(sym, positions[sym].entry),
                       all_ts[-1], "END_OF_TEST")
    equity_vals[-1] = nav_now()

    if progress_cb:
        progress_cb(0.98, "Computing metrics")

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_times), name="nav")
    result.equity = equity
    result.trades = trades
    result.metrics = compute_metrics(equity, trades, initial_cash, fees_total)

    if bench_daily is not None:
        b = bench_daily[(bench_daily["timestamp"] >= equity.index[0]) &
                        (bench_daily["timestamp"] <= equity.index[-1])]
        if len(b) > 1:
            bench_series = b.set_index("timestamp")["close"]
            result.benchmark = bench_series / bench_series.iloc[0] * initial_cash
            result.metrics["benchmark_return_pct"] = float(
                (bench_series.iloc[-1] / bench_series.iloc[0] - 1) * 100)
            result.metrics["benchmark_symbol"] = bench_sym

    if progress_cb:
        progress_cb(1.0, "Done")
    return result


def compute_metrics(equity: pd.Series, trades: list, initial_cash: float, fees_total: float) -> dict:
    m = {}
    final = float(equity.iloc[-1])
    m["initial_cash"] = initial_cash
    m["final_nav"] = final
    m["total_return_pct"] = (final / initial_cash - 1) * 100

    span_days = max(1.0, (equity.index[-1] - equity.index[0]).total_seconds() / 86400)
    m["period_days"] = span_days
    if final > 0:
        m["cagr_pct"] = ((final / initial_cash) ** (365.0 / span_days) - 1) * 100
    else:
        m["cagr_pct"] = -100.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    m["max_drawdown_pct"] = float(drawdown.min() * 100)

    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) > 2 and rets.std() > 0:
        m["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(252))
    else:
        m["sharpe"] = 0.0

    m["n_trades"] = len(trades)
    m["fees_total"] = fees_total
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    m["win_rate_pct"] = len(wins) / len(trades) * 100 if trades else 0.0
    gross_win = sum(t.pnl_usd for t in wins)
    gross_loss = -sum(t.pnl_usd for t in losses)
    m["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    m["avg_win_usd"] = gross_win / len(wins) if wins else 0.0
    m["avg_loss_usd"] = -gross_loss / len(losses) if losses else 0.0
    r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
    m["expectancy_r"] = float(np.mean(r_vals)) if r_vals else 0.0
    return m


def trades_to_frame(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "symbol": t.symbol, "entry_time": t.entry_time, "exit_time": t.exit_time,
        "entry": t.entry, "exit": t.exit, "qty": t.qty, "pnl_usd": t.pnl_usd,
        "pnl_pct": t.pnl_pct, "r_multiple": t.r_multiple, "exit_reason": t.exit_reason,
        "score": t.score,
    } for t in trades])
