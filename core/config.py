"""Strategy/risk configuration shared by the live worker, backtester and UI.

All tunables live in one dataclass so the backtester can run the exact same
parameter set as the live bot (or any experimental variation).
"""
import os
from dataclasses import dataclass, field, asdict

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSM", "AVGO", "ASML",
    "AMZN", "GOOGL", "META", "TSLA", "QQQ", "SPY",
]

# yfinance crypto tickers trade 24/7 and use the -USD suffix
CRYPTO_SUFFIX = "-USD"


def normalize_symbol(sym: str) -> str:
    """Migrates legacy 'AAPL/USD' config entries to plain yfinance tickers."""
    sym = sym.strip()
    if "/" in sym:
        base, quote = sym.split("/", 1)
        if base.upper() in ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK"):
            return f"{base.upper()}-USD"
        return base.upper()
    return sym.upper()


def parse_watchlist(raw: str):
    if not raw:
        return list(DEFAULT_WATCHLIST)
    symbols = [normalize_symbol(s) for s in raw.split(",") if s.strip()]
    # de-dup, preserve order
    seen, out = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or list(DEFAULT_WATCHLIST)


@dataclass
class StrategyConfig:
    # --- entry / trend gates ---
    market_filter_enabled: bool = True      # benchmark must be above its 200d EMA
    benchmark_equity: str = "SPY"
    benchmark_crypto: str = "BTC-USD"
    rsi_entry_min: float = 35.0             # never buy an oversold knife
    rsi_entry_max: float = 68.0             # never chase overbought
    volume_expansion_ratio: float = 1.2     # vol >= ratio * 20-bar SMA scores a point
    support_proximity_pct: float = 2.0      # "near support" = within this % above a swing low
    min_confluence_score: int = 65          # entries below this score are skipped entirely

    # --- exits / stops ---
    atr_mult_sl: float = 2.5                # initial stop = entry - atr_mult_sl * ATR14(1h)
    sl_min_pct: float = 1.5                 # clamp stop distance into [min, max] % of entry
    sl_max_pct: float = 8.0
    rr_ratio: float = 2.0                   # take profit = entry + rr_ratio * R
    breakeven_at_r: float = 1.0             # move stop to entry after +1R
    trail_at_r: float = 1.5                 # start chandelier trail after +1.5R
    trail_atr_mult: float = 2.5             # trail distance = mult * ATR at entry
    max_holding_days: int = 0               # 0 = disabled time stop

    # --- position sizing / portfolio risk ---
    risk_per_trade_pct: float = 1.0         # % of NAV lost if the initial stop is hit
    max_position_pct: float = 20.0          # max notional per position as % of NAV
    max_total_exposure_pct: float = 80.0    # max invested notional as % of NAV
    max_open_positions: int = 5
    daily_loss_limit_pct: float = 3.0       # circuit breaker threshold from day-start NAV
    cooldown_hours: float = 24.0            # per-symbol lockout after a stop-loss exit
    min_notional_usd: float = 100.0

    # --- execution costs ---
    fee_pct: float = 0.0005                 # per side
    slippage_pct: float = 0.0005            # per side

    # --- AI sentiment ---
    ai_enabled: bool = True
    ai_veto_score: int = -3                 # sentiment <= veto blocks the entry
    ai_confirm_score: int = 3               # sentiment >= confirm scores confluence points

    watchlist: list = field(default_factory=lambda: list(DEFAULT_WATCHLIST))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_db(cls):
        """Builds the config from the SQLite config table with env fallbacks."""
        from core.db import get_config

        def cfg(key, default, cast=float):
            raw = get_config(key)
            if raw is None or raw == "":
                raw = os.getenv(key.upper(), None)
            if raw is None or raw == "":
                return default
            try:
                if cast is bool:
                    return str(raw).strip().lower() in ("1", "true", "yes", "on")
                return cast(raw)
            except (TypeError, ValueError):
                return default

        c = cls()
        c.market_filter_enabled = cfg("market_filter_enabled", c.market_filter_enabled, bool)
        c.rsi_entry_min = cfg("rsi_entry_min", c.rsi_entry_min)
        c.rsi_entry_max = cfg("rsi_entry_max", c.rsi_entry_max)
        c.min_confluence_score = int(cfg("min_confluence_score", c.min_confluence_score))
        c.atr_mult_sl = cfg("atr_mult_sl", c.atr_mult_sl)
        c.sl_min_pct = cfg("sl_min_pct", c.sl_min_pct)
        c.sl_max_pct = cfg("sl_max_pct", c.sl_max_pct)
        c.rr_ratio = cfg("rr_ratio", c.rr_ratio)
        c.breakeven_at_r = cfg("breakeven_at_r", c.breakeven_at_r)
        c.trail_at_r = cfg("trail_at_r", c.trail_at_r)
        c.trail_atr_mult = cfg("trail_atr_mult", c.trail_atr_mult)
        c.max_holding_days = int(cfg("max_holding_days", c.max_holding_days))
        # legacy key "risk_percentage" kept so the Telegram /risk command still works
        c.risk_per_trade_pct = cfg("risk_per_trade_pct", cfg("risk_percentage", c.risk_per_trade_pct))
        c.max_position_pct = cfg("max_position_pct", c.max_position_pct)
        c.max_total_exposure_pct = cfg("max_total_exposure_pct", c.max_total_exposure_pct)
        c.max_open_positions = int(cfg("max_open_positions", c.max_open_positions))
        c.daily_loss_limit_pct = cfg("daily_loss_limit_pct", c.daily_loss_limit_pct)
        c.cooldown_hours = cfg("cooldown_hours", c.cooldown_hours)
        c.fee_pct = cfg("fee_pct", c.fee_pct)
        c.slippage_pct = cfg("slippage_pct", c.slippage_pct)
        c.ai_enabled = cfg("ai_enabled", c.ai_enabled, bool)
        c.ai_veto_score = int(cfg("ai_veto_score", c.ai_veto_score))
        c.ai_confirm_score = int(cfg("min_ai_sentiment_threshold", c.ai_confirm_score))
        c.watchlist = parse_watchlist(get_config("selected_assets") or os.getenv("SELECTED_ASSETS", ""))
        return c
