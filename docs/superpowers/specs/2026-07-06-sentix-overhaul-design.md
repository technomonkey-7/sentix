# Sentix v2 Overhaul — Design

Date: 2026-07-06
Status: Approved autonomously (user granted full authority and is unavailable for review).

## Problem

The current bot has algorithm faults that lose money:

1. RSI crossing **below** 30 is treated as a BUY (buying falling knives).
2. No trend/regime filter — 1h buy signals fire in daily downtrends and bear markets.
3. "Risk %" is actually notional size (2% of NAV per trade → ~0.06% real risk at stop). Not risk-based sizing.
4. Fixed % SL/TP ignores per-asset volatility (no ATR).
5. Guardian thread and analysis thread both read-modify-write the whole portfolio without a transaction → lost/duplicated cash.
6. No max positions, exposure cap, daily-loss circuit breaker, or post-stop cooldown.
7. No market-hours awareness; SL/TP evaluated on stale prices at night/weekends.
8. No backtesting — the strategy cannot be validated.
9. Simulated fallbacks (fake candles, random sentiment) can reach decision paths.
10. 2,511-line monolithic UI.

## Goals

- A defensible long-only swing strategy with regime gates, ATR risk, and true risk-based sizing.
- Portfolio-level risk management (caps, circuit breaker, cooldowns).
- A backtesting engine sharing the exact live strategy code.
- Atomic, fee-aware trade accounting.
- Market-hours + data-freshness guards; crypto (24/7) support as a new market.
- A rebuilt, modular, bilingual (TR/EN) Streamlit UI with equity curve and backtest pages.
- Keep Telegram bot and FastAPI compatibility where cheap.

Non-goals: real brokerage execution (paper only), short selling, options, non-USD currencies (BIST deferred).

## Architecture

Pure-function strategy core shared by live worker and backtester:

```
core/indicators.py   pure indicator math (EMA/SMA/RSI/MACD/ATR/swings/patterns), no DB
core/strategy.py     evaluate_entry / evaluate_exit / build_frames — pure, returns dataclasses with reasons
core/risk.py         position sizing, portfolio caps, circuit breaker, cooldown, market calendar, freshness
core/accounting.py   atomic execute_buy/execute_sell (single SQLite transaction), NAV, equity snapshots
core/backtest.py     event-driven simulator over 1h bars using strategy.* ; metrics + equity curve
core/data_fetcher.py UTC timestamps; 1h/4h/daily fetch; news RSS; NO synthetic fallback data
core/db.py           schema + migrations (new: equity_history, cooldowns, signals; trades: fees, pnl_usd,
                     high_watermark, trail_dist, r_multiple, entry_atr)
worker.py            slow loop (analysis) + fast guardian loop, using the modules above
ai/sentiment_analyzer.py  kept; fallback = neutral 0 flagged simulated (no random scores)
ui/app.py + ui/translations.py   rebuilt Streamlit app (Overview / Charts / Backtest / Settings / Logs)
```

Symbols are raw yfinance tickers (`AAPL`, `QQQ`, `BTC-USD`). Legacy `X/USD` config values are migrated on read. Crypto (`-USD` suffix) trades 24/7; equities respect NYSE regular hours.

## Strategy (long-only swing)

Timeframes: 1h execution, 4h and daily context (daily fetched separately; 4h resampled from 1h in UTC).

**Mandatory entry gates (all must pass):**
1. Data fresh (last 1h candle < 2 intervals old) and market open for the symbol.
2. Daily regime: close > EMA200(d) and EMA50(d) > EMA200(d).
3. Benchmark regime (SPY for equities, BTC-USD for crypto): close > EMA200(d). Config-toggleable.
4. Trigger on last completed 1h candle: EMA20 cross-up, MACD signal-line cross-up, or RSI recovery up through 35.
5. RSI(1h) in [35, 68] — never buy oversold knives or chase overbought.
6. No open position in symbol, symbol not in cooldown, portfolio caps pass, circuit breaker clear.
7. AI veto: sentiment <= -3 blocks (when AI available).

**Confluence score** (base 50 + factors; sets risk multiplier):
- +15 4h trend aligned (close > EMA50(4h); half credit for MACD hist > 0 only)
- +10 volume >= 1.2x SMA20(vol)
- +10 near swing support (within 2% above) or bullish candle pattern
- +15 AI sentiment >= threshold
- Multiplier: 50–64 → 0.5x, 65–79 → 0.75x, >=80 → 1.0x.

**Exits:**
- Initial stop: entry − 2.5×ATR14(1h), clamped to [1.5%, 8%] of entry. TP: entry + 2.0R (config `rr_ratio`).
- Breakeven at +1R; chandelier trail after +1.5R: SL = max(SL, high-watermark − trail_dist) where trail_dist = 2.5×ATR at entry (stored on the trade).
- Regime-break exit: daily close < EMA200(d).
- Optional time stop (`max_holding_days`, default off).

**Sizing:** risk_amount = NAV × risk_per_trade_pct(default 1%) × conf_mult; qty = risk_amount / (entry − stop); notional capped at min(20% NAV, 98% cash); skip if notional < $100. Fills apply fee (0.05%) + slippage (0.05%).

**Portfolio guards:** max 5 open positions; total exposure ≤ 80% NAV; daily loss ≥ 3% from day-start NAV trips circuit breaker until next session; 24h cooldown per symbol after a stop-loss exit.

## Backtester

- Per symbol: 1h history (≤730d, default 365d) + daily history (5y) for regime warmup.
- Signals on completed bars; fills next bar open ± slippage; intra-bar SL/TP with conservative same-bar rule (SL first).
- Same gates as live minus AI (sentiment None → factor 0, no veto).
- Metrics: total return, CAGR, max drawdown, Sharpe (daily), win rate, profit factor, expectancy (R), exposure, trades, fees; equity + drawdown series; SPY buy-and-hold comparison.

## Accounting

`execute_buy/execute_sell` run balance check + portfolio mutation + trade row in a single `BEGIN IMMEDIATE` transaction. PnL is fee-inclusive (USD and %). Guardian and worker both go through these. Equity snapshots recorded to `equity_history` each cycle (dedup ~5min).

## UI

Streamlit, bilingual TR/EN (Istanbul timezone display), tabs:
1. **Overview** — NAV metrics, equity curve, open positions with live PnL/stop distance, signal scanner table (per-gate status per symbol), recent trades.
2. **Charts** — candles + EMA + trade markers, RSI/MACD panes, latest AI run for the asset.
3. **Backtest** — parameter form → run → metric cards, equity/drawdown chart, trades table.
4. **Settings** — watchlist, risk params, strategy toggles, AI keys/models, Telegram, reset portfolio.
5. **Logs** — filterable event log + AI audit trail.
Sidebar: bot start/stop, status heartbeat, language, auto-refresh.

## Compatibility

- Keep `run_worker_cycle(force=)`, `calculate_total_nav`, `validate_api_key_for_start` signatures (Telegram + FastAPI import them).
- Keep config keys `bot_running`, `selected_assets`, `risk_percentage` (now = risk-per-trade %); `/sltp` Telegram command remaps to `atr_mult_sl` / `rr_ratio`.
- DB migrations are additive; `is_active=1` still marks open positions.

## Testing

Unit tests: indicators (known-value checks), strategy gates/triggers (synthetic frames), risk sizing/caps/calendar, accounting atomicity (concurrent threads), backtest smoke on synthetic data. Verification: full test run + launch worker cycle once + Streamlit boot.
