# Sentix v2 | AI-Assisted Algorithmic Swing-Trading Platform

Sentix is a modular, Docker-ready **paper-trading** platform for US stocks, ETFs and crypto. Version 2 is a ground-up rebuild of the trading engine around three principles:

1. **Trade only with the trend** — multi-timeframe regime gates keep the bot out of downtrends and bear markets.
2. **Risk is a number, not a vibe** — every position is sized so a stop-out loses a fixed, small percentage of the portfolio (ATR-based stops, true risk-based sizing).
3. **Never trust a strategy you haven't backtested** — the built-in backtester runs the *exact same strategy code* as the live worker over historical data.

> ⚠️ Sentix executes **paper trades only** (a simulated $10,000 portfolio). Nothing here is financial advice; backtest results do not guarantee future returns.

---

## What changed vs v1 (why v1 lost money)

| v1 fault | v2 fix |
|---|---|
| RSI crossing *below* 30 triggered a BUY (buying falling knives) | Only upward momentum triggers: MACD cross-up, EMA20 reclaim, RSI *recovery* through 35 |
| No trend filter — bought 1h signals inside daily downtrends | Daily regime gate (close > EMA200, EMA50 > EMA200) + benchmark filter (SPY/BTC > EMA200) |
| "Risk %" was actually notional size (real risk ≈ 0.06 % NAV) | True sizing: `qty = (NAV × risk%) / (entry − stop)` |
| Fixed 2 % stops ignored volatility | Stops = 2.5–3.5 × ATR(14), clamped, with breakeven + chandelier trailing |
| Guardian & analysis threads raced on the portfolio (cash could vanish) | All fills are single SQLite transactions (`BEGIN IMMEDIATE`), tested under concurrency |
| No exposure caps, circuit breaker or cooldowns | Max positions, per-position & total exposure caps, daily-loss circuit breaker, post-stop cooldown |
| Traded 24/7 on stale night/weekend prices | Market-hours calendar + session-aware data-freshness guard (crypto stays 24/7) |
| Random "simulated" sentiment & fake candles reached decisions | All synthetic fallbacks removed; no data ⇒ no trade |
| No backtesting at all | Event-driven backtester + metrics + UI page, sharing the live strategy code |

---

## Strategy (long-only swing)

**Entry — all gates must pass:**
- Market open for the symbol & data fresh (session-aware; blocks holidays/dead feeds)
- Daily uptrend: close > EMA200 **and** EMA50 > EMA200
- Benchmark uptrend (SPY for equities, BTC-USD for crypto) — configurable
- Trigger on the last **completed** 1h candle: MACD cross-up / EMA20 reclaim / RSI recovery
- RSI(1h) between 35 and 68 (no knives, no chasing)
- Confluence score ≥ threshold — factors: 4h trend (+15), volume expansion (+10), near support or bullish pattern (+10), AI sentiment (+15)
- AI veto: Gemini sentiment ≤ −3 blocks the trade (AI optional; bot runs technical-only without a key)
- Not in cooldown, circuit breaker clear, portfolio caps OK

**Position sizing:** risk 1 % of NAV per trade (× 0.5–1.0 confidence multiplier), capped at 20 % NAV per position, 80 % total exposure, max 5 open positions.

**Exit:**
- Initial stop: entry − 4.5×ATR(14) (clamped 3–8 %) — wide stops beat tight ones in the tuning backtests
- Target: 1.5R; breakeven at +1R; chandelier trail (high-watermark − 2.5×ATR) after +1.5R
- Regime break (daily close < EMA200) closes the position; optional time stop
- Daily-loss circuit breaker: −3 % from day-start NAV halts new entries until the next session; a stop-out puts the symbol on a 24h cooldown

**Reference backtest** (10-symbol tech/ETF watchlist, hourly bars, fees + slippage included, AI disabled, run 2026-07-06):

| Window | Return | Max DD | Sharpe | Trades | Win rate | Profit factor | Expectancy |
|---|---|---|---|---|---|---|---|
| 12 months | +8.3 % | −10.9 % | 0.62 | 128 | 40.6 % | 1.18 | +0.13R |
| 22 months | +15.6 % | −11.0 % | 0.67 | 212 | 42.5 % | 1.19 | +0.16R |

Honest context: SPY buy-and-hold returned more over the same strong bull-market windows (+22 % / +41 %). The system's value is **positive expectancy with controlled risk** — capped drawdowns, a circuit breaker, and regime gates that move to cash in downtrends, which buy-and-hold cannot do. Re-run the backtest yourself from the dashboard before trusting any parameter change.

---

## Components

```
core/
├── indicators.py    Pure indicator math (EMA/RSI/MACD/ATR, swings, patterns)
├── strategy.py      Entry/exit evaluation — pure functions, fully unit-tested
├── risk.py          Sizing, caps, market calendar, freshness guards
├── accounting.py    Atomic fee-aware paper fills, NAV, circuit breaker
├── backtest.py      Event-driven backtester (no lookahead, next-bar fills)
├── data_fetcher.py  yfinance OHLCV (UTC) + Google News RSS
├── db.py            SQLite schema, equity history, signals audit, cooldowns
├── config.py        StrategyConfig — one dataclass for live AND backtest
└── telegram_bot.py  Remote control & trade notifications
ai/sentiment_analyzer.py  Batched Gemini news sentiment (optional)
worker.py            Analysis loop (5 min) + SL/TP guardian loop (60 s)
ui/app.py            Streamlit dashboard (TR/EN)
api/main.py          FastAPI REST endpoints
```

Run the tests any time: `python -m unittest core.test_strategy core.test_risk core.test_accounting core.test_backtest`

---

## Quick start

```bash
cp .env.example .env          # add GEMINI_API_KEY / Telegram tokens if you want AI + alerts
python -m venv .venv && .venv\Scripts\activate   # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python worker.py              # terminal 1: trading engine
streamlit run ui/app.py       # terminal 2: dashboard at http://localhost:8501
```

Docker: `docker-compose up --build -d` (worker + UI; optional Gluetun VPN block in the compose file).

**Watchlist** accepts any yfinance ticker: `AAPL`, `QQQ`, `BTC-USD`, `ETH-USD`… Crypto trades around the clock; equities trade NYSE regular hours only.

---

## Dashboard

- **Portfolio** — NAV, equity curve, open positions with live PnL and stop distance, closed trades with R-multiples
- **Signal Scanner** — per-symbol gate-by-gate checklist showing exactly why the bot did or didn't trade
- **Charts** — candles with EMA/RSI/MACD panes and trade markers
- **Backtest** — run the strategy over up to ~23 months of hourly data, compare with SPY buy-and-hold, inspect every simulated trade
- **Settings** — watchlist, risk parameters, AI keys, bot start/stop
- **Logs** — full event log + AI audit trail

## Telegram commands

`/portfolio` `/positions` `/trades` `/assets` `/ai_status` `/logs` `/trigger` `/pause` `/resume`
`/risk 1.0` — set risk-per-trade (% NAV lost at stop) · `/sltp 2.5 2.0` — set ATR stop multiple & reward:risk ratio

---

## License

**Sentix Personal and Non-Commercial License (SPNCL-1.0)** — personal, educational and research use only; commercial use, paid hosting and fund management are prohibited. See [LICENSE](LICENSE).
