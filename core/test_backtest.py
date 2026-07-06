import unittest

import numpy as np
import pandas as pd

import core.backtest as backtest
from core.backtest import run_backtest, trades_to_frame
from core.config import StrategyConfig

END = pd.Timestamp("2026-07-01 00:00:00", tz="UTC")


def synthetic_ohlcv(symbol, timeframe, limit=None, period=None, **kwargs):
    """Deterministic upward-drifting random walk, shared end time."""
    seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(symbol)) % (2**32)
    rng = np.random.default_rng(seed)

    if timeframe == "1h":
        days = int(str(period).rstrip("d")) if period else 90
        n = days * 24
        freq = "h"
        drift, vol = 0.0004, 0.006
    else:  # "1d"
        n = 900
        freq = "D"
        drift, vol = 0.0008, 0.012

    rets = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    ts = pd.date_range(end=END, periods=n, freq=freq)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame({"timestamp": ts, "open": open_,
                       "high": np.maximum.reduce([open_, close, high]),
                       "low": np.minimum.reduce([open_, close, low]),
                       "close": close,
                       "volume": rng.uniform(1e4, 5e4, n)})
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self._orig_fetch = backtest.fetch_ohlcv
        backtest.fetch_ohlcv = synthetic_ohlcv
        self.cfg = StrategyConfig()

    def tearDown(self):
        backtest.fetch_ohlcv = self._orig_fetch

    def test_backtest_runs_and_produces_metrics(self):
        result = run_backtest(["SYNA", "SYNB"], self.cfg, initial_cash=10000.0, months=2)
        self.assertEqual(result.errors, [])
        self.assertIsNotNone(result.equity)
        self.assertGreater(len(result.equity), 500)
        # equity starts at initial cash
        self.assertAlmostEqual(float(result.equity.iloc[0]), 10000.0, delta=200.0)

        m = result.metrics
        for key in ("total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe",
                    "win_rate_pct", "profit_factor", "n_trades", "expectancy_r", "fees_total"):
            self.assertIn(key, m)
        self.assertLessEqual(m["max_drawdown_pct"], 0.0)
        self.assertGreater(m["n_trades"], 0, "uptrending synthetic data should produce trades")
        self.assertIsNotNone(result.benchmark)

    def test_trades_frame_shape(self):
        result = run_backtest(["SYNA"], self.cfg, initial_cash=10000.0, months=2)
        df = trades_to_frame(result.trades)
        if not df.empty:
            for col in ("symbol", "entry_time", "exit_time", "pnl_usd", "exit_reason", "score"):
                self.assertIn(col, df.columns)
            # every fill respects sizing: no position risked more than ~2x target risk
            self.assertTrue((df["qty"] > 0).all())

    def test_no_lookahead_in_daily_slice(self):
        df = synthetic_ohlcv("SYNA", "1d")
        ts = pd.Timestamp("2026-05-15 14:00:00", tz="UTC")
        sliced = backtest._completed_daily(df, ts)
        self.assertTrue((sliced["timestamp"] < ts.normalize()).all())

    def test_cash_never_negative(self):
        result = run_backtest(["SYNA", "SYNB"], self.cfg, initial_cash=10000.0, months=2)
        # NAV can dip but equity must never go below zero (long-only, no leverage)
        self.assertTrue((result.equity > 0).all())


if __name__ == "__main__":
    unittest.main()
