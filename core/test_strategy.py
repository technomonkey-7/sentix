import unittest

import numpy as np
import pandas as pd

from core.config import StrategyConfig
from core.indicators import add_indicators, swing_levels, bullish_patterns, rsi
from core.strategy import (
    detect_trigger, trigger_mask, evaluate_entry, evaluate_exit, update_trailing_stop,
)


def make_1h_frame(n=60, close=None, **overrides):
    """Builds a 1h frame with explicit indicator columns for deterministic tests."""
    if close is None:
        close = [100.0] * n
    elif len(close) < n:
        close = [close[0]] * (n - len(close)) + list(close)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC"),
        "open": [c - 0.1 for c in close],
        "high": [c + 0.5 for c in close],
        "low": [c - 0.5 for c in close],
        "close": close,
        "volume": [10000.0] * n,
        "ema": [100.0] * n,
        "ema50": [100.0] * n,
        "rsi": [50.0] * n,
        "macd": [0.0] * n,
        "macd_signal": [0.0] * n,
        "macd_hist": [0.0] * n,
        "atr": [2.0] * n,
        "vol_sma": [10000.0] * n,
    })
    for col, values in overrides.items():
        df.loc[df.index[-len(values):], col] = values
    return df


def make_daily_frame(uptrend=True):
    n = 5
    if uptrend:
        base = {"close": 120.0, "ema50": 115.0, "ema200": 100.0}
    else:
        base = {"close": 90.0, "ema50": 95.0, "ema200": 100.0}
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-05-28", periods=n, freq="D", tz="UTC"),
        "close": [base["close"]] * n,
        "ema50": [base["ema50"]] * n,
        "ema200": [base["ema200"]] * n,
    })


class TestTriggers(unittest.TestCase):
    def test_rsi_crossing_below_30_is_NOT_a_buy(self):
        """Regression for the v1 falling-knife bug: RSI dropping through 30
        must never be treated as an entry trigger."""
        df = make_1h_frame(rsi=[45.0, 32.0, 28.0, 28.0])
        side, _ = detect_trigger(df)
        self.assertIsNone(side)

    def test_rsi_recovery_through_35_is_a_buy(self):
        df = make_1h_frame(rsi=[28.0, 30.0, 36.0, 36.0])
        side, detail = detect_trigger(df)
        self.assertEqual(side, "RSI_RECOVERY")
        self.assertIn("35", detail)

    def test_macd_cross_up_triggers(self):
        df = make_1h_frame(macd=[0.0, -0.1, 0.3, 0.3], macd_signal=[0.0, 0.1, 0.2, 0.2])
        side, _ = detect_trigger(df)
        self.assertEqual(side, "MACD_CROSS_UP")

    def test_macd_cross_below_noise_threshold_ignored(self):
        # separation of 0.001 on a 100 close = 0.001% < 0.005% minimum
        df = make_1h_frame(macd=[0.0, 0.0999, 0.201, 0.201], macd_signal=[0.0, 0.1, 0.2, 0.2])
        side, _ = detect_trigger(df)
        self.assertIsNone(side)

    def test_ema_cross_up_triggers(self):
        df = make_1h_frame(close=[99.0, 99.0, 101.0, 101.0])
        side, _ = detect_trigger(df)
        self.assertEqual(side, "EMA20_CROSS_UP")

    def test_macd_bearish_cross_is_not_an_entry(self):
        df = make_1h_frame(macd=[0.0, 0.3, 0.1, 0.1], macd_signal=[0.0, 0.2, 0.2, 0.2])
        side, _ = detect_trigger(df)
        self.assertIsNone(side)

    def test_vectorized_mask_matches_scalar_detection(self):
        np.random.seed(11)
        n = 300
        close = 100 + np.cumsum(np.random.normal(0.03, 0.9, n))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "open": close - 0.1, "high": close + 0.7, "low": close - 0.7,
            "close": close, "volume": np.random.uniform(1e4, 5e4, n),
        })
        df = add_indicators(df)
        mask = trigger_mask(df)
        for i in range(40, n):
            scalar = detect_trigger(df.iloc[: i + 1], completed_idx=-1)[0] is not None
            self.assertEqual(scalar, bool(mask.iloc[i]), f"mismatch at bar {i}")


class TestEvaluateEntry(unittest.TestCase):
    def setUp(self):
        # min_confluence_score=50 so gate tests exercise the base path;
        # the quality gate has its own dedicated test below
        self.cfg = StrategyConfig(min_confluence_score=50)
        # a valid MACD-cross entry setup on the completed candle
        self.df_1h = make_1h_frame(macd=[0.0, -0.1, 0.3, 0.3], macd_signal=[0.0, 0.1, 0.2, 0.2])
        self.daily_up = make_daily_frame(uptrend=True)
        self.daily_down = make_daily_frame(uptrend=False)

    def test_enters_in_full_uptrend(self):
        sig = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_up, self.cfg)
        self.assertTrue(sig.should_enter, sig.reason)
        self.assertLess(sig.stop, 100.0)
        self.assertGreater(sig.take_profit, 100.0)
        self.assertGreaterEqual(sig.score, 50)
        self.assertIn(sig.conf_mult, (0.5, 0.75, 1.0))

    def test_daily_downtrend_blocks_entry(self):
        sig = evaluate_entry(self.df_1h, None, self.daily_down, self.daily_up, self.cfg)
        self.assertFalse(sig.should_enter)
        failed = [g.name for g in sig.gates if not g.passed]
        self.assertIn("daily_regime", failed)

    def test_bear_market_benchmark_blocks_entry(self):
        sig = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_down, self.cfg)
        self.assertFalse(sig.should_enter)
        failed = [g.name for g in sig.gates if not g.passed]
        self.assertIn("market_regime", failed)

    def test_overbought_rsi_blocks_entry(self):
        df = make_1h_frame(macd=[0.0, -0.1, 0.3, 0.3], macd_signal=[0.0, 0.1, 0.2, 0.2],
                           rsi=[75.0, 75.0, 75.0, 75.0])
        sig = evaluate_entry(df, None, self.daily_up, self.daily_up, self.cfg)
        self.assertFalse(sig.should_enter)
        failed = [g.name for g in sig.gates if not g.passed]
        self.assertIn("rsi_band", failed)

    def test_ai_veto_blocks_entry(self):
        sig = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_up, self.cfg, sentiment=-5)
        self.assertFalse(sig.should_enter)
        failed = [g.name for g in sig.gates if not g.passed]
        self.assertIn("ai_veto", failed)

    def test_positive_sentiment_raises_score(self):
        base = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_up, self.cfg, sentiment=None)
        boosted = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_up, self.cfg, sentiment=5)
        self.assertEqual(boosted.score, base.score + 15)

    def test_min_confluence_gate_blocks_weak_setups(self):
        strict = StrategyConfig(min_confluence_score=80)
        sig = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_up, strict)
        self.assertFalse(sig.should_enter)
        failed = [g.name for g in sig.gates if not g.passed]
        self.assertIn("confluence", failed)

    def test_stop_respects_clamp_bounds(self):
        sig = evaluate_entry(self.df_1h, None, self.daily_up, self.daily_up, self.cfg)
        dist_pct = (100.0 - sig.stop) / 100.0 * 100
        self.assertGreaterEqual(dist_pct, self.cfg.sl_min_pct - 1e-9)
        self.assertLessEqual(dist_pct, self.cfg.sl_max_pct + 1e-9)
        # TP = entry + rr * risk distance
        self.assertAlmostEqual(sig.take_profit, 100.0 + self.cfg.rr_ratio * (100.0 - sig.stop), places=6)


class TestExitsAndTrailing(unittest.TestCase):
    def setUp(self):
        # rr_ratio pinned to 2.0: these fixtures assume R = (tp-entry)/2
        self.cfg = StrategyConfig(rr_ratio=2.0)

    def test_regime_break_exit(self):
        sig = evaluate_exit({"timestamp": "2026-06-01T00:00:00+00:00"},
                            make_daily_frame(uptrend=False), self.cfg)
        self.assertTrue(sig.should_exit)
        self.assertEqual(sig.code, "REGIME_BREAK")

    def test_no_exit_in_uptrend(self):
        sig = evaluate_exit({"timestamp": "2026-06-01T00:00:00+00:00"},
                            make_daily_frame(uptrend=True), self.cfg)
        self.assertFalse(sig.should_exit)

    def test_time_stop(self):
        cfg = StrategyConfig(max_holding_days=5)
        sig = evaluate_exit({"timestamp": "2026-06-01T00:00:00+00:00"},
                            make_daily_frame(uptrend=True), cfg,
                            now=pd.Timestamp("2026-06-10T00:00:00+00:00").to_pydatetime())
        self.assertTrue(sig.should_exit)
        self.assertEqual(sig.code, "TIME_STOP")

    def test_trailing_stop_never_moves_down(self):
        pos = {"price": 100.0, "stop_loss": 97.0, "take_profit": 106.0,
               "trail_dist": 2.0, "high_watermark": 100.0}
        stop, hwm = update_trailing_stop(pos, 98.0, self.cfg)
        self.assertEqual(stop, 97.0)
        self.assertEqual(hwm, 100.0)

    def test_breakeven_move_at_1r(self):
        # R = (106-100)/2 = 3 -> breakeven armed at price >= 103
        pos = {"price": 100.0, "stop_loss": 97.0, "take_profit": 106.0,
               "trail_dist": 5.0, "high_watermark": 100.0}
        stop, _ = update_trailing_stop(pos, 103.5, self.cfg)
        self.assertGreaterEqual(stop, 100.0)

    def test_chandelier_trail_after_1_5r(self):
        # trail armed at price >= 104.5; trail = hwm - 2.0
        pos = {"price": 100.0, "stop_loss": 97.0, "take_profit": 106.0,
               "trail_dist": 2.0, "high_watermark": 100.0}
        stop, hwm = update_trailing_stop(pos, 105.5, self.cfg)
        self.assertAlmostEqual(stop, 103.5)
        self.assertAlmostEqual(hwm, 105.5)


class TestIndicators(unittest.TestCase):
    def test_rsi_extremes(self):
        rally = pd.Series(np.linspace(100, 200, 60))
        self.assertGreater(float(rsi(rally).iloc[-1]), 95.0)
        crash = pd.Series(np.linspace(200, 100, 60))
        self.assertLess(float(rsi(crash).iloc[-1]), 5.0)

    def test_add_indicators_columns(self):
        np.random.seed(3)
        close = 100 + np.cumsum(np.random.normal(0, 1, 250))
        df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": 1000.0})
        out = add_indicators(df)
        for col in ("ema", "ema50", "ema200", "rsi", "macd", "macd_signal", "macd_hist", "atr", "vol_sma"):
            self.assertIn(col, out.columns)
            self.assertFalse(pd.isna(out[col].iloc[-1]), f"{col} is NaN at the end")
        self.assertGreater(float(out["atr"].iloc[-1]), 0.0)

    def test_swing_levels(self):
        df = pd.DataFrame({
            "high": [100.0] * 3 + [150.0] + [100.0] * 7,
            "low": [100.0] * 7 + [50.0] + [100.0] * 3,
            "open": [100.0] * 11, "close": [100.0] * 11,
        })
        supports, resistances = swing_levels(df, window=2)
        self.assertIn(50.0, supports)
        self.assertIn(150.0, resistances)

    def test_bullish_engulfing(self):
        df = pd.DataFrame({
            "open": [100.0, 94.0, 100.0], "close": [95.0, 101.0, 100.0],
            "high": [101.0, 102.0, 100.0], "low": [94.0, 93.0, 100.0],
        })
        self.assertIn("Bullish Engulfing", bullish_patterns(df))


if __name__ == "__main__":
    unittest.main()
