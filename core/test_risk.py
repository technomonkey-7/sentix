import unittest
from datetime import datetime, timezone

from core.config import StrategyConfig, normalize_symbol, parse_watchlist
from core.risk import (
    clamp_stop, compute_position_size, data_is_fresh, is_crypto,
    is_market_open, market_minutes_elapsed, next_session_start,
)


def utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


class TestMarketCalendar(unittest.TestCase):
    def test_crypto_always_open(self):
        self.assertTrue(is_market_open("BTC-USD", utc(2026, 7, 4, 3, 0)))   # Saturday night
        self.assertTrue(is_crypto("ETH-USD"))
        self.assertFalse(is_crypto("AAPL"))

    def test_equity_regular_hours(self):
        # Tuesday 2026-07-07 15:00 UTC == 11:00 New York (EDT) -> open
        self.assertTrue(is_market_open("AAPL", utc(2026, 7, 7, 15, 0)))
        # Tuesday 12:00 UTC == 08:00 NY -> pre-market, closed
        self.assertFalse(is_market_open("AAPL", utc(2026, 7, 7, 12, 0)))
        # Tuesday 21:00 UTC == 17:00 NY -> after hours, closed
        self.assertFalse(is_market_open("AAPL", utc(2026, 7, 7, 21, 0)))
        # Saturday -> closed
        self.assertFalse(is_market_open("AAPL", utc(2026, 7, 11, 15, 0)))

    def test_market_minutes_skip_weekend(self):
        # Friday 2026-07-10 19:00 UTC (15:00 NY) to Sunday: only 60 session minutes elapsed
        elapsed = market_minutes_elapsed(utc(2026, 7, 10, 19, 0), utc(2026, 7, 12, 12, 0))
        self.assertAlmostEqual(elapsed, 60.0, delta=1.0)

    def test_freshness_over_weekend(self):
        # Candle from Friday 15:00 NY is still fresh on Sunday (market never opened)
        self.assertTrue(data_is_fresh("AAPL", utc(2026, 7, 10, 19, 0), 60, utc(2026, 7, 12, 12, 0)))
        # ...but stale by Monday 10:31 NY (60 Friday minutes + 61 Monday minutes > 120)
        self.assertFalse(data_is_fresh("AAPL", utc(2026, 7, 10, 19, 0), 60, utc(2026, 7, 13, 14, 31)))

    def test_crypto_freshness_is_wall_clock(self):
        self.assertTrue(data_is_fresh("BTC-USD", utc(2026, 7, 12, 11, 0), 60, utc(2026, 7, 12, 12, 30)))
        self.assertFalse(data_is_fresh("BTC-USD", utc(2026, 7, 12, 9, 0), 60, utc(2026, 7, 12, 12, 30)))

    def test_next_session_start_rolls_over_weekend(self):
        # From Friday 20:00 UTC the next session is Monday 09:30 NY
        nxt = next_session_start(utc(2026, 7, 10, 20, 0))
        self.assertEqual(nxt.astimezone(timezone.utc).weekday(), 0)


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyConfig()  # 1% risk, 20% pos cap, 80% exposure cap, 5 max pos

    def test_risk_based_quantity(self):
        # NAV 10k, risk 1% = $100, stop distance $2 -> 50 shares = $5000 notional,
        # capped at 20% NAV = $2000 -> 20 shares
        r = compute_position_size(10000, 10000, 100.0, 98.0, 1.0, 0, 0, self.cfg)
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.risk_amount, 100.0)
        self.assertAlmostEqual(r.notional, 2000.0)
        self.assertAlmostEqual(r.quantity, 20.0)

    def test_uncapped_when_stop_is_wide(self):
        # stop distance $8 -> 12.5 shares = $1250 notional, under all caps
        r = compute_position_size(10000, 10000, 100.0, 92.0, 1.0, 0, 0, self.cfg)
        self.assertAlmostEqual(r.quantity, 12.5)
        self.assertAlmostEqual(r.notional, 1250.0)

    def test_confidence_scales_risk(self):
        full = compute_position_size(10000, 10000, 100.0, 92.0, 1.0, 0, 0, self.cfg)
        half = compute_position_size(10000, 10000, 100.0, 92.0, 0.5, 0, 0, self.cfg)
        self.assertAlmostEqual(half.notional, full.notional / 2)

    def test_exposure_cap(self):
        # exposure cap 80% of 10k = 8000; already 7500 invested -> only 500 room
        r = compute_position_size(10000, 10000, 100.0, 92.0, 1.0, 7500, 0, self.cfg)
        self.assertAlmostEqual(r.notional, 500.0)

    def test_cash_cap(self):
        r = compute_position_size(10000, 300.0, 100.0, 92.0, 1.0, 0, 0, self.cfg)
        self.assertAlmostEqual(r.notional, 294.0)  # 98% of cash

    def test_max_positions_blocks(self):
        r = compute_position_size(10000, 10000, 100.0, 92.0, 1.0, 0, 5, self.cfg)
        self.assertFalse(r.ok)

    def test_min_notional_blocks_dust(self):
        r = compute_position_size(10000, 50.0, 100.0, 92.0, 1.0, 0, 0, self.cfg)
        self.assertFalse(r.ok)

    def test_invalid_stop_blocks(self):
        r = compute_position_size(10000, 10000, 100.0, 101.0, 1.0, 0, 0, self.cfg)
        self.assertFalse(r.ok)

    def test_clamp_stop_bounds(self):
        cfg = self.cfg
        # too tight (0.5%) -> widened to 1.5%
        self.assertAlmostEqual(clamp_stop(100.0, 99.5, cfg), 98.5)
        # too wide (12%) -> tightened to 8%
        self.assertAlmostEqual(clamp_stop(100.0, 88.0, cfg), 92.0)
        # in range stays
        self.assertAlmostEqual(clamp_stop(100.0, 96.0, cfg), 96.0)


class TestSymbols(unittest.TestCase):
    def test_legacy_migration(self):
        self.assertEqual(normalize_symbol("AAPL/USD"), "AAPL")
        self.assertEqual(normalize_symbol("BTC/USD"), "BTC-USD")
        self.assertEqual(normalize_symbol("btc-usd"), "BTC-USD")
        self.assertEqual(normalize_symbol(" nvda "), "NVDA")

    def test_parse_watchlist_dedup(self):
        out = parse_watchlist("AAPL/USD, AAPL, MSFT")
        self.assertEqual(out, ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
