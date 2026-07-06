import os
import tempfile
import threading
import unittest

import core.db as db
from core import accounting
from core.config import StrategyConfig


class TempDbTestCase(unittest.TestCase):
    """Redirects core.db to a throwaway SQLite file for the duration of a test."""

    def setUp(self):
        self._orig_db_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp(prefix="sentix_test_")
        db.DB_PATH = os.path.join(self._tmpdir, "test.db")
        db.init_db()
        self.cfg = StrategyConfig(fee_pct=0.001, slippage_pct=0.0, cooldown_hours=24)

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        # best effort cleanup (WAL side files may linger on Windows)
        for name in os.listdir(self._tmpdir):
            try:
                os.remove(os.path.join(self._tmpdir, name))
            except OSError:
                pass
        try:
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def cash(self):
        return db.get_portfolio()["USD"]["balance"]


class TestExecuteBuySell(TempDbTestCase):
    def test_buy_deducts_exact_cost(self):
        trade = accounting.execute_buy("AAPL", 100.0, 10.0, self.cfg,
                                       stop_loss=95.0, take_profit=110.0)
        self.assertIsNotNone(trade)
        # cost = 10 * 100 + 0.1% fee = 1001.0
        self.assertAlmostEqual(self.cash(), 10000.0 - 1001.0, places=6)
        port = db.get_portfolio()
        self.assertAlmostEqual(port["AAPL"]["balance"], 10.0)
        pos = db.get_active_position("AAPL")
        self.assertIsNotNone(pos)
        self.assertEqual(pos["is_active"], 1)
        self.assertAlmostEqual(pos["fees"], 1.0)

    def test_double_buy_rejected(self):
        self.assertIsNotNone(accounting.execute_buy("AAPL", 100.0, 5.0, self.cfg,
                                                    stop_loss=95.0, take_profit=110.0))
        self.assertIsNone(accounting.execute_buy("AAPL", 100.0, 5.0, self.cfg,
                                                 stop_loss=95.0, take_profit=110.0))

    def test_insufficient_cash_shrinks_or_rejects(self):
        # try to buy $50,000 worth with $10,000 cash -> shrinks to fit
        trade = accounting.execute_buy("MSFT", 500.0, 100.0, self.cfg,
                                       stop_loss=480.0, take_profit=540.0)
        self.assertIsNotNone(trade)
        self.assertGreater(self.cash(), 0.0)

    def test_sell_closes_position_with_fee_inclusive_pnl(self):
        accounting.execute_buy("AAPL", 100.0, 10.0, self.cfg,
                               stop_loss=95.0, take_profit=110.0)
        result = accounting.execute_sell("AAPL", 110.0, self.cfg, trade_type="TAKE_PROFIT",
                                         reason="test")
        self.assertIsNotNone(result)
        # buy cost 1001.0; proceeds 1100 - 1.1 fee = 1098.9; pnl = 97.9
        self.assertAlmostEqual(result["pnl_usd"], 97.9, places=6)
        self.assertAlmostEqual(self.cash(), 10000.0 - 1001.0 + 1098.9, places=6)
        self.assertIsNone(db.get_active_position("AAPL"))
        port = db.get_portfolio()
        self.assertAlmostEqual(port["AAPL"]["balance"], 0.0)

    def test_sell_without_holdings_rejected(self):
        self.assertIsNone(accounting.execute_sell("TSLA", 200.0, self.cfg))

    def test_stop_loss_exit_sets_cooldown(self):
        accounting.execute_buy("NVDA", 100.0, 5.0, self.cfg, stop_loss=95.0, take_profit=110.0)
        accounting.execute_sell("NVDA", 95.0, self.cfg, trade_type="STOP_LOSS", reason="stop")
        self.assertIsNotNone(db.get_cooldown("NVDA"))

    def test_take_profit_exit_sets_no_cooldown(self):
        accounting.execute_buy("NVDA", 100.0, 5.0, self.cfg, stop_loss=95.0, take_profit=110.0)
        accounting.execute_sell("NVDA", 110.0, self.cfg, trade_type="TAKE_PROFIT", reason="tp")
        self.assertIsNone(db.get_cooldown("NVDA"))

    def test_slippage_applied(self):
        cfg = StrategyConfig(fee_pct=0.0, slippage_pct=0.01)
        trade = accounting.execute_buy("AMD", 100.0, 1.0, cfg, stop_loss=95.0, take_profit=110.0)
        self.assertAlmostEqual(trade["price"], 101.0)

    def test_concurrent_trades_conserve_cash(self):
        """The v1 race: concurrent read-modify-write lost/duplicated cash.
        v2 uses one transaction per fill, so totals must stay exact."""
        cfg = StrategyConfig(fee_pct=0.0, slippage_pct=0.0, cooldown_hours=0)
        symbols = [f"SYM{i}" for i in range(8)]

        def cycle(sym):
            for _ in range(3):
                accounting.execute_buy(sym, 100.0, 2.0, cfg, stop_loss=95.0, take_profit=110.0)
                accounting.execute_sell(sym, 100.0, cfg, trade_type="MANUAL", reason="t")

        threads = [threading.Thread(target=cycle, args=(s,)) for s in symbols]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # zero fees, zero slippage, flat price -> cash must be exactly the start
        self.assertAlmostEqual(self.cash(), 10000.0, places=4)
        for s in symbols:
            self.assertIsNone(db.get_active_position(s))


class TestNavAndCircuitBreaker(TempDbTestCase):
    def test_nav_uses_price_overrides(self):
        accounting.execute_buy("AAPL", 100.0, 10.0, StrategyConfig(fee_pct=0.0, slippage_pct=0.0),
                               stop_loss=95.0, take_profit=110.0)
        nav, cash, exposure = accounting.calculate_nav({"AAPL": 120.0})
        self.assertAlmostEqual(exposure, 1200.0)
        self.assertAlmostEqual(nav, 9000.0 + 1200.0)

    def test_circuit_breaker_trips_on_daily_loss(self):
        cfg = StrategyConfig(daily_loss_limit_pct=3.0)
        active, _ = accounting.check_circuit_breaker(10000.0, cfg)   # sets day baseline
        self.assertFalse(active)
        active, _ = accounting.check_circuit_breaker(9800.0, cfg)    # -2%: fine
        self.assertFalse(active)
        active, detail = accounting.check_circuit_breaker(9690.0, cfg)  # -3.1%: trip
        self.assertTrue(active, detail)
        active, _ = accounting.check_circuit_breaker(10500.0, cfg)   # stays tripped until next session
        self.assertTrue(active)


if __name__ == "__main__":
    unittest.main()
