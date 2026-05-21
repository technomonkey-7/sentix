import unittest
import pandas as pd
import numpy as np
from core.math_engine import calculate_indicators, check_triggers

class TestMathEngine(unittest.TestCase):

    def setUp(self):
        # Create a mock DataFrame with enough data points to compute indicators (min 30 required)
        self.limit = 40
        np.random.seed(42)
        close_prices = [100.0 + i * 0.5 + np.random.normal(0, 0.2) for i in range(self.limit)]
        
        self.df = pd.DataFrame({
            'open': [c - 0.2 for c in close_prices],
            'high': [c + 0.5 for c in close_prices],
            'low': [c - 0.5 for c in close_prices],
            'close': close_prices,
            'volume': [100.0] * self.limit
        })

    def test_calculate_indicators(self):
        """Verifies that EMA, RSI, and MACD indicators are computed successfully and appended as columns."""
        df_indicators = calculate_indicators(self.df)
        
        # Check that indicator columns are present in the resulting DataFrame
        self.assertIn('ema', df_indicators.columns)
        self.assertIn('rsi', df_indicators.columns)
        self.assertIn('macd', df_indicators.columns)
        self.assertIn('macd_signal', df_indicators.columns)
        self.assertIn('macd_hist', df_indicators.columns)
        
        # Check that indicator values for the latest rows are valid floats (not all NaN)
        self.assertFalse(df_indicators['ema'].iloc[-5:].isna().all())
        self.assertFalse(df_indicators['rsi'].iloc[-5:].isna().all())
        self.assertFalse(df_indicators['macd'].iloc[-5:].isna().all())

    def test_check_triggers_no_signal(self):
        """Verifies that neutral states return no crossover triggers."""
        # Create a stable DataFrame where indicators don't cross any threshold
        df_stable = pd.DataFrame({
            'close': [100.0] * 5,
            'ema': [100.0] * 5,
            'rsi': [50.0] * 5,
            'macd': [0.0] * 5,
            'macd_signal': [0.0] * 5
        })
        side, reason = check_triggers(df_stable)
        self.assertIsNone(side)
        self.assertEqual(reason, "No indicator crossovers detected.")

    def test_check_triggers_macd_golden_cross(self):
        """Verifies that a MACD Golden Cross (MACD line crosses above Signal line) triggers a BUY."""
        # index -3 (prev): MACD <= Signal (e.g. MACD=0.1, Signal=0.2)
        # index -2 (curr): MACD > Signal  (e.g. MACD=0.3, Signal=0.2)
        df_mock = pd.DataFrame({
            'close': [100.0] * 5,
            'ema': [100.0] * 5,
            'rsi': [50.0] * 5,
            'macd': [0.0, 0.0, 0.1, 0.3, 0.3],
            'macd_signal': [0.0, 0.0, 0.2, 0.2, 0.2]
        })
        side, reason = check_triggers(df_mock)
        self.assertEqual(side, "BUY")
        self.assertIn("MACD Bullish Cross", reason)

    def test_check_triggers_macd_death_cross(self):
        """Verifies that a MACD Death Cross (MACD line crosses below Signal line) triggers a SELL."""
        # index -3 (prev): MACD >= Signal (e.g. MACD=0.3, Signal=0.2)
        # index -2 (curr): MACD < Signal  (e.g. MACD=0.1, Signal=0.2)
        df_mock = pd.DataFrame({
            'close': [100.0] * 5,
            'ema': [100.0] * 5,
            'rsi': [50.0] * 5,
            'macd': [0.0, 0.0, 0.3, 0.1, 0.1],
            'macd_signal': [0.0, 0.0, 0.2, 0.2, 0.2]
        })
        side, reason = check_triggers(df_mock)
        self.assertEqual(side, "SELL")
        self.assertIn("MACD Bearish Cross", reason)

    def test_check_triggers_rsi_oversold_cross(self):
        """Verifies that RSI dropping below 30 triggers a BUY."""
        # index -3 (prev): RSI >= 30 (e.g. 32)
        # index -2 (curr): RSI < 30 (e.g. 28)
        df_mock = pd.DataFrame({
            'close': [100.0] * 5,
            'ema': [100.0] * 5,
            'rsi': [50.0, 50.0, 32.0, 28.0, 28.0],
            'macd': [0.0] * 5,
            'macd_signal': [0.0] * 5
        })
        side, reason = check_triggers(df_mock)
        self.assertEqual(side, "BUY")
        self.assertIn("RSI Oversold Entry", reason)

    def test_check_triggers_rsi_oversold_recovery(self):
        """Verifies that RSI crossing back above 30 triggers a BUY."""
        # index -3 (prev): RSI < 30 (e.g. 28)
        # index -2 (curr): RSI >= 30 (e.g. 31)
        df_mock = pd.DataFrame({
            'close': [100.0] * 5,
            'ema': [100.0] * 5,
            'rsi': [50.0, 50.0, 28.0, 31.0, 31.0],
            'macd': [0.0] * 5,
            'macd_signal': [0.0] * 5
        })
        side, reason = check_triggers(df_mock)
        self.assertEqual(side, "BUY")
        self.assertIn("RSI Oversold Recovery", reason)

    def test_check_triggers_ema_bullish_cross(self):
        """Verifies that price crossing above EMA 20 triggers a BUY."""
        # index -3 (prev): close <= ema (e.g. close=99, ema=100)
        # index -2 (curr): close > ema  (e.g. close=101, ema=100)
        df_mock = pd.DataFrame({
            'close': [100.0, 100.0, 99.0, 101.0, 101.0],
            'ema': [100.0] * 5,
            'rsi': [50.0] * 5,
            'macd': [0.0] * 5,
            'macd_signal': [0.0] * 5
        })
        side, reason = check_triggers(df_mock)
        self.assertEqual(side, "BUY")
        self.assertIn("EMA Crossover (Price broke above 20 EMA)", reason)

    def test_check_triggers_ema_bearish_cross(self):
        """Verifies that price crossing below EMA 20 triggers a SELL."""
        # index -3 (prev): close >= ema (e.g. close=101, ema=100)
        # index -2 (curr): close < ema  (e.g. close=99, ema=100)
        df_mock = pd.DataFrame({
            'close': [100.0, 100.0, 101.0, 99.0, 99.0],
            'ema': [100.0] * 5,
            'rsi': [50.0] * 5,
            'macd': [0.0] * 5,
            'macd_signal': [0.0] * 5
        })
        side, reason = check_triggers(df_mock)
        self.assertEqual(side, "SELL")
        self.assertIn("EMA Crossover (Price fell below 20 EMA)", reason)

if __name__ == '__main__':
    unittest.main()
