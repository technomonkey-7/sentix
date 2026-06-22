import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add workspace root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import init_db, get_connection, log_event, save_config, get_config
import worker

def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))

class TestWorkerCycleExecution(unittest.TestCase):
    
    def setUp(self):
        # Initialize db
        init_db()
        # Set some configurations for the test
        save_config("selected_assets", "NVDA/USD,AAPL/USD")
        save_config("bot_running", "true")
        save_config("live_mode", "false")
        save_config("risk_percentage", "2.0")
        save_config("stop_loss_pct", "2.0")
        save_config("take_profit_pct", "5.0")
        save_config("trailing_stop_loss_pct", "2.0")
        save_config("vpn_check_enabled", "false")
        
    @patch('worker.validate_api_key_for_start', return_value=True)
    @patch('ai.sentiment_analyzer.analyze_sentiment_batch')
    def test_worker_cycle_buy_sell(self, mock_sentiment_batch, mock_api_key):
        """Simulates worker cycle where one asset triggers a BUY and another a SELL/Neutral, verifying scoring and execution."""
        
        # Configure mocked sentiment response
        mock_sentiment_batch.return_value = [
            {
                "symbol": "NVDA/USD",
                "sentiment_score": 4,
                "reason": "Strong positive stock news, GPU demand growing.",
                "digest": "GPU demand high.",
                "is_simulated": False
            },
            {
                "symbol": "AAPL/USD",
                "sentiment_score": -4,
                "reason": "Regulatory headwinds and weaker sales guidance.",
                "digest": "Regulatory issues.",
                "is_simulated": False
            }
        ]
        
        # Let's mock fetch_ohlcv to return custom DataFrames that will trigger specific behaviors
        # For NVDA: let's trigger EMA Bullish Crossover and make 4H trend, volume, and patterns confirmed to get score 4/4
        # For AAPL: let's trigger EMA Bearish Crossover and make 4H trend bearish to test confluence exit
        
        import pandas as pd
        import numpy as np
        
        # Helper to build mock DF with indicators
        def get_mock_df(symbol, close_trend, is_bullish_trend=True):
            limit = 40
            timestamps = [f"2026-06-22 {10+i:02d}:00:00" for i in range(limit)]
            
            # Simple price series
            close_prices = []
            current = 100.0
            for i in range(limit):
                if i >= limit - 3:
                    # Crossovers at the end
                    if close_trend == "up":
                        current += 2.0
                    else:
                        current -= 2.0
                else:
                    current += 0.1
                close_prices.append(current)
                
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': [c - 0.5 for c in close_prices],
                'high': [c + 1.0 for c in close_prices],
                'low': [c - 1.0 for c in close_prices],
                'close': close_prices,
                'volume': [50000.0] * limit,
                '_is_simulated': False
            })
            
            # Add indicators manually to control crossovers
            df['ema'] = [100.0] * limit
            df['rsi'] = [50.0] * limit
            df['macd'] = [0.0] * limit
            df['macd_signal'] = [0.0] * limit
            df['macd_hist'] = [0.0] * limit
            
            # Set specific crossover at index -2 and -3
            if close_trend == "up":
                # Price breaks above EMA 20 (close_p <= ema_p, close_c >= ema_c * 1.0005)
                # ema_p = 100.0, close_p = 99.0
                # ema_c = 100.0, close_c = 101.0
                df.loc[limit-3, 'close'] = 99.0
                df.loc[limit-3, 'ema'] = 100.0
                df.loc[limit-2, 'close'] = 101.0
                df.loc[limit-2, 'ema'] = 100.0
                
                # Make volume expanding: latest volume > mean(20)
                df.loc[limit-2, 'volume'] = 100000.0
            else:
                # Price breaks below EMA 20 (close_p >= ema_p, close_c <= ema_c * 0.9995)
                # ema_p = 100.0, close_p = 101.0
                # ema_c = 100.0, close_c = 99.0
                df.loc[limit-3, 'close'] = 101.0
                df.loc[limit-3, 'ema'] = 100.0
                df.loc[limit-2, 'close'] = 99.0
                df.loc[limit-2, 'ema'] = 100.0
                
            return df
            
        nvda_df = get_mock_df("NVDA/USD", "up")
        aapl_df = get_mock_df("AAPL/USD", "down")
        
        def mock_fetch_ohlcv(symbol, timeframe, limit=100):
            if "NVDA" in symbol:
                return nvda_df.copy()
            else:
                return aapl_df.copy()
                
        # Patch fetch_ohlcv in worker
        with patch('worker.fetch_ohlcv', side_effect=mock_fetch_ohlcv):
            print("\n--- Running worker cycle mock test ---")
            
            # Reset portfolio to have USD balance
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE portfolio SET balance = 10000.0 WHERE asset = 'USD'")
            # Remove any holdings
            cursor.execute("DELETE FROM portfolio WHERE asset != 'USD'")
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM logs")
            conn.commit()
            conn.close()
            
            # Execute worker cycle
            worker._execute_cycle_logic(force=True)
            
            # Fetch active trades
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades")
            trades = [dict(r) for r in cursor.fetchall()]
            
            cursor.execute("SELECT * FROM portfolio")
            portfolio = [dict(r) for r in cursor.fetchall()]
            
            cursor.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 20")
            logs = [dict(r) for r in cursor.fetchall()]
            conn.close()
            
            safe_print("\n--- PORTFOLIO STATE ---")
            for p in portfolio:
                safe_print(f"Asset: {p['asset']} | Balance: {p['balance']} | Avg Price: {p['avg_entry_price']}")
                
            safe_print("\n--- TRADES RECORDED ---")
            for t in trades:
                safe_print(f"ID: {t['id']} | Asset: {t['asset']} | Side: {t['side']} | Price: {t['price']} | Sentiment: {t['sentiment_score']} | SL: {t['stop_loss']} | TP: {t['take_profit']} | Reason: {t['reason']}")
                
            safe_print("\n--- RECENT LOGS ---")
            for l in reversed(logs):
                safe_print(f"[{l['level']}] [{l['module']}] {l['message']}")
                
            # Assertions
            # NVDA should have a recorded BUY trade since score is high enough (at least 2: 4H trend + volume + sentiment 4)
            nvda_buys = [t for t in trades if t['asset'] == "NVDA/USD" and t['side'] == "BUY"]
            self.assertEqual(len(nvda_buys), 1)
            self.assertEqual(nvda_buys[0]['sentiment_score'], 4)
            # Check stop loss and take profit are set
            self.assertIsNotNone(nvda_buys[0]['stop_loss'])
            self.assertIsNotNone(nvda_buys[0]['take_profit'])
            
            # AAPL should NOT have a SELL execution because there is no open position to sell (empty holdings)
            # But let's check that the logs mentioned it or skipped it safely
            aapl_sells = [t for t in trades if t['asset'] == "AAPL/USD" and t['side'] == "SELL"]
            self.assertEqual(len(aapl_sells), 0)

            # Let's set an active position for AAPL and test confluence exit!
            conn = get_connection()
            cursor = conn.cursor()
            # Insert active position in portfolio and trade log
            cursor.execute("INSERT OR REPLACE INTO portfolio (asset, balance, avg_entry_price) VALUES ('AAPL', 10.0, 100.0)")
            cursor.execute("""
                INSERT INTO trades (timestamp, asset, side, price, amount, trade_type, sentiment_score, reason, stop_loss, take_profit, is_active)
                VALUES ('2026-06-22 10:00:00', 'AAPL/USD', 'BUY', 100.0, 10.0, 'AI_HYBRID', 3, 'Initial entry', 98.0, 105.0, 1)
            """)
            conn.commit()
            conn.close()
            
            print("\n--- Running worker cycle mock test (with AAPL active position for Confluence Exit) ---")
            
            # We must reset analyzed timestamp to allow re-analysis
            worker._last_analyzed_timestamps = {}
            
            worker._execute_cycle_logic(force=True)
            
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades")
            trades = [dict(r) for r in cursor.fetchall()]
            cursor.execute("SELECT * FROM portfolio")
            portfolio = [dict(r) for r in cursor.fetchall()]
            conn.close()
            
            safe_print("\n--- PORTFOLIO STATE AFTER CONFLUENCE EXIT ---")
            for p in portfolio:
                safe_print(f"Asset: {p['asset']} | Balance: {p['balance']} | Avg Price: {p['avg_entry_price']}")
                
            safe_print("\n--- TRADES RECORDED AFTER CONFLUENCE EXIT ---")
            for t in trades:
                safe_print(f"ID: {t['id']} | Asset: {t['asset']} | Side: {t['side']} | Price: {t['price']} | PnL: {t.get('pnl')} | Active: {t['is_active']} | Reason: {t['reason']}")
            
            # AAPL should be sold and position closed (is_active = 0)
            aapl_pos = [p for p in portfolio if p['asset'] == 'AAPL']
            self.assertEqual(aapl_pos[0]['balance'], 0.0)
            
            # There should be a closing trade entry or updated trades table
            closed_trades = [t for t in trades if t['asset'] == "AAPL/USD" and t['is_active'] == 0 and t['side'] == "BUY"]
            self.assertEqual(len(closed_trades), 1)
            self.assertIsNotNone(closed_trades[0]['pnl'])

if __name__ == '__main__':
    unittest.main()
