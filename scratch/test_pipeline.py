import sys
import os
import json
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.math_engine import confirm_with_higher_tf
import ai.sentiment_analyzer as sa
from ai.sentiment_analyzer import (
    DigestResult,
    BatchDigestResponse,
    SentimentResult,
    BatchSentimentResponse,
    analyze_sentiment_batch
)

class TestSentixOptimization(unittest.TestCase):

    def setUp(self):
        # Reset sa globals
        sa._api_keys = None
        sa._current_key_idx = 0

    # ==========================================
    # 1. MATH ENGINE DOUBLE-CHECK TESTS
    # ==========================================
    def test_confirm_with_higher_tf_buy_bullish_and_volume(self):
        # BUY: 4h trend is bullish (Close > EMA), 1h volume is expanding
        # Index -2 is the completed candle. Volume at -2 must be high.
        # Mean volume of tail(20) will be: (19*10 + 20)/20 = 10.5
        # Volume at -2 is 20, which is >= 10.5
        df_1h = pd.DataFrame({
            'volume': [10.0] * 18 + [20.0, 10.0],
            'close': [100.0] * 20
        })
        # Index -2 is the completed candle. Close must be > EMA.
        df_4h = pd.DataFrame({
            'close': [100.0] * 18 + [105.0, 100.0],
            'ema': [100.0] * 20
        })
        confirmed, reason = confirm_with_higher_tf(df_1h, df_4h, "BUY")
        self.assertTrue(confirmed, f"Failed: {reason}")
        self.assertIn("Confirmed", reason)

    def test_confirm_with_higher_tf_buy_macd_bullish(self):
        # BUY: 4h trend close <= EMA but MACD > Signal, 1h volume is expanding
        df_1h = pd.DataFrame({
            'volume': [10.0] * 18 + [20.0, 10.0],
            'close': [100.0] * 20
        })
        df_4h = pd.DataFrame({
            'close': [95.0] * 20,
            'ema': [100.0] * 20,
            'macd': [0.0] * 18 + [1.5, 0.0],
            'macd_signal': [0.0] * 18 + [1.0, 0.0]
        })
        confirmed, reason = confirm_with_higher_tf(df_1h, df_4h, "BUY")
        self.assertTrue(confirmed, f"Failed: {reason}")
        self.assertIn("Confirmed", reason)

    def test_confirm_with_higher_tf_buy_bearish_4h(self):
        # BUY: 4h trend close <= EMA and MACD <= Signal, 1h volume is expanding
        df_1h = pd.DataFrame({
            'volume': [10.0] * 18 + [20.0, 10.0],
            'close': [100.0] * 20
        })
        df_4h = pd.DataFrame({
            'close': [95.0] * 20,
            'ema': [100.0] * 20,
            'macd': [0.0] * 18 + [0.5, 0.0],
            'macd_signal': [0.0] * 18 + [1.0, 0.0]
        })
        confirmed, reason = confirm_with_higher_tf(df_1h, df_4h, "BUY")
        self.assertFalse(confirmed, f"Failed: {reason}")
        self.assertIn("4h trend is not bullish", reason)

    def test_confirm_with_higher_tf_buy_low_volume(self):
        # BUY: 4h trend is bullish, but 1h volume is below average (not expanding)
        # Volume at completed candle (-2) is 5.0. Mean is (19*15 + 5)/20 = 14.5.
        # 5.0 < 14.5, so volume check fails.
        df_1h = pd.DataFrame({
            'volume': [15.0] * 18 + [5.0, 15.0],
            'close': [100.0] * 20
        })
        df_4h = pd.DataFrame({
            'close': [100.0] * 18 + [105.0, 100.0],
            'ema': [100.0] * 20
        })
        confirmed, reason = confirm_with_higher_tf(df_1h, df_4h, "BUY")
        self.assertFalse(confirmed, f"Failed: {reason}")
        self.assertIn("volume", reason)
        self.assertIn("below 20-period average", reason)

    def test_confirm_with_higher_tf_sell_bearish(self):
        # SELL: 4h trend is bearish (Close < EMA at completed candle -2)
        df_1h = pd.DataFrame({'volume': [10.0]*20}) # Volume doesn't affect SELL
        df_4h = pd.DataFrame({
            'close': [100.0] * 18 + [95.0, 100.0],
            'ema': [100.0] * 20
        })
        confirmed, reason = confirm_with_higher_tf(df_1h, df_4h, "SELL")
        self.assertTrue(confirmed, f"Failed: {reason}")
        self.assertIn("Confirmed", reason)

    def test_confirm_with_higher_tf_sell_bullish_rejected(self):
        # SELL: 4h trend is bullish (Close >= EMA and MACD >= Signal at completed candle -2)
        df_1h = pd.DataFrame({'volume': [10.0]*20})
        df_4h = pd.DataFrame({
            'close': [100.0] * 18 + [105.0, 100.0],
            'ema': [100.0] * 20,
            'macd': [0.0] * 18 + [1.5, 0.0],
            'macd_signal': [0.0] * 18 + [1.0, 0.0]
        })
        confirmed, reason = confirm_with_higher_tf(df_1h, df_4h, "SELL")
        self.assertFalse(confirmed, f"Failed: {reason}")
        self.assertIn("4h trend is not bearish", reason)

    # ==========================================
    # 2. KEY ROTATION & RETRY LOGIC TESTS
    # ==========================================
    @patch('ai.sentiment_analyzer.load_api_keys')
    def test_key_rotation_on_429(self, mock_load_keys):
        # Mock keys file to return three fake keys
        mock_load_keys.return_value = ["KEY_A", "KEY_B", "KEY_C"]
        
        # Track active keys used in each attempt
        keys_attempted = []

        def mock_generate(client, *args, **kwargs):
            # Record current active key index
            active_key = sa._api_keys[sa._current_key_idx]
            keys_attempted.append(active_key)
            
            # Fail with 429 for the first two keys, succeed for the third one
            if active_key == "KEY_A":
                raise ValueError("ResourceExhausted: 429 Rate Limit Exceeded")
            elif active_key == "KEY_B":
                raise ValueError("API Error code 429 occurred")
            elif active_key == "KEY_C":
                mock_response = MagicMock()
                mock_response.text = "Success!"
                return mock_response
            raise ValueError("Unexpected key")

        # Run call_gemini_with_retry
        res = sa.call_gemini_with_retry(mock_generate)
        
        # Verify the key sequence attempted
        self.assertEqual(keys_attempted, ["KEY_A", "KEY_B", "KEY_C"])
        self.assertEqual(res.text, "Success!")
        # Current index should remain at the successful key (index 2 / KEY_C)
        self.assertEqual(sa._current_key_idx, 2)

    @patch('ai.sentiment_analyzer.load_api_keys')
    def test_key_rotation_all_fail(self, mock_load_keys):
        # Mock keys file
        mock_load_keys.return_value = ["KEY_A", "KEY_B"]
        
        def mock_generate(client, *args, **kwargs):
            raise ValueError("ResourceExhausted error on " + sa._api_keys[sa._current_key_idx])

        # Since all keys fail, it should raise the last exception
        with self.assertRaises(ValueError) as context:
            sa.call_gemini_with_retry(mock_generate)
            
        self.assertIn("ResourceExhausted error on KEY_", str(context.exception))

    # ==========================================
    # 3. STRUCTURED BATCH SENTIMENT ANALYSIS
    # ==========================================
    @patch('ai.sentiment_analyzer.load_api_keys')
    @patch('ai.sentiment_analyzer.save_ai_run')
    @patch('google.genai.Client')
    def test_analyze_sentiment_batch_success(self, mock_client_class, mock_save_ai_run, mock_load_keys):
        # Mock load_api_keys to enable client creation
        mock_load_keys.return_value = ["TEST_KEY"]
        
        # Mock Gemini Client and generator response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Configure return values for step 1 (summarization) and step 2 (scoring)
        mock_resp_step1 = MagicMock()
        mock_resp_step1.text = json.dumps({
            "digests": [
                {"symbol": "BTC/USDT", "digest": "Positive developments: Bitcoin has institutional inflows. Macro is steady."},
                {"symbol": "ETH/USDT", "digest": "Neutral news: Ethereum is undergoing minor upgrades, gas fees stable."}
            ]
        })
        
        mock_resp_step2 = MagicMock()
        mock_resp_step2.text = json.dumps({
            "results": [
                {"symbol": "BTC/USDT", "sentiment_score": 5, "reason": "Strong institutional inflow signals bullish momentum."},
                {"symbol": "ETH/USDT", "sentiment_score": 0, "reason": "No major positive catalysts, neutral sideways sentiment."}
            ]
        })
        
        # Mock generate_content call sequence:
        # First call is Step 1 (Summarizer)
        # Second call is Step 2 (Sentiment Analyzer)
        mock_client.models.generate_content.side_effect = [mock_resp_step1, mock_resp_step2]
        
        candidates = [
            {"symbol": "BTC/USDT", "news_items": [{"title": "BTC rises", "description": "ETFs buying lots of BTC"}]},
            {"symbol": "ETH/USDT", "news_items": [{"title": "ETH upgrade", "description": "Gas optimization upgrade completed successfully"}]}
        ]
        
        results = analyze_sentiment_batch(candidates)
        
        # Check that two results are returned
        self.assertEqual(len(results), 2)
        
        # Verify first result (BTC/USDT)
        self.assertEqual(results[0]["symbol"], "BTC/USDT")
        self.assertEqual(results[0]["sentiment_score"], 5)
        self.assertEqual(results[0]["reason"], "Strong institutional inflow signals bullish momentum.")
        self.assertEqual(results[0]["digest"], "Positive developments: Bitcoin has institutional inflows. Macro is steady.")
        
        # Verify second result (ETH/USDT)
        self.assertEqual(results[1]["symbol"], "ETH/USDT")
        self.assertEqual(results[1]["sentiment_score"], 0)
        self.assertEqual(results[1]["reason"], "No major positive catalysts, neutral sideways sentiment.")
        self.assertEqual(results[1]["digest"], "Neutral news: Ethereum is undergoing minor upgrades, gas fees stable.")
        
        # Ensure database audit log was called
        self.assertEqual(mock_save_ai_run.call_count, 2)

    @patch('ai.sentiment_analyzer.get_gemini_client')
    @patch('ai.sentiment_analyzer.save_ai_run')
    def test_analyze_sentiment_batch_fallback_on_client_failure(self, mock_save_ai_run, mock_get_client):
        # Force client connection failure to trigger simulation path
        mock_get_client.return_value = None
        
        # Fallback to simulated sentiment should run for BTC and ETH
        candidates = [
            {"symbol": "BTC/USDT", "news_items": [{"title": "Bitcoin surges on ETF inflows", "description": "Major hedge funds disclose spot allocation additions."}]},
            {"symbol": "ETH/USDT", "news_items": [{"title": "Ethereum correction risk", "description": "Slight drop as options expire. Bearish pressure continues."}]}
        ]
        
        results = analyze_sentiment_batch(candidates)
        
        self.assertEqual(len(results), 2)
        
        # Check simulated scoring is returned appropriately
        self.assertEqual(results[0]["symbol"], "BTC/USDT")
        self.assertTrue(results[0]["sentiment_score"] >= 3) # Positive keywords match
        self.assertIn("Simulated", results[0]["reason"])
        
        self.assertEqual(results[1]["symbol"], "ETH/USDT")
        self.assertTrue(results[1]["sentiment_score"] <= -3) # Negative keywords match
        self.assertIn("Simulated", results[1]["reason"])

if __name__ == '__main__':
    unittest.main()
