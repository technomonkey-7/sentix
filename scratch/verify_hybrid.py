import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock core db functions to avoid database pollution
import core.db as db

# Save original functions
orig_record_trade = db.record_trade
orig_close_active_position = db.close_active_position
orig_update_portfolio = db.update_portfolio
orig_get_portfolio = db.get_portfolio

# Simple mocks
mock_trades_recorded = []
mock_positions_closed = []
mock_portfolio = {
    "USD": {"balance": 10000.0, "avg_entry_price": 0.0},
    "BTC": {"balance": 0.0, "avg_entry_price": 0.0}
}

def mock_get_portfolio():
    return mock_portfolio

def mock_update_portfolio(portfolio):
    global mock_portfolio
    mock_portfolio = portfolio

def mock_record_trade(asset, side, price, amount, trade_type, sentiment_score=None, reason=None, stop_loss=None, take_profit=None, is_active=0):
    mock_trades_recorded.append({
        "asset": asset,
        "side": side,
        "price": price,
        "amount": amount,
        "trade_type": trade_type,
        "sentiment_score": sentiment_score,
        "reason": reason,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "is_active": is_active
    })

def mock_close_active_position(asset, sell_price, trade_type, reason=None):
    mock_positions_closed.append({
        "asset": asset,
        "sell_price": sell_price,
        "trade_type": trade_type,
        "reason": reason
    })
    return 5.0 # Mock PnL %

db.get_portfolio = mock_get_portfolio
db.update_portfolio = mock_update_portfolio
db.record_trade = mock_record_trade
db.close_active_position = mock_close_active_position

# Mocks for news and sentiment
import core.data_fetcher as df
import ai.sentiment_analyzer as sa

def run_test():
    global mock_portfolio
    print("=== STARTING SENTIX HYBRID STRATEGY LOGIC TESTS ===")
    
    # Test Parameters
    risk_pct = 2.0
    sl_pct = 3.0
    tp_pct = 6.0
    sentiment_threshold = 3
    
    # We will simulate the execution logic from worker.py for different candidates
    
    def simulate_worker_decision(candidate, fetched_news, sa_result_batch, fetch_exception=False, sa_exception=False):
        global mock_trades_recorded, mock_positions_closed, mock_portfolio
        mock_trades_recorded = []
        mock_positions_closed = []
        
        asset = candidate["asset"]
        trigger_side = candidate["trigger_side"]
        trigger_reason = candidate["trigger_reason"]
        completed_timestamp = candidate["completed_timestamp"]
        current_price = candidate["current_price"]
        asset_ticker = candidate["asset_ticker"]
        active_pos = candidate["active_pos"]
        
        # 1. Scrape news logic with failsafes
        batch_input = []
        local_sentiment_overrides = {}
        
        # Simulate fetch_asset_news
        if fetch_exception:
            # Emulate exception
            news = None
            print(f"[TEST LOG] Exception fetching news for {asset}: Mocked error")
        else:
            news = fetched_news
            
        if not news:
            print(f"[TEST LOG] No live news fetched for {asset}. Failsafe: Proceeding with neutral sentiment (0) without AI call.")
            local_sentiment_overrides[asset] = {
                "symbol": asset,
                "sentiment_score": 0,
                "reason": "Failsafe: No news articles could be fetched. Defaulted to neutral sentiment.",
                "digest": "No news articles could be fetched.",
                "is_simulated": False
            }
        else:
            batch_input.append({
                "symbol": asset,
                "news_items": news
            })
            
        # 2. Call batch sentiment logic
        ai_results = []
        if batch_input:
            if sa_exception:
                print(f"[TEST LOG] Batch sentiment analysis failed: Mocked AI error. Failsafe: Defaulting to 0.")
                for item in batch_input:
                    local_sentiment_overrides[item["symbol"]] = {
                        "symbol": item["symbol"],
                        "sentiment_score": 0,
                        "reason": "Failsafe: AI model call failed (Mocked AI error). Defaulted to neutral sentiment.",
                        "digest": "AI analysis error fallback.",
                        "is_simulated": False
                    }
            else:
                ai_results = sa_result_batch
                
        # Combine
        ai_map = {res["symbol"]: res for res in ai_results}
        for sym, res in local_sentiment_overrides.items():
            ai_map[sym] = res
            
        # 3. Trade Execution Logic
        portfolio = db.get_portfolio()
        total_nav = 10000.0 # Mock NAV
        
        usd_balance = portfolio.get("USD", {}).get("balance", 0.0)
        asset_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
        
        ai_res = ai_map.get(asset)
        if not ai_res:
            print(f"[TEST LOG] No AI sentiment results returned for {asset}. Skipping.")
            return
            
        sentiment_score = ai_res["sentiment_score"]
        ai_reason = ai_res["reason"]
        
        print(f"[TEST LOG] Gemini Sentiment outcome for {asset}: {sentiment_score} (Reason: '{ai_reason}')")
        
        if trigger_side == "BUY":
            if active_pos or usd_balance < 10.0:
                print(f"[TEST LOG] Skipping BUY: active_pos={active_pos}, usd_balance={usd_balance}")
                return
                
            # VETO check
            if sentiment_score <= -2:
                print(f"[TEST LOG] [VETO] Technical BUY filtered out by Gemini AI. Sentiment is bearish ({sentiment_score}). Veto triggered. Reason: {ai_reason}")
                return
                
            # Dynamic Sizing
            if sentiment_score >= sentiment_threshold:
                applied_risk_pct = risk_pct
                sizing_reason = f"Full risk size applied due to positive AI sentiment ({sentiment_score} >= threshold {sentiment_threshold})."
            else:
                applied_risk_pct = risk_pct / 2.0
                sizing_reason = f"Half risk size applied due to neutral AI sentiment ({sentiment_score} < threshold {sentiment_threshold})."
                
            trade_value = total_nav * (applied_risk_pct / 100.0)
            trade_value = min(trade_value, usd_balance)
            
            print(f"[TEST LOG] Sizing Outcome: applied_risk_pct={applied_risk_pct}%, trade_value=${trade_value:.2f} ({sizing_reason})")
            
            fee_pct = 0.001
            trade_net_value = trade_value * (1 - fee_pct)
            amount_to_buy = trade_net_value / current_price
            
            portfolio["USD"]["balance"] = usd_balance - trade_value
            portfolio[asset_ticker] = {
                "balance": amount_to_buy,
                "avg_entry_price": current_price
            }
            db.update_portfolio(portfolio)
            db.record_trade(
                asset=asset,
                side="BUY",
                price=current_price,
                amount=amount_to_buy,
                trade_type="AI_HYBRID",
                sentiment_score=sentiment_score,
                reason=f"Technical: {trigger_reason} | AI: {ai_reason} | Sizing: {sizing_reason}",
                stop_loss=current_price * (1 - sl_pct / 100.0),
                take_profit=current_price * (1 + tp_pct / 100.0),
                is_active=1
            )
            
        elif trigger_side == "SELL":
            if asset_balance <= 0.0001:
                print(f"[TEST LOG] Skipping SELL: asset_balance={asset_balance}")
                return
                
            # VETO check
            if sentiment_score >= 3:
                print(f"[TEST LOG] [VETO] Technical SELL filtered out by Gemini AI. Extremely bullish sentiment ({sentiment_score}). Veto triggered. Reason: {ai_reason}")
                return
                
            amount_to_sell = asset_balance
            trade_value = amount_to_sell * current_price
            fee_pct = 0.001
            trade_net_value = trade_value * (1 - fee_pct)
            
            portfolio["USD"]["balance"] = usd_balance + trade_net_value
            portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
            db.update_portfolio(portfolio)
            
            if active_pos:
                db.close_active_position(asset, current_price, "AI_HYBRID", f"Bearish crossover executed. AI sentiment: {sentiment_score}")
            else:
                db.record_trade(
                    asset=asset,
                    side="SELL",
                    price=current_price,
                    amount=amount_to_sell,
                    trade_type="AI_HYBRID",
                    sentiment_score=sentiment_score,
                    reason=f"Technical: {trigger_reason} | AI sentiment: {sentiment_score}",
                    is_active=0
                )

    # --- TEST CASES ---
    
    # Reset balance
    mock_portfolio = {
        "USD": {"balance": 10000.0, "avg_entry_price": 0.0},
        "BTC": {"balance": 0.0, "avg_entry_price": 0.0}
    }
    
    print("\n--- CASE 1: BUY Crossover, No News Scraped (Failsafe) ---")
    candidate_1 = {
        "asset": "BTC/USDT",
        "trigger_side": "BUY",
        "trigger_reason": "EMA Crossover (Price broke above 20 EMA)",
        "completed_timestamp": "2026-05-27 20:00:00",
        "current_price": 60000.0,
        "asset_ticker": "BTC",
        "active_pos": None
    }
    simulate_worker_decision(candidate_1, fetched_news=None, sa_result_batch=[])
    assert len(mock_trades_recorded) == 1, "Should have executed buy"
    assert mock_trades_recorded[0]["trade_type"] == "AI_HYBRID"
    assert "Half risk size" in mock_trades_recorded[0]["reason"], "Should apply half risk for neutral news"
    print("Result: PASS (Successfully executed with half risk size)")
    
    print("\n--- CASE 2: BUY Crossover, Extremely Bullish News (Full Size) ---")
    mock_portfolio = {
        "USD": {"balance": 10000.0, "avg_entry_price": 0.0},
        "BTC": {"balance": 0.0, "avg_entry_price": 0.0}
    }
    simulate_worker_decision(
        candidate_1, 
        fetched_news=[{"title": "BTC breakout", "description": "breakout positive"}],
        sa_result_batch=[{"symbol": "BTC/USDT", "sentiment_score": 5, "reason": "Extremely positive breakouts"}]
    )
    assert len(mock_trades_recorded) == 1
    assert "Full risk size" in mock_trades_recorded[0]["reason"], "Should apply full risk for score >= 3"
    print("Result: PASS (Successfully executed with full risk size)")
    
    print("\n--- CASE 3: BUY Crossover, Bearish News (AI Veto) ---")
    mock_portfolio = {
        "USD": {"balance": 10000.0, "avg_entry_price": 0.0},
        "BTC": {"balance": 0.0, "avg_entry_price": 0.0}
    }
    simulate_worker_decision(
        candidate_1, 
        fetched_news=[{"title": "BTC hack", "description": "systemic risk negative"}],
        sa_result_batch=[{"symbol": "BTC/USDT", "sentiment_score": -4, "reason": "Security hack concerns"}]
    )
    assert len(mock_trades_recorded) == 0, "Bearish news should veto BUY"
    print("Result: PASS (Veto successfully prevented trade)")
    
    print("\n--- CASE 4: SELL Crossover, Bearish News (Immediate Exit) ---")
    mock_portfolio = {
        "USD": {"balance": 0.0, "avg_entry_price": 0.0},
        "BTC": {"balance": 0.1, "avg_entry_price": 60000.0}
    }
    candidate_2 = {
        "asset": "BTC/USDT",
        "trigger_side": "SELL",
        "trigger_reason": "EMA Crossover (Price fell below 20 EMA)",
        "completed_timestamp": "2026-05-27 21:00:00",
        "current_price": 61000.0,
        "asset_ticker": "BTC",
        "active_pos": {"id": 42, "price": 60000.0, "amount": 0.1}
    }
    simulate_worker_decision(
        candidate_2,
        fetched_news=[{"title": "BTC drop", "description": "negative correction"}],
        sa_result_batch=[{"symbol": "BTC/USDT", "sentiment_score": -3, "reason": "Bearish correction"}]
    )
    assert len(mock_positions_closed) == 1, "Should exit position"
    print("Result: PASS (Position successfully closed)")
    
    print("\n--- CASE 5: SELL Crossover, Extremely Bullish News (Vetoed Exit) ---")
    mock_portfolio = {
        "USD": {"balance": 0.0, "avg_entry_price": 0.0},
        "BTC": {"balance": 0.1, "avg_entry_price": 60000.0}
    }
    simulate_worker_decision(
        candidate_2,
        fetched_news=[{"title": "BTC institutional support", "description": "huge bullish news"}],
        sa_result_batch=[{"symbol": "BTC/USDT", "sentiment_score": 4, "reason": "Institutional breakout"}]
    )
    assert len(mock_positions_closed) == 0, "Should veto sell due to extremely bullish news"
    print("Result: PASS (Veto successfully prevented exit on highly bullish news)")
    
    print("\n=== ALL TEST CASES COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_test()
