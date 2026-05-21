import os
import json
import random
from google import genai
from google.genai import types
from core.db import log_event, get_config, save_ai_run
from dotenv import load_dotenv

# Load env variables on startup
load_dotenv()

def get_gemini_client(api_key=None):
    """
    Sets up and returns a Google GenAI client instance.
    Fetches the API key from database config, env vars, or returns None if missing.
    """
    # 1. Check passed key, 2. Check DB config, 3. Check environment variable
    key = api_key or get_config("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() == "":
        return None
    
    try:
        client = genai.Client(api_key=key.strip())
        return client
    except Exception as e:
        log_event("ERROR", "AI_MODULE", f"Failed to configure Gemini client: {e}")
        return None

def analyze_sentiment(symbol="BTC/USDT", news_items=None):
    """
    Main entry point for Gemini-based sentiment analysis.
    Uses a hybrid two-step model approach to optimize costs and output high-quality JSON.
    """
    if not news_items or len(news_items) == 0:
        log_event("WARNING", "AI_MODULE", f"No news articles provided to analyze for {symbol}.")
        return {"sentiment_score": 0, "reason": "No recent news articles were found to analyze."}

    # Fetch models from config or env, with fallback defaults
    summarizer_model = get_config("summarizer_model") or os.getenv("SUMMARIZER_MODEL", "gemini-3.1-flash-lite")
    sentiment_model = get_config("sentiment_model") or os.getenv("SENTIMENT_MODEL", "gemini-3.5-flash")

    client = get_gemini_client()
    
    # If no API client is available, run simulation mode
    if client is None:
        log_event("WARNING", "AI_MODULE", f"No Gemini API key configured. Executing simulated sentiment engine for {symbol}...")
        return run_simulated_sentiment(symbol, news_items)

    try:
        # ---- STEP 1: Summarize and clean news using Flash model ----
        log_event("INFO", "AI_MODULE", f"Step 1: Summarizing {len(news_items)} news articles for {symbol} using '{summarizer_model}'...")
        
        news_text = "\n\n".join([
            f"Title: {item['title']}\nSnippet: {item['description']}"
            for item in news_items
        ])
        
        step1_prompt = f"""
        You are a highly efficient assistant for a quantitative trading platform. 
        Your task is to analyze, clean, filter out noise or spam, and synthesize a concise 2-paragraph news digest for the cryptocurrency {symbol.split('/')[0]}.
        Highlight the major positive developments, regulatory issues, and macroeconomic trends contained in the articles.
        
        Raw News Articles:
        {news_text}
        
        Concise Digest:
        """
        
        flash_response = client.models.generate_content(
            model=summarizer_model,
            contents=step1_prompt
        )
        digest = flash_response.text.strip()
        
        log_event("SUCCESS", "AI_MODULE", f"Step 1 Complete. News summarized successfully.")

        # ---- STEP 2: Structural JSON sentiment evaluation using Pro model (with Flash fallback) ----
        log_event("INFO", "AI_MODULE", f"Step 2: Performing JSON Sentiment evaluation using '{sentiment_model}'...")
        
        step2_prompt = f"""
        You are a seasoned quantitative financial analyst and sentiment analysis engine.
        Analyze the following synthesized cryptocurrency news digest and evaluate the near-term market sentiment for {symbol.split('/')[0]}.
        
        You MUST strictly return a JSON object with two fields:
        1. 'sentiment_score': an integer between -10 (extremely bearish, panic, crash imminent) and 10 (extremely bullish, FOMO, massive breakout).
        2. 'reason': a crisp, single-sentence explanation of why you gave this score based only on the digest.
        
        Digest:
        {digest}
        
        JSON response:
        """

        # Set up generation config for strict JSON output
        json_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

        # Attempt to run using the larger model (e.g. gemini-2.5-pro)
        try:
            pro_response = client.models.generate_content(
                model=sentiment_model,
                contents=step2_prompt,
                config=json_config
            )
            result_text = pro_response.text.strip()
        except Exception as pro_err:
            log_event("WARNING", "AI_MODULE", f"Failed calling '{sentiment_model}': {pro_err}. Falling back to '{summarizer_model}' for JSON mode.")
            # Fallback to Flash model
            pro_response = client.models.generate_content(
                model=summarizer_model,
                contents=step2_prompt,
                config=json_config
            )
            result_text = pro_response.text.strip()

        # Parse output safely
        sentiment_data = json.loads(result_text)
        
        # Verify schema integrity
        if "sentiment_score" not in sentiment_data or "reason" not in sentiment_data:
            raise KeyError("JSON response missing required keys.")
            
        sentiment_score = int(sentiment_data["sentiment_score"])
        # Clamp score between -10 and 10
        sentiment_score = max(-10, min(10, sentiment_score))
        reason = sentiment_data["reason"]
        
        log_event("SUCCESS", "AI_MODULE", f"Step 2 Complete. Sentiment Score: {sentiment_score}. Reason: '{reason}'")
        
        # Save run to SQLite audit log
        save_ai_run(symbol, digest, sentiment_score, reason)
        
        return {
            "sentiment_score": sentiment_score,
            "reason": reason,
            "digest": digest
        }

    except Exception as e:
        log_event("ERROR", "AI_MODULE", f"Gemini API execution error: {e}. Falling back to heuristic sentiment analyzer.")
        return run_simulated_sentiment(symbol, news_items)

def run_simulated_sentiment(symbol="BTC/USDT", news_items=None):
    """
    Intelligent simulated fallback that scans keywords in the articles to 
    produce an appropriate score and single-sentence explanation.
    Allows complete app preview and offline verification.
    """
    if not news_items:
        return {"sentiment_score": 0, "reason": "No news articles to simulate."}
        
    score = 0
    positive_words = ["breakout", "rally", "upgrade", "inflow", "institutional", "gain", "growth", "support", "bullish", "scale"]
    negative_words = ["correction", "concern", "liquidations", "regulatory", "headwinds", "cautious", "drop", "bearish", "crash", "fall"]
    
    pos_count = 0
    neg_count = 0
    
    for item in news_items:
        text = (item["title"] + " " + item["description"]).lower()
        for pw in positive_words:
            if pw in text:
                pos_count += 1
        for nw in negative_words:
            if nw in text:
                neg_count += 1
                
    if pos_count > neg_count:
        score = random.randint(3, 7)
        reason = f"Simulated sentiment evaluates to positive bullish drift (+{score}) due to institutional chatter and breakout potential."
    elif neg_count > pos_count:
        score = random.randint(-7, -3)
        reason = f"Simulated sentiment evaluates to negative bearish drift ({score}) following short-term correction risks and liquidation concerns."
    else:
        score = random.randint(-2, 2)
        reason = f"Simulated sentiment evaluates to neutral/indecisive ({score}) as positive tech upgrades are offset by macroeconomic headwinds."
        
    digest = "Simulated Digest:\n" + "\n".join([f"- {i['title']}" for i in news_items[:3]])
    
    # Save simulation to SQLite audit log
    save_ai_run(symbol, digest, score, reason)
    
    return {
        "sentiment_score": score,
        "reason": reason,
        "digest": digest
    }

if __name__ == "__main__":
    # Test execution
    mock_news = [
        {"title": "Bitcoin surges over $70k on institutional support", "description": "ETFs show record daily inflows as major financial groups increase spot allocation."},
        {"title": "Concerns over macroeconomic headwinds grow", "description": "Markets consolidate as traders await global rate announcements."}
    ]
    res = analyze_sentiment("BTC/USDT", mock_news)
    print(res)
