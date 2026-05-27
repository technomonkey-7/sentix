import os
import json
import random
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from core.db import log_event, get_config, save_ai_run
from dotenv import load_dotenv

# Load env variables on startup
load_dotenv()

# Structured output Pydantic schemas
class DigestResult(BaseModel):
    symbol: str = Field(description="The trading symbol, e.g., BTC/USDT")
    digest: str = Field(description="A detailed, comprehensive multi-paragraph news digest highlighting key developments, specific news details, regulatory impacts, and macroeconomic factors.")

class BatchDigestResponse(BaseModel):
    digests: List[DigestResult]

class SentimentResult(BaseModel):
    symbol: str = Field(description="The trading symbol, e.g., BTC/USDT")
    sentiment_score: int = Field(description="Sentiment score from -10 (extremely bearish, panic, crash imminent) to 10 (extremely bullish, FOMO, massive breakout)")
    reason: str = Field(description="A single-sentence explanation for the score based strictly on the news digest")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]


# Global API Keys cache and rotation state
_api_keys = None
_current_key_idx = 0

def load_api_keys():
    """
    Loads Gemini API keys from gemini_keys.txt in the project root.
    Returns a list of keys.
    """
    keys = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys_file = os.path.join(project_root, "gemini_keys.txt")
    if os.path.exists(keys_file):
        try:
            with open(keys_file, "r", encoding="utf-8") as f:
                for line in f:
                    k = line.strip()
                    if k and not k.startswith("#"):
                        keys.append(k)
        except Exception as e:
            log_event("ERROR", "AI_MODULE", f"Failed to read gemini_keys.txt: {e}")
    return keys

def get_gemini_client(api_key=None):
    """
    Sets up and returns a Google GenAI client instance.
    Fetches the API key from database config, env vars, or keys pool.
    """
    global _api_keys, _current_key_idx
    if api_key and api_key.strip():
        return genai.Client(api_key=api_key.strip())

    # Check key pool
    if _api_keys is None:
        _api_keys = load_api_keys()

    if _api_keys:
        key = _api_keys[_current_key_idx]
        return genai.Client(api_key=key)

    # Fallback to single config or env
    key = get_config("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() == "":
        return None
    
    try:
        return genai.Client(api_key=key.strip())
    except Exception as e:
        log_event("ERROR", "AI_MODULE", f"Failed to configure Gemini client: {e}")
        return None

def get_current_key_name():
    """Returns the descriptive name of the currently active key."""
    global _api_keys, _current_key_idx
    if _api_keys:
        return f"pool_key_{_current_key_idx + 1}"
    return "default_config"

def rotate_key():
    """
    Rotates the active key in the key pool.
    Returns True if rotation happened, False otherwise.
    """
    global _api_keys, _current_key_idx
    if _api_keys is None:
        _api_keys = load_api_keys()

    if _api_keys and len(_api_keys) > 1:
        _current_key_idx = (_current_key_idx + 1) % len(_api_keys)
        log_event("WARNING", "AI_MODULE", f"Rotated to Gemini API key index {_current_key_idx + 1}/{len(_api_keys)}")
        return True
    return False

def call_gemini_with_retry(func, *args, **kwargs):
    """
    Calls a function with a Gemini client. Retries with key rotation if 429 occurs.
    """
    global _api_keys
    if _api_keys is None:
        _api_keys = load_api_keys()

    num_keys = len(_api_keys) if _api_keys else 1
    last_error = None

    for attempt in range(num_keys + 1):
        client = get_gemini_client()
        key_name = get_current_key_name()
        if client is None:
            raise ValueError("No Gemini API client could be configured.")

        try:
            return func(client, *args, **kwargs)
        except Exception as e:
            is_rate_limit = False
            if isinstance(e, APIError):
                if e.code == 429 or "429" in str(e) or "ResourceExhausted" in str(e):
                    is_rate_limit = True
            elif "429" in str(e) or "ResourceExhausted" in str(e):
                is_rate_limit = True

            if is_rate_limit:
                log_event("WARNING", "AI_MODULE", f"Rate limit (429) hit on Gemini client '{key_name}'.")
                rotated = rotate_key()
                if not rotated:
                    # Only one key is available. Wait and retry once.
                    if attempt == 0:
                        log_event("INFO", "AI_MODULE", "Only one API key configured. Waiting 10 seconds before retry...")
                        time.sleep(10)
                        continue
                    else:
                        raise e
                else:
                    # Key rotated. Wait a brief duration before retrying.
                    log_event("INFO", "AI_MODULE", "Sleeping 2 seconds before retrying with rotated API key...")
                    time.sleep(2)
                last_error = e
            else:
                log_event("ERROR", "AI_MODULE", f"Gemini API call failed with error: {e}")
                raise e
    if last_error:
        raise last_error
    raise RuntimeError("Gemini API call retry loop ended without result.")

def analyze_sentiment_batch(candidates: List[Dict[str, Any]], sentiment_model_override=None) -> List[Dict[str, Any]]:
    """
    Analyzes news sentiment for a batch of candidate symbols in a single Gemini call.
    Each candidate is a dict: {"symbol": str, "news_items": list}
    
    Uses structured output JSON schemas to ensure reliable parsing.
    """
    if not candidates:
        return []

    live_mode = (get_config("live_mode") or os.getenv("LIVE_MODE", "false")).lower() == "true"

    try:
        test_client = get_gemini_client()
    except Exception as e:
        log_event("ERROR", "AI_MODULE", f"Failed to get Gemini client: {e}")
        if live_mode:
            raise RuntimeError(f"LIVE MODE: Failed to configure Gemini client: {e}")
        test_client = None

    if test_client is None:
        if live_mode:
            log_event("ERROR", "AI_MODULE", "Gemini client is not configured in LIVE MODE. Refusing simulation fallback.")
            raise RuntimeError("LIVE MODE: No Gemini client configured. Gemini API key might be missing or invalid.")
        log_event("WARNING", "AI_MODULE", "No Gemini client configured. Running simulated fallbacks for all candidates.")
        results = []
        for c in candidates:
            sim = run_simulated_sentiment(c["symbol"], c.get("news_items"))
            results.append({
                "symbol": c["symbol"],
                "sentiment_score": sim["sentiment_score"],
                "reason": sim["reason"],
                "digest": sim["digest"],
                "is_simulated": True
            })
        return results

    summarizer_model = get_config("summarizer_model") or os.getenv("SUMMARIZER_MODEL", "gemini-3.1-flash-lite")
    sentiment_model = sentiment_model_override or get_config("sentiment_model") or os.getenv("SENTIMENT_MODEL", "gemini-3.5-flash")

    # Filter out candidates with empty news
    valid_candidates = []
    results = []
    for c in candidates:
        symbol = c["symbol"]
        news_items = c.get("news_items")
        if not news_items or len(news_items) == 0:
            log_event("WARNING", "AI_MODULE", f"No news articles provided to analyze for {symbol}.")
            results.append({
                "symbol": symbol,
                "sentiment_score": 0,
                "reason": "No recent news articles were found to analyze.",
                "digest": "No recent news articles were found to analyze.",
                "is_simulated": True
            })
        else:
            valid_candidates.append(c)

    if not valid_candidates:
        return results

    # Step 1: Batch News Summarization
    log_event("INFO", "AI_MODULE", f"Step 1: Batch summarizing news for {len(valid_candidates)} symbols using '{summarizer_model}'...")
    
    news_blocks = []
    for c in valid_candidates:
        symbol = c["symbol"]
        news_text = "\n\n".join([
            f"Title: {item['title']}\nSnippet: {item['description']}"
            for item in c["news_items"]
        ])
        news_blocks.append(f"Asset Symbol: {symbol}\nRaw news:\n{news_text}")
        
    combined_news_prompt = f"""
    You are a highly detailed and precise quantitative research assistant for a trading platform. 
    Your task is to analyze, clean, filter out noise or spam, and synthesize a detailed, comprehensive multi-paragraph news digest for each cryptocurrency asset.
    
    CRITICAL INSTRUCTIONS:
    - Do NOT make the digest too short, brief, or generic. Other quantitative models rely on the richness of this digest to perform sentiment analysis.
    - Provide a thorough coverage of major positive developments, technological updates, institutional inflows, regulatory changes, and macroeconomic factors.
    - Include specific facts, percentages, named entities, and dates where available in the raw articles.
    - Ensure both positive opportunities and potential downside risks/worries are clearly articulated in detail.
    
    Here is the raw news by symbol:
    
    {"\n\n===Next Asset Symbol===\n\n".join(news_blocks)}
    """
    
    digest_map = {}
    try:
        def do_summarize(c_client):
            return c_client.models.generate_content(
                model=summarizer_model,
                contents=combined_news_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchDigestResponse
                )
            )
            
        summary_resp = call_gemini_with_retry(do_summarize)
        digest_data = json.loads(summary_resp.text.strip())
        
        for d in digest_data.get("digests", []):
            digest_map[d["symbol"]] = d["digest"]
            
        log_event("SUCCESS", "AI_MODULE", "Step 1 Complete. News summarized successfully.")
    except Exception as e:
        if live_mode:
            log_event("ERROR", "AI_MODULE", f"Batch summarization failed in LIVE MODE: {e}. Refusing simulation/placeholder fallback.")
            raise RuntimeError(f"LIVE MODE: Batch summarization failed: {e}")
        log_event("ERROR", "AI_MODULE", f"Batch summarization failed: {e}. Falling back to individual placeholder digests.")
        for c in valid_candidates:
            digest_map[c["symbol"]] = f"Summarization failed: {e}"

    # Step 2: Batch Sentiment Scoring with a Skeptical Analyst persona
    log_event("INFO", "AI_MODULE", f"Step 2: Performing Batch Sentiment evaluation using '{sentiment_model}'...")
    
    digest_blocks = []
    for symbol, digest in digest_map.items():
        digest_blocks.append(f"Asset Symbol: {symbol}\nDigest:\n{digest}")
        
    skeptical_prompt = f"""
    You are a highly skeptical quantitative financial analyst. 
    Analyze the following synthesized cryptocurrency news digests and evaluate the near-term market sentiment for each asset.
    
    IMPORTANT CRITERIA FOR YOUR ANALYSIS:
    - Actively look for risks, regulatory headwinds, macro factors, and potential false breakouts.
    - Ignore speculative retail hype, promotional marketing releases, and sensationalist clickbait titles.
    - Demand solid, verified fundamentals.
    - If the news for an asset is weak, vague, ambiguous, or lacks solid backing, bias your score toward 0 (Neutral).
    - Only assign strong positive scores (>= 3) or strong negative scores (<= -3) if there is clear, verified fundamental evidence.
    
    Here are the digests by symbol:
    
    {"\n\n===Next Asset Symbol===\n\n".join(digest_blocks)}
    """
    
    try:
        def do_scoring(c_client):
            return c_client.models.generate_content(
                model=sentiment_model,
                contents=skeptical_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchSentimentResponse
                )
            )
            
        score_resp = call_gemini_with_retry(do_scoring)
        score_data = json.loads(score_resp.text.strip())
        
        # Match scores with digests and append
        received_symbols = set()
        for res in score_data.get("results", []):
            symbol = res["symbol"]
            score = max(-10, min(10, int(res["sentiment_score"])))
            reason = res["reason"]
            digest = digest_map.get(symbol, "")
            
            # Save run to SQLite audit log
            save_ai_run(symbol, digest, score, reason)
            
            # Check if this candidate's news was simulated
            news_items = next((c.get("news_items", []) for c in valid_candidates if c["symbol"] == symbol), [])
            has_simulated_news = any(item.get("_is_simulated", False) for item in news_items)
            
            results.append({
                "symbol": symbol,
                "sentiment_score": score,
                "reason": reason,
                "digest": digest,
                "is_simulated": has_simulated_news
            })
            received_symbols.add(symbol)
            
        # Handle cases where some symbols were missed by the model output
        for c in valid_candidates:
            symbol = c["symbol"]
            if symbol not in received_symbols:
                news_items = c.get("news_items", [])
                has_simulated_news = any(item.get("_is_simulated", False) for item in news_items)
                results.append({
                    "symbol": symbol,
                    "sentiment_score": 0,
                    "reason": "Model response did not return results for this asset. Defaulted to neutral.",
                    "digest": digest_map.get(symbol, ""),
                    "is_simulated": has_simulated_news
                })
                
        log_event("SUCCESS", "AI_MODULE", "Step 2 Complete. Batch sentiment evaluated.")
        return results
        
    except Exception as e:
        if live_mode:
            log_event("ERROR", "AI_MODULE", f"Batch sentiment scoring failed in LIVE MODE: {e}. Refusing simulation fallback.")
            raise RuntimeError(f"LIVE MODE: Batch sentiment scoring failed: {e}")
        log_event("ERROR", "AI_MODULE", f"Batch sentiment scoring failed: {e}. Falling back to simulation.")
        for c in valid_candidates:
            symbol = c["symbol"]
            sim = run_simulated_sentiment(symbol, c["news_items"])
            results.append({
                "symbol": symbol,
                "sentiment_score": sim["sentiment_score"],
                "reason": sim["reason"],
                "digest": sim["digest"],
                "is_simulated": True
            })
        return results

def analyze_sentiment(symbol="BTC/USDT", news_items=None, sentiment_model_override=None):
    """
    Main entry point for Gemini-based sentiment analysis.
    Refactored to call the structured batch logic under the hood to ensure consistent optimization.
    """
    results = analyze_sentiment_batch([{"symbol": symbol, "news_items": news_items}], sentiment_model_override)
    if results:
        return results[0]
    return {"sentiment_score": 0, "reason": "Sentiment analysis failed.", "digest": "", "is_simulated": True}

def run_simulated_sentiment(symbol="BTC/USDT", news_items=None):
    """
    Intelligent simulated fallback that scans keywords in the articles to 
    produce an appropriate score and single-sentence explanation.
    Allows complete app preview and offline verification.
    """
    if not news_items:
        return {"sentiment_score": 0, "reason": "No news articles to simulate.", "digest": "No news articles to simulate."}
        
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
