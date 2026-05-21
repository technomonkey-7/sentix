from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Dynamic project root resolver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db import (
    get_portfolio, get_all_active_positions, get_logs, 
    get_latest_ai_run, get_config, save_config, get_trades, get_latest_candles
)
from worker import run_worker_cycle

app = FastAPI(title="Sentix Premium API")

# Setup CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfigUpdate(BaseModel):
    key: str
    value: str

@app.get("/api/status")
def get_status():
    return {"status": "online", "version": "1.0.0"}

@app.get("/api/portfolio")
def get_portfolio_api():
    port = get_portfolio()
    # Calculate free USD
    usd_data = port.get('USD', {'balance': 0.0})
    free_usd = usd_data['balance']
    return {"portfolio": port, "free_usd": free_usd}

@app.get("/api/positions")
def get_positions_api():
    positions = get_all_active_positions()
    return {"positions": positions, "count": len(positions)}

@app.get("/api/logs")
def get_logs_api(limit: int = 100):
    return {"logs": get_logs(limit)}

@app.get("/api/trades")
def get_trades_api(limit: int = 50):
    return {"trades": get_trades(limit)}

@app.get("/api/config/{key}")
def get_config_api(key: str):
    return {"key": key, "value": get_config(key)}

@app.get("/api/config")
def get_all_config():
    # Helper to get all basic configs needed for frontend display
    keys = ["risk_percentage", "stop_loss_pct", "take_profit_pct", "min_ai_sentiment_threshold", "selected_assets", "summarizer_model", "sentiment_model"]
    res = {}
    for k in keys:
        res[k] = get_config(k)
    return res

@app.post("/api/config")
def update_config_api(config: ConfigUpdate):
    save_config(config.key, config.value)
    return {"success": True}

@app.post("/api/trigger_analysis")
def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_worker_cycle, force=True)
    return {"success": True, "message": "Analysis cycle triggered in background"}

@app.get("/api/chart/{asset}/{interval}")
def get_chart_data(asset: str, interval: str, limit: int = 100):
    # format asset for DB like BTC-USDT -> BTC/USDT
    asset_formatted = asset.replace("-", "/")
    return {"candles": get_latest_candles(asset_formatted, interval, limit)}

@app.get("/api/ai_run/{asset}")
def get_ai_run(asset: str):
    asset_formatted = asset.replace("-", "/")
    run = get_latest_ai_run(asset_formatted)
    if not run:
        raise HTTPException(status_code=404, detail="No AI run found")
    return {"ai_run": run}
