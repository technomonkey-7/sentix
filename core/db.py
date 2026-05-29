import sqlite3
import os
import json
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.db")

def get_connection():
    """Returns a SQLite database connection with custom configurations."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent reads while writing — critical for worker + UI coexistence
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def init_db():
    """Initializes the SQLite database with all production tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Candles Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candles (
        timestamp TEXT,
        asset TEXT,
        interval TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        rsi REAL,
        macd REAL,
        macd_signal REAL,
        macd_hist REAL,
        ema REAL,
        PRIMARY KEY (timestamp, asset, interval)
    )
    """)

    # Trades Table (Paper Trading Orders)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        asset TEXT,
        side TEXT,          -- BUY / SELL
        price REAL,
        amount REAL,
        trade_type TEXT,    -- TECHNICAL / AI_CONFIRMED
        sentiment_score INTEGER,
        reason TEXT,
        stop_loss REAL,
        take_profit REAL,
        pnl REAL,
        is_active INTEGER DEFAULT 0
    )
    """)

    # Migration for existing database schemas: check if columns exist, if not add them
    cursor.execute("PRAGMA table_info(trades)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    migration_cols = {
        "stop_loss": "REAL",
        "take_profit": "REAL",
        "pnl": "REAL",
        "is_active": "INTEGER DEFAULT 0"
    }
    for col_name, col_def in migration_cols.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")

    # Portfolio Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        asset TEXT PRIMARY KEY, -- 'USD' or asset tickers like 'BTC', 'ETH', 'SOL'
        balance REAL DEFAULT 0.0,
        avg_entry_price REAL DEFAULT 0.0
    )
    """)

    # AI Runs Table (Audit Log of Gemini calls)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        asset TEXT,
        news_digest TEXT,
        sentiment_score INTEGER,
        reason TEXT
    )
    """)

    # Logs Table (Live Logging visible in Streamlit)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        level TEXT,         -- INFO, WARNING, ERROR, SUCCESS
        module TEXT,        -- WORKER, UI, AI, MATH
        message TEXT
    )
    """)

    # Config Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    # Initialize default portfolio balance ($10,000 USD)
    cursor.execute("SELECT balance FROM portfolio WHERE asset = 'USD'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO portfolio (asset, balance, avg_entry_price) VALUES ('USD', 10000.0, 0.0)")

    conn.commit()
    conn.close()

# ----------------- DB Operations Helpers -----------------

def log_event(level, module, message):
    """Logs an event into the SQLite logs table and prints it to stdout."""
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        print(f"[{timestamp}] [{level}] [{module}] {message}", flush=True)
    except UnicodeEncodeError:
        safe_msg = message.encode('ascii', 'replace').decode('ascii')
        print(f"[{timestamp}] [{level}] [{module}] {safe_msg}", flush=True)
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, level, module, message) VALUES (?, ?, ?, ?)",
            (timestamp, level, module, message)
        )
        # Auto-cleanup: keep only the most recent 5000 log entries to prevent unbounded growth
        cursor.execute("""
            DELETE FROM logs WHERE id NOT IN (
                SELECT id FROM logs ORDER BY id DESC LIMIT 5000
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        try:
            print(f"FAILED TO WRITE LOG TO DB: {e}", flush=True)
        except Exception:
            pass

def save_candles(candles_list):
    """Saves a batch of candle data into the database, clearing old ones first to prevent mixing simulated/real data."""
    if not candles_list:
        return
    asset = candles_list[0]['asset']
    interval = candles_list[0]['interval']
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Clear existing candles for this asset and interval to prevent mixing simulated/real candles
    cursor.execute("DELETE FROM candles WHERE asset = ? AND interval = ?", (asset, interval))
    
    for c in candles_list:
        cursor.execute("""
        INSERT OR REPLACE INTO candles (
            timestamp, asset, interval, open, high, low, close, volume, rsi, macd, macd_signal, macd_hist, ema
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c['timestamp'], c['asset'], c['interval'], c['open'], c['high'], c['low'], c['close'], c['volume'],
            c.get('rsi'), c.get('macd'), c.get('macd_signal'), c.get('macd_hist'), c.get('ema')
        ))
    conn.commit()
    conn.close()

def get_latest_candles(asset, interval, limit=100):
    """Fetches the latest candles ordered by timestamp ascending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM candles 
        WHERE asset = ? AND interval = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (asset, interval, limit))
    rows = cursor.fetchall()
    conn.close()
    # Reverse to make it ascending (chronological)
    return [dict(r) for r in reversed(rows)]

def get_portfolio():
    """Gets all portfolio balances."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM portfolio")
    rows = cursor.fetchall()
    conn.close()
    return {r['asset']: {'balance': r['balance'], 'avg_entry_price': r['avg_entry_price']} for r in rows}

def update_portfolio(portfolio_dict):
    """Updates portfolio balances."""
    conn = get_connection()
    cursor = conn.cursor()
    for asset, data in portfolio_dict.items():
        cursor.execute("""
            INSERT OR REPLACE INTO portfolio (asset, balance, avg_entry_price)
            VALUES (?, ?, ?)
        """, (asset, data['balance'], data['avg_entry_price']))
    conn.commit()
    conn.close()

def record_trade(asset, side, price, amount, trade_type, sentiment_score=None, reason=None, stop_loss=None, take_profit=None, pnl=None, is_active=0):
    """Records a simulated paper trade and updates portfolio balances."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Save to trades table
    cursor.execute("""
        INSERT INTO trades (timestamp, asset, side, price, amount, trade_type, sentiment_score, reason, stop_loss, take_profit, pnl, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, asset, side, price, amount, trade_type, sentiment_score, reason, stop_loss, take_profit, pnl, is_active))

    conn.commit()
    conn.close()

    try:
        from core.telegram_bot import send_trade_notification
        send_trade_notification(
            asset=asset,
            side=side,
            price=price,
            amount=amount,
            trade_type=trade_type,
            sentiment_score=sentiment_score,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=pnl
        )
    except Exception as tg_err:
        try:
            print(f"Telegram notification error: {tg_err}", flush=True)
        except Exception:
            pass

def get_active_position(asset):
    """
    Retrieves the active trade position for a given asset (is_active = 1).
    Returns a dictionary of the active trade or None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM trades 
        WHERE asset = ? AND is_active = 1 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (asset,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_active_positions():
    """
    Retrieves all active trade positions (is_active = 1).
    Returns a list of dictionaries.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM trades 
        WHERE is_active = 1 
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def close_active_position(asset, sell_price, trade_type, reason=None):
    """
    Closes the active BUY trade for an asset:
    1. Calculates the P&L percentage.
    2. Updates the original BUY trade's is_active to 0 and writes the P&L.
    3. Records a new SELL trade with the calculated P&L.
    """
    active_pos = get_active_position(asset)
    if not active_pos:
        return None

    buy_price = active_pos['price']
    buy_amount = active_pos['amount']
    
    # Calculate P&L %: (sell_price - buy_price) / buy_price * 100
    pnl_pct = ((sell_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0.0
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Update active BUY trade to inactive
    cursor.execute("""
        UPDATE trades 
        SET is_active = 0, pnl = ? 
        WHERE id = ?
    """, (pnl_pct, active_pos['id']))
    
    conn.commit()
    conn.close()
    
    # 2. Record new SELL trade
    record_trade(
        asset=asset,
        side="SELL",
        price=sell_price,
        amount=buy_amount,
        trade_type=trade_type,
        sentiment_score=None,
        reason=reason,
        pnl=pnl_pct,
        is_active=0
    )
    
    return pnl_pct

def save_ai_run(asset, news_digest, sentiment_score, reason):
    """Saves a Gemini AI model run for analytical auditing."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO ai_runs (timestamp, asset, news_digest, sentiment_score, reason)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, asset, news_digest, sentiment_score, reason))
    conn.commit()
    conn.close()

def get_latest_ai_run(asset):
    """Gets the most recent AI evaluation for an asset."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM ai_runs 
        WHERE asset = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (asset,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_trades(limit=50):
    """Gets the history of trades."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_logs(limit=100):
    """Gets the log messages."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_config(key, default=None):
    """Fetches a configuration value by key."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else default

def save_config(key, value):
    """Saves a configuration key-value pair."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

# Initialize DB on import if running directly
if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
