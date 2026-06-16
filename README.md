# Sentix | AI-Powered Algorithmic Trading & Sentiment Analysis Platform

Sentix is a modular, production-ready, and Docker-compatible algorithmic trading and news sentiment analysis platform. It combines deterministic technical indicator math with state-of-the-art Generative AI (Google Gemini API) to confirm trading decisions.

To maximize operational cost-efficiency, the platform utilizes a **decoupled SQLite data architecture** and **conditional AI triggering**, ensuring that the Google Gemini API is only invoked when potential technical crossover triggers are met on historical charts.

---

## 🚀 Key Advanced Features

### 1. Dual-Loop Background Worker (`worker.py`)
The worker service operates using a dual-speed asynchronous architecture to ensure safety and precision:
- **SL/TP Guardian Loop (Fast Loop - Default: 30s):** Queries real-time prices via CCXT ticker calls every 30 seconds. If an active position breaches its configured Stop-Loss or Take-Profit margin, the guardian immediately liquidates the position in paper trading.
- **Market Analysis Scheduler Loop (Slow Loop - Default: 300s):** Runs every 5 minutes to fetch the latest completed 1-hour candle data, calculate technical indicators, evaluate crossover triggers, and trigger Gemini analysis when a signal is active.

### 2. Multi-Factor Validation (Double-Check Math Filters)
To filter out false breakouts and fake signals, the math engine implements a two-stage filter:
- **4-Hour Timeframe Trend Filter:** When a 1-hour buy signal is triggered, it must be supported by a bullish trend on the 4-hour timeframe (Price > EMA 20 or MACD Golden Cross) to be approved.
- **Volume Verification Filter:** The volume of the latest completed 1-hour candle must exceed the 20-period simple moving average of volume (Volume MA).

### 3. Token-Efficient Batch AI Sentiment Engine (Gemini API)
- **Modern Google GenAI SDK Support:** Integrates the latest `google-genai` library with full support for Gemini 3.5, 3.1 Flash Lite, 2.5 Pro, and newer models.
- **Two-Step AI Processing & Vetos:**
  - **Step 1 (Summarization):** Google News RSS articles (latest 5-10 items) are cleaned, de-noised, and synthesized into a comprehensive piyasa summary using `gemini-3.1-flash-lite` to minimize token overhead.
  - **Step 2 (Structured Scoring):** The summary is evaluated by the advanced model in **JSON Mode** (enforced by Pydantic schemas) to produce a structured sentiment rating between -10 (extremely bearish/panic) and +10 (extremely bullish/FOMO).
- **AI Veto Logic:** If technical indicators trigger a BUY but the AI sentiment is bearish (<= -2), the buy order is vetoed and cancelled. If indicators trigger a SELL but AI sentiment is extremely bullish (>= 3), the sell order is vetoed to let profits run.
- **API Key Pool Rotation:** Loads a pool of Gemini API keys from `gemini_keys.txt`. If a key hits a `429 ResourceExhausted` (Rate Limit) error, the platform rotates to the next key automatically.
- **Failsafe Offline Simulation:** In the absence of an API key or network connection, the platform automatically switches to an advanced simulation mode (scanning article descriptions for bullish/bearish keywords) to preserve complete dashboard functionality.
- **Dynamic Position Sizing:** Based on how many confirmation factors are met (Base trigger, 4H Trend, Volume, Positive AI Sentiment), the position risk size is dynamically scaled between 25% and 100% of the target risk percentage.

### 4. VPN Guardian (Location & Identity Protection)
- The worker daemon communicates with the host Unix socket (`/var/run/docker.sock`) to monitor the status of the Gluetun VPN container (`sentix_vpn`).
- If the VPN drops or the external IP address leaks, the bot **instantly halts all trading actions and SL/TP checks** in safe mode. It immediately alerts the user via Telegram, appending the latest VPN logs for error analysis.

### 5. Interactive Telegram Bot Integration
Manage and monitor your trading bot remotely:
- **Instant Notifications:** Receives real-time alerts for executed buys, sells, SL/TP liquidations, and VPN state updates.
- **Remote Controller Commands:** Supports authorized remote polling commands for querying system status, metrics, adjusting risk settings, and manual cycle execution.

### 6. Dual-UI Dashboard Architecture
Choose the dashboard that fits your use case:
- **Streamlit Cyberpunk Panel (`ui/app.py`):** A password-protected, glassmorphic UI featuring Plotly subplots, custom CSS overrides, and a detailed step-by-step indicator checklist to inspect bot rules.
- **FastAPI Backend (`api/main.py`) & Web UI (`frontend/`):** A lightweight SaaS-style dashboard written in vanilla HTML/CSS/JS that communicates asynchronously with the FastAPI REST API.

---

## 📂 Directory Structure

```
sentix/
├── core/
│   ├── db.py                 # SQLite database schema, connections, and transactional helpers
│   ├── data_fetcher.py       # ccxt candlestick puller and built-in RSS news scraper
│   ├── math_engine.py        # Technical indicator calculations and crossover logic
│   └── telegram_bot.py       # Telegram notification routing and remote command polling
├── ai/
│   └── sentiment_analyzer.py # Batch Gemini API client with rotation and simulation fallbacks
├── ui/
│   └── app.py                # Premium dark-mode Streamlit dashboard with custom CSS & Plotly
├── api/
│   └── main.py               # REST API endpoints serving data to the HTML frontend
├── frontend/
│   ├── index.html            # Modern HTML dashboard landing page
│   ├── style.css             # Glassmorphic dark styling rules
│   └── main.js               # JavaScript routing, REST sync, and TradingView layout
├── worker.py                 # Core background loop coordinator (Dual-Loop Scheduler)
├── Dockerfile                # Production Docker container definition
├── docker-compose.yml        # Orchestrates worker, streamlit dashboard, and Gluetun VPN
├── gemini_keys.txt.example   # Template for Gemini API key rotation pool
├── requirements.txt          # Python library dependencies
└── LICENSE                   # Personal Use License agreement (Commercial use prohibited)
```

---

## 🛠️ Local Setup & Execution

### 1. Prerequisite Environment
Create your `.env` configuration file by copying the template:
```bash
cp .env.example .env
```
Edit `.env` to supply your **Google Gemini API Key**, **Telegram Bot Token**, and **Chat ID**. If using key rotation, create a `gemini_keys.txt` file in the root directory and write one API key per line.

### 2. Python Virtual Environment Setup
Ensure you have Python 3.10+ installed:
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux / macOS)
source .venv/bin/activate

# Install dependencies
    pip install -r requirements.txt
```

### 3. Running Locally

#### Option A: Background Worker + Streamlit Dashboard
1. **Launch the Background Worker & Telegram Bot Listener:**
   ```bash
   python worker.py
   ```
2. **Launch the Streamlit UI** (in a separate terminal):
   ```bash
   streamlit run ui/app.py
   ```
   Open your browser to `http://localhost:8501`.

#### Option B: Background Worker + FastAPI Backend + Web UI
1. **Launch the Background Worker:**
   ```bash
   python worker.py
   ```
2. **Launch the FastAPI Server:**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
3. Open `frontend/index.html` in your web browser, or host it using any local HTTP static server. It will automatically connect to the FastAPI REST API at `http://localhost:8000/api`.

---

## 🐳 Production Deployment with Docker

The platform is designed to run in a fully self-contained Docker environment behind **Gluetun VPN** to hide the server's real IP address from exchanges.

1. Configure your VPN provider credentials inside `.env` (ProtonVPN, NordVPN, etc., are supported).
2. Start the full stack with a single command:
   ```bash
   docker-compose up --build -d
   ```
3. Monitor the system logs remotely:
   ```bash
   # View combined logs
   docker-compose logs -f
   
   # View only AI trading worker logs
   docker-compose logs -f worker
   ```

---

## 🤖 Telegram Bot Commands

You can send the following commands to your bot via Telegram:

- `/portfolio` - View current holdings, token allocation, and total Net Asset Value (NAV).
- `/positions` - List all active spot positions, purchase prices, and real-time PnL.
- `/trades` - Display the last 5 executed trades.
- `/assets` - List all active trading pairs currently being scanned.
- `/vpn` - Check VPN container connection status, external IP address, and proxies.
- `/vpn_logs` - Fetch the last 25 lines of stdout/stderr logs from the `sentix_vpn` container.
- `/ai_status` - Display latest active Gemini sentiment scores and reasons for selected assets.
- `/logs` - Show the last 5 logs saved in the SQLite event logs table.
- `/trigger` - Manually trigger an analysis cycle immediately in the background.
- `/pause` - Pause the automatic trading scheduler (SL/TP guardians remain active for protection).
- `/resume` - Resume the automatic trading scheduler.
- `/risk [pct]` - Update the trade risk percentage based on total NAV (e.g., `/risk 2.5`).
- `/sltp [sl] [tp]` - Update the Stop-Loss and Take-Profit ratios dynamically (e.g., `/sltp 3.0 6.0`).

---

## 📄 License & Commercial Restrictions

This software is licensed under the **Sentix Personal and Non-Commercial License Agreement (SPNCL-1.0)**.

### Terms of Use:
- **Individual/Personal Use:** You are permitted to execute, inspect, and modify the source code of this Software for private, individual, educational, and non-commercial research purposes.
- **Commercial Restriction:** Any commercial use, redistribution, or commercialization of the Software or its derivatives is strictly prohibited. You may NOT host the software as a paid service (SaaS), manage funds for clients using this software, or rebrand and republish it as your own commercial product.
- For full legal terms, please review the [LICENSE](https://github.com/technomonkey-7/sentix/blob/main/LICENSE) file in the root directory.
