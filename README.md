# Sentix | AI-Powered Algorithmic Trading & Sentiment Analysis Platform

Sentix is a modular, production-ready, and Docker-compatible algorithmic trading and news sentiment analysis platform. It combines deterministic technical indicator math with state-of-the-art Generative AI to confirm trading decisions. 

To maximize operational cost-efficiency, the platform utilizes a **decoupled SQLite data architecture** and **conditional AI triggering**, ensuring that the Google Gemini API is only invoked when potential technical crossover triggers are met.

---

## Key Features

1. **Decoupled Service Design**:
   - **Background Worker (`worker.py`)**: Runs on a periodic schedule to fetch OHLCV candle data, compute indicator math, scrape live news feeds, coordinate trading decisions, and execute paper orders.
   - **Streamlit UI Dashboard (`ui/app.py`)**: A gorgeous, glassmorphic dark-themed user interface to visually inspect live portfolio metrics, interactive technical charts, bot runtime logs, and the raw Gemini AI sentiment audit trails.
   - Both modules operate independently, communicating solely via a shared, transactional SQLite database (`data.db`).

2. **Token-Efficient Hybrid AI**:
   - **Conditional Evaluation**: The Gemini API is **never** polled on a loop. It is only called when a physical crossover trigger is active on the latest completed candle.
   - **Step 1 (De-noising)**: Utilizes `gemini-2.5-flash` to scrape, filter out spam, and summarize the latest 5-10 news articles from Google News RSS.
   - **Step 2 (Structured Score)**: Passes the clean digest to `gemini-2.5-pro` using **JSON Mode** to output a crisp sentiment rating between -10 and 10.
   - **Dynamic Model Configs**: Active models can be customized in the UI (defaulting to the latest Gemini 2.5 and 2.0 editions).
   - **Intelligent Fallback**: In the absence of an API key or network connection, the platform automatically drops into an advanced simulation mode to preserve complete UI functionality.

3. **Deterministic Math Engine**:
   - Indicator calculations are handled entirely with zero-AI determinism using `pandas-ta` (**RSI 14**, **MACD 12,26,9**, **EMA 20**).
   - Prevents repainting issues by running signals strictly on **fully completed candles** (index `-2`).

---

## Directory Structure

```
sentix/
├── core/
│   ├── __init__.py
│   ├── db.py                 # SQLite database schema, connections, and transactional helpers
│   ├── data_fetcher.py       # ccxt candlestick puller and built-in RSS news scraper
│   └── math_engine.py        # Technical indicator calculations and crossover logic
├── ai/
│   ├── __init__.py
│   └── sentiment_analyzer.py # Two-step Gemini API client with model fallbacks
├── ui/
│   ├── __init__.py
│   └── app.py                # Premium dark-mode Streamlit dashboard with custom CSS & Plotly
├── worker.py                 # Core background loop runner coordinating decisions
├── Dockerfile                # Production Docker container definition
├── docker-compose.yml        # Orchestrates separate ui and worker containers
├── requirements.txt          # Python dependency manifests
└── README.md                 # Complete platform documentation
```

---

## Local Setup & Execution

### 1. Prerequisite Environment
Create your `.env` configuration file by copying the template:
```bash
cp .env.example .env
```
Edit `.env` to supply your **Google Gemini API Key**:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

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
For a local test run:

1. **Launch the Background Worker**:
   ```bash
   python worker.py
   ```
2. **Launch the Streamlit UI** (in a separate terminal):
   ```bash
   streamlit run ui/app.py
   ```
   Open your browser to `http://localhost:8501`.

---

## Production Deployment with Docker

The platform is designed to be fully self-contained and deployable on a **DigitalOcean Linux VPS** using Docker.

### 1. Single Command Build & Run
From the root directory:
```bash
docker-compose up --build -d
```

This starts:
- `sentix_worker` container executing the scheduler in the background.
- `sentix_ui` container exposing the dashboard on port `8501`.
- A shared bind-mount mapping between both containers to synchronize `data.db` instantly.

### 2. Logs Monitoring on DigitalOcean
To monitor the system logs remotely:
```bash
# View combined logs
docker-compose logs -f

# View only AI trading activity logs
docker-compose logs -f worker
```
All print logs are flushed directly to standard output, making them fully compatible with tools like PM2, Docker logging drivers, or systemd services.
