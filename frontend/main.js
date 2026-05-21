// API Base URL (FastAPI)
const API_URL = 'http://127.0.0.1:8000/api';

// DOM Elements
const navItems = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');
const statusDot = document.querySelector('.pulse-dot');
const statusText = document.getElementById('conn-status');

const els = {
  navValue: document.getElementById('nav-value'),
  freeUsd: document.getElementById('free-usd-value'),
  posCount: document.getElementById('open-positions-count'),
  posList: document.getElementById('positions-list'),
  logs: document.getElementById('terminal-logs'),
  syncBtn: document.getElementById('sync-tick-btn')
};

// State
let pollInterval = null;

// Initialization
async function init() {
  setupTabs();
  setupInteractions();
  
  // Check backend connection
  await checkBackendStatus();
  
  // Start Polling (every 3 seconds to feel real-time without WS complexity for now)
  pollInterval = setInterval(fetchDashboardData, 3000);
  fetchDashboardData();
  
  // TODO FOR USER: Initialize chart
  initUserChart();
}

// ----------------------------------------------------
// UI Logic
// ----------------------------------------------------
function setupTabs() {
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      // Reset active classes
      navItems.forEach(nav => nav.classList.remove('active'));
      tabContents.forEach(tab => tab.classList.remove('active'));
      
      // Set active
      item.classList.add('active');
      const targetId = `tab-${item.dataset.tab}`;
      document.getElementById(targetId).classList.add('active');
    });
  });
}

function setupInteractions() {
  els.syncBtn.addEventListener('click', async () => {
    els.syncBtn.textContent = '⏳ Syncing...';
    try {
      await fetch(`${API_URL}/trigger_analysis`, { method: 'POST' });
      // Instantly fetch data
      setTimeout(fetchDashboardData, 1000);
    } catch(e) {
      console.error(e);
    } finally {
      els.syncBtn.textContent = '🚀 Force Sync Tick';
    }
  });
}

// ----------------------------------------------------
// Data Fetching
// ----------------------------------------------------
async function checkBackendStatus() {
  try {
    const res = await fetch(`${API_URL}/status`);
    if(res.ok) {
      statusDot.classList.add('online');
      statusText.textContent = 'Backend Connected';
    }
  } catch(e) {
    statusDot.classList.remove('online');
    statusText.textContent = 'Backend Offline';
  }
}

async function fetchDashboardData() {
  try {
    // Parallel fetching for speed
    const [portRes, posRes, logsRes] = await Promise.all([
      fetch(`${API_URL}/portfolio`).then(r => r.json()),
      fetch(`${API_URL}/positions`).then(r => r.json()),
      fetch(`${API_URL}/logs?limit=30`).then(r => r.json())
    ]);

    // Update Metrics
    updateMetrics(portRes, posRes);
    
    // Update Positions List
    updatePositions(posRes.positions);
    
    // Update Logs
    updateLogs(logsRes.logs);

  } catch (error) {
    statusDot.classList.remove('online');
    statusText.textContent = 'Connection Error';
    console.error("Polling failed:", error);
  }
}

function updateMetrics(portData, posData) {
  statusDot.classList.add('online');
  statusText.textContent = 'Live Feed Active';

  // Format money
  const formatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
  
  els.freeUsd.textContent = formatter.format(portData.free_usd);
  els.posCount.textContent = posData.count;
  
  // Very rough NAV calc for UI (Normally backend does this exactly)
  let totalNav = portData.free_usd;
  posData.positions.forEach(p => {
    totalNav += (p.price * p.amount); // Approximation using entry price
  });
  els.navValue.textContent = formatter.format(totalNav);
}

function updatePositions(positions) {
  if (positions.length === 0) {
    els.posList.innerHTML = `<div style="color:var(--text-muted); text-align:center; margin-top:2rem;">No active positions</div>`;
    return;
  }
  
  els.posList.innerHTML = positions.map(p => `
    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid ${p.pnl >= 0 ? 'var(--neon-green)' : 'var(--neon-red)'}">
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <strong>${p.asset}</strong>
        <span class="${p.pnl >= 0 ? 'neon-green' : 'neon-red'}">${p.pnl ? p.pnl.toFixed(2) : '0.00'}%</span>
      </div>
      <div style="font-size: 0.8rem; color: var(--text-muted); display:flex; justify-content:space-between;">
        <span>Entry: $${p.price.toFixed(2)}</span>
        <span>Amt: ${p.amount.toFixed(4)}</span>
      </div>
    </div>
  `).join('');
}

function updateLogs(logs) {
  els.logs.innerHTML = logs.map(l => `
    <div class="log-row">
      <span class="timestamp">[${new Date(l.timestamp).toLocaleTimeString()}]</span>
      <span class="log-${l.level}">[${l.level}]</span>
      <span>[${l.module}]</span>
      <span style="color: #fff;">${l.message}</span>
    </div>
  `).join('');
}

// ----------------------------------------------------
// 🚀 DEVELOPER AREA: YOUR TURN TO SHINE!
// ----------------------------------------------------
/* 
  Hello! You mentioned you wanted to learn/develop the charting.
  Here is your workspace for TradingView Lightweight Charts.

  Step 1: I've included the library in index.html via CDN.
  Step 2: You will use `LightweightCharts.createChart()` and attach it to `document.getElementById('tvchart-container')`.
  Step 3: You can fetch candle data from the API I built: `fetch('/api/chart/BTC-USDT/1h')`.
  
  Here is a skeleton function to get you started. Try completing it!
*/
function initUserChart() {
  const container = document.getElementById('tvchart-container');
  // Clear the placeholder
  container.innerHTML = '';
  
  // TODO: Create chart instance
  // const chart = LightweightCharts.createChart(container, {
  //   layout: { background: { type: 'solid', color: 'transparent' }, textColor: '#d1d4dc' },
  //   grid: { vertLines: { color: 'rgba(42, 46, 57, 0)' }, horzLines: { color: 'rgba(42, 46, 57, 0.6)' } },
  // });
  
  // TODO: Create a candlestick series
  // const candleSeries = chart.addCandlestickSeries({ ...colors });

  // TODO: Fetch data from our new FastAPI endpoint and map it
  // Example fetch:
  /*
    fetch('http://127.0.0.1:8000/api/chart/BTC-USDT/1h')
      .then(res => res.json())
      .then(data => {
         // Map API data (timestamp, open, high, low, close) to TradingView format
         // Note: TradingView wants timestamp as unix seconds.
         const formattedData = data.candles.map(c => ({
            time: new Date(c.timestamp).getTime() / 1000,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close
         }));
         // Set data to chart
         candleSeries.setData(formattedData);
      });
  */
  
  container.innerHTML = `<div class="chart-placeholder">
    <p style="text-align:center; line-height: 1.5;">
       <strong>Developer Space</strong><br>
       Uncomment the code in <code>main.js -> initUserChart()</code> to render TradingView Charts!<br>
       You can customize colors, crosshairs, and more.
    </p>
  </div>`;
}

// Boot
init();
