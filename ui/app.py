import sys
import os
# Dynamic project root resolver to ensure imports work in all execution contexts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Import database, worker, and math dependencies
from core.db import (
    get_connection, get_portfolio, get_trades, get_logs, 
    get_latest_ai_run, get_config, save_config, log_event,
    get_latest_candles, get_active_position, get_all_active_positions, 
    close_active_position, update_portfolio, record_trade
)
from worker import run_worker_cycle, calculate_total_nav
from core.data_fetcher import fetch_ohlcv, fetch_asset_news
from core.math_engine import calculate_indicators, check_triggers
from ai.sentiment_analyzer import analyze_sentiment

# Load env configurations
load_dotenv()

from datetime import timezone, timedelta
ISTANBUL_TZ = timezone(timedelta(hours=3))

# Bilingual Translations Dictionary
T = {
    "TR": {
        "title": "Sentix | Premium SaaS Kontrol Paneli",
        "subtitle": "Premium Kaldıraçsız Yapay Zeka Algoritmik Trading ve Gerçek Zamanlı Pozisyon Koruyucu Kontrol Paneli",
        "select_pairs": "Aktif İşlem Çiftlerini Seçin",
        "select_pairs_help": "Botun arka planda tarayacağı ve otomatik sinyal arayacağı Binance spot işlem çiftleri.",
        "sum_model": "Özetleme Modeli (Aşırı Token Cimrisi - Adım 1)",
        "sum_model_help": "Google News makalelerini hızlıca okuyup temiz bir piyasa özeti çıkaran Flash modeli.",
        "sent_model": "Duygu Analiz Modeli (Gelişmiş - Adım 2)",
        "sent_model_help": "Piyasa özetine bakarak -10 ile +10 arası duygu skoru ve mantıksal açıklama üreten Pro modeli.",
        "risk_pct": "NAV İşlem Büyüklüğü (Risk %)",
        "risk_pct_help": "Her bir alım emrinin toplam cüzdan büyüklüğünüze (NAV) oranı. Kasa yönetimi için %1-%5 önerilir.",
        "sl_pct": "Stop-Loss (Zarar Durdur) Oranı %",
        "sl_pct_help": "Fiyat giriş seviyesinden bu yüzde kadar düştüğünde işlem anında otomatik olarak kapatılarak zarar sınırlandırılır.",
        "tp_pct": "Take-Profit (Kâr Al) Oranı %",
        "tp_pct_help": "Fiyat giriş seviyesinden bu yüzde kadar yükseldiğinde işlem anında kapatılarak kâr realize edilir.",
        "exchange_name": "Aktif Borsa (CCXT)",
        "exchange_name_help": "Piyasa verilerinin çekileceği borsa (örn: binance, bybit, kraken, okx, gateio, coinbase).",
        "min_sentiment": "Min Yapay Zeka Duygu Eşiği",
        "min_sentiment_help": "Teknik sinyal oluştuktan sonra, işlemin onaylanması için gereken minimum yapay zeka duygu skoru (örn: 3).",
        "quick_actions": "⚙️ Hızlı Eylemler",
        "sync_tick": "🚀 Sync Tick",
        "sync_tick_help": "Veri tabanını anlık günceller, indikatörleri hesaplar ve arka plan tarama döngüsünü tetikler.",
        "reset_port": "🔄 Portföyü Sıfırla",
        "reset_port_help": "Sanal bakiyeyi $10,000 USD yapar, tüm açık işlemleri, mumları ve log geçmişini temizler.",
        "warning_select_pair": "⚠️ Lütfen sol panelden en az bir aktif işlem çifti seçin.",
        "nav_metrics": "Portföy Durumu (NAV Hesaplama)",
        "free_usd": "USD Serbest Bakiye",
        "free_usd_help": "Cüzdanınızdaki boşta duran, kullanılabilir nakit dolar miktarı.",
        "portfolio_nav": "Portföy NAV Değeri",
        "portfolio_nav_help": "Toplam portföy net değeri: Serbest USD + Açık kripto pozisyonlarının anlık piyasa değeri.",
        "net_return": "Toplam Net Getiri",
        "net_return_help": "Başlangıç bakiyesi ($10,000) üzerinden elde ettiğiniz kâr/zarar yüzdesi.",
        "open_pos_count": "Açık Pozisyonlar",
        "open_pos_count_help": "Şu anda taşımakta olduğunuz aktif spot cüzdan varlıklarının sayısı.",
        "bot_active": "🟢 BOT AKTİF - SİNYAL BEKLENİYOR",
        "bot_active_desc": "Matematik motoru 1H Binance spot grafiklerinde RSI, EMA ve MACD kesişimlerini 7/24 izliyor...",
        "bot_offline": "🔴 BOT ÇEVRİMDIŞI - ARKA PLAN ÇALIŞMIYOR",
        "bot_offline_desc": "Matematik motoru ve pozisyon koruyucu aktif değil. Lütfen terminalden 'python worker.py' komutunu çalıştırın.",
        "in_pos": "🔴 POZİSYONDAYIM: {asset} ({arrow} {pnl:+.2f}%)",
        "pos_details": "Giriş: ${entry:,.2f} | Fiyat: ${live:,.2f} | Stop-Loss: ${sl:,.2f} | Take-Profit: ${tp:,.2f}",
        "tab_a": "📈 Canlı Takip Paneli",
        "tab_b": "🔬 Manuel Analiz ve Test",
        "tab_c": "📜 Loglar ve Şeffaflık",
        "chart_title": "{asset} Grafik Analizi (Zaman Dilimi: Europe/Istanbul UTC+3)",
        "crossover_inspector": "📊 Matematiksel İndikatör Sinyal Analizörü",
        "crossover_inspector_desc": "Python arka planda matematiksel indikatörleri hesaplayıp tarıyor. Aşağıdan anlık durumları görebilirsiniz:",
        "col_asset": "Kripto Varlık",
        "col_price": "Son Fiyat",
        "col_rsi": "RSI (14) Durumu",
        "col_macd": "MACD Crossover",
        "col_ema": "Fiyat vs EMA 20",
        "col_decision": "Bot Kararı / Durum",
        "decision_neutral": "❌ AI PAS GEÇİLDİ (Sinyal Yok)",
        "decision_buy": "🟢 ALIM TETİKLENDİ",
        "decision_sell": "🔴 SATIM TETİKLENDİ",
        "decision_in_pos": "🔒 POZİSYON KİLİTLİ (Açık Pozisyon Var)",
        "indicator_explanation": "💡 **Nasıl Yorumlanır?** Alım için RSI <= 30 veya MACD yukarı kesişim veya fiyatın EMA 20 üzerine çıkması gerekir. Bu teknik koşullar sağlanmadan yapay zeka tetiklenmez.",
        "run_manual_btn": "⚡ Manuel Analiz Et",
        "manual_spinner": "Piyasa verileri toplanıyor ve Gemini ile analiz ediliyor...",
        "manual_report_title": "📊 Yapay Zeka Finansal Raporu ({asset})",
        "indicator_analysis": "🔬 Teknik Göstergeler Analizi",
        "indicator": "İndikatör",
        "val": "Değer",
        "status": "Durum",
        "ai_sentiment_synthesis": "🤖 Yapay Zeka Haber Duygusu Sentezi",
        "news_digest_title": "Haber Özet Özeti (Flash Model):",
        "ai_reason_title": "Büyük Model Duygu Yorumu (Pro Model):",
        "news_feed_title": "📰 Taranan En Güncel Haberler",
        "news_freshness": "Yayınlanma: {time}",
        "manual_order_placement": "🛒 Manuel İşlem Gönderme",
        "pos_exists_warn": "⚠️ Bu coin için aktif açık pozisyonunuz bulunuyor. Yeni alım yapamazsınız, pozisyonu kapatabilirsiniz.",
        "close_pos_btn": "💰 Pozisyonu Manuel Kapat (SELL)",
        "buy_order_btn": "🛒 Manuel Alım Yap (BUY) - SL/TP Korumalı",
        "sell_order_btn": "💰 Manuel SAT (SELL)",
        "order_success": "Emir başarıyla gerçekleştirildi!",
        "order_fail": "Emir gerçekleştirilemedi: {e}",
        "action_logs": "🤖 Canlı İşlem Günlüğü (SQLite Logs)",
        "ai_thought_logs": "🧠 AI Düşünce Günlüğü (AI Audit Trail)",
        "ai_no_run": "Henüz Gemini AI analizi gerçekleşmedi. Teknik kırılımlar gerçekleştikçe veya manuel analiz yapıldığında AI kayıtları burada listelenir.",
        "news_digest_ai": "Derlenmiş Haber Özeti (Flash Modeli):",
        "ai_decision_json": "Büyük Model Duygu Skoru ve Kararı (JSON):",
        "reason": "Gerekçe",
        "score": "Duygu Skoru",
        "api_key_help": "Gemini API işlemlerini yetkilendirmek için kullanılan şifrelenmiş anahtar.",
        "active_holdings": "📦 Aktif Varlık Bilgileri",
        "balance": "Bakiyesi",
        "cost": "Maliyet",
        "size": "Büyüklük",
        "analyze_on_chart": "Grafik Üzerinde İncele",
        "recent_trades_sim": "📜 Son Gerçekleşen İşlemler (Simüle)",
        "no_trades_yet": "💡 Henüz bir işlem gerçekleşmedi. Bot piyasadaki indikatör kesişimlerini takip ediyor.",
        "coin_to_analyze": "Analiz Edilecek Coin",
        "asset_type": "Varlık Tipi",
        "portfolio_risk_share": "Portföy Risk Payı",
        "insufficient_bal_err": "Bakiye yetersiz! Alım işlemi gerçekleştirmek için en az $10 USD serbest bakiye gereklidir.",
        "transparency_logs": "📜 Platform Şeffaflığı ve Güvenli Log Kayıtları",
        "system_active_waiting": "Sistem aktif. İlk analiz tetiklenmesi bekleniyor...",
        "col_time": "Zaman",
        "col_amount": "Miktar",
        "col_value": "Değer",
        "col_sl": "SL Sınırı",
        "col_tp": "TP Sınırı",
        "col_pnl": "Kâr/Zarar %",
        "col_desc": "Açıklama",
        "api_key_updated": "🔑 API Anahtarı güncellendi ve kaydedildi!",
        "active_pairs_updated": "📂 Aktif işlem çiftleri güncellendi!",
        "sync_success": "Senkronizasyon döngüsü başarıyla çalıştı!",
        "reset_success": "Portföy başarıyla sıfırlandı!",
        "auto_refresh": "⏱️ Otomatik Yenileme",
        "auto_refresh_help": "Aktif olduğunda kontrol paneli seçilen saniyede bir otomatik olarak yenilenir ve en güncel verileri çeker.",
        "refresh_interval": "Yenileme Sıklığı (Saniye)",
        "col_pair": "İşlem Çifti",
        "col_side": "İşlem Yönü",
        "checklist_title": "🔍 Detaylı İndikatör Sinyal Kontrol Listesi",
        "checklist_desc": "Seçili kripto varlık için botun alım/satım kurallarını tek tek denetleyin. Hangi kuralın gerçekleşmediğini ve neden yapay zekanın uykuda olduğunu buradan görebilirsiniz.",
        "checklist_select_asset": "İncelenecek Kripto Varlık",
        "chk_buy_rules": "🟢 ALIM (BUY) KOŞULLARI KONTROLÜ",
        "chk_sell_rules": "🔴 SATIM (SELL) KOŞULLARI KONTROLÜ",
        "chk_rsi_buy": "RSI Aşırı Satım: RSI 30'un altında olmalı veya 30'u yukarı kesmeli. Mevcut RSI: {rsi_c} (Önceki: {rsi_p})",
        "chk_macd_buy": "MACD Altın Kesişim: MACD çizgisi Sinyal çizgisini yukarı kesmeli. Mevcut MACD: {macd_c}, Sinyal: {sig_c} (Önceki MACD: {macd_p}, Sinyal: {sig_p})",
        "chk_ema_buy": "EMA 20 Boğa Kırılımı: Fiyat EMA 20'nin üzerine çıkmalı. Fiyat: {close_c}, EMA 20: {ema_c} (Önceki Fiyat: {close_p}, EMA: {ema_p})",
        "chk_pos_buy": "Portföy Durumu: Bu varlıkta açık pozisyon olmamalıdır.",
        "chk_rsi_sell": "RSI Aşırı Alım: RSI 70'in üzerinde olmalı veya 70'i aşağı kesmeli. Mevcut RSI: {rsi_c} (Önceki: {rsi_p})",
        "chk_macd_sell": "MACD Ölüm Kesişim: MACD çizgisi Sinyal çizgisini aşağı kesmeli. Mevcut MACD: {macd_c}, Sinyal: {sig_c} (Önceki MACD: {macd_p}, Sinyal: {sig_p})",
        "chk_ema_sell": "EMA 20 Ayı Kırılımı: Fiyat EMA 20'nin altına inmeli. Fiyat: {close_c}, EMA 20: {ema_c} (Önceki Fiyat: {close_p}, EMA: {ema_p})",
        "chk_pos_sell": "Portföy Durumu: Bu varlıkta açık bir pozisyon bulunmalıdır.",
        "chk_active_pos_yes": "Açık pozisyon VAR (Giriş: {entry}, Güncel Fiyat: {live})",
        "chk_active_pos_no": "Açık pozisyon YOK",
        "rule_met": "✅ Gerçekleşti",
        "rule_not_met": "❌ Gerçekleşmedi",
        "ai_status_label": "Yapay Zeka Durumu",
        "ai_status_sleeping": "💤 UYKU MODU (Teknik tetikleyici yok, Gemini çalıştırılmadı - Token tasarrufu: %100)",
        "ai_status_waking": "⚡ AKTİF UYANIK (Teknik tetikleyici var! Gemini haberleri tarayıp karar verecek)",
        "reset_settings": "⚙️ Ayarları Sıfırla",
        "reset_settings_help": "Tüm işlem parametrelerini (risk %, SL, TP, AI eşiği, fee) optimal varsayılan değerlere sıfırlar.",
        "reset_settings_success": "Tüm ayarlar optimal varsayılanlara sıfırlandı!",
        "last_update_label": "Son Güncelleme"
    },
    "EN": {
        "title": "Sentix | Premium SaaS Control Panel",
        "subtitle": "Premium Leverage-Free AI Algorithmic Trading & Real-Time Position Guardian Control Panel",
        "select_pairs": "Select Active Trading Pairs",
        "select_pairs_help": "Binance spot trading pairs that the bot scans and searches for crossover signals in the background.",
        "sum_model": "Summarization Model (Step 1 - Token Efficient)",
        "sum_model_help": "The fast, low-cost Gemini Flash model used to clean, summarize, and digest scraped Google News articles.",
        "sent_model": "Sentiment Analysis Model (Step 2 - Advanced)",
        "sent_model_help": "The powerful Gemini Pro model used to evaluate the news digest and output a -10 to +10 sentiment score with a logical explanation.",
        "risk_pct": "NAV Trade Size (Risk %)",
        "risk_pct_help": "Size of each BUY order as a percentage of your total Net Asset Value (NAV). 1%-5% is recommended for risk safety.",
        "sl_pct": "Stop-Loss Percentage %",
        "sl_pct_help": "If the price drops below this percentage from your entry level, the position is instantly closed to limit losses.",
        "tp_pct": "Take-Profit Percentage %",
        "tp_pct_help": "If the price rises above this percentage from your entry level, the position is instantly closed to lock in profits.",
        "exchange_name": "Active Exchange (CCXT)",
        "exchange_name_help": "The exchange from which market data will be fetched (e.g., binance, bybit, kraken, okx, gateio, coinbase).",
        "min_sentiment": "Min AI Sentiment Threshold",
        "min_sentiment_help": "The minimum sentiment score (e.g. 3) required from Gemini to approve and execute a trade after a technical trigger.",
        "quick_actions": "⚙️ Quick Actions",
        "sync_tick": "🚀 Sync Tick",
        "sync_tick_help": "Force updates the database, calculates indicators, and triggers the background scanning loop immediately.",
        "reset_port": "🔄 Reset Portfolio",
        "reset_port_help": "Resets virtual balance to $10,000 USD and wipes out all trades, candles, and log histories.",
        "warning_select_pair": "⚠️ Please select at least one active trading pair in the Config Panel sidebar.",
        "nav_metrics": "Portfolio Status (NAV Calculation)",
        "free_usd": "USD Free Balance",
        "free_usd_help": "The available, unallocated cash dollar amount in your portfolio.",
        "portfolio_nav": "Portfolio NAV Value",
        "portfolio_nav_help": "Total Net Asset Value: Free USD + current market valuation of all active crypto holdings.",
        "net_return": "Total Net Return",
        "net_return_help": "The profit/loss percentage achieved based on your initial starting capital of $10,000 USD.",
        "open_pos_count": "Open Positions",
        "open_pos_count_help": "Number of active crypto spot holdings currently in your portfolio.",
        "bot_active": "🟢 BOT ACTIVE - WAITING FOR SIGNAL",
        "bot_active_desc": "Math engine is scanning 1H Binance spot charts for RSI, EMA, and MACD crossovers 24/7...",
        "bot_offline": "🔴 BOT OFFLINE - BACKGROUND WORKER NOT RUNNING",
        "bot_offline_desc": "Math engine and position guardian are inactive. Please run 'python worker.py' from your terminal.",
        "in_pos": "🔴 IN POSITION: {asset} ({arrow} {pnl:+.2f}%)",
        "pos_details": "Entry: ${entry:,.2f} | Price: ${live:,.2f} | Stop-Loss: ${sl:,.2f} | Take-Profit: ${tp:,.2f}",
        "tab_a": "📈 Live Monitoring & Indicators Analysis",
        "tab_b": "🔬 Manual Analysis & Test Panel",
        "tab_c": "📜 Logs & Transparency",
        "chart_title": "{asset} Chart Analysis (Timezone: Europe/Istanbul UTC+3)",
        "crossover_inspector": "📊 Mathematical Indicator Signal Analyzer",
        "crossover_inspector_desc": "Python is calculating mathematical indicators and scanning in the background. Live indicators state:",
        "col_asset": "Crypto Asset",
        "col_price": "Last Price",
        "col_rsi": "RSI (14) State",
        "col_macd": "MACD Crossover",
        "col_ema": "Price vs EMA 20",
        "col_decision": "Bot Decision / Status",
        "decision_neutral": "❌ AI BYPASSED (No Crossover)",
        "decision_buy": "🟢 BUY TRIGGERED",
        "decision_sell": "🔴 SELL TRIGGERED",
        "decision_in_pos": "🔒 LOCKED (Position Active)",
        "indicator_explanation": "💡 **How to interpret?** A buy trigger requires RSI <= 30, MACD crossing up, or price breaking above the 20 EMA. AI is not called until these technical criteria are met.",
        "run_manual_btn": "⚡ Run Manual Analysis",
        "manual_spinner": "Gathering market data and analyzing with Gemini...",
        "manual_report_title": "📊 AI Financial Report ({asset})",
        "indicator_analysis": "🔬 Technical Indicators Analysis",
        "indicator": "Indicator",
        "val": "Value",
        "status": "Status",
        "ai_sentiment_synthesis": "🤖 AI News Sentiment Synthesis",
        "news_digest_title": "News Summary Digest (Flash Model):",
        "ai_reason_title": "Large Model Sentiment Explanation (Pro Model):",
        "news_feed_title": "📰 Scraped Latest News Feed",
        "news_freshness": "Published: {time}",
        "manual_order_placement": "🛒 Manual Order Placement",
        "pos_exists_warn": "⚠️ You have an active open position for this coin. You cannot open a new position, but you can close the active one.",
        "close_pos_btn": "💰 Close Position Manually (SELL)",
        "buy_order_btn": "🛒 Manual BUY - SL/TP Protected",
        "sell_order_btn": "💰 Manual SELL",
        "order_success": "Order executed successfully!",
        "order_fail": "Order execution failed: {e}",
        "action_logs": "🤖 Live Action Log (SQLite Logs)",
        "ai_thought_logs": "🧠 AI Thought Log (AI Audit Trail)",
        "ai_no_run": "No Gemini AI analysis has occurred yet. AI records will be logged here as technical crossovers occur or when a manual report is run.",
        "news_digest_ai": "Scraped News Digest (Flash Model):",
        "ai_decision_json": "Large Model Sentiment Score & Verdict (JSON):",
        "reason": "Reason",
        "score": "Sentiment Score",
        "api_key_help": "Encrypted key used to authorize Google Gemini API requests.",
        "active_holdings": "📦 Active Holdings Info",
        "balance": "Balance",
        "cost": "Cost",
        "size": "Size",
        "analyze_on_chart": "Analyze on Chart",
        "recent_trades_sim": "📜 Recent Executed Trades (Simulated)",
        "no_trades_yet": "💡 No trades have executed yet. The bot is monitoring indicator crossovers.",
        "coin_to_analyze": "Coin to Analyze",
        "asset_type": "Asset Type",
        "portfolio_risk_share": "Portfolio Risk Share",
        "insufficient_bal_err": "Insufficient balance! A minimum of $10 USD free balance is required to execute a BUY.",
        "transparency_logs": "📜 Platform Transparency & Secure Event Logs",
        "system_active_waiting": "System active. Waiting for the first analysis tick...",
        "col_time": "Time",
        "col_amount": "Amount",
        "col_value": "Value",
        "col_sl": "SL Limit",
        "col_tp": "TP Limit",
        "col_pnl": "PnL %",
        "col_desc": "Reason",
        "api_key_updated": "🔑 API Key updated and saved!",
        "active_pairs_updated": "📂 Active trading pairs updated!",
        "sync_success": "Sync cycle ran successfully!",
        "reset_success": "Portfolio reset successfully!",
        "auto_refresh": "⏱️ Auto-Refresh",
        "auto_refresh_help": "When enabled, the control panel automatically reloads every N seconds to pull the latest values and chart ticks.",
        "refresh_interval": "Refresh Interval (Seconds)",
        "col_pair": "Trading Pair",
        "col_side": "Side",
        "checklist_title": "🔍 Detailed Indicator Signal Checklist",
        "checklist_desc": "Verify the bot's trading rules step-by-step for the selected asset. See exactly which condition failed and why the AI remains in sleep mode.",
        "checklist_select_asset": "Crypto Asset to Inspect",
        "chk_buy_rules": "🟢 BUY CONDITIONS CHECKLIST",
        "chk_sell_rules": "🔴 SELL CONDITIONS CHECKLIST",
        "chk_rsi_buy": "RSI Oversold: RSI must drop below 30 or cross back above 30. Current RSI: {rsi_c} (Previous: {rsi_p})",
        "chk_macd_buy": "MACD Golden Cross: MACD line must cross above Signal line. Current MACD: {macd_c}, Signal: {sig_c} (Prev MACD: {macd_p}, Signal: {sig_p})",
        "chk_ema_buy": "EMA 20 Bullish Cross: Price must break above EMA 20. Price: {close_c}, EMA 20: {ema_c} (Prev Price: {close_p}, EMA: {ema_p})",
        "chk_pos_buy": "Portfolio Status: You must not have an open position in this asset.",
        "chk_rsi_sell": "RSI Overbought: RSI must exceed 70 or cross back below 70. Current RSI: {rsi_c} (Previous: {rsi_p})",
        "chk_macd_sell": "MACD Death Cross: MACD line must cross below Signal line. Current MACD: {macd_c}, Signal: {sig_c} (Prev MACD: {macd_p}, Signal: {sig_p})",
        "chk_ema_sell": "EMA 20 Bearish Cross: Price must fall below EMA 20. Price: {close_c}, EMA 20: {ema_c} (Prev Price: {close_p}, EMA: {ema_p})",
        "chk_pos_sell": "Portfolio Status: You must have an active position in this asset.",
        "chk_active_pos_yes": "Active position EXISTS (Entry: {entry}, Live Price: {live})",
        "chk_active_pos_no": "No active position",
        "rule_met": "✅ Met",
        "rule_not_met": "❌ Not Met",
        "ai_status_label": "AI Status",
        "ai_status_sleeping": "💤 SLEEP MODE (No technical trigger, Gemini bypassed - Token saving: 100%)",
        "ai_status_waking": "⚡ AWAKE & SCROLLING (Technical trigger detected! Gemini will scan news and decide)",
        "reset_settings": "⚙️ Reset Settings",
        "reset_settings_help": "Resets all trading parameters (risk %, SL, TP, AI threshold, fee) to optimally calculated defaults.",
        "reset_settings_success": "All settings reset to optimal defaults!",
        "last_update_label": "Last Update"
    }
}

def t(key, **kwargs):
    lang = st.session_state.get("lang", "TR")
    text = T[lang].get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

def get_news_relative_time(pub_date_str):
    import email.utils
    from datetime import datetime, timezone as tz, timedelta
    try:
        dt = email.utils.parsedate_to_datetime(pub_date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.utc)
            
        # Localize to Europe/Istanbul (UTC+3)
        istanbul_tz = tz(timedelta(hours=3))
        local_dt = dt.astimezone(istanbul_tz)
        local_time_str = local_dt.strftime("%H:%M")
        
        now_utc = datetime.now(tz.utc)
        diff = now_utc - dt
        diff_minutes = int(diff.total_seconds() / 60)
        
        lang = st.session_state.get("lang", "TR")
        if diff_minutes < 1:
            rel = "Şimdi" if lang == "TR" else "Just now"
        elif diff_minutes < 60:
            rel = f"{diff_minutes} dakika önce" if lang == "TR" else f"{diff_minutes}m ago"
        elif diff_minutes < 1440:
            diff_hours = diff_minutes // 60
            rel = f"{diff_hours} saat önce" if lang == "TR" else f"{diff_hours}h ago"
        else:
            diff_days = diff_minutes // 1440
            rel = f"{diff_days} gün önce" if lang == "TR" else f"{diff_days}d ago"
            
        return f"{rel} ({local_time_str})"
    except Exception:
        return pub_date_str

# Page Configurations
st.set_page_config(
    page_title="Sentix | Premium SaaS Control Panel",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication Gate
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD")
if ACCESS_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        # Custom CSS for Cyberpunk Login Page
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }
            .stApp {
                background: radial-gradient(circle at 50% 50%, #0d1527 0%, #05070c 100%) !important;
                color: #E2E8F0 !important;
            }
            .login-card {
                background: rgba(255, 255, 255, 0.02) !important;
                backdrop-filter: blur(25px) !important;
                -webkit-backdrop-filter: blur(25px) !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
                border-radius: 16px !important;
                padding: 2.5rem !important;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
                text-align: center;
                margin-top: 15vh;
            }
            .login-title {
                background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                letter-spacing: 0.05em;
            }
            .login-subtitle {
                color: #64748B;
                font-size: 0.95rem;
                margin-bottom: 2rem;
            }
            /* Form inputs styling override for login */
            div[data-baseweb="input"] > input {
                background-color: rgba(255, 255, 255, 0.02) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 8px !important;
                color: #FFFFFF !important;
            }
        </style>
        """, unsafe_allow_html=True)

        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.markdown("""
            <div class="login-card">
                <div class="login-title">⚡ SENTIX TERMINAL</div>
                <div class="login-subtitle">Giriş Yetkilendirme / Secure Access Portal</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_gate", clear_on_submit=False):
                pwd = st.text_input("Access Password / Giriş Şifresi", type="password")
                submit = st.form_submit_button("UNLOCK TERMINAL / GİRİŞ YAP")
                
                if submit:
                    if pwd == ACCESS_PASSWORD:
                        st.session_state["authenticated"] = True
                        st.success("Access Granted. Initializing console...")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("Access Denied. Invalid password key.")
            
            st.markdown("<p style='text-align: center; color: #334155; font-size: 0.8rem; margin-top: 2rem;'>Protected by Sentix AI & Glassmorphic Security Layer</p>", unsafe_allow_html=True)
        st.stop()

# Custom TradingView Lightweight Charts HTML/JS Code Generator
def get_tradingview_html(asset, candles, trades, active_pos):
    # Pre-process candles data to JSON
    df = pd.DataFrame(candles)
    if df.empty:
        return "<h4 style='color:#94A3B8; text-align:center;'>No candle data available</h4>"
        
    # Convert timestamps to unix epoch seconds (integer)
    df['time'] = pd.to_datetime(df['timestamp']).astype('datetime64[ns]').astype('int64') // 10**9
    
    # Fill NaN values for JSON safety
    df = df.fillna(0)
    
    # Format data for chart
    candles_json = []
    ema_json = []
    rsi_json = []
    macd_json = []
    macd_signal_json = []
    macd_hist_json = []
    
    for _, r in df.iterrows():
        t = int(r['time'])
        candles_json.append({
            'time': t,
            'open': float(r['open']),
            'high': float(r['high']),
            'low': float(r['low']),
            'close': float(r['close'])
        })
        if 'ema' in r and r['ema'] > 0:
            ema_json.append({'time': t, 'value': float(r['ema'])})
        if 'rsi' in r and r['rsi'] > 0:
            rsi_json.append({'time': t, 'value': float(r['rsi'])})
        if 'macd' in r:
            macd_json.append({'time': t, 'value': float(r['macd'])})
        if 'macd_signal' in r:
            macd_signal_json.append({'time': t, 'value': float(r['macd_signal'])})
        if 'macd_hist' in r:
            macd_hist_json.append({'time': t, 'value': float(r['macd_hist'])})

    # Prepare trades markers
    markers_json = []
    if trades is not None and not trades.empty:
        for _, t_row in trades.iterrows():
            try:
                # Align trade time to the hour to match candle timestamps
                t_time = pd.to_datetime(t_row['timestamp'])
                t_time_rounded = t_time.round('H')
                t_unix = int(t_time_rounded.timestamp())
                
                side = t_row['side']
                p = float(t_row['price'])
                
                if side == 'BUY':
                    markers_json.append({
                        'time': t_unix,
                        'position': 'belowBar',
                        'color': '#00F2FE',
                        'shape': 'arrowUp',
                        'text': f"BUY @ ${p:,.2f}"
                    })
                else:
                    markers_json.append({
                        'time': t_unix,
                        'position': 'aboveBar',
                        'color': '#FF0055',
                        'shape': 'arrowDown',
                        'text': f"SELL @ ${p:,.2f}"
                    })
            except Exception:
                pass

    # Sort markers by time (required by TradingView Lightweight Charts)
    markers_json = sorted(markers_json, key=lambda x: x['time'])

    # Prepare active position targets (lines)
    active_pos_json = {}
    if active_pos:
        active_pos_json = {
            'entry': float(active_pos['price']),
            'sl': float(active_pos['stop_loss']),
            'tp': float(active_pos['take_profit'])
        }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background-color: transparent;
                overflow: hidden;
                font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            .chart-wrapper {{
                display: flex;
                flex-direction: column;
                gap: 6px;
                height: 100%;
                padding: 5px;
                box-sizing: border-box;
            }}
            .chart-container {{
                width: 100%;
                position: relative;
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 8px;
                overflow: hidden;
            }}
            .chart-label {{
                position: absolute;
                top: 8px;
                left: 10px;
                z-index: 10;
                color: #64748B;
                font-size: 10px;
                font-weight: 600;
                pointer-events: none;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
        </style>
    </head>
    <body>
        <div class="chart-wrapper">
            <div id="price-chart" class="chart-container" style="flex: 2.8; min-height: 240px;">
                <div class="chart-label">{asset} - Price & EMA 20</div>
            </div>
            <div id="rsi-chart" class="chart-container" style="flex: 1; min-height: 90px;">
                <div class="chart-label">RSI (14)</div>
            </div>
            <div id="macd-chart" class="chart-container" style="flex: 1.2; min-height: 110px;">
                <div class="chart-label">MACD Crossover</div>
            </div>
        </div>

        <script>
            const candleData = {json.dumps(candles_json)};
            const emaData = {json.dumps(ema_json)};
            const rsiData = {json.dumps(rsi_json)};
            const macdData = {json.dumps(macd_json)};
            const macdSignalData = {json.dumps(macd_signal_json)};
            const macdHistData = {json.dumps(macd_hist_json)};
            const markersData = {json.dumps(markers_json)};
            const activePos = {json.dumps(active_pos_json)};

            const theme = {{
                bg: '#07090e',
                text: '#94A3B8',
                grid: 'rgba(255, 255, 255, 0.015)',
                green: '#10B981',
                red: '#EF4444',
                blue: '#00F2FE',
                purple: '#D946EF',
                pink: '#EC4899',
                gold: '#F59E0B'
            }};

            const chartOptions = () => ({{
                layout: {{
                    background: {{ type: 'solid', color: '#07090e' }},
                    textColor: theme.text,
                    fontSize: 10,
                    fontFamily: 'Outfit, sans-serif'
                }},
                grid: {{
                    vertLines: {{ color: theme.grid }},
                    horzLines: {{ color: theme.grid }}
                }},
                timeScale: {{
                    borderColor: 'rgba(255, 255, 255, 0.03)',
                    timeVisible: true,
                    secondsVisible: false
                }},
                rightPriceScale: {{
                    borderColor: 'rgba(255, 255, 255, 0.03)',
                }},
                crosshair: {{
                    mode: 0,
                    vertLine: {{
                        color: 'rgba(255, 255, 255, 0.12)',
                        style: 3,
                        labelBackgroundColor: '#1E293B'
                    }},
                    horzLine: {{
                        color: 'rgba(255, 255, 255, 0.12)',
                        style: 3,
                        labelBackgroundColor: '#1E293B'
                    }}
                }}
            }});

            // 1. Create Price Chart
            const priceContainer = document.getElementById('price-chart');
            const priceChart = LightweightCharts.createChart(priceContainer, chartOptions());
            
            const candleSeries = priceChart.addCandlestickSeries({{
                upColor: theme.green,
                downColor: theme.red,
                borderUpColor: theme.green,
                borderDownColor: theme.red,
                wickUpColor: theme.green,
                wickDownColor: theme.red
            }});
            candleSeries.setData(candleData);

            // Add EMA 20
            const emaSeries = priceChart.addLineSeries({{
                color: theme.blue,
                lineWidth: 1.5,
                title: 'EMA 20'
            }});
            emaSeries.setData(emaData);

            // Add Markers (Trades)
            if (markersData.length > 0) {{
                candleSeries.setMarkers(markersData);
            }}

            // Add Active Position Target Lines
            if (activePos.entry) {{
                candleSeries.createPriceLine({{
                    price: activePos.entry,
                    color: '#00F2FE',
                    lineWidth: 1.5,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'ENTRY: $' + activePos.entry.toFixed(2)
                }});
                candleSeries.createPriceLine({{
                    price: activePos.sl,
                    color: '#FF0055',
                    lineWidth: 1.5,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'SL: $' + activePos.sl.toFixed(2)
                }});
                candleSeries.createPriceLine({{
                    price: activePos.tp,
                    color: '#10B981',
                    lineWidth: 1.5,
                    lineStyle: 2,
                    axisLabelVisible: true,
                    title: 'TP: $' + activePos.tp.toFixed(2)
                }});
            }}

            // 2. Create RSI Chart
            const rsiContainer = document.getElementById('rsi-chart');
            const rsiChart = LightweightCharts.createChart(rsiContainer, {{
                ...chartOptions(),
                timeScale: {{
                    ...chartOptions().timeScale,
                    visible: false
                }}
            }});
            const rsiSeries = rsiChart.addLineSeries({{
                color: theme.gold,
                lineWidth: 1.2,
                title: 'RSI'
            }});
            rsiSeries.setData(rsiData);
            
            // Add RSI bounds lines (30, 70)
            rsiSeries.createPriceLine({{ price: 30, color: theme.red, lineWidth: 1, lineStyle: 2, axisLabelVisible: true }});
            rsiSeries.createPriceLine({{ price: 70, color: theme.green, lineWidth: 1, lineStyle: 2, axisLabelVisible: true }});
            rsiChart.priceScale('right').applyOptions({{
                autoScale: false,
                scaleMargins: {{ top: 0.1, bottom: 0.1 }}
            }});

            // 3. Create MACD Chart
            const macdContainer = document.getElementById('macd-chart');
            const macdChart = LightweightCharts.createChart(macdContainer, chartOptions());
            
            const macdSeries = macdChart.addLineSeries({{
                color: '#3B82F6',
                lineWidth: 1.2,
                title: 'MACD'
            }});
            macdSeries.setData(macdData);
            
            const signalSeries = macdChart.addLineSeries({{
                color: '#EC4899',
                lineWidth: 1.2,
                title: 'Signal'
            }});
            signalSeries.setData(macdSignalData);

            const histSeries = macdChart.addHistogramSeries({{
                color: 'rgba(16, 185, 129, 0.4)',
                priceFormat: {{ type: 'volume' }},
                priceScaleId: 'overlay'
            }});
            
            const histData = macdHistData.map(d => ({{
                time: d.time,
                value: d.value,
                color: d.value >= 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
            }}));
            histSeries.setData(histData);

            // Synchronize timescales
            const priceTimeScale = priceChart.timeScale();
            const rsiTimeScale = rsiChart.timeScale();
            const macdTimeScale = macdChart.timeScale();

            let isSyncing = false;

            const syncTimescales = (sourceTimeScale, targetTimeScales) => {{
                sourceTimeScale.subscribeVisibleTimeRangeChange(() => {{
                    if (isSyncing) return;
                    isSyncing = true;
                    const range = sourceTimeScale.getVisibleRange();
                    targetTimeScales.forEach(ts => ts.setVisibleRange(range));
                    isSyncing = false;
                }});
            }};

            syncTimescales(priceTimeScale, [rsiTimeScale, macdTimeScale]);
            syncTimescales(rsiTimeScale, [priceTimeScale, macdTimeScale]);
            syncTimescales(macdTimeScale, [priceTimeScale, rsiTimeScale]);

            // Resize handler
            const resizeHandler = () => {{
                priceChart.resize(priceContainer.clientWidth, priceContainer.clientHeight);
                rsiChart.resize(rsiContainer.clientWidth, rsiContainer.clientHeight);
                macdChart.resize(macdContainer.clientWidth, macdContainer.clientHeight);
            }};
            window.addEventListener('resize', resizeHandler);
            setTimeout(resizeHandler, 100);
        </script>
    </body>
    </html>
    """
    return html_content

# Custom Premium Cyberpunk Glassmorphic Theme Injection
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background Override */
    .stApp {
        background: radial-gradient(circle at 80% 10%, #0d1527 0%, #05070c 100%) !important;
        color: #E2E8F0 !important;
    }
    
    /* Hide standard Streamlit decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.75) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem !important;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Interactive status banners */
    .glow-active-pos {
        background: rgba(239, 68, 68, 0.03) !important;
        border: 1.5px solid #FF0055 !important;
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.2) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.5rem !important;
        animation: pulse-red 2.5s infinite;
    }
    
    .glow-active-pos-green {
        background: rgba(16, 185, 129, 0.03) !important;
        border: 1.5px solid #10B981 !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.5rem !important;
        animation: pulse-green 2.5s infinite;
    }
    
    .glow-waiting {
        background: rgba(0, 242, 254, 0.03) !important;
        border: 1.5px solid #00F2FE !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.15) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    .glow-offline {
        background: rgba(239, 68, 68, 0.05) !important;
        border: 1.5px solid #EF4444 !important;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        margin-bottom: 1.5rem !important;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 5px rgba(255, 0, 85, 0.1); }
        50% { box-shadow: 0 0 20px rgba(255, 0, 85, 0.35); }
        100% { box-shadow: 0 0 5px rgba(255, 0, 85, 0.1); }
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.1); }
        50% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.35); }
        100% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.1); }
    }
    
    /* Glassmorphic Cards */
    .card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.35) !important;
    }
    
    .card-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Tabs Overrides */
    div[data-testid="stTabBar"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 4px !important;
        margin-bottom: 1.5rem !important;
    }
    
    button[data-baseweb="tab"] {
        color: #64748B !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    
    button[aria-selected="true"] {
        color: #00F2FE !important;
        background: rgba(0, 242, 254, 0.08) !important;
        font-weight: 600 !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.1) !important;
    }
    
    /* Form controls styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > input, div[data-baseweb="input"] > div > input, textarea {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        transition: all 0.2s ease;
    }
    
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > input:hover, textarea:hover {
        border-color: rgba(0, 242, 254, 0.3) !important;
    }
    
    /* Button Overrides */
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        color: #00F2FE !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(79, 172, 254, 0.2) 100%) !important;
        border-color: #00F2FE !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* Logs Panel styling */
    .log-container {
        background: rgba(5, 7, 12, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px;
        font-family: 'Consolas', monospace;
        padding: 1.2rem;
        height: 400px;
        overflow-y: auto;
        font-size: 0.82rem;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.6);
    }
    
    .log-line {
        margin-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.015);
        padding-bottom: 0.2rem;
    }
    
    .log-info { color: #94A3B8; }
    .log-warning { color: #F59E0B; }
    .log-success { color: #10B981; font-weight: 600; }
    .log-error { color: #EF4444; font-weight: 600; }
    .log-math { color: #A855F7; }
    
    /* JSON Display styling */
    .json-container {
        background: rgba(5, 7, 12, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px;
        padding: 1.2rem;
        color: #34D399;
        font-family: 'Consolas', monospace;
        font-size: 0.82rem;
        overflow-x: auto;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.6);
    }

    .report-card {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #1F2937;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00F2FE;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATIONS -----------------

# Language Selector at the very top of sidebar
if "lang" not in st.session_state:
    st.session_state["lang"] = "TR"

lang_opt = st.sidebar.selectbox(
    "🌐 Dil Seçimi / Language",
    options=["Türkçe", "English"],
    index=0 if st.session_state["lang"] == "TR" else 1
)
st.session_state["lang"] = "TR" if lang_opt == "Türkçe" else "EN"

st.sidebar.markdown(f"<h2 style='text-align: center; color: #FFFFFF;'>⚡ {t('title').split('|')[0].strip()}</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin-top: 0; margin-bottom: 1.5rem; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

# Gemini API Key Input
env_key = os.getenv("GEMINI_API_KEY", "")
db_key = get_config("gemini_api_key", "")
active_key = db_key if db_key else env_key

gemini_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=active_key,
    type="password",
    help=t("api_key_help")
)

if gemini_key != db_key:
    save_config("gemini_api_key", gemini_key)
    st.sidebar.success(t("api_key_updated"))

# Active Trading Assets Configuration
ALL_SUPPORTED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", 
    "NEAR/USDT", "MATIC/USDT", "SUI/USDT", "APT/USDT", "OP/USDT", 
    "ARB/USDT", "LTC/USDT", "TRX/USDT", "XLM/USDT", "UNI/USDT", 
    "ATOM/USDT", "INJ/USDT", "TIA/USDT", "GRT/USDT", "FET/USDT", 
    "RNDR/USDT", "SHIB/USDT", "ETC/USDT", "FIL/USDT", "ICP/USDT"
]
env_assets = os.getenv("SELECTED_ASSETS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,DOGE/USDT,AVAX/USDT,LINK/USDT,DOT/USDT")
db_assets = get_config("selected_assets", "")
active_assets_str = db_assets if db_assets else env_assets

asset_list = st.sidebar.multiselect(
    t("select_pairs"),
    options=ALL_SUPPORTED_PAIRS,
    default=[a.strip() for a in active_assets_str.split(",") if a.strip() in ALL_SUPPORTED_PAIRS],
    help=t("select_pairs_help")
)

if not asset_list:
    asset_list = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

if ",".join(asset_list) != db_assets:
    save_config("selected_assets", ",".join(asset_list))
    st.sidebar.info(t("active_pairs_updated"))

# Active Exchange Configuration
env_exchange = os.getenv("EXCHANGE_NAME", "binance")
db_exchange = get_config("exchange_name", "")
active_exchange = db_exchange if db_exchange else env_exchange

exchange_name_input = st.sidebar.text_input(
    t("exchange_name"),
    value=active_exchange,
    help=t("exchange_name_help")
)

if exchange_name_input != db_exchange:
    save_config("exchange_name", exchange_name_input)
    st.sidebar.info("Borsa tercihi güncellendi!" if st.session_state["lang"] == "TR" else "Exchange preference updated!")

# AI Models Settings
summarizer_model_default = os.getenv("SUMMARIZER_MODEL", "gemini-3.1-flash-lite")
sentiment_model_default = os.getenv("SENTIMENT_MODEL", "gemini-2.5-pro")

summarizer_model = st.sidebar.selectbox(
    t("sum_model"),
    options=["gemini-3.1-flash-lite","gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
    index=["gemini-3.1-flash-lite","gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"].index(get_config("summarizer_model", summarizer_model_default)),
    help=t("sum_model_help")
)

sentiment_model = st.sidebar.selectbox(
    t("sent_model"),
    options=["gemini-3.1-flash-lite","gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
    index=["gemini-3.1-flash-lite","gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"].index(get_config("sentiment_model", sentiment_model_default)),
    help=t("sent_model_help")
)

if summarizer_model != get_config("summarizer_model"):
    save_config("summarizer_model", summarizer_model)
if sentiment_model != get_config("sentiment_model"):
    save_config("sentiment_model", sentiment_model)

# Sizing & Safety Settings
risk_percentage = st.sidebar.slider(
    t("risk_pct"),
    min_value=1.0,
    max_value=5.0,
    value=float(get_config("risk_percentage", "2.0")),
    step=0.5,
    help=t("risk_pct_help")
)
if str(risk_percentage) != get_config("risk_percentage"):
    save_config("risk_percentage", risk_percentage)

stop_loss_pct = st.sidebar.slider(
    t("sl_pct"),
    min_value=1.0,
    max_value=10.0,
    value=float(get_config("stop_loss_pct", "3.0")),
    step=0.5,
    help=t("sl_pct_help")
)
if str(stop_loss_pct) != get_config("stop_loss_pct"):
    save_config("stop_loss_pct", stop_loss_pct)

take_profit_pct = st.sidebar.slider(
    t("tp_pct"),
    min_value=2.0,
    max_value=20.0,
    value=float(get_config("take_profit_pct", "6.0")),
    step=0.5,
    help=t("tp_pct_help")
)
if str(take_profit_pct) != get_config("take_profit_pct"):
    save_config("take_profit_pct", take_profit_pct)

sentiment_threshold = st.sidebar.slider(
    t("min_sentiment"),
    min_value=1,
    max_value=10,
    value=int(get_config("min_ai_sentiment_threshold", "3")),
    help=t("min_sentiment_help")
)
if str(sentiment_threshold) != get_config("min_ai_sentiment_threshold"):
    save_config("min_ai_sentiment_threshold", sentiment_threshold)

# Action Operations Area
st.sidebar.markdown(f"### {t('quick_actions')}")

col_action1, col_action2 = st.sidebar.columns(2)

with col_action1:
    if st.button(t("sync_tick"), width="stretch", help=t("sync_tick_help")):
        with st.spinner("Processing analysis..." if st.session_state["lang"] == "EN" else "Analiz yürütülüyor..."):
            try:
                run_worker_cycle(force=True)
                st.toast(t("sync_success"), icon="✅")
                time_now = datetime.now().strftime("%H:%M:%S")
                st.success(f"Tick completed at {time_now}" if st.session_state["lang"] == "EN" else f"Tick tamamlandı: {time_now}")
                st.rerun()
            except Exception as e:
                st.error(f"Sync failed: {e}")

with col_action2:
    if st.button(t("reset_port"), width="stretch", help=t("reset_port_help")):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio")
            cursor.execute("INSERT INTO portfolio (asset, balance, avg_entry_price) VALUES ('USD', 10000.0, 0.0)")
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM ai_runs")
            cursor.execute("DELETE FROM candles")
            conn.commit()
            conn.close()
            log_event("WARNING", "UI_DASHBOARD", "Portfolio balance and trade database reset by user.")
            st.toast(t("reset_success"), icon="🔄")
            st.success("State reset successful." if st.session_state["lang"] == "EN" else "Durum başarıyla sıfırlandı.")
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

# Reset Settings to Optimal Defaults Button
if st.sidebar.button(t("reset_settings"), width="stretch", help=t("reset_settings_help")):
    try:
        # Optimal default settings calculated for 1H candle, Binance spot, moderate risk
        optimal_defaults = {
            "risk_percentage": "2.0",       # 2% NAV per trade — conservative position sizing
            "stop_loss_pct": "3.0",         # 3% SL — tight enough to limit drawdown, wide enough to avoid noise
            "take_profit_pct": "6.0",       # 6% TP — 2:1 reward/risk ratio (industry standard minimum)
            "min_ai_sentiment_threshold": "3",  # Score >= 3 out of 10 — filters noise but allows moderate signals
            "fee_pct": "0.001",             # 0.1% — Binance default spot fee (VIP0 maker/taker)
            "summarizer_model": "gemini-2.5-flash",
            "sentiment_model": "gemini-2.5-pro",
            "live_mode": "false",
        }
        for key, value in optimal_defaults.items():
            save_config(key, value)
        log_event("INFO", "UI_DASHBOARD", "All settings reset to optimal defaults by user.")
        st.toast(t("reset_settings_success"), icon="⚙️")
        st.rerun()
    except Exception as e:
        st.error(f"Reset settings failed: {e}")

# Auto-Refresh Control Section
st.sidebar.markdown(f"### {t('auto_refresh')}")

# Initialize session state for auto-refresh if not set
if "auto_refresh_enabled" not in st.session_state:
    st.session_state["auto_refresh_enabled"] = False
if "auto_refresh_interval" not in st.session_state:
    st.session_state["auto_refresh_interval"] = 30

enable_auto_refresh = st.sidebar.checkbox(
    t("auto_refresh"), 
    value=st.session_state["auto_refresh_enabled"],
    key="auto_refresh_checkbox",
    help=t("auto_refresh_help")
)
st.session_state["auto_refresh_enabled"] = enable_auto_refresh

# Manual Refresh Button
_refresh_btn_label = "Refresh Dashboard" if st.session_state.get("lang", "TR") == "EN" else "Paneli Yenile"
if st.sidebar.button(_refresh_btn_label, width="stretch", key="dashboard_refresh_btn"):
    st.rerun()

if enable_auto_refresh:
    refresh_interval = st.sidebar.slider(
        t("refresh_interval"),
        min_value=10,
        max_value=120,
        value=st.session_state["auto_refresh_interval"],
        step=5,
        key="auto_refresh_slider"
    )
    st.session_state["auto_refresh_interval"] = refresh_interval
    
    # Use streamlit-autorefresh for a clean, CORS-safe, soft page rerun
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_interval * 1000, key="dashboard_autorefresh")

if ACCESS_PASSWORD and st.session_state.get("authenticated", False):
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    if st.sidebar.button("🔓 Log Out / Çıkış Yap", width="stretch", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.rerun()

st.sidebar.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
_now_formatted = datetime.now(ISTANBUL_TZ).strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(
    "<p style='text-align: center; color: #64748B; font-size: 0.8rem;'>"
    "Sentix SaaS Platform v1.2.0<br>"
    "Dual-Loop SL/TP Guardian Active<br>"
    f"{t('last_update_label')}: {_now_formatted} UTC+3"
    "</p>", 
    unsafe_allow_html=True
)

# ----------------- MAIN DOCK ARCHITECTURE -----------------

# Page Header
st.markdown("<h1 class='title-gradient'>⚡ Sentix</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{t('subtitle')}</p>", unsafe_allow_html=True)

# Current Time & Last Data Update Bar
_last_data_update = "—"
try:
    _conn = get_connection()
    _cur = _conn.cursor()
    _cur.execute("SELECT MAX(timestamp) FROM candles")
    _row = _cur.fetchone()
    _conn.close()
    if _row and _row[0]:
        _ts_str = str(_row[0])
        try:
            # Candles table timestamp is saved in UTC format: %Y-%m-%d %H:%M:%S
            if "T" in _ts_str:
                _ts_str = _ts_str.replace("T", " ")
            if "Z" in _ts_str:
                _ts_str = _ts_str.replace("Z", "")
            if "." in _ts_str:
                _ts_str = _ts_str.split(".")[0]
            
            _dt_utc = datetime.strptime(_ts_str, "%Y-%m-%d %H:%M:%S")
            _dt_utc = _dt_utc.replace(tzinfo=timezone.utc)
            _dt_istanbul = _dt_utc.astimezone(ISTANBUL_TZ)
            _last_data_update = _dt_istanbul.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            _last_data_update = _ts_str
except Exception:
    pass

# Translations and states for HTML
_clock_label = "Saat" if st.session_state.get("lang", "TR") == "TR" else "Time"
_data_label = "Veri Güncelleme" if st.session_state.get("lang", "TR") == "TR" else "Data Update"
_auto_label = "Otomatik Yenileme Aktif" if st.session_state.get("lang", "TR") == "TR" else "Auto-Refresh Active"
_auto_status_html = ""
if st.session_state.get("auto_refresh_enabled", False):
    _interval = st.session_state.get("auto_refresh_interval", 30)
    _auto_status_html = f'<div class="badge">🔄 {_auto_label} ({_interval}s)</div>'

# Render live-ticking client-side clock inside an iframe to prevent server-side rerun overhead
st.iframe(
    f"""
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            color: #E2E8F0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 0.82rem;
            overflow: hidden;
        }}
        .bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(17, 24, 39, 0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 8px 16px;
            box-sizing: border-box;
            height: 38px;
        }}
        .left {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .badge {{
            background: rgba(0, 242, 254, 0.15);
            color: #00F2FE;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 500;
        }}
        .clock-container {{
            color: #94A3B8;
        }}
        .clock-val {{
            color: #E2E8F0;
            font-weight: 600;
        }}
        .date-val {{
            color: #64748B;
            font-weight: normal;
        }}
        .data-container {{
            color: #94A3B8;
        }}
        .data-val {{
            color: #00F2FE;
            font-weight: 600;
        }}
    </style>
    <div class="bar">
        <div class="left">
            <span class="clock-container">🕐 {_clock_label}: <span id="clock" class="clock-val">--:--:--</span></span>
            <span class="data-container">📊 {_data_label}: <span class="data-val">{_last_data_update}</span></span>
        </div>
        {_auto_status_html}
    </div>
    <script>
        function updateClock() {{
            var now = new Date();
            var utc = now.getTime() + (now.getTimezoneOffset() * 60000);
            // Istanbul is UTC+3
            var istanbulTime = new Date(utc + (3600000 * 3));
            
            var hh = String(istanbulTime.getHours()).padStart(2, '0');
            var mm = String(istanbulTime.getMinutes()).padStart(2, '0');
            var ss = String(istanbulTime.getSeconds()).padStart(2, '0');
            
            var dd = String(istanbulTime.getDate()).padStart(2, '0');
            var mo = String(istanbulTime.getMonth() + 1).padStart(2, '0');
            var yyyy = istanbulTime.getFullYear();
            
            var clockEl = document.getElementById('clock');
            if (clockEl) {{
                clockEl.innerHTML = hh + ':' + mm + ':' + ss + ' <span class="date-val">(' + dd + '.' + mo + '.' + yyyy + ')</span>';
            }}
        }}
        setInterval(updateClock, 1000);
        updateClock();
    </script>
    """,
    height=40,
)

# Safety check for active asset list
if not asset_list:
    st.warning("⚠️ Please select at least one active trading pair in the Config Panel sidebar.")
    st.stop()

# ----------------- PORTFOLIO CALCULATIONS -----------------
portfolio = get_portfolio()
trades = get_trades(limit=50)
logs = get_logs(limit=100)
active_positions_list = get_all_active_positions()
total_nav = calculate_total_nav(portfolio)
usd_bal = portfolio.get("USD", {}).get("balance", 10000.0)
profit_pct = ((total_nav - 10000.0) / 10000.0) * 100

# Check if background worker is active
worker_heartbeat_str = get_config("worker_heartbeat")
is_worker_active = False
if worker_heartbeat_str:
    try:
        from datetime import timezone
        heartbeat_time = datetime.fromisoformat(worker_heartbeat_str)
        # Use timezone-aware comparison to avoid deprecation warnings
        if heartbeat_time.tzinfo is None:
            time_diff = (datetime.utcnow() - heartbeat_time).total_seconds()
        else:
            time_diff = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
        # Increased threshold to 180s to prevent false offline alerts during long analysis runs
        if time_diff <= 180:
            is_worker_active = True
    except Exception:
        pass

# ----------------- STATEFUL HEADER STATUS BADGE -----------------
# 1. Show warning if bot is offline
if not is_worker_active:
    st.markdown(f"""
    <div class="glow-offline">
        <h3 style="margin: 0; color: #EF4444; font-size: 1.15rem; font-weight: 700;">
            {t("bot_offline")}
        </h3>
        <p style="margin: 0.3rem 0 0 0; color: #E2E8F0; font-size: 0.85rem;">
            {t("bot_offline_desc")}
        </p>
    </div>
    """, unsafe_allow_html=True)

# 2. Show active positions PnL cards
if active_positions_list:
    # Build glowing status indicator for all active holdings
    status_lines = []
    for pos in active_positions_list:
        ticker = pos['asset'].split("/")[0]
        entry = pos['price']
        qty = pos['amount']
        sl = pos['stop_loss']
        tp = pos['take_profit']
        
        # Look up price in candles
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT close FROM candles WHERE asset = ? ORDER BY timestamp DESC LIMIT 1", (pos['asset'],))
        crow = cursor.fetchone()
        conn.close()
        live_price = float(crow[0]) if crow else entry
        
        real_pnl_pct = ((live_price - entry) / entry) * 100
        pnl_class = "glow-active-pos-green" if real_pnl_pct >= 0 else "glow-active-pos"
        arrow = "▲" if real_pnl_pct >= 0 else "▼"
        status_lines.append(f"""
        <div class="{pnl_class}">
            <h3 style="margin: 0; color: #FFFFFF; font-size: 1.15rem; font-weight: 700;">
                {t("in_pos", asset=pos['asset'], arrow=arrow, pnl=real_pnl_pct)}
            </h3>
            <p style="margin: 0.3rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                {t("pos_details", entry=entry, live=live_price, sl=sl, tp=tp)}
            </p>
        </div>
        """)
    st.markdown("".join(status_lines), unsafe_allow_html=True)
elif is_worker_active:
    # Bot is active, no active positions
    st.markdown(f"""
    <div class="glow-waiting">
        <h3 style="margin: 0; color: #00F2FE; font-size: 1.15rem; font-weight: 700;">
            {t("bot_active")}
        </h3>
        <p style="margin: 0.3rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
            {t("bot_active_desc")}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- TABS CREATION -----------------
tab_a, tab_b, tab_c = st.tabs([
    t("tab_a"), 
    t("tab_b"), 
    t("tab_c")
])

# ==========================================
#          TAB A: CANLI TAKIP PANELI
# ==========================================
with tab_a:
    # 4 Card metrics grid
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💵 {t("free_usd")}</div>
            <div class="card-value">${usd_bal:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💼 {t("portfolio_nav")}</div>
            <div class="card-value">${total_nav:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m3:
        p_badge = "badge-positive" if profit_pct >= 0 else "badge-negative"
        sign = "+" if profit_pct >= 0 else ""
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📈 {t("net_return")}</div>
            <div class="card-value">
                {sign}{profit_pct:.2f}%
                <span class="badge {p_badge}">{sign}{profit_pct:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m4:
        active_cnt = len(active_positions_list)
        protection_text = "GÜVENLİK AKTİF" if st.session_state["lang"] == "TR" else "PROTECTION ACTIVE"
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🛡️ {t("open_pos_count")}</div>
            <div class="card-value">
                {active_cnt} / {len(asset_list)}
                <span class="badge badge-neutral">{protection_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Active Asset Holdings Grid
    asset_values_breakdown = {}
    for asset_key, data in portfolio.items():
        if asset_key == "USD" or data["balance"] <= 0.0001:
            continue
            
        pair_name = f"{asset_key}/USDT"
        conn_val = get_connection()
        c_val = conn_val.cursor()
        c_val.execute("SELECT close FROM candles WHERE asset = ? ORDER BY timestamp DESC LIMIT 1", (pair_name,))
        c_row = c_val.fetchone()
        conn_val.close()
        
        val_price = float(c_row[0]) if c_row else data["avg_entry_price"]
        holding_value = data["balance"] * val_price
        asset_values_breakdown[asset_key] = {
            "balance": data["balance"],
            "value": holding_value,
            "avg_entry": data["avg_entry_price"],
            "live_price": val_price
        }

    if asset_values_breakdown:
        st.markdown(f"### {t('active_holdings')}")
        holding_cols = st.columns(len(asset_values_breakdown))
        for idx, (ticker, details) in enumerate(asset_values_breakdown.items()):
            with holding_cols[idx]:
                pnl_val = (details['live_price'] - details['avg_entry']) * details['balance']
                pnl_pct = ((details['live_price'] - details['avg_entry']) / details['avg_entry'] * 100) if details['avg_entry'] > 0 else 0.0
                
                pnl_color = "color: #10B981;" if pnl_val >= 0 else "color: #EF4444;"
                pnl_text = f"+${pnl_val:,.2f} (+{pnl_pct:.2f}%)" if pnl_val >= 0 else f"-${abs(pnl_val):,.2f} ({pnl_pct:.2f}%)"
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.2rem; text-align: center;">
                    <h4 style="margin: 0; color: #94A3B8; font-size: 0.85rem; font-weight: 500;">{ticker} {t('balance')}</h4>
                    <p style="margin: 0.4rem 0 0 0; font-size: 1.6rem; font-weight: 700; color: #FFFFFF;">{details['balance']:.4f}</p>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #64748B;">{t('cost')}: ${details['avg_entry']:,.2f} | {t('size')}: ${details['value']:,.2f}</p>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem; {pnl_color} font-weight: 600;">{pnl_text}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Chart overlay Selection
    selected_asset = st.selectbox(
        t("analyze_on_chart"),
        options=asset_list,
        key="main_chart_pair"
    )

    # Load candles for chart
    candles = get_latest_candles(selected_asset, "1h", limit=120)
    
    if not candles:
        st.info("💡 Bu asset için veritabanında veri bulunmuyor. Sidebar'daki '🚀 Sync Tick' butonuna tıklayarak verileri indirebilirsiniz.")
    else:
        # Load past trades for markers
        db_trades = get_connection()
        c_trades = db_trades.cursor()
        c_trades.execute("SELECT * FROM trades WHERE asset = ? ORDER BY timestamp ASC", (selected_asset,))
        raw_trades = c_trades.fetchall()
        db_trades.close()
        
        trades_df = pd.DataFrame([dict(r) for r in raw_trades]) if raw_trades else pd.DataFrame()
        active_pos_for_chart = get_active_position(selected_asset)

        # Generate TradingView Lightweight Charts components
        html_code = get_tradingview_html(selected_asset, candles, trades_df, active_pos_for_chart)
        
        st.markdown(f"<h4 style='text-align: center; color: #FFFFFF; font-family: Outfit, sans-serif; margin-bottom: 0.5rem;'>📈 {t('chart_title', asset=selected_asset)}</h4>", unsafe_allow_html=True)
        st.iframe(src=html_code, height=530)

    # ----------------- CROSSOVER BYPASS INSPECTOR -----------------
    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)
    st.markdown(f"### 📊 {t('crossover_inspector')}")
    st.markdown(f"<p style='color: #94A3B8; font-size: 0.85rem;'>{t('crossover_inspector_desc')}</p>", unsafe_allow_html=True)
    
    rows_html = ""
    for asset in asset_list:
        # Load latest candles from DB
        asset_candles = get_latest_candles(asset, "1h", limit=50)
        
        if not asset_candles or len(asset_candles) < 5:
            no_data_str = "Sync Gerekli" if st.session_state["lang"] == "TR" else "Sync Required"
            rows_html += f"""
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 10px; font-weight: 600; color: #FFFFFF;">{asset}</td>
                <td style="padding: 10px; color: #94A3B8;">N/A</td>
                <td style="padding: 10px; color: #94A3B8;">{no_data_str}</td>
                <td style="padding: 10px; color: #94A3B8;">N/A</td>
                <td style="padding: 10px; color: #94A3B8;">N/A</td>
                <td style="padding: 10px;"><span class="badge badge-neutral">{no_data_str}</span></td>
            </tr>
            """
            continue
            
        # Convert to DataFrame and compute indicators
        df_ast = pd.DataFrame(asset_candles)
        df_ast = calculate_indicators(df_ast)
        
        # Extract latest completed and previous completed candle
        c_curr = df_ast.iloc[-2]
        c_prev = df_ast.iloc[-3]
        
        close_c = float(c_curr["close"])
        close_p = float(c_prev["close"])
        
        rsi_val = c_curr.get("rsi")
        rsi_prev = c_prev.get("rsi")
        macd_val = c_curr.get("macd")
        macd_prev = c_prev.get("macd")
        sig_val = c_curr.get("macd_signal")
        sig_prev = c_prev.get("macd_signal")
        ema_val = c_curr.get("ema")
        ema_prev = c_prev.get("ema")
        
        # 1. RSI status text
        if rsi_val is not None:
            if rsi_val <= 30:
                rsi_str = f"<span style='color: #10B981; font-weight: 600;'>🟢 {rsi_val:.2f} ({'Aşırı Satım' if st.session_state['lang'] == 'TR' else 'Oversold'})</span>"
            elif rsi_val >= 70:
                rsi_str = f"<span style='color: #EF4444; font-weight: 600;'>🔴 {rsi_val:.2f} ({'Aşırı Alım' if st.session_state['lang'] == 'TR' else 'Overbought'})</span>"
            else:
                rsi_str = f"<span style='color: #94A3B8;'>⚪ {rsi_val:.2f} ({'Nötr' if st.session_state['lang'] == 'TR' else 'Neutral'})</span>"
        else:
            rsi_str = "N/A"
            
        # 2. MACD crossover status
        if macd_val is not None and sig_val is not None:
            macd_diff = macd_val - sig_val
            macd_bull_cross = (macd_prev <= sig_prev) and (macd_val > sig_val)
            macd_bear_cross = (macd_prev >= sig_prev) and (macd_val < sig_val)
            
            if macd_bull_cross:
                macd_str = f"<span style='color: #10B981; font-weight: 600;'>🟢 {'Boğa Kesişimi' if st.session_state['lang'] == 'TR' else 'Bullish Cross'}</span>"
            elif macd_bear_cross:
                macd_str = f"<span style='color: #EF4444; font-weight: 600;'>🔴 {'Ayı Kesişimi' if st.session_state['lang'] == 'TR' else 'Bearish Cross'}</span>"
            else:
                macd_str = f"<span style='color: #94A3B8;'>⚪ {'Nötr' if st.session_state['lang'] == 'TR' else 'Neutral'} ({macd_diff:+.4f})</span>"
        else:
            macd_str = "N/A"
            
        # 3. EMA status text
        if ema_val is not None:
            ema_bull_cross = (close_p <= ema_prev) and (close_c > ema_val)
            ema_bear_cross = (close_p >= ema_prev) and (close_c < ema_val)
            
            if ema_bull_cross:
                ema_str = f"<span style='color: #10B981; font-weight: 600;'>🟢 {'Yukarı Kırılım' if st.session_state['lang'] == 'TR' else 'Cross Above'}</span>"
            elif ema_bear_cross:
                ema_str = f"<span style='color: #EF4444; font-weight: 600;'>🔴 {'Altına Kırılım' if st.session_state['lang'] == 'TR' else 'Cross Below'}</span>"
            elif close_c > ema_val:
                ema_str = f"<span style='color: #10B981;'>📈 {'EMA Üzerinde' if st.session_state['lang'] == 'TR' else 'Above EMA'}</span>"
            else:
                ema_str = f"<span style='color: #EF4444;'>📉 {'EMA Altında' if st.session_state['lang'] == 'TR' else 'Below EMA'}</span>"
        else:
            ema_str = "N/A"
            
        # 4. Check active position and triggers
        active_pos_check = get_active_position(asset)
        trigger_side, trigger_reason = check_triggers(df_ast)
        
        if active_pos_check:
            decision_badge = f"<span class='badge' style='background-color: rgba(148,163,184,0.15); color: #94A3B8; border: 1px solid rgba(148,163,184,0.2);'>{t('decision_in_pos')}</span>"
        elif trigger_side == "BUY":
            decision_badge = f"<span class='badge' style='background-color: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.2); font-weight: 600;'>{t('decision_buy')}</span>"
        elif trigger_side == "SELL":
            decision_badge = f"<span class='badge' style='background-color: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.2); font-weight: 600;'>{t('decision_sell')}</span>"
        else:
            decision_badge = f"<span class='badge badge-neutral'>{t('decision_neutral')}</span>"
            
        rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 10px; font-weight: 600; color: #FFFFFF;">{asset}</td>
            <td style="padding: 10px; font-weight: 500; color: #E2E8F0;">${close_c:,.2f}</td>
            <td style="padding: 10px;">{rsi_str}</td>
            <td style="padding: 10px;">{macd_str}</td>
            <td style="padding: 10px;">{ema_str}</td>
            <td style="padding: 10px;">{decision_badge}</td>
        </tr>
        """
        
    # Calculate dynamic height based on number of rows (each row ~45px + header ~50px + footer ~50px + padding)
    table_row_count = len(asset_list)
    table_height = 50 + (table_row_count * 45) + 60 + 40  # header + rows + footer + padding
    
    inspector_table = f"""
    <div style="background: rgba(17, 24, 39, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.2rem; font-family: 'Outfit', sans-serif;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; color: #E2E8F0;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #94A3B8; font-weight: 600;">
                    <th style="padding: 10px;">{t('col_asset')}</th>
                    <th style="padding: 10px;">{t('col_price')}</th>
                    <th style="padding: 10px;">{t('col_rsi')}</th>
                    <th style="padding: 10px;">{t('col_macd')}</th>
                    <th style="padding: 10px;">{t('col_ema')}</th>
                    <th style="padding: 10px;">{t('col_decision')}</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <p style="font-size: 0.8rem; color: #64748B; margin-top: 1rem; margin-bottom: 0;">{t('indicator_explanation')}</p>
    </div>
    """
    # Use st.html() instead of st.markdown() — Streamlit 1.57.0 strips <table> from markdown
    st.html(inspector_table)

    # Detailed Indicator Signal Checklist Section
    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)
    st.markdown(f"### {t('checklist_title')}")
    
    col_chk_sel, _ = st.columns([3, 1])
    with col_chk_sel:
        chk_asset = st.selectbox(t('checklist_select_asset'), options=asset_list, key="chk_asset_sel")
        
    chk_candles = get_latest_candles(chk_asset, "1h", limit=50)
    if not chk_candles or len(chk_candles) < 5:
        st.info("💡 Bu asset için veritabanında yeterli analiz verisi yok." if st.session_state["lang"] == "TR" else "💡 Insufficient historical candle data for checklist evaluation.")
    else:
        df_chk = pd.DataFrame(chk_candles)
        df_chk = calculate_indicators(df_chk)
        
        c_curr = df_chk.iloc[-2]
        c_prev = df_chk.iloc[-3]
        
        close_c = float(c_curr["close"])
        close_p = float(c_prev["close"])
        
        rsi_val = float(c_curr['rsi']) if (c_curr.get('rsi') is not None and not pd.isna(c_curr.get('rsi'))) else None
        rsi_prev = float(c_prev['rsi']) if (c_prev.get('rsi') is not None and not pd.isna(c_prev.get('rsi'))) else None
        macd_val = float(c_curr['macd']) if (c_curr.get('macd') is not None and not pd.isna(c_curr.get('macd'))) else None
        macd_prev = float(c_prev['macd']) if (c_prev.get('macd') is not None and not pd.isna(c_prev.get('macd'))) else None
        sig_val = float(c_curr['macd_signal']) if (c_curr.get('macd_signal') is not None and not pd.isna(c_curr.get('macd_signal'))) else None
        sig_prev = float(c_prev['macd_signal']) if (c_prev.get('macd_signal') is not None and not pd.isna(c_prev.get('macd_signal'))) else None
        ema_val = float(c_curr['ema']) if (c_curr.get('ema') is not None and not pd.isna(c_curr.get('ema'))) else None
        ema_prev = float(c_prev['ema']) if (c_prev.get('ema') is not None and not pd.isna(c_prev.get('ema'))) else None
        
        active_pos_check = get_active_position(chk_asset)
        trigger_side, trigger_reason = check_triggers(df_chk)
        
        # Rules evaluation
        rsi_buy_met = (rsi_val is not None) and (rsi_val <= 30 or (rsi_prev is not None and rsi_prev < 30 and rsi_val >= 30))
        macd_buy_met = (macd_val is not None and sig_val is not None and macd_prev is not None and sig_prev is not None) and (macd_prev <= sig_prev and macd_val > sig_val)
        ema_buy_met = (ema_val is not None and ema_prev is not None) and (close_p <= ema_prev and close_c > ema_val)
        pos_buy_met = active_pos_check is None

        rsi_sell_met = (rsi_val is not None) and (rsi_val >= 70 or (rsi_prev is not None and rsi_prev > 70 and rsi_val <= 70))
        macd_sell_met = (macd_val is not None and sig_val is not None and macd_prev is not None and sig_prev is not None) and (macd_prev >= sig_prev and macd_val < sig_val)
        ema_sell_met = (ema_val is not None and ema_prev is not None) and (close_p >= ema_prev and close_c < ema_val)
        pos_sell_met = active_pos_check is not None
        
        # Pre-format values
        rsi_c_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
        rsi_p_str = f"{rsi_prev:.2f}" if rsi_prev is not None else "N/A"
        macd_c_str = f"{macd_val:.4f}" if macd_val is not None else "N/A"
        sig_c_str = f"{sig_val:.4f}" if sig_val is not None else "N/A"
        macd_p_str = f"{macd_prev:.4f}" if macd_prev is not None else "N/A"
        sig_p_str = f"{sig_prev:.4f}" if sig_prev is not None else "N/A"
        ema_c_str = f"${ema_val:,.2f}" if ema_val is not None else "N/A"
        ema_p_str = f"${ema_prev:,.2f}" if ema_prev is not None else "N/A"
        close_c_str = f"${close_c:,.2f}"
        close_p_str = f"${close_p:,.2f}"
        
        with st.container():
            st.markdown(f"<div style='background: rgba(17, 24, 39, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 1.2rem;'>{t('checklist_desc')}</p>", unsafe_allow_html=True)
            
            # Display AI Status banner inside expander
            if trigger_side:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10B981; border-radius: 8px; padding: 0.8rem; margin-bottom: 1.2rem;">
                    <span style="color: #10B981; font-weight: bold; font-size: 0.9rem;">🤖 {t('ai_status_label')}:</span>
                    <span style="color: #FFFFFF; font-size: 0.9rem; margin-left: 5px;">{t('ai_status_waking')}</span>
                    <p style="margin: 0.3rem 0 0 0; color: #94A3B8; font-size: 0.8rem;">Trigger: {trigger_reason}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(148, 163, 184, 0.08); border: 1px solid #64748B; border-radius: 8px; padding: 0.8rem; margin-bottom: 1.2rem;">
                    <span style="color: #94A3B8; font-weight: bold; font-size: 0.9rem;">🤖 {t('ai_status_label')}:</span>
                    <span style="color: #E2E8F0; font-size: 0.9rem; margin-left: 5px;">{t('ai_status_sleeping')}</span>
                </div>
                """, unsafe_allow_html=True)

            col_chk1, col_chk2 = st.columns(2)
            
            with col_chk1:
                st.markdown(f"<h5 style='color: #10B981; margin-bottom: 0.8rem;'>{t('chk_buy_rules')}</h5>", unsafe_allow_html=True)
                
                # 1. RSI Buy
                chk_icon = "🟢" if rsi_buy_met else "⚪"
                lbl_met = t('rule_met') if rsi_buy_met else t('rule_not_met')
                st.markdown(f"{chk_icon} **RSI:** {t('chk_rsi_buy', rsi_c=rsi_c_str, rsi_p=rsi_p_str)} <span style='font-size:0.8rem; color:{'#10B981' if rsi_buy_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span>", unsafe_allow_html=True)
                
                # 2. MACD Buy
                chk_icon = "🟢" if macd_buy_met else "⚪"
                lbl_met = t('rule_met') if macd_buy_met else t('rule_not_met')
                st.markdown(f"{chk_icon} **MACD:** {t('chk_macd_buy', macd_c=macd_c_str, sig_c=sig_c_str, macd_p=macd_p_str, sig_p=sig_p_str)} <span style='font-size:0.8rem; color:{'#10B981' if macd_buy_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span>", unsafe_allow_html=True)
                
                # 3. EMA Buy
                chk_icon = "🟢" if ema_buy_met else "⚪"
                lbl_met = t('rule_met') if ema_buy_met else t('rule_not_met')
                st.markdown(f"{chk_icon} **EMA 20:** {t('chk_ema_buy', close_c=close_c_str, ema_c=ema_c_str, close_p=close_p_str, ema_p=ema_p_str)} <span style='font-size:0.8rem; color:{'#10B981' if ema_buy_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span>", unsafe_allow_html=True)
                
                # 4. Position Buy
                chk_icon = "🟢" if pos_buy_met else "⚪"
                lbl_met = t('rule_met') if pos_buy_met else t('rule_not_met')
                pos_desc = t('chk_active_pos_no') if pos_buy_met else t('chk_active_pos_yes', entry=active_pos_check['price'], live=close_c)
                st.markdown(f"{chk_icon} **{t('chk_pos_buy')}** <span style='font-size:0.8rem; color:{'#10B981' if pos_buy_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span><br><span style='font-size:0.8rem; color:#94A3B8;'>{pos_desc}</span>", unsafe_allow_html=True)
                
            with col_chk2:
                st.markdown(f"<h5 style='color: #EF4444; margin-bottom: 0.8rem;'>{t('chk_sell_rules')}</h5>", unsafe_allow_html=True)
                
                # 1. RSI Sell
                chk_icon = "🔴" if rsi_sell_met else "⚪"
                lbl_met = t('rule_met') if rsi_sell_met else t('rule_not_met')
                st.markdown(f"{chk_icon} **RSI:** {t('chk_rsi_sell', rsi_c=rsi_c_str, rsi_p=rsi_p_str)} <span style='font-size:0.8rem; color:{'#10B981' if rsi_sell_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span>", unsafe_allow_html=True)
                
                # 2. MACD Sell
                chk_icon = "🔴" if macd_sell_met else "⚪"
                lbl_met = t('rule_met') if macd_sell_met else t('rule_not_met')
                st.markdown(f"{chk_icon} **MACD:** {t('chk_macd_sell', macd_c=macd_c_str, sig_c=sig_c_str, macd_p=macd_p_str, sig_p=sig_p_str)} <span style='font-size:0.8rem; color:{'#10B981' if macd_sell_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span>", unsafe_allow_html=True)
                
                # 3. EMA Sell
                chk_icon = "🔴" if ema_sell_met else "⚪"
                lbl_met = t('rule_met') if ema_sell_met else t('rule_not_met')
                st.markdown(f"{chk_icon} **EMA 20:** {t('chk_ema_sell', close_c=close_c_str, ema_c=ema_c_str, close_p=close_p_str, ema_p=ema_p_str)} <span style='font-size:0.8rem; color:{'#10B981' if ema_sell_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span>", unsafe_allow_html=True)
                
                # 4. Position Sell
                chk_icon = "🔴" if pos_sell_met else "⚪"
                lbl_met = t('rule_met') if pos_sell_met else t('rule_not_met')
                pos_desc = t('chk_active_pos_yes', entry=active_pos_check['price'], live=close_c) if pos_sell_met else t('chk_active_pos_no')
                st.markdown(f"{chk_icon} **{t('chk_pos_sell')}** <span style='font-size:0.8rem; color:{'#10B981' if pos_sell_met else '#EF4444'}; font-weight:600;'>({lbl_met})</span><br><span style='font-size:0.8rem; color:#94A3B8;'>{pos_desc}</span>", unsafe_allow_html=True)
                
            st.markdown(f"</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.05);'>", unsafe_allow_html=True)

    # Simulated Trade History Table
    st.markdown(f"### {t('recent_trades_sim')}")
    if not trades:
        st.info(t("no_trades_yet"))
    else:
        trades_rows = []
        for t_row in trades:
            try:
                t_parsed = datetime.fromisoformat(t_row['timestamp'])
                if t_parsed.tzinfo is None:
                    t_parsed = t_parsed.replace(tzinfo=timezone.utc)
                t_parsed = t_parsed.astimezone(ISTANBUL_TZ)
                t_time = t_parsed.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                t_time = t_row['timestamp'][:19].replace('T', ' ')
            
            val = t_row['price'] * t_row['amount']
            pnl_val = f"{t_row['pnl']:+.2f}%" if t_row['pnl'] is not None else ("Açık Pozisyon" if st.session_state["lang"] == "TR" else "Open Position")
            trades_rows.append({
                t("col_time"): t_time,
                t("col_pair"): t_row['asset'],
                t("col_side"): t_row['side'],
                t("col_price"): f"${t_row['price']:,.2f}",
                t("col_amount"): f"{t_row['amount']:.4f}",
                t("col_value"): f"${val:,.2f}",
                t("col_sl"): f"${t_row['stop_loss']:,.2f}" if t_row['stop_loss'] else "N/A",
                t("col_tp"): f"${t_row['take_profit']:,.2f}" if t_row['take_profit'] else "N/A",
                t("col_pnl"): pnl_val,
                t("col_desc"): t_row['reason']
            })
        df_trades = pd.DataFrame(trades_rows)
        st.dataframe(df_trades, width="stretch", hide_index=True)

# ==========================================
#        TAB B: MANUEL ANALIZ VE TEST
# ==========================================
with tab_b:
    st.markdown(f"### 🔬 {'Canlı Yapay Zeka Finansal Raporlama' if st.session_state['lang'] == 'TR' else 'Live AI Financial Reporting'}")
    st.markdown(
        "İstediğiniz coin için indikatör durumunu hesaplayıp, son 24 saatteki haberleri Gemini ile taratarak özelleştirilmiş finansal rapor oluşturabilirsiniz."
        if st.session_state["lang"] == "TR" else
        "Compute indicator crossover states for any asset and scrape the latest 24h news with Google News RSS and Gemini to synthesize a personalized AI financial report."
    )

    col_b_sel, col_b_run = st.columns([3, 1])
    with col_b_sel:
        manual_pair = st.selectbox(t("coin_to_analyze"), options=ALL_SUPPORTED_PAIRS, key="manual_pair_sel")
    with col_b_run:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        btn_run = st.button(t("run_manual_btn"), width="stretch")

    if btn_run:
        with st.spinner(t("manual_spinner")):
            try:
                # 1. Fetch live CCXT candles
                m_candles_df = fetch_ohlcv(symbol=manual_pair, timeframe="1h", limit=100)
                m_candles_df = calculate_indicators(m_candles_df)
                
                # 2. Extract technical metrics
                completed_c = m_candles_df.iloc[-2]
                close_price = float(completed_c["close"])
                rsi_val = completed_c.get("rsi")
                macd_val = completed_c.get("macd")
                sig_val = completed_c.get("macd_signal")
                ema_val = completed_c.get("ema")
                
                # Calculate crossover recommendation
                trigger_side, trigger_reason = check_triggers(m_candles_df)
                
                # 3. Fetch News RSS feeds
                news_items = fetch_asset_news(manual_pair, limit=10)
                
                # 4. Request Gemini sentiment report
                ai_report = analyze_sentiment(manual_pair, news_items)
                
                sentiment_score = ai_report.get("sentiment_score", 0)
                ai_reason = ai_report.get("reason", "N/A")
                digest = ai_report.get("digest", "N/A")
                
                # Save report variables to Session State so order form can access them
                st.session_state["manual_report"] = {
                    "asset": manual_pair,
                    "price": close_price,
                    "rsi": rsi_val,
                    "macd": macd_val,
                    "signal": sig_val,
                    "ema": ema_val,
                    "trigger_side": trigger_side,
                    "trigger_reason": trigger_reason,
                    "sentiment_score": sentiment_score,
                    "ai_reason": ai_reason,
                    "digest": digest,
                    "news_items": news_items
                }
                
                st.success("Rapor başarıyla üretildi!" if st.session_state["lang"] == "TR" else "AI Report compiled successfully!")
            except Exception as e:
                st.error(f"Hata: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Render Report if it exists in session state
    if "manual_report" in st.session_state:
        rep = st.session_state["manual_report"]
        
        # Verify if asset ticker is selected
        asset_ticker = rep["asset"].split("/")[0]
        
        # Pre-format values to avoid f-string format specifier errors with conditional statements
        rsi_str = f"{rep['rsi']:.2f}" if rep["rsi"] is not None else "N/A"
        macd_diff_val = (rep["macd"] - rep["signal"]) if (rep["macd"] is not None and rep["signal"] is not None) else None
        macd_diff_str = f"{macd_diff_val:.4f}" if macd_diff_val is not None else "N/A"
        
        # CSS Styling for Report card
        sentiment_lbl = "GEMINI DUYGU SKORU" if st.session_state["lang"] == "TR" else "GEMINI SENTIMENT VERDICT"
        st.markdown(f"""
        <div class="report-card">
            <h2 style="margin:0 0 1rem 0; color:#00F2FE;">📊 {t('manual_report_title', asset=rep['asset'])}</h2>
            <div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:1.5rem;">
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:1rem; border-radius:10px; min-width:180px;">
                    <span style="color:#94A3B8; font-size:0.8rem; display:block;">{t('col_price').upper()}</span>
                    <strong style="font-size:1.4rem; color:#FFFFFF;">${rep['price']:,.2f}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:1rem; border-radius:10px; min-width:180px;">
                    <span style="color:#94A3B8; font-size:0.8rem; display:block;">RSI DEĞERİ / RSI VALUE</span>
                    <strong style="font-size:1.4rem; color:#FFFFFF;">{rsi_str}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:1rem; border-radius:10px; min-width:180px;">
                    <span style="color:#94A3B8; font-size:0.8rem; display:block;">MACD FARK / MACD DIFF</span>
                    <strong style="font-size:1.4rem; color:#FFFFFF;">{macd_diff_str}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); padding:1rem; border-radius:10px; min-width:180px;">
                    <span style="color:#94A3B8; font-size:0.8rem; display:block;">{sentiment_lbl}</span>
                    <strong style="font-size:1.4rem; color:#00F2FE;">{rep['sentiment_score']:+d} / 10</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Columns inside report
        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            st.markdown(f"#### {t('indicator_analysis')}")
            m_df_desc = []
            if rep['rsi'] is not None:
                if st.session_state["lang"] == "TR":
                    rsi_desc = "Aşırı Satım (Boğa Sinyali)" if rep['rsi'] <= 30 else ("Aşırı Alım (Ayı Riski)" if rep['rsi'] >= 70 else "Nötr Bölge")
                else:
                    rsi_desc = "Oversold (Bullish Signal)" if rep['rsi'] <= 30 else ("Overbought (Bearish Risk)" if rep['rsi'] >= 70 else "Neutral Zone")
                m_df_desc.append({t("indicator"): "RSI (14)", t("val"): f"{rep['rsi']:.2f}", t("status"): rsi_desc})
                
            if rep['macd'] is not None and rep['signal'] is not None:
                macd_diff = rep['macd'] - rep['signal']
                if st.session_state["lang"] == "TR":
                    macd_desc = "Boğa Sinyali (Yukarı Kesişim)" if macd_diff > 0 else "Ayı Sinyali (Aşağı Kesişim)"
                else:
                    macd_desc = "Bullish Cross (Above Signal)" if macd_diff > 0 else "Bearish Cross (Below Signal)"
                m_df_desc.append({t("indicator"): "MACD Crossover", t("val"): f"MACD: {rep['macd']:.4f} | Sig: {rep['signal']:.4f}", t("status"): macd_desc})
                
            if rep['ema'] is not None:
                if st.session_state["lang"] == "TR":
                    ema_desc = "Fiyat EMA Üzerinde (Yükseliş Trendi)" if rep['price'] > rep['ema'] else "Fiyat EMA Altında (Düşüş Trendi)"
                else:
                    ema_desc = "Price Above EMA (Bullish Trend)" if rep['price'] > rep['ema'] else "Price Below EMA (Bearish Trend)"
                m_df_desc.append({t("indicator"): "EMA 20", t("val"): f"${rep['ema']:,.2f}", t("status"): ema_desc})
                
            st.dataframe(pd.DataFrame(m_df_desc), width="stretch", hide_index=True)
            
            # Technical Trigger
            if rep['trigger_side']:
                st.warning(f"💥 **Piyasa Tetikleyici Tespit Edildi / Trigger:** {rep['trigger_reason']}")
            else:
                st.info(
                    "ℹ️ **Teknik Tetikleyici:** Bu mum periyodunda herhangi bir indikatör kesişim sinyali bulunmuyor."
                    if st.session_state["lang"] == "TR" else
                    "ℹ️ **Technical Trigger:** No mathematical indicator crossovers detected for this completed candle period."
                )

        with col_rep2:
            st.markdown(f"#### {t('ai_sentiment_synthesis')}")
            st.markdown(f"**{t('news_digest_title')}**")
            st.info(rep['digest'])
            st.markdown(f"**{t('ai_reason_title')}**")
            st.success(rep['ai_reason'])

        # Scraped news feed visualizer
        if "news_items" in rep and rep["news_items"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"#### {t('news_feed_title')}")
            for news_item in rep["news_items"]:
                rel_time = get_news_relative_time(news_item["pub_date"])
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <a href="{news_item['link']}" target="_blank" style="color: #00F2FE; font-weight: 600; text-decoration: none;">{news_item['title']}</a>
                    <p style="color: #94A3B8; font-size: 0.8rem; margin: 0.2rem 0 0 0;">{news_item['description']}</p>
                    <span style="color: #64748B; font-size: 0.75rem;">⏱️ {t('news_freshness', time=rel_time)}</span>
                </div>
                """, unsafe_allow_html=True)

        # Order Form Controls
        st.markdown("<hr style='border-color:rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        st.markdown(f"#### {t('manual_order_placement')}")
        
        # Check active positions for this asset
        pos_exists = get_active_position(rep['asset'])
        
        col_frm1, col_frm2 = st.columns(2)
        with col_frm1:
            st.write(f"{t('asset_type')}: **{asset_ticker}** | {t('col_price')}: **${rep['price']:,.2f}**")
            # Calculate manual size based on NAV
            man_risk_pct = float(get_config("risk_percentage") or 2.0)
            man_size = total_nav * (man_risk_pct / 100.0)
            man_size = min(man_size, usd_bal)
            st.write(f"{t('portfolio_risk_share')} (%{man_risk_pct}): **${man_size:.2f} USD**")

        with col_frm2:
            if pos_exists:
                # Active position exists, allow manual SELL
                st.error(t("pos_exists_warn"))
                btn_manual_sell = st.button(t("close_pos_btn"), width="stretch", type="primary")
                if btn_manual_sell:
                    # Execute manual close
                    pnl_pct = close_active_position(rep['asset'], rep['price'], "MANUAL", "Manual trade closed by user via Tab B control panel.")
                    
                    # Update portfolio balance
                    p_balance = portfolio.get(asset_ticker, {}).get("balance", 0.0)
                    trade_val = p_balance * rep['price']
                    fee_pct = 0.001
                    trade_net = trade_val * (1 - fee_pct)
                    
                    portfolio["USD"]["balance"] = usd_bal + trade_net
                    portfolio[asset_ticker] = {"balance": 0.0, "avg_entry_price": 0.0}
                    update_portfolio(portfolio)
                    
                    log_event("SUCCESS", "UI_DASHBOARD", f"🛒 MANUEL SATIŞ TAMAMLANDI: {asset_ticker} pozisyonu ${rep['price']:.2f} fiyatından kapatıldı. PnL: {pnl_pct:+.2f}%")
                    st.toast("Pozisyon başarıyla kapatıldı!" if st.session_state["lang"] == "TR" else "Position liquidated manually!", icon="💰")
                    del st.session_state["manual_report"]
                    st.rerun()
            else:
                # No active position, allow manual BUY
                btn_manual_buy = st.button(t("buy_order_btn"), width="stretch")
                if btn_manual_buy:
                    if usd_bal < 10.0:
                        st.error(t("insufficient_bal_err"))
                    else:
                          trade_net = man_size * (1 - 0.001)
                          qty = trade_net / rep['price']
                          
                          # Stop loss / Take profit targets
                          m_sl = float(get_config("stop_loss_pct") or 3.0)
                          m_tp = float(get_config("take_profit_pct") or 6.0)
                          sl_price = rep['price'] * (1 - m_sl / 100.0)
                          tp_price = rep['price'] * (1 + m_tp / 100.0)
                          
                          # Update portfolio memory
                          portfolio["USD"]["balance"] = usd_bal - man_size
                          existing_ast = portfolio.get(asset_ticker, {"balance": 0.0, "avg_entry_price": 0.0})
                          new_bal = existing_ast["balance"] + qty
                          new_avg = ((existing_ast["balance"] * existing_ast["avg_entry_price"]) + (qty * rep['price'])) / new_bal if new_bal > 0 else rep['price']
                          
                          portfolio[asset_ticker] = {
                              "balance": new_bal,
                              "avg_entry_price": new_avg
                          }
                          update_portfolio(portfolio)
                          
                          # Record BUY
                          record_trade(
                              asset=rep['asset'],
                              side="BUY",
                              price=rep['price'],
                              amount=qty,
                              trade_type="TECHNICAL" if not rep['trigger_side'] else "AI_CONFIRMED",
                              sentiment_score=rep['sentiment_score'],
                              reason=f"Manual execution via Tab B dashboard. Tech: {rep['trigger_reason']} | AI: {rep['ai_reason']}",
                              stop_loss=sl_price,
                              take_profit=tp_price,
                              is_active=1
                          )
                          
                          log_event("SUCCESS", "UI_DASHBOARD", f"🛒 MANUEL ALIM GERÇEKLEŞTİ: {qty:.4f} {asset_ticker} alındı. Fiyat: ${rep['price']:.2f} | SL: ${sl_price:.2f}, TP: ${tp_price:.2f}")
                          st.toast("Manuel alım başarıyla gerçekleşti!" if st.session_state["lang"] == "TR" else "Manual BUY executed successfully!", icon="🛒")
                          del st.session_state["manual_report"]
                          st.rerun()

# ==========================================
#        TAB C: LOGLAR VE ŞEFFAFLIK
# ==========================================
with tab_c:
    st.markdown(f"### {t('transparency_logs')}")
    
    col_log1, col_log2 = st.columns(2)
    
    with col_log1:
        st.markdown(f"#### {t('action_logs')}")
        if not logs:
            st.markdown(
                f"<div class='log-container'><span class='log-info'>{t('system_active_waiting')}</span></div>", 
                unsafe_allow_html=True
            )
        else:
            log_html = "<div class='log-container'>"
            for l in logs:
                level_cls = "log-info"
                if l['level'] == "WARNING": level_cls = "log-warning"
                elif l['level'] == "SUCCESS": level_cls = "log-success"
                elif l['level'] == "ERROR": level_cls = "log-error"
                elif l['level'] == "INFO" and l['module'] == "MATH": level_cls = "log-math"
                
                try:
                    t_parsed = datetime.fromisoformat(l['timestamp'])
                    if t_parsed.tzinfo is None:
                        t_parsed = t_parsed.replace(tzinfo=timezone.utc)
                    t_parsed = t_parsed.astimezone(ISTANBUL_TZ)
                    t_str = t_parsed.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    t_str = l['timestamp'][:19]
                    
                log_html += f"<div class='log-line'><span class='log-info'>[{t_str}]</span> <span class='{level_cls}'>[{l['level']}]</span> <span style='color: #60A5FA;'>[{l['module']}]</span> <span class='{level_cls}'>{l['message']}</span></div>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
            
    with col_log2:
        st.markdown(f"#### {t('ai_thought_logs')}")
        
        # Load AI runs history
        conn_ai = get_connection()
        cursor_ai = conn_ai.cursor()
        cursor_ai.execute("SELECT * FROM ai_runs ORDER BY timestamp DESC LIMIT 20")
        raw_ai = cursor_ai.fetchall()
        conn_ai.close()
        
        ai_runs = [dict(r) for r in raw_ai]
        
        if not ai_runs:
            st.markdown(
                f"<div class='json-container'>{t('ai_no_run')}</div>",
                unsafe_allow_html=True
            )
        else:
            for idx, run in enumerate(ai_runs):
                try:
                    t_parsed = datetime.fromisoformat(run['timestamp'])
                    if t_parsed.tzinfo is None:
                        t_parsed = t_parsed.replace(tzinfo=timezone.utc)
                    t_parsed = t_parsed.astimezone(ISTANBUL_TZ)
                    run_time_str = t_parsed.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    run_time_str = run['timestamp'][:19].replace('T', ' ')
                    
                with st.expander(f"🤖 {run['asset']} - {run_time_str} ({t('score')}: {run['sentiment_score']:+d})", expanded=(idx==0)):
                    st.markdown(f"**{t('news_digest_ai')}**")
                    st.info(run['news_digest'])
                    
                    st.markdown(f"**{t('ai_decision_json')}**")
                    run_json = {
                        "sentiment_score": run["sentiment_score"],
                        "reason": run["reason"]
                    }
                    st.markdown(f"<div class='json-container'>{json.dumps(run_json, indent=2)}</div>", unsafe_allow_html=True)
