import os
import time
import threading
import requests
from datetime import datetime, timezone

# Resolve dynamic imports lazily to prevent circular imports
def get_db_helpers():
    from core.db import (
        get_config, save_config, get_portfolio, get_trades, get_logs,
        get_all_active_positions, get_latest_ai_run, log_event
    )
    from worker import calculate_total_nav, run_worker_cycle
    from core.data_fetcher import check_vpn_connection
    return get_config, save_config, get_portfolio, get_trades, get_logs, get_all_active_positions, get_latest_ai_run, log_event, calculate_total_nav, run_worker_cycle, check_vpn_connection

def send_telegram_message(message: str, parse_mode="Markdown") -> bool:
    """
    Sends a message to the Telegram channel/chat configured via environment variables.
    Supports proxy configuration through Gluetun and falls back to no proxy on failure.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode
    }
    
    # Proxy configuration from environment (Gluetun stack)
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
        
    # Attempt direct connection first, fallback to proxy
    for use_proxy in [False, True]:
        try:
            resp = requests.post(url, json=payload, proxies=proxies if use_proxy else {}, timeout=10)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
            
    return False

def send_trade_notification(asset, side, price, amount, trade_type, sentiment_score=None, reason=None, stop_loss=None, take_profit=None, pnl=None):
    """
    Formats and routes trade execution notifications (BUY/SELL) to Telegram.
    """
    asset_ticker = asset.split("/")[0]
    value = amount * price
    
    if side.upper() == "BUY":
        score_str = f"{sentiment_score}/10" if sentiment_score is not None else "Yok"
        sl_val = f"${stop_loss:,.2f}" if stop_loss else "Yok"
        tp_val = f"${take_profit:,.2f}" if take_profit else "Yok"
        msg = (
            "🛒 *YENİ İŞLEM AÇILDI (BUY)*\n\n"
            f"• *Varlık:* `{asset}`\n"
            f"• *İşlem Yönü:* ALIM (BUY)\n"
            f"• *İşlem Tipi:* `{trade_type}`\n"
            f"• *Giriş Fiyatı:* `${price:,.2f}`\n"
            f"• *Miktar:* `{amount:.4f} {asset_ticker}`\n"
            f"• *İşlem Hacmi:* `${value:,.2f}`\n"
            f"• *Zarar Durdur (SL):* `{sl_val}`\n"
            f"• *Kâr Al (TP):* `{tp_val}`\n"
            f"• *Yapay Zeka Skoru:* `{score_str}`\n"
            f"• *Gerekçe:* {reason or 'Belirtilmedi'}"
        )
    else: # SELL
        pnl_str = f"{pnl:+.2f}%" if pnl is not None else "Belirtilmedi"
        msg = (
            "💰 *İŞLEM KAPATILDI (SELL)*\n\n"
            f"• *Varlık:* `{asset}`\n"
            f"• *İşlem Yönü:* SATIM (SELL)\n"
            f"• *Kapatma Tipi:* `{trade_type}`\n"
            f"• *Çıkış Fiyatı:* `${price:,.2f}`\n"
            f"• *Miktar:* `{amount:.4f} {asset_ticker}`\n"
            f"• *İşlem Hacmi:* `${value:,.2f}`\n"
            f"• *Net Kâr/Zarar (PnL):* `{pnl_str}`\n"
            f"• *Gerekçe:* {reason or 'Belirtilmedi'}"
        )
        
    send_telegram_message(msg)

def telegram_polling_loop():
    """
    Background polling loop to receive updates and process commands.
    Only authorized updates originating from TELEGRAM_CHAT_ID are executed.
    """
    get_config, save_config, get_portfolio, get_trades, get_logs, get_all_active_positions, get_latest_ai_run, log_event, calculate_total_nav, run_worker_cycle, check_vpn_connection = get_db_helpers()
    
    offset = None
    log_event("INFO", "TELEGRAM", "Telegram bot listener thread started.")
    
    while True:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            time.sleep(10)
            continue
            
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 30}
        if offset:
            params["offset"] = offset
            
        # Proxy configurations
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
            
        try:
            resp = None
            for use_proxy in [False, True]:
                try:
                    resp = requests.get(url, params=params, proxies=proxies if use_proxy else {}, timeout=35)
                    if resp.status_code == 200:
                        break
                except Exception:
                    pass
                    
            if resp is None or resp.status_code != 200:
                time.sleep(10)
                continue
                
            data = resp.json()
            if not data.get("ok"):
                time.sleep(10)
                continue
                
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                    
                chat_id = message.get("chat", {}).get("id")
                configured_chat_id = os.getenv("TELEGRAM_CHAT_ID")
                
                # Basic authentication gate
                if not configured_chat_id:
                    # Capture chat ID automatically on first /start if .env is empty (helper for easy setup)
                    text = message.get("text", "")
                    if text.startswith("/start"):
                        send_telegram_message(f"🤖 Sentix Bot paired successfully! Chat ID: `{chat_id}`. Please add this Chat ID to your .env file as: TELEGRAM_CHAT_ID={chat_id}")
                        log_event("SUCCESS", "TELEGRAM", f"Telegram bot automatically paired with Chat ID: {chat_id}")
                        continue
                
                if str(chat_id) != str(configured_chat_id):
                    continue
                    
                text = message.get("text", "").strip()
                if not text:
                    continue
                    
                process_command(text)
                
        except Exception as e:
            time.sleep(10)

def process_command(text: str):
    """
    Executes business logic for Telegram commands.
    """
    get_config, save_config, get_portfolio, get_trades, get_logs, get_all_active_positions, get_latest_ai_run, log_event, calculate_total_nav, run_worker_cycle, check_vpn_connection = get_db_helpers()
    
    cmd_parts = text.split(maxsplit=2)
    # Strip bot username if present (e.g. /help@MyBot -> /help)
    cmd = cmd_parts[0].lower().split('@')[0]
    
    if cmd == "/start" or cmd == "/help":
        msg = (
            "🤖 *Sentix Algoritmik Hisse Senedi Trading Paneli*\n\n"
            "Kullanabileceğiniz komutlar:\n"
            "📈 `/portfolio` - Portföy durumu ve Net Değer (NAV)\n"
            "📦 `/positions` - Açık aktif hisse pozisyonları ve anlık PnL\n"
            "📜 `/trades` - Son 5 işlem geçmişi\n"
            "📂 `/assets` - İzlenen aktif hisseleri/endeksleri listeler\n"
            "🔒 `/vpn` - VPN bağlantı durumu ve IP adresi\n"
            "🔒 `/vpn_logs` - Sunucudaki VPN bağlantı loglarını listeler\n"
            "🧠 `/ai_status` - Sadece aktif duygu skoru olan hisseleri listeler\n"
            "📝 `/logs` - Son 5 platform log kaydı\n"
            "🚀 `/trigger` - Manuel analiz döngüsünü tetikler\n"
            "⏸️ `/pause` - Otomatik alım-satım döngüsünü duraklatır (durdurur)\n"
            "▶️ `/resume` - Otomatik alım-satım döngüsünü başlatır\n"
            "⚙️ `/risk` `yüzde` - NAV işlem büyüklüğü yüzdesini ayarlar (örn: `/risk 2.5`)\n"
            "🛡️ `/sltp` `sl` `tp` - Stop-Loss ve Take-Profit oranlarını günceller (örn: `/sltp 2.0 5.0`)"
        )
        send_telegram_message(msg)
        
    elif cmd == "/portfolio":
        portfolio = get_portfolio()
        total_nav = calculate_total_nav(portfolio)
        free_usd = portfolio.get("USD", {}).get("balance", 0.0)
        
        # Calculate return from starting $10,000 USD
        initial_usd = 10000.0
        net_return = ((total_nav - initial_usd) / initial_usd) * 100
        
        msg = (
            "📈 *Sentix Portföy Durumu*\n\n"
            f"• *Toplam NAV:* `${total_nav:,.2f}`\n"
            f"• *Kullanılabilir USD:* `${free_usd:,.2f}`\n"
            f"• *Net Getiri:* `{net_return:+.2f}%`\n\n"
            "*Varlık Dağılımı:*\n"
        )
        
        has_assets = False
        for asset, data in portfolio.items():
            if asset == "USD":
                continue
            bal = data.get("balance", 0.0)
            avg_price = data.get("avg_entry_price", 0.0)
            if bal > 0:
                has_assets = True
                msg += f"• *{asset}:* `{bal:.4f}` (Ort. Maliyet: `${avg_price:,.2f}`)\n"
                
        if not has_assets:
            msg += "_Hisse senedi varlığı bulunmuyor (Tümü USD'de)._"
            
        send_telegram_message(msg)
        
    elif cmd == "/positions":
        positions = get_all_active_positions()
        if not positions:
            send_telegram_message("📦 *Açık Pozisyon Bulunmuyor.*")
            return
            
        msg = "📦 *Sentix Aktif Pozisyonlar:*\n\n"
        for pos in positions:
            asset = pos['asset']
            buy_price = pos['price']
            amount = pos['amount']
            sl = pos.get('stop_loss')
            tp = pos.get('take_profit')
            
            # Fetch latest price of this asset
            current_price = buy_price
            try:
                from core.db import get_connection
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT close FROM candles WHERE asset = ? ORDER BY timestamp DESC LIMIT 1", (asset,))
                row = cursor.fetchone()
                if row:
                    current_price = float(row[0])
                conn.close()
            except Exception:
                pass
                
            pnl = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0.0
            
            msg += (
                f"• *{asset}*\n"
                f"  Giriş: `${buy_price:,.2f}` | Güncel: `${current_price:,.2f}`\n"
                f"  Miktar: `{amount:.4f}` | PnL: `{pnl:+.2f}%`\n"
                f"  SL: `${sl:,.2f}` | TP: `${tp:,.2f}`\n\n"
            )
        send_telegram_message(msg)
        
    elif cmd == "/trades":
        trades = get_trades(limit=5)
        if not trades:
            send_telegram_message("📜 *İşlem geçmişi bulunmuyor.*")
            return
            
        msg = "📜 *Son 5 İşlem Geçmişi:*\n\n"
        for t_val in trades:
            side = t_val['side']
            asset = t_val['asset']
            price = t_val['price']
            amount = t_val['amount']
            pnl = t_val.get('pnl')
            ts = t_val['timestamp']
            
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%d/%m %H:%M")
            except Exception:
                time_str = ts
                
            emoji = "🛒" if side == "BUY" else "💰"
            pnl_str = f" | PnL: `{pnl:+.2f}%`" if pnl is not None else ""
            
            msg += f"{emoji} *{side}* `{asset}` | Fiyat: `${price:,.2f}` | Mkt: `{amount:.4f}`{pnl_str} ({time_str})\n"
            
        send_telegram_message(msg)
        
    elif cmd == "/vpn":
        status = get_config("vpn_status", "unknown")
        ip_str = "Bilinmiyor"
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
            
        try:
            resp_ip = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=5)
            if resp_ip.status_code == 200:
                ip_str = resp_ip.json().get("origin", "Bilinmiyor")
        except Exception:
            pass
            
        status_emoji = "🟢" if status == "connected" else "🔴"
        msg = (
            "🔒 *VPN Bağlantı Durumu*\n\n"
            f"• *Sunucu Bağlantısı:* {status_emoji} {status.upper()}\n"
            f"• *Dış IP Adresi:* `{ip_str}`\n"
            f"• *Proxy Sunucusu:* `{http_proxy or 'Yok'}`"
        )
        send_telegram_message(msg)

    elif cmd == "/assets":
        assets_str = get_config("selected_assets", "")
        active_assets = [a.strip() for a in assets_str.split(",") if a.strip()]
        
        if not active_assets:
            send_telegram_message("📂 *İzlenen aktif hisse senedi bulunmuyor.*")
            return
            
        msg = "📂 *İzlenen Hisseler / Endeksler:*\n\n"
        for idx, asset in enumerate(active_assets, 1):
            msg += f"{idx}. `{asset}`\n"
            
        send_telegram_message(msg)
        
    elif cmd == "/ai_status":
        assets_str = get_config("selected_assets", "")
        active_assets = [a.strip() for a in assets_str.split(",") if a.strip()]
        
        paused = get_config("bot_paused", "false") == "true"
        paused_str = "⏸️ Duraklatıldı" if paused else "🟢 Aktif İzlemede"
        
        msg = (
            "🧠 *İndikatör & Gemini AI Durumu*\n"
            f"• *Otomatik Alım-Satım:* {paused_str}\n\n"
        )
        
        has_runs = False
        for asset in active_assets:
            ai_run = get_latest_ai_run(asset)
            if not ai_run:
                continue
            has_runs = True
            
            from core.db import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candles WHERE asset = ? ORDER BY timestamp DESC LIMIT 1", (asset,))
            row = cursor.fetchone()
            conn.close()
            
            rsi_str = "N/A"
            macd_str = "N/A"
            ema_str = "N/A"
            price_str = "N/A"
            
            if row:
                row_dict = dict(row)
                price_str = f"${row_dict['close']:,.2f}"
                if row_dict.get('rsi') is not None:
                    rsi_str = f"{row_dict['rsi']:.2f}"
                if row_dict.get('macd') is not None and row_dict.get('macd_signal') is not None:
                    diff = row_dict['macd'] - row_dict['macd_signal']
                    macd_str = f"Diff: {diff:+.4f}"
                if row_dict.get('ema') is not None:
                    ema_str = f"${row_dict['ema']:,.2f}"
            
            ai_score = f"{ai_run['sentiment_score']}/10"
                
            msg += (
                f"• *{asset}* | Son Fiyat: `{price_str}`\n"
                f"  RSI: `{rsi_str}` | EMA: `{ema_str}`\n"
                f"  MACD: `{macd_str}`\n"
                f"  Gemini Duygu Skoru: `{ai_score}`\n\n"
            )
            
        if not has_runs:
            msg += "_Aktif yapay zeka duygu analizi kaydı bulunmuyor (Gemini henüz tetiklenmedi)._"
            
        send_telegram_message(msg)
        
    elif cmd == "/logs":
        logs = get_logs(limit=5)
        if not logs:
            send_telegram_message("📝 *Log kaydı bulunmuyor.*")
            return
            
        msg = "📝 *Son 5 Sistem Günlüğü:*\n\n"
        for l in logs:
            lvl = l['level']
            mod = l['module']
            message_text = l['message']
            ts = l['timestamp']
            
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%H:%M:%S")
            except Exception:
                time_str = ts
                
            emoji = "🔴" if lvl == "ERROR" else ("🟡" if lvl == "WARNING" else ("🟢" if lvl == "SUCCESS" else "ℹ️"))
            msg += f"{emoji} `[{time_str}]` `[{mod}]` {message_text}\n"
            
        send_telegram_message(msg)
        
    elif cmd == "/trigger":
        send_telegram_message("🚀 *Manuel analiz döngüsü arka planda başlatılıyor...*")
        
        def run_trigger_async():
            try:
                run_worker_cycle(force=True)
                send_telegram_message("✅ *Manuel analiz döngüsü başarıyla tamamlandı!*")
            except Exception as e:
                send_telegram_message(f"❌ *Analiz döngüsü başarısız oldu:* {e}")
                
        threading.Thread(target=run_trigger_async, daemon=True).start()
        
    elif cmd == "/pause":
        save_config("bot_running", "false")
        log_event("WARNING", "TELEGRAM", "Automatic trading paused (stopped) by Telegram command.")
        send_telegram_message("⏸️ *Otomatik alım-satım döngüsü DURDURULDU!* (Bot tamamen uykudadır.)")
        
    elif cmd == "/resume":
        from worker import validate_api_key_for_start
        if validate_api_key_for_start():
            save_config("bot_running", "true")
            log_event("SUCCESS", "TELEGRAM", "Automatic trading started by Telegram command.")
            send_telegram_message("▶️ *Otomatik alım-satım döngüsü BAŞLATILDI!*")
        else:
            log_event("ERROR", "TELEGRAM", "Attempted to start bot without valid API Key.")
            send_telegram_message("❌ *Hata:* Google Gemini API anahtarı bulunamadı! Bot başlatılamıyor.")
        
    elif cmd == "/risk":
        if len(cmd_parts) < 2:
            send_telegram_message("⚠️ *Lütfen bir risk yüzdesi belirtin.* Örn: `/risk 2.5`")
            return
        try:
            val = float(cmd_parts[1])
            if not (1.0 <= val <= 5.0):
                send_telegram_message("⚠️ *Risk yüzdesi 1.0 ile 5.0 arasında olmalıdır.*")
                return
            save_config("risk_percentage", val)
            log_event("SUCCESS", "TELEGRAM", f"NAV risk percentage updated to {val}% via Telegram command.")
            send_telegram_message(f"⚙️ *İşlem Risk Büyüklüğü (Risk %) güncellendi:* {val}% of NAV")
        except ValueError:
            send_telegram_message("⚠️ *Geçersiz yüzde biçimi. Lütfen sayısal bir değer girin.*")
            
    elif cmd == "/sltp":
        if len(cmd_parts) < 2:
            send_telegram_message("⚠️ *Lütfen Stop-Loss ve Take-Profit oranlarını belirtin.* Örn: `/sltp 3.0 6.0`")
            return
        try:
            sub_parts = cmd_parts[1].split()
            if len(sub_parts) != 2:
                send_telegram_message("⚠️ *Lütfen hem SL hem TP oranlarını girin.* Örn: `/sltp 3.0 6.0`")
                return
            sl_val = float(sub_parts[0])
            tp_val = float(sub_parts[1])
            
            if not (1.0 <= sl_val <= 10.0) or not (2.0 <= tp_val <= 20.0):
                send_telegram_message("⚠️ *SL oranı %1.0-%10.0, TP oranı %2.0-%20.0 arasında olmalıdır.*")
                return
                
            save_config("stop_loss_pct", sl_val)
            save_config("take_profit_pct", tp_val)
            log_event("SUCCESS", "TELEGRAM", f"SL/TP percentages updated to {sl_val}% / {tp_val}% via Telegram command.")
            send_telegram_message(f"🛡️ *Pozisyon Koruma Limitleri güncellendi:*\n• *Stop-Loss:* `{sl_val}%`\n• *Take-Profit:* `{tp_val}%`")
        except ValueError:
            send_telegram_message("⚠️ *Geçersiz oran biçimi. Lütfen sayısal değerler girin.*")
            
    elif cmd == "/vpn_logs":
        send_telegram_message("🔒 *VPN günlükleri Docker üzerinden alınıyor...*")
        logs = get_vpn_logs(tail=25)
        send_telegram_message(f"🔒 *VPN Günlükleri (Son 25 Satır):*\n\n```\n{logs}\n```")
            
    else:
        send_telegram_message("⚠️ *Bilinmeyen komut.* Yardım almak için /help yazabilirsiniz.")

def get_vpn_logs(tail=20) -> str:
    """
    Reads the stdout/stderr logs of the Gluetun VPN container ('sentix_vpn')
    using the Docker daemon UNIX socket.
    """
    import socket
    import http.client
    
    # Custom HTTP connection over Unix Socket
    class UnixHTTPConnection(http.client.HTTPConnection):
        def __init__(self, path):
            super().__init__('localhost')
            self.path = path
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.path)
            
    socket_path = '/var/run/docker.sock'
    if not os.path.exists(socket_path):
        return f"Docker soketi ({socket_path}) bulunamadı. Lütfen docker-compose.yml dosyasında soketin monte edildiğinden emin olun."
        
    try:
        conn = UnixHTTPConnection(socket_path)
        # Tail logs
        url = f"/containers/sentix_vpn/logs?stdout=true&stderr=true&tail={tail}"
        conn.request('GET', url)
        resp = conn.getresponse()
        
        if resp.status != 200:
            conn.close()
            return f"Docker API hatası: HTTP {resp.status} - {resp.reason}"
            
        data = resp.read()
        conn.close()
        
        # Clean up Docker log stream frames
        lines = []
        i = 0
        while i < len(data):
            if i + 8 > len(data):
                break
            header = data[i:i+8]
            size = int.from_bytes(header[4:8], byteorder='big')
            if i + 8 + size > len(data):
                line_bytes = data[i+8:]
                lines.append(line_bytes.decode('utf-8', errors='ignore'))
                break
            line_bytes = data[i+8:i+8+size]
            lines.append(line_bytes.decode('utf-8', errors='ignore'))
            i += 8 + size
            
        logs_text = "".join(lines).strip()
        return logs_text if logs_text else "VPN log kaydı bulunmuyor."
    except Exception as e:
        return f"Docker günlükleri çekilemedi: {e}"

def start_telegram_bot():
    """
    Launches the Telegram polling loop in a background thread.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        # Silently skip starting if Telegram bot is not configured
        return
    t = threading.Thread(target=telegram_polling_loop, daemon=True, name="TelegramBotThread")
    t.start()
