import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from core.db import (
    init_db, get_trades, get_logs, get_latest_ai_run,
    get_config, save_config, get_latest_candles, get_all_active_positions,
    get_equity_history, get_latest_signals, get_connection, log_event,
)
from core.accounting import calculate_nav, execute_buy, execute_sell
from core.config import StrategyConfig, DEFAULT_WATCHLIST, parse_watchlist, normalize_symbol
from core.data_fetcher import fetch_ohlcv, fetch_realtime_price, fetch_asset_news
from core.indicators import add_indicators
from core.backtest import run_backtest, trades_to_frame
from core.risk import is_market_open, is_crypto
from worker import run_worker_cycle, validate_api_key_for_start

from ui.translations import t
from ui import components as C

load_dotenv()

INITIAL_CASH = 10000.0
HEARTBEAT_STALE_SECONDS = 120


# =========================================================================
# Session state / page setup
# =========================================================================

def _init_session_state():
    defaults = {
        "lang": "TR",
        "authenticated": False,
        "auto_refresh": True,
        "refresh_interval": 30,
        "bt_result": None,
        "bt_params": None,
        "bt_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _inject_css():
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 10px 14px 6px 14px;
        }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        section[data-testid="stSidebar"] .stButton button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _check_password() -> bool:
    """Optional password gate driven by the ACCESS_PASSWORD env var."""
    access_password = os.getenv("ACCESS_PASSWORD")
    if not access_password:
        return True
    if st.session_state.get("authenticated"):
        return True

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(f"### {t('password_title')}")
        st.caption(t("password_desc"))
        with st.form("login_form"):
            pwd = st.text_input(t("password_label"), type="password")
            submitted = st.form_submit_button(t("password_submit"), width="stretch")
        if submitted:
            if pwd == access_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error(t("password_wrong"))
    return False


# =========================================================================
# Sidebar
# =========================================================================

def _render_sidebar():
    with st.sidebar:
        st.markdown(f"## 📈 {t('app_title')}")
        st.caption(t("app_tagline"))

        lang_options = ["TR", "EN"]
        idx = lang_options.index(st.session_state.lang) if st.session_state.lang in lang_options else 0
        chosen = st.radio(t("lang_label"), lang_options, index=idx, horizontal=True, key="lang_selector")
        if chosen != st.session_state.lang:
            st.session_state.lang = chosen
            st.rerun()

        st.divider()

        # ---- bot status ----
        st.markdown(f"**{t('sidebar_status_title')}**")
        bot_running = get_config("bot_running", "false") == "true"
        heartbeat_raw = get_config("worker_heartbeat")
        age = C.seconds_since(heartbeat_raw) if heartbeat_raw else float("inf")

        if bot_running and age <= HEARTBEAT_STALE_SECONDS:
            st.success(f"🟢 {t('status_running')}")
        elif bot_running:
            st.warning(f"🟡 {t('status_stale')}")
        else:
            st.error(f"🔴 {t('status_stopped')}")

        if heartbeat_raw:
            st.caption(t("heartbeat_label", time=C.to_istanbul_str(heartbeat_raw, "%H:%M:%S"),
                        rel=C.relative_time(heartbeat_raw, st.session_state.lang)))
        else:
            st.caption(t("heartbeat_never"))

        c1, c2 = st.columns(2)
        with c1:
            if st.button(t("btn_start"), disabled=bot_running, key="btn_start_bot"):
                save_config("bot_running", "true")
                log_event("INFO", "UI", "Bot started from dashboard.")
                st.toast(t("bot_started"))
                st.rerun()
        with c2:
            if st.button(t("btn_stop"), disabled=not bot_running, key="btn_stop_bot"):
                save_config("bot_running", "false")
                log_event("INFO", "UI", "Bot stopped from dashboard.")
                st.toast(t("bot_stopped"))
                st.rerun()

        if st.button(t("btn_run_now"), key="btn_run_now", width="stretch"):
            with st.spinner(t("run_now_spinner")):
                try:
                    run_worker_cycle(force=True)
                    st.success(t("run_now_success"))
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(t("run_now_error", error=str(e)))

        try:
            cfg_ai = get_config("ai_enabled", "true") == "true"
            if cfg_ai and not validate_api_key_for_start():
                st.caption(t("ai_key_missing_warn"))
        except Exception:
            pass

        st.divider()

        # ---- NAV quick figure ----
        try:
            nav, cash, _exposure = calculate_nav()
        except Exception as e:
            nav, cash = 0.0, 0.0
            st.caption(t("loading_error", error=str(e)))
        st.metric(t("sidebar_nav_label"), C.fmt_usd(nav))
        st.caption(f"{t('sidebar_cash_label')}: {C.fmt_usd(cash)}")

        st.divider()

        # ---- auto-refresh controls ----
        st.markdown(f"**{t('sidebar_autorefresh_title')}**")
        st.session_state.auto_refresh = st.checkbox(
            t("autorefresh_toggle"), value=st.session_state.auto_refresh, key="autorefresh_toggle_cb")
        st.session_state.refresh_interval = st.slider(
            t("autorefresh_interval"), min_value=10, max_value=120,
            value=st.session_state.refresh_interval, step=5, key="autorefresh_interval_slider")
        if st.session_state.bt_result is not None:
            st.caption(t("autorefresh_paused_backtest"))

        st.divider()

        # ---- market status ----
        st.markdown(f"**{t('market_status_title')}**")
        try:
            us_open = is_market_open("AAPL")
            st.write(t("market_open") if us_open else t("market_closed"))
        except Exception:
            pass
        st.caption(t("crypto_always_open"))

    return nav, cash


def _maybe_autorefresh():
    backtest_showing = st.session_state.bt_result is not None or st.session_state.bt_running
    if st.session_state.auto_refresh and not backtest_showing:
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="global_autorefresh")


# =========================================================================
# Tab 1: Portfolio
# =========================================================================

def _pair_closed_trades(trades: list, limit: int = 20):
    """Pairs each SELL row with the most recent BUY row of the same asset
    that preceded it, so closed positions can show both entry and exit."""
    buys = [t for t in trades if t.get("side") == "BUY"]
    sells = [t for t in trades if t.get("side") == "SELL"]
    paired = []
    for sell in sells:
        candidates = [b for b in buys if b["asset"] == sell["asset"] and b["timestamp"] <= sell["timestamp"]]
        buy = max(candidates, key=lambda b: b["timestamp"]) if candidates else None
        if buy is not None:
            buys.remove(buy)
        paired.append((buy, sell))
        if len(paired) >= limit:
            break
    return paired


def _live_prices_for(open_positions: list) -> dict:
    """Best-effort current price per open position. Uses real-time quotes
    when there are few enough positions to keep it snappy, otherwise falls
    back to the latest stored candle close."""
    prices = {}
    use_live = 0 < len(open_positions) <= 5
    for pos in open_positions:
        asset = pos["asset"]
        price = None
        if use_live:
            try:
                price = fetch_realtime_price(asset)
            except Exception:
                price = None
        if price is None:
            try:
                candles = get_latest_candles(asset, "1h", 1)
                if candles:
                    price = candles[-1]["close"]
            except Exception:
                price = None
        prices[asset] = price if price is not None else pos["price"]
    return prices


def render_portfolio_tab(cfg: StrategyConfig):
    try:
        nav, cash, _exposure = calculate_nav()
        open_positions = get_all_active_positions()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("metric_nav"), C.fmt_usd(nav))
        c2.metric(t("metric_cash"), C.fmt_usd(cash))
        ret_pct = (nav / INITIAL_CASH - 1) * 100 if INITIAL_CASH else 0.0
        c3.metric(t("metric_return"), C.fmt_pct(ret_pct), help=t("metric_return_help"))
        c4.metric(t("metric_open_positions"), str(len(open_positions)))

        st.subheader(t("equity_curve_title"))
        try:
            history = get_equity_history()
        except Exception as e:
            history = []
            st.warning(t("loading_error", error=str(e)))
        if not history:
            st.info(t("equity_curve_empty"))
        else:
            st.plotly_chart(C.equity_area_chart(history, INITIAL_CASH), width="stretch")

        st.subheader(t("open_positions_title"))
        if not open_positions:
            st.info(t("open_positions_empty"))
        else:
            live_prices = _live_prices_for(open_positions)
            rows = []
            for pos in open_positions:
                asset = pos["asset"]
                entry = float(pos["price"])
                price = float(live_prices.get(asset) or entry)
                qty = float(pos["amount"] or 0)
                pnl_pct = (price / entry - 1) * 100 if entry else 0.0
                pnl_usd = (price - entry) * qty
                stop = pos.get("stop_loss")
                stop_dist = ((price - float(stop)) / price * 100) if stop and price else None
                rows.append({
                    t("col_asset"): asset,
                    t("col_entry"): entry,
                    t("col_current"): price,
                    t("col_qty"): qty,
                    t("col_pnl_pct"): pnl_pct,
                    t("col_pnl_usd"): pnl_usd,
                    t("col_stop"): float(stop) if stop else None,
                    t("col_stop_dist"): stop_dist,
                    t("col_target"): float(pos["take_profit"]) if pos.get("take_profit") else None,
                    t("col_score"): pos.get("confluence_score"),
                    t("col_time"): C.to_istanbul_str(pos["timestamp"]),
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df, width="stretch", hide_index=True,
                column_config={
                    t("col_entry"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_current"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_qty"): st.column_config.NumberColumn(format="%.4f"),
                    t("col_pnl_pct"): st.column_config.NumberColumn(format="%.2f%%"),
                    t("col_pnl_usd"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_stop"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_stop_dist"): st.column_config.NumberColumn(format="%.2f%%"),
                    t("col_target"): st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            st.markdown(f"**{t('manual_close_title')}**")
            for pos in open_positions:
                asset = pos["asset"]
                cc1, cc2, cc3 = st.columns([2, 3, 2])
                cc1.write(f"**{asset}**")
                cc2.caption(f"{t('col_qty')}: {float(pos['amount'] or 0):.4f} @ {C.fmt_usd(pos['price'])}")
                if cc3.button(t("btn_close_position"), key=f"close_{pos['id']}"):
                    try:
                        price = fetch_realtime_price(asset)
                        if price is None:
                            st.error(t("price_unavailable"))
                        else:
                            result = execute_sell(asset, price, cfg, trade_type="MANUAL",
                                                  reason="Manual close from UI")
                            if result:
                                st.success(t("manual_trade_success"))
                                st.rerun()
                            else:
                                st.error(t("manual_trade_fail_generic"))
                    except Exception as e:
                        st.error(t("manual_trade_fail", error=str(e)))

        with st.expander(t("manual_trade_title")):
            open_assets = {p["asset"] for p in open_positions}
            buyable = [a for a in (cfg.watchlist or DEFAULT_WATCHLIST) if a not in open_assets]
            if not buyable:
                st.caption(t("manual_trade_no_assets"))
            else:
                asset_pick = st.selectbox(t("manual_trade_asset"), buyable, key="manual_buy_asset")
                usd_amount = st.number_input(t("manual_trade_usd"), min_value=10.0, value=500.0,
                                             step=50.0, key="manual_buy_usd")
                if st.button(t("manual_trade_fetch_price"), key="manual_buy_fetch"):
                    try:
                        price = fetch_realtime_price(asset_pick)
                    except Exception:
                        price = None
                    if price is None:
                        st.error(t("manual_trade_price_fail"))
                    else:
                        st.session_state["manual_buy_quote"] = {"asset": asset_pick, "price": price}

                quote = st.session_state.get("manual_buy_quote")
                if quote and quote["asset"] == asset_pick:
                    price = quote["price"]
                    st.caption(f"{t('manual_trade_quote')}: {C.fmt_usd(price)}")
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        stop = st.number_input(t("manual_trade_stop"), value=round(price * 0.97, 2),
                                               key="manual_buy_stop")
                    with mc2:
                        tp = st.number_input(t("manual_trade_tp"), value=round(price * 1.06, 2),
                                             key="manual_buy_tp")
                    qty = usd_amount / price if price else 0.0
                    st.caption(t("manual_trade_qty", qty=f"{qty:.6f}"))
                    if st.button(t("manual_trade_confirm"), key="manual_buy_confirm"):
                        try:
                            trade = execute_buy(asset_pick, price, qty, cfg, trade_type="MANUAL",
                                                reason="Manual UI order", stop_loss=stop, take_profit=tp)
                            if trade:
                                st.success(t("manual_trade_success"))
                                st.session_state.pop("manual_buy_quote", None)
                                st.rerun()
                            else:
                                st.error(t("manual_trade_fail_generic"))
                        except Exception as e:
                            st.error(t("manual_trade_fail", error=str(e)))

        st.subheader(t("closed_trades_title"))
        try:
            trades = get_trades(limit=200)
        except Exception as e:
            trades = []
            st.warning(t("loading_error", error=str(e)))
        paired = _pair_closed_trades(trades, limit=20)
        if not paired:
            st.info(t("closed_trades_empty"))
        else:
            rows = []
            for buy, sell in paired:
                rows.append({
                    t("col_asset"): sell["asset"],
                    t("col_side"): t("side_sell"),
                    t("col_qty"): float(sell["amount"] or 0),
                    t("col_entry"): float(buy["price"]) if buy else None,
                    t("col_exit"): float(sell["price"]),
                    t("col_pnl_usd"): sell.get("pnl_usd"),
                    t("col_pnl_pct"): sell.get("pnl"),
                    t("col_r_multiple"): sell.get("r_multiple"),
                    t("col_exit_reason"): sell.get("trade_type"),
                    t("col_time"): C.to_istanbul_str(sell["timestamp"]),
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df, width="stretch", hide_index=True,
                column_config={
                    t("col_qty"): st.column_config.NumberColumn(format="%.4f"),
                    t("col_entry"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_exit"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_pnl_usd"): st.column_config.NumberColumn(format="$%.2f"),
                    t("col_pnl_pct"): st.column_config.NumberColumn(format="%.2f%%"),
                    t("col_r_multiple"): st.column_config.NumberColumn(format="%.2f"),
                },
            )
    except Exception as e:
        st.error(t("tab_error", error=str(e)))


# =========================================================================
# Tab 2: Signal Scanner
# =========================================================================

_DECISION_KEY = {"ENTER": "decision_enter", "HOLD": "decision_hold",
                 "SKIP": "decision_skip", "EXIT": "decision_exit"}


def render_signals_tab():
    try:
        st.subheader(t("signals_title"))
        st.caption(t("signals_desc"))
        try:
            signals = get_latest_signals()
        except Exception as e:
            signals = []
            st.warning(t("loading_error", error=str(e)))

        if not signals:
            st.info(t("signals_empty"))
            return

        overview = []
        for s in signals:
            decision = s.get("decision") or "SKIP"
            overview.append({
                t("col_asset"): s["asset"],
                "_decision_raw": decision,
                t("col_score"): s.get("score"),
                t("col_current"): s.get("price"),
                t("col_time"): C.to_istanbul_str(s["timestamp"]),
            })
        overview_df = pd.DataFrame(overview)
        display_df = overview_df.drop(columns=["_decision_raw"]).copy()
        display_df.insert(1, "Karar" if st.session_state.lang == "TR" else "Decision",
                          [t(_DECISION_KEY.get(d, "decision_skip")) for d in overview_df["_decision_raw"]])
        st.dataframe(
            display_df, width="stretch", hide_index=True,
            column_config={t("col_current"): st.column_config.NumberColumn(format="$%.2f")},
        )

        st.divider()
        for s in signals:
            decision = s.get("decision") or "SKIP"
            try:
                details = json.loads(s.get("details") or "{}")
            except (TypeError, ValueError):
                details = {}
            label = t("expander_label", asset=s["asset"],
                      decision=t(_DECISION_KEY.get(decision, "decision_skip")), score=s.get("score") or 0)
            with st.expander(label):
                gates = details.get("gates") or []
                if gates:
                    st.markdown(f"**{t('gates_title')}**")
                    for g in gates:
                        mark = t("gate_pass") if g.get("passed") else t("gate_fail")
                        st.markdown(f"{mark} **{g.get('name')}** — {g.get('detail')}")

                factors = details.get("factors") or []
                st.markdown(f"**{t('factors_title')}**")
                if factors:
                    for f in factors:
                        st.markdown(f"- {f}")
                else:
                    st.caption(t("no_factors"))

                if details.get("reason"):
                    st.markdown(f"**{t('reason_title')}:** {details['reason']}")
                if details.get("sizing"):
                    st.markdown(f"**{t('sizing_title')}:** {details['sizing']}")
                if details.get("sentiment") is not None:
                    st.markdown(f"**{t('sentiment_title')}:** {details['sentiment']}")
    except Exception as e:
        st.error(t("tab_error", error=str(e)))


# =========================================================================
# Tab 3: Charts
# =========================================================================

def _load_chart_frame(asset: str, timeframe: str):
    """Returns (df, used_fallback). df has an indicator set and a display-tz
    'timestamp' column, or None if no data is available at all."""
    df = None
    try:
        df = fetch_ohlcv(asset, timeframe, limit=300)
    except Exception:
        df = None

    used_fallback = False
    if df is not None and not df.empty:
        df = add_indicators(df)
    elif timeframe == "1h":
        used_fallback = True
        try:
            candles = get_latest_candles(asset, "1h", 300)
        except Exception:
            candles = []
        if not candles:
            return None, used_fallback
        df = pd.DataFrame(candles)
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["timestamp"] = ts.fillna(pd.Timestamp.now(tz="UTC"))
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            return None, used_fallback
    else:
        return None, used_fallback

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(C.ISTANBUL)
    return df, used_fallback


def render_charts_tab(cfg: StrategyConfig):
    try:
        st.subheader(t("charts_title"))
        watchlist = cfg.watchlist or list(DEFAULT_WATCHLIST)
        c1, c2 = st.columns([2, 1])
        with c1:
            asset = st.selectbox(t("select_asset"), watchlist, key="chart_asset")
        with c2:
            timeframe = st.selectbox(t("select_timeframe"), ["1h", "4h", "1d"], key="chart_timeframe")

        try:
            st.caption(t("crypto_always_open") if is_crypto(asset)
                      else (t("market_open") if is_market_open(asset) else t("market_closed")))
        except Exception:
            pass

        df, used_fallback = _load_chart_frame(asset, timeframe)
        if used_fallback:
            st.warning(t("chart_fetch_fail"))

        if df is None or df.empty:
            st.info(t("chart_no_data"))
        else:
            try:
                trades = get_trades(limit=500)
            except Exception:
                trades = []
            buy_markers, sell_markers = [], []
            lo, hi = df["timestamp"].min(), df["timestamp"].max()
            for tr in trades:
                if tr["asset"] != asset:
                    continue
                ts = C.parse_utc(tr["timestamp"])
                if ts is None:
                    continue
                ts_local = ts.astimezone(C.ISTANBUL)
                if not (lo <= ts_local <= hi):
                    continue
                point = (ts_local, float(tr["price"]))
                if tr["side"] == "BUY":
                    buy_markers.append(point)
                elif tr["side"] == "SELL":
                    sell_markers.append(point)

            fig = C.candlestick_chart(df, buy_markers=buy_markers, sell_markers=sell_markers)
            st.plotly_chart(fig, width="stretch")

        st.divider()
        col_ai, col_news = st.columns(2)
        with col_ai:
            st.markdown(f"**{t('ai_run_title')}**")
            try:
                ai_run = get_latest_ai_run(asset)
            except Exception:
                ai_run = None
            if not ai_run:
                st.info(t("ai_no_run"))
            else:
                score = ai_run.get("sentiment_score")
                color = C.color_for(score if score is not None else 0)
                st.markdown(f"{t('ai_score_label')}: <span style='color:{color};font-weight:700'>{score}</span>",
                           unsafe_allow_html=True)
                if ai_run.get("news_digest"):
                    st.caption(f"{t('ai_digest_label')}: {ai_run['news_digest']}")
                if ai_run.get("reason"):
                    st.caption(f"{t('ai_reason_label')}: {ai_run['reason']}")
                st.caption(f"{t('ai_run_time')}: {C.to_istanbul_str(ai_run.get('timestamp'))}")

        with col_news:
            st.markdown(f"**{t('news_title')}**")
            try:
                news = fetch_asset_news(asset, limit=5)
            except Exception:
                news = []
            if not news:
                st.info(t("news_empty"))
            else:
                for item in news:
                    rel = C.relative_time(item.get("pub_date"), st.session_state.lang)
                    st.markdown(f"- [{item['title']}]({item['link']})  \n  <small>{rel}</small>",
                               unsafe_allow_html=True)
    except Exception as e:
        st.error(t("tab_error", error=str(e)))


# =========================================================================
# Tab 4: Backtest
# =========================================================================

def render_backtest_tab(cfg: StrategyConfig):
    st.subheader(t("backtest_title"))
    st.caption(t("backtest_caption"))

    watchlist_options = sorted(set(DEFAULT_WATCHLIST) | set(cfg.watchlist or []))
    default_selection = cfg.watchlist or list(DEFAULT_WATCHLIST)

    with st.form("backtest_form"):
        symbols = st.multiselect(t("bt_symbols_label"), watchlist_options, default=default_selection)
        c1, c2 = st.columns(2)
        with c1:
            months = st.slider(t("bt_months_label"), 1, 23, 12)
            risk_pct = st.slider(t("bt_risk_label"), 0.25, 3.0, float(cfg.risk_per_trade_pct), 0.25)
            atr_mult = st.slider(t("bt_atr_label"), 1.0, 5.0, float(cfg.atr_mult_sl), 0.5)
        with c2:
            initial_cash = st.number_input(t("bt_cash_label"), min_value=100.0, value=10000.0, step=500.0)
            rr_ratio = st.slider(t("bt_rr_label"), 1.0, 4.0, float(cfg.rr_ratio), 0.5)
            market_filter = st.checkbox(t("bt_market_filter_label"), value=cfg.market_filter_enabled)
        run_clicked = st.form_submit_button(t("bt_run_button"), width="stretch")

    clear_clicked = False
    if st.session_state.bt_result is not None:
        clear_clicked = st.button(t("bt_clear_button"))
    if clear_clicked:
        st.session_state.bt_result = None
        st.session_state.bt_params = None
        st.rerun()

    if run_clicked:
        if not symbols:
            st.warning(t("bt_no_symbols_warn"))
        else:
            bt_cfg = StrategyConfig.from_db()
            bt_cfg.risk_per_trade_pct = risk_pct
            bt_cfg.atr_mult_sl = atr_mult
            bt_cfg.rr_ratio = rr_ratio
            bt_cfg.market_filter_enabled = market_filter
            bt_cfg.watchlist = symbols

            st.session_state.bt_running = True
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _progress_cb(frac, message):
                progress_bar.progress(min(max(frac, 0.0), 1.0))
                status_text.caption(message)

            try:
                with st.spinner(t("bt_running_spinner")):
                    result = run_backtest(symbols, bt_cfg, initial_cash=initial_cash,
                                          months=months, progress_cb=_progress_cb)
                st.session_state.bt_result = result
                st.session_state.bt_params = {
                    "symbols": symbols, "months": months, "initial_cash": initial_cash,
                }
            except Exception as e:
                st.error(t("bt_error", error=str(e)))
            finally:
                st.session_state.bt_running = False
                progress_bar.empty()
                status_text.empty()

    result = st.session_state.bt_result
    if result is None:
        return

    st.subheader(t("bt_metrics_title"))
    m = result.metrics or {}
    row1 = st.columns(4)
    row1[0].metric(t("metric_total_return"), C.fmt_pct(m.get("total_return_pct", 0)))
    row1[1].metric(t("metric_cagr"), C.fmt_pct(m.get("cagr_pct", 0)))
    row1[2].metric(t("metric_max_dd"), C.fmt_pct(m.get("max_drawdown_pct", 0), signed=False))
    row1[3].metric(t("metric_sharpe"), C.fmt_num(m.get("sharpe", 0)))
    row2 = st.columns(4)
    row2[0].metric(t("metric_win_rate"), C.fmt_pct(m.get("win_rate_pct", 0), signed=False))
    pf = m.get("profit_factor", 0)
    row2[1].metric(t("metric_profit_factor"), "∞" if pf == float("inf") else C.fmt_num(pf))
    row2[2].metric(t("metric_n_trades"), str(m.get("n_trades", 0)))
    row2[3].metric(t("metric_expectancy_r"), C.fmt_num(m.get("expectancy_r", 0)))

    if result.equity is not None and len(result.equity) > 0:
        st.subheader(t("bt_equity_vs_bench_title"))
        bench_label = m.get("benchmark_symbol", "Benchmark")
        st.plotly_chart(C.backtest_equity_chart(result.equity, result.benchmark, bench_label),
                        width="stretch")

        st.subheader(t("bt_drawdown_title"))
        st.plotly_chart(C.drawdown_chart(result.equity), width="stretch")

    st.subheader(t("bt_trades_title"))
    trades_df = trades_to_frame(result.trades)
    if trades_df.empty:
        st.info(t("bt_trades_empty"))
    else:
        trades_df = trades_df.rename(columns={
            "symbol": t("col_bt_symbol"), "entry_time": t("col_bt_entry_time"),
            "exit_time": t("col_bt_exit_time"), "entry": t("col_bt_entry"), "exit": t("col_bt_exit"),
            "qty": t("col_bt_qty"), "pnl_usd": t("col_bt_pnl_usd"), "pnl_pct": t("col_bt_pnl_pct"),
            "r_multiple": t("col_bt_r"), "exit_reason": t("col_bt_reason"), "score": t("col_bt_score"),
        })
        st.dataframe(
            trades_df, width="stretch", hide_index=True,
            column_config={
                t("col_bt_entry"): st.column_config.NumberColumn(format="$%.2f"),
                t("col_bt_exit"): st.column_config.NumberColumn(format="$%.2f"),
                t("col_bt_qty"): st.column_config.NumberColumn(format="%.4f"),
                t("col_bt_pnl_usd"): st.column_config.NumberColumn(format="$%.2f"),
                t("col_bt_pnl_pct"): st.column_config.NumberColumn(format="%.2f%%"),
                t("col_bt_r"): st.column_config.NumberColumn(format="%.2f"),
            },
        )

    if result.errors:
        st.subheader(t("bt_errors_title"))
        for err in result.errors:
            st.warning(err)


# =========================================================================
# Tab 5: Settings
# =========================================================================

def render_settings_tab(cfg: StrategyConfig):
    st.subheader(t("settings_title"))

    with st.expander(t("exp_watchlist"), expanded=True):
        current = parse_watchlist(get_config("selected_assets") or "")
        in_default = [s for s in current if s in DEFAULT_WATCHLIST]
        extra_current = [s for s in current if s not in DEFAULT_WATCHLIST]
        chosen = st.multiselect(t("watchlist_multiselect"), DEFAULT_WATCHLIST, default=in_default,
                                key="settings_watchlist")
        extra_text = st.text_input(t("watchlist_extra"), value=", ".join(extra_current),
                                   help=t("watchlist_extra_help"), key="settings_watchlist_extra")
        if st.button(t("save_button"), key="save_watchlist"):
            extra = [normalize_symbol(s) for s in extra_text.split(",") if s.strip()]
            merged, seen = [], set()
            for s in chosen + extra:
                if s not in seen:
                    seen.add(s)
                    merged.append(s)
            save_config("selected_assets", ",".join(merged) if merged else ",".join(DEFAULT_WATCHLIST))
            st.success(t("settings_saved"))
            st.toast(t("settings_saved"))

    with st.expander(t("exp_risk")):
        risk_per_trade = st.slider(t("risk_per_trade"), 0.25, 3.0, float(cfg.risk_per_trade_pct), 0.25,
                                   key="settings_risk_per_trade")
        max_open = st.slider(t("max_open_positions"), 1, 10, int(cfg.max_open_positions), 1,
                             key="settings_max_open")
        max_pos_pct = st.slider(t("max_position_pct"), 5.0, 50.0, float(cfg.max_position_pct), 1.0,
                                key="settings_max_pos_pct")
        max_exposure = st.slider(t("max_total_exposure_pct"), 20.0, 100.0, float(cfg.max_total_exposure_pct), 5.0,
                                 key="settings_max_exposure")
        daily_loss = st.slider(t("daily_loss_limit_pct"), 1.0, 10.0, float(cfg.daily_loss_limit_pct), 0.5,
                              key="settings_daily_loss")
        cooldown = st.slider(t("cooldown_hours"), 0, 72, int(cfg.cooldown_hours), 1,
                             key="settings_cooldown")
        if st.button(t("save_button"), key="save_risk"):
            save_config("risk_per_trade_pct", risk_per_trade)
            save_config("max_open_positions", max_open)
            save_config("max_position_pct", max_pos_pct)
            save_config("max_total_exposure_pct", max_exposure)
            save_config("daily_loss_limit_pct", daily_loss)
            save_config("cooldown_hours", cooldown)
            st.success(t("settings_saved"))
            st.toast(t("settings_saved"))

    with st.expander(t("exp_strategy")):
        atr_mult = st.slider(t("atr_mult_sl"), 1.0, 5.0, float(cfg.atr_mult_sl), 0.5, key="settings_atr_mult")
        rr_ratio = st.slider(t("rr_ratio"), 1.0, 4.0, float(cfg.rr_ratio), 0.5, key="settings_rr_ratio")
        trail_mult = st.slider(t("trail_atr_mult"), 1.0, 5.0, float(cfg.trail_atr_mult), 0.5,
                               key="settings_trail_mult")
        max_holding = st.slider(t("max_holding_days"), 0, 30, int(cfg.max_holding_days), 1,
                                key="settings_max_holding")
        market_filter = st.checkbox(t("market_filter_enabled"), value=cfg.market_filter_enabled,
                                    key="settings_market_filter")
        c1, c2 = st.columns(2)
        with c1:
            fee_pct = st.number_input(t("fee_pct"), min_value=0.0, max_value=1.0, value=float(cfg.fee_pct),
                                      step=0.0001, format="%.4f", key="settings_fee_pct")
        with c2:
            slippage_pct = st.number_input(t("slippage_pct"), min_value=0.0, max_value=1.0,
                                           value=float(cfg.slippage_pct), step=0.0001, format="%.4f",
                                           key="settings_slippage_pct")
        if st.button(t("save_button"), key="save_strategy"):
            save_config("atr_mult_sl", atr_mult)
            save_config("rr_ratio", rr_ratio)
            save_config("trail_atr_mult", trail_mult)
            save_config("max_holding_days", max_holding)
            save_config("market_filter_enabled", "true" if market_filter else "false")
            save_config("fee_pct", fee_pct)
            save_config("slippage_pct", slippage_pct)
            st.success(t("settings_saved"))
            st.toast(t("settings_saved"))

    with st.expander(t("exp_ai")):
        ai_enabled = st.checkbox(t("ai_enabled"), value=cfg.ai_enabled, key="settings_ai_enabled")
        existing_key = get_config("gemini_api_key") or ""
        st.caption(t("gemini_api_key_set") if existing_key else t("gemini_api_key_missing"))
        new_key = st.text_input(t("gemini_api_key"), value="", type="password", key="settings_gemini_key")
        summarizer_model = st.text_input(t("summarizer_model"),
                                         value=get_config("summarizer_model", "gemini-3.1-flash-lite"),
                                         key="settings_summarizer_model")
        sentiment_model = st.text_input(t("sentiment_model"),
                                        value=get_config("sentiment_model", "gemini-3.5-flash"),
                                        key="settings_sentiment_model")
        min_sentiment = st.slider(t("min_ai_sentiment_threshold"), 1, 10, int(cfg.ai_confirm_score),
                                  key="settings_min_sentiment")
        news_freshness = st.slider(t("news_freshness_hours"), 6, 72,
                                   int(get_config("news_freshness_hours", "24")), key="settings_news_freshness")
        if st.button(t("save_button"), key="save_ai"):
            save_config("ai_enabled", "true" if ai_enabled else "false")
            if new_key.strip():
                save_config("gemini_api_key", new_key.strip())
            save_config("summarizer_model", summarizer_model)
            save_config("sentiment_model", sentiment_model)
            save_config("min_ai_sentiment_threshold", min_sentiment)
            save_config("news_freshness_hours", news_freshness)
            st.success(t("settings_saved"))
            st.toast(t("settings_saved"))

    with st.expander(t("exp_system")):
        try:
            interval_default = int(get_config("simulation_interval_seconds", "300"))
        except (TypeError, ValueError):
            interval_default = 300
        sim_interval = st.slider(t("simulation_interval_seconds"), 60, 3600, interval_default, 30,
                                 key="settings_sim_interval")
        if st.button(t("save_button"), key="save_system"):
            save_config("simulation_interval_seconds", sim_interval)
            st.success(t("settings_saved"))
            st.toast(t("settings_saved"))

    st.divider()
    st.subheader(t("danger_zone_title"))
    st.warning(t("reset_desc"))
    confirm = st.checkbox(t("reset_confirm_checkbox"), key="reset_confirm")
    if st.button(t("reset_button"), type="primary", key="reset_button"):
        if not confirm:
            st.warning(t("reset_need_confirm"))
        else:
            try:
                conn = get_connection()
                cur = conn.cursor()
                for table in ("trades", "candles", "logs", "equity_history", "signals", "cooldowns"):
                    cur.execute(f"DELETE FROM {table}")
                cur.execute("UPDATE portfolio SET balance = 10000, avg_entry_price = 0 WHERE asset = 'USD'")
                cur.execute("DELETE FROM portfolio WHERE asset != 'USD'")
                conn.commit()
                conn.close()
                st.session_state.bt_result = None
                st.session_state.bt_params = None
                log_event("WARNING", "UI", "Portfolio reset from dashboard.")
                st.success(t("reset_success"))
                st.rerun()
            except Exception as e:
                st.error(t("loading_error", error=str(e)))


# =========================================================================
# Tab 6: Logs
# =========================================================================

_LEVEL_EMOJI = {"INFO": "🔵", "WARNING": "🟡", "ERROR": "🔴", "SUCCESS": "🟢"}


def render_logs_tab():
    try:
        st.subheader(t("logs_title"))
        try:
            all_logs = get_logs(limit=1000)
        except Exception as e:
            all_logs = []
            st.warning(t("loading_error", error=str(e)))

        if not all_logs:
            st.info(t("logs_empty"))
        else:
            modules = sorted({l.get("module") or "-" for l in all_logs})
            levels = ["INFO", "WARNING", "ERROR", "SUCCESS"]

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                level_filter = st.multiselect(t("level_filter_label"), levels, default=levels,
                                              key="logs_level_filter")
            with c2:
                module_filter = st.multiselect(t("module_filter_label"), modules, default=modules,
                                               key="logs_module_filter")
            with c3:
                limit = st.slider(t("limit_label"), 10, 500, 100, 10, key="logs_limit")

            filtered = [l for l in all_logs
                       if (l.get("level") in level_filter) and ((l.get("module") or "-") in module_filter)]
            filtered = filtered[:limit]

            rows = [{
                t("col_time"): C.to_istanbul_str(l["timestamp"], "%Y-%m-%d %H:%M:%S"),
                t("col_level"): f"{_LEVEL_EMOJI.get(l.get('level'), '⚪')} {l.get('level')}",
                t("col_module"): l.get("module"),
                t("col_message"): l.get("message"),
            } for l in filtered]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)

        st.divider()
        st.subheader(t("ai_audit_title"))
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT * FROM ai_runs ORDER BY id DESC LIMIT 20")
            ai_runs = [dict(r) for r in cur.fetchall()]
            conn.close()
        except Exception as e:
            ai_runs = []
            st.warning(t("loading_error", error=str(e)))

        if not ai_runs:
            st.info(t("ai_audit_empty"))
        else:
            for run in ai_runs:
                label = t("ai_audit_label", time=C.to_istanbul_str(run.get("timestamp")),
                          asset=run.get("asset"), score=run.get("sentiment_score"))
                with st.expander(label):
                    if run.get("news_digest"):
                        st.markdown(f"**{t('ai_digest_label')}:** {run['news_digest']}")
                    if run.get("reason"):
                        st.markdown(f"**{t('ai_reason_label')}:** {run['reason']}")
    except Exception as e:
        st.error(t("tab_error", error=str(e)))


# =========================================================================
# Main
# =========================================================================

def main():
    st.set_page_config(page_title="Sentix", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")
    _init_session_state()
    _inject_css()

    if not _check_password():
        return

    try:
        init_db()
    except Exception as e:
        st.error(t("loading_error", error=str(e)))
        return

    try:
        cfg = StrategyConfig.from_db()
    except Exception:
        cfg = StrategyConfig()

    _render_sidebar()
    _maybe_autorefresh()

    tabs = st.tabs([
        t("tab_portfolio"), t("tab_signals"), t("tab_charts"),
        t("tab_backtest"), t("tab_settings"), t("tab_logs"),
    ])
    with tabs[0]:
        render_portfolio_tab(cfg)
    with tabs[1]:
        render_signals_tab()
    with tabs[2]:
        render_charts_tab(cfg)
    with tabs[3]:
        render_backtest_tab(cfg)
    with tabs[4]:
        render_settings_tab(cfg)
    with tabs[5]:
        render_logs_tab()


if __name__ == "__main__":
    main()
