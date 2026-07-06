"""Presentation helpers for the Sentix dashboard: timestamp/number formatting
and Plotly chart builders. No Streamlit calls live here — this module only
turns data into strings and figures so ``ui/app.py`` stays focused on layout.
"""
from __future__ import annotations

import email.utils
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ISTANBUL = ZoneInfo("Europe/Istanbul")

# ----------------- palette -----------------
COLOR_UP = "#00D18F"
COLOR_DOWN = "#FF5C5C"
COLOR_NEUTRAL = "#8A97A5"
COLOR_ACCENT = "#3AA0FF"
COLOR_GRID = "#1E2630"
PAPER_BG = "rgba(0,0,0,0)"

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PAPER_BG,
    font=dict(color="#E6EDF3"),
    margin=dict(l=40, r=20, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
)


# ----------------- timestamps -----------------

def parse_utc(raw):
    """Parses a DB timestamp (ISO string, 'YYYY-MM-DD HH:MM:SS', or datetime)
    into a tz-aware UTC datetime. Returns None on failure."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, pd.Timestamp):
        dt = raw.to_pydatetime()
    else:
        s = str(raw).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # RFC 822 (e.g. Google News RSS pubDate: "Sun, 06 Jul 2026 10:00:00 GMT")
                    dt = email.utils.parsedate_to_datetime(s)
                except (TypeError, ValueError):
                    return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_istanbul_str(raw, fmt: str = "%Y-%m-%d %H:%M") -> str:
    dt = parse_utc(raw)
    if dt is None:
        return "-"
    return dt.astimezone(ISTANBUL).strftime(fmt)


def relative_time(raw, lang: str = "TR") -> str:
    """'5 dk önce' / '5m ago' style relative label."""
    dt = parse_utc(raw)
    if dt is None:
        return "-"
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 0:
        secs = 0
    if secs < 60:
        return "az önce" if lang == "TR" else "just now"
    mins = int(secs // 60)
    if mins < 60:
        return f"{mins} dk önce" if lang == "TR" else f"{mins}m ago"
    hours = int(mins // 60)
    if hours < 24:
        return f"{hours} sa önce" if lang == "TR" else f"{hours}h ago"
    days = int(hours // 24)
    return f"{days} gün önce" if lang == "TR" else f"{days}d ago"


def seconds_since(raw) -> float:
    dt = parse_utc(raw)
    if dt is None:
        return float("inf")
    return (datetime.now(timezone.utc) - dt).total_seconds()


# ----------------- number formatting -----------------

def fmt_usd(x, decimals: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


def fmt_usd_signed(x, decimals: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.{decimals}f}"


def fmt_pct(x, decimals: int = 2, signed: bool = True) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if signed:
        return f"{x:+.{decimals}f}%"
    return f"{x:.{decimals}f}%"


def fmt_num(x, decimals: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{x:,.{decimals}f}"


def color_for(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return COLOR_NEUTRAL
    if v > 0:
        return COLOR_UP
    if v < 0:
        return COLOR_DOWN
    return COLOR_NEUTRAL


# ----------------- charts -----------------

def equity_area_chart(equity_history: list[dict], initial_cash: float) -> go.Figure:
    """Area chart of NAV over time vs. a flat initial-cash baseline. The fill
    color reflects whether the strategy currently sits above or below the
    starting balance."""
    fig = go.Figure()
    if not equity_history:
        fig.update_layout(**CHART_LAYOUT, height=360,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    xs = [parse_utc(r["timestamp"]).astimezone(ISTANBUL) for r in equity_history]
    ys = [float(r["nav"]) for r in equity_history]
    final_up = ys[-1] >= initial_cash
    line_color = COLOR_UP if final_up else COLOR_DOWN
    fill_rgba = "rgba(0,209,143,0.15)" if final_up else "rgba(255,92,92,0.15)"

    fig.add_trace(go.Scatter(
        x=xs, y=[initial_cash] * len(xs), mode="lines", name="Başlangıç",
        line=dict(color=COLOR_GRID, width=1, dash="dot"), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="NAV",
        line=dict(color=line_color, width=2), fill="tonexty", fillcolor=fill_rgba,
    ))
    fig.update_layout(**CHART_LAYOUT, height=380,
                      xaxis=dict(gridcolor=COLOR_GRID, showgrid=True),
                      yaxis=dict(gridcolor=COLOR_GRID, showgrid=True, tickprefix="$"))
    return fig


def candlestick_chart(df: pd.DataFrame, buy_markers=None, sell_markers=None,
                      show_ema=True) -> go.Figure:
    """4-row chart: price+EMA candles, volume, RSI, MACD. Expects a frame
    with a tz-aware 'timestamp' column already converted to display tz."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03,
        specs=[[{"secondary_y": False}]] * 4,
    )

    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
        increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
        name="Fiyat", showlegend=False,
    ), row=1, col=1)

    if show_ema and "ema" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema"], name="EMA20",
                                 line=dict(color=COLOR_ACCENT, width=1.3)), row=1, col=1)
    if show_ema and "ema50" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema50"], name="EMA50",
                                 line=dict(color="#C48EFF", width=1.3)), row=1, col=1)

    if buy_markers:
        fig.add_trace(go.Scatter(
            x=[m[0] for m in buy_markers], y=[m[1] for m in buy_markers],
            mode="markers", name="AL", marker=dict(symbol="triangle-up", size=12, color=COLOR_UP,
                                                    line=dict(width=1, color="#FFFFFF")),
        ), row=1, col=1)
    if sell_markers:
        fig.add_trace(go.Scatter(
            x=[m[0] for m in sell_markers], y=[m[1] for m in sell_markers],
            mode="markers", name="SAT", marker=dict(symbol="triangle-down", size=12, color=COLOR_DOWN,
                                                     line=dict(width=1, color="#FFFFFF")),
        ), row=1, col=1)

    if "volume" in df.columns:
        vol_colors = [COLOR_UP if c >= o else COLOR_DOWN for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], marker_color=vol_colors,
                             name="Hacim", showlegend=False, opacity=0.7), row=2, col=1)

    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rsi"], name="RSI",
                                 line=dict(color="#FFC24B", width=1.3), showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line=dict(color=COLOR_DOWN, width=1, dash="dot"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color=COLOR_UP, width=1, dash="dot"), row=3, col=1)

    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd"], name="MACD",
                                 line=dict(color=COLOR_ACCENT, width=1.3), showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["macd_signal"], name="Sinyal",
                                 line=dict(color="#FFC24B", width=1.2), showlegend=False), row=4, col=1)
        if "macd_hist" in df.columns:
            hist_colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in df["macd_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df["timestamp"], y=df["macd_hist"], marker_color=hist_colors,
                                 name="Histogram", showlegend=False, opacity=0.6), row=4, col=1)

    fig.update_layout(**CHART_LAYOUT, height=760, xaxis4=dict(rangeslider=dict(visible=False)),
                      xaxis=dict(rangeslider=dict(visible=False)))
    for r in range(1, 5):
        fig.update_xaxes(gridcolor=COLOR_GRID, row=r, col=1)
        fig.update_yaxes(gridcolor=COLOR_GRID, row=r, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def backtest_equity_chart(equity: pd.Series, benchmark: pd.Series | None,
                          benchmark_label: str = "Benchmark") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Strateji",
                             line=dict(color=COLOR_UP, width=2)))
    if benchmark is not None and len(benchmark) > 1:
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, mode="lines",
                                 name=benchmark_label, line=dict(color=COLOR_NEUTRAL, width=1.5, dash="dash")))
    fig.update_layout(**CHART_LAYOUT, height=400,
                      xaxis=dict(gridcolor=COLOR_GRID), yaxis=dict(gridcolor=COLOR_GRID, tickprefix="$"))
    return fig


def drawdown_chart(equity: pd.Series) -> go.Figure:
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown",
                             line=dict(color=COLOR_DOWN, width=1.5), fill="tozeroy",
                             fillcolor="rgba(255,92,92,0.18)"))
    fig.update_layout(**CHART_LAYOUT, height=260,
                      xaxis=dict(gridcolor=COLOR_GRID), yaxis=dict(gridcolor=COLOR_GRID, ticksuffix="%"))
    return fig


def empty_figure(message: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**CHART_LAYOUT, height=300, xaxis=dict(visible=False), yaxis=dict(visible=False))
    if message:
        fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=COLOR_NEUTRAL))
    return fig
