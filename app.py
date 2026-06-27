"""
Live watchlist dashboard.

Run with:  streamlit run app.py

Layers used:
    config.py      -> what to watch and alert on
    data.py        -> fetch OHLCV
    indicators.py  -> compute signals
    alerts.py      -> edge-triggered alerts + notifications
    paper.py       -> optional Alpaca paper trading
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as C
import data as D
import indicators as I
import alerts as A
from paper import PaperBroker

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


st.set_page_config(page_title="Quant Watchlist", layout="wide")

# --- persistent state ----------------------------------------------------
if "alert_state" not in st.session_state:
    st.session_state.alert_state = {}      # de-dup memory
if "alert_feed" not in st.session_state:
    st.session_state.alert_feed = []       # rolling log of fired alerts

broker = PaperBroker(
    C.ALPACA_API_KEY, C.ALPACA_SECRET_KEY, C.ALPACA_PAPER_URL, C.ALPACA_ENABLED
)

# --- auto refresh --------------------------------------------------------
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=C.REFRESH_SECONDS * 1000, key="auto")
else:
    st.caption("Install streamlit-autorefresh for live polling; using manual refresh.")
    st.button("Refresh now")


@st.cache_data(ttl=C.REFRESH_SECONDS, show_spinner=False)
def load(tickers, period, interval):
    return D.get_watchlist(list(tickers), period=period, interval=interval)


hist = load(tuple(C.WATCHLIST), C.DATA_PERIOD, C.DATA_INTERVAL)

# --- compute snapshots + evaluate alerts ---------------------------------
rows, snapshots = [], {}
for tk, df in hist.items():
    ind = I.compute_indicators(df, C.INDICATOR_CFG)
    snapshots[tk] = ind
    trend = I.trend_label(ind)

    for fired in A.evaluate(tk, ind, C.ALERT_RULES, st.session_state.alert_state):
        A.notify(fired, C.WEBHOOK_URL)
        st.session_state.alert_feed.insert(0, fired)
        if (
            C.AUTO_PAPER_TRADE
            and broker.is_enabled
            and fired["name"] in C.PAPER_TRADE_ON
        ):
            res = broker.market_buy(tk, C.PAPER_ORDER_QTY)
            fired["paper_order"] = res
    st.session_state.alert_feed = st.session_state.alert_feed[:100]

    rows.append(
        {
            "Ticker": tk,
            "Price": round(ind["price"], 2),
            "% Chg": round(ind["pct_change"], 2) if pd.notna(ind["pct_change"]) else None,
            "RSI": round(ind["rsi"], 1) if pd.notna(ind["rsi"]) else None,
            "MACD hist": round(ind["macd_hist"], 3),
            "ATR %": round(ind["atr_pct"], 2) if pd.notna(ind["atr_pct"]) else None,
            "Support": round(ind["support"], 2),
            "Resistance": round(ind["resistance"], 2),
            "Vol x": round(ind["vol_ratio"], 2) if pd.notna(ind["vol_ratio"]) else None,
            "Trend": trend,
        }
    )

st.title("Quant watchlist")
st.caption(
    f"{len(rows)} symbols · {C.DATA_INTERVAL} bars · refresh {C.REFRESH_SECONDS}s · "
    f"paper trading {'ON' if broker.is_enabled else 'off'}"
)

# --- watchlist table -----------------------------------------------------
if rows:
    table = pd.DataFrame(rows).set_index("Ticker")

    def color_trend(v):
        return {
            "bullish": "color: #1d9e75; font-weight: 600",
            "bearish": "color: #e24b4a; font-weight: 600",
        }.get(v, "color: #888780")

    def color_pct(v):
        if v is None:
            return ""
        return "color: #1d9e75" if v >= 0 else "color: #e24b4a"

    styled = (
        table.style.map(color_trend, subset=["Trend"])
        .map(color_pct, subset=["% Chg"])
        .format(na_rep="—")
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.warning("No data returned. Check your symbols / internet connection.")

# --- two columns: detail chart + alert feed ------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Detail")
    if snapshots:
        sel = st.selectbox("Symbol", list(snapshots.keys()))
        df = hist[sel]
        ind = snapshots[sel]
        if _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name=sel,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=I.ema(df["Close"], C.INDICATOR_CFG["ema_fast"]),
                    line=dict(width=1, dash="dot"), name=f"EMA{C.INDICATOR_CFG['ema_fast']}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=I.sma(df["Close"], C.INDICATOR_CFG["sma_fast"]),
                    line=dict(width=1), name=f"SMA{C.INDICATOR_CFG['sma_fast']}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=I.sma(df["Close"], C.INDICATOR_CFG["sma_slow"]),
                    line=dict(width=1), name=f"SMA{C.INDICATOR_CFG['sma_slow']}",
                )
            )
            for label in ("38.2%", "50%", "61.8%"):
                fig.add_hline(
                    y=ind["fib"][label], line_dash="dot", line_width=1,
                    annotation_text=f"fib {label}", annotation_position="right",
                )
            fig.add_hline(y=ind["support"], line_color="#1d9e75", line_width=1)
            fig.add_hline(y=ind["resistance"], line_color="#e24b4a", line_width=1)
            fig.update_layout(
                height=460, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False, showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df["Close"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${ind['price']:.2f}", f"{ind['pct_change']:.2f}%")
        c2.metric("RSI", f"{ind['rsi']:.1f}")
        c3.metric("Support", f"${ind['support']:.2f}")
        c4.metric("Resistance", f"${ind['resistance']:.2f}")

with right:
    st.subheader("Alerts")
    if st.session_state.alert_feed:
        feed = pd.DataFrame(st.session_state.alert_feed)[
            ["time", "ticker", "name", "price"]
        ]
        st.dataframe(feed, use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts yet. They appear here as conditions trigger.")

    if broker.is_enabled:
        st.subheader("Paper account")
        acct = broker.account()
        if acct:
            a, b = st.columns(2)
            a.metric("Equity", f"${acct['equity']:,.0f}")
            b.metric("Buying power", f"${acct['buying_power']:,.0f}")
        pos = broker.positions()
        if pos:
            st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
