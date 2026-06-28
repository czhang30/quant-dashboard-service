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

import datetime as dt

import pandas as pd
import streamlit as st

import config as C
import data as D
import indicators as I
import alerts as A
import advisor as Adv
import positions as P
from paper import PaperBroker

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


st.set_page_config(page_title="Quant Watchlist", layout="wide")

# --- persistent state ----------------------------------------------------
if "alert_state" not in st.session_state:
    st.session_state.alert_state = {}      # de-dup memory
if "alert_feed" not in st.session_state:
    st.session_state.alert_feed = []       # rolling log of fired alerts
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(C.WATCHLIST)

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


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_option(ticker: str, direction: str, price: float) -> dict | None:
    """Cache options recommendations for 15 min per ticker/direction/price."""
    return Adv.best_option(ticker, direction, price)


@st.cache_data(ttl=60, show_spinner=False)
def _monitor_price(ticker: str, option_type: str, strike: float, expiry: str) -> float | None:
    """Re-fetch a specific option's mid-price every 60 s for P&L monitoring."""
    return Adv.current_option_price(ticker, option_type, strike, expiry)


# --- sidebar: watchlist management ---------------------------------------
with st.sidebar:
    with st.expander("Indicators", expanded=False):
        data_interval = st.selectbox(
            "Data interval",
            options=["1d", "1wk", "1h", "30m", "15m", "5m"],
            index=["1d", "1wk", "1h", "30m", "15m", "5m"].index(C.DATA_INTERVAL),
        )
        fib_lookback = st.slider(
            "Fib lookback (bars)", min_value=5, max_value=200,
            value=C.INDICATOR_CFG["fib_lookback"], step=5,
        )

    with st.expander("Watchlist", expanded=False):
        add_col, btn_col = st.columns([3, 1])
        new_ticker = add_col.text_input("Add ticker", placeholder="e.g. AAPL", label_visibility="collapsed")
        if btn_col.button("Add", use_container_width=True):
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
                st.rerun()

        to_remove = []
        for tk in list(st.session_state.watchlist):
            col_tk, col_rm = st.columns([4, 1])
            col_tk.write(tk)
            if col_rm.button("✕", key=f"rm_{tk}"):
                to_remove.append(tk)
        for tk in to_remove:
            st.session_state.watchlist.remove(tk)
        if to_remove:
            st.rerun()

hist    = load(tuple(st.session_state.watchlist), C.DATA_PERIOD, data_interval)
hist_lt = load(tuple(st.session_state.watchlist), "2y", data_interval)
cs_mom    = I.cross_sectional_momentum(hist)
cs_mom_lt = I.cross_sectional_momentum(hist_lt)

# --- compute snapshots + evaluate alerts ---------------------------------
rows, snapshots = [], {}
indicator_cfg = {**C.INDICATOR_CFG, "fib_lookback": fib_lookback}
for tk, df in hist.items():
    ind = I.compute_indicators(df, indicator_cfg)
    ind.update(cs_mom.get(tk, {}))
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

    mom_rank = ind.get("mom_rank")
    mom_total = ind.get("mom_rank_total")
    rows.append(
        {
            "Ticker":    tk,
            "Price":     round(ind["price"], 2),
            "% Chg":     round(ind["pct_change"], 2) if pd.notna(ind["pct_change"]) else None,
            "RSI":       round(ind["rsi"], 1) if pd.notna(ind["rsi"]) else None,
            "%B":        round(ind["bb_pct_b"], 2) if pd.notna(ind.get("bb_pct_b", float("nan"))) else None,
            "Z":         round(ind["zscore"], 2) if pd.notna(ind.get("zscore", float("nan"))) else None,
            "RVol%":     round(ind["realized_vol"], 1) if pd.notna(ind.get("realized_vol", float("nan"))) else None,
            "ADX":       round(ind["adx"], 1) if pd.notna(ind.get("adx", float("nan"))) else None,
            "Mom Rnk":   f"{mom_rank}/{mom_total}" if mom_rank else None,
            "MACD hist": round(ind["macd_hist"], 3),
            "ATR %":     round(ind["atr_pct"], 2) if pd.notna(ind["atr_pct"]) else None,
            "Vol x":     round(ind["vol_ratio"], 2) if pd.notna(ind["vol_ratio"]) else None,
            "Trend":     trend,
        }
    )

# Long-term snapshots: 2y data, 252-bar fib window (≈ 52-week range)
lt_indicator_cfg = {**C.INDICATOR_CFG, "fib_lookback": 252, "sr_lookback": 60}
lt_snapshots: dict = {}
for tk, df_lt in hist_lt.items():
    ind_lt = I.compute_indicators(df_lt, lt_indicator_cfg)
    ind_lt.update(cs_mom_lt.get(tk, {}))
    lt_snapshots[tk] = ind_lt

st.title("Quant watchlist")
st.caption(
    f"{len(rows)} symbols · {data_interval} bars · refresh {C.REFRESH_SECONDS}s · "
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

    def color_pct_b(v):
        if v is None:
            return ""
        if v <= 0.05:
            return "color: #1d9e75; font-weight: 600"
        if v >= 0.95:
            return "color: #e24b4a; font-weight: 600"
        return ""

    def color_zscore(v):
        if v is None:
            return ""
        if v <= -2:
            return "color: #1d9e75; font-weight: 600"
        if v >= 2:
            return "color: #e24b4a; font-weight: 600"
        return ""

    def color_adx(v):
        if v is None:
            return ""
        return "color: #f5a623; font-weight: 600" if v >= 25 else "color: #888780"

    styled = (
        table.style.map(color_trend, subset=["Trend"])
        .map(color_pct, subset=["% Chg"])
        .map(color_pct_b, subset=["%B"])
        .map(color_zscore, subset=["Z"])
        .map(color_adx, subset=["ADX"])
        .format(na_rep="—")
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.warning("No data returned. Check your symbols / internet connection.")

# --- detail chart (full width) -------------------------------------------
st.subheader("Detail")
if snapshots:
    sym_col, sr_col = st.columns([1, 1])
    sel = sym_col.selectbox("Symbol", list(snapshots.keys()))
    sr_method = sr_col.selectbox(
        "S/R Method",
        ["Price Extremes", "Swing Clusters", "Volume Profile", "Fractals", "Moving Averages", "KDE"],
    )
    df = hist[sel]
    ind = snapshots[sel]
    sr_levels = I.compute_sr(df, sr_method, indicator_cfg)

    if _HAS_PLOTLY:
        macd_line, signal_line, macd_hist = I.macd(df["Close"])
        hist_colors = ["#1d9e75" if v >= 0 else "#e24b4a" for v in macd_hist]
        bb_up_s, _, bb_lo_s, _, _ = I.bollinger_bands(df["Close"])

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.04,
        )
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=sel,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb_up_s,
                line=dict(width=0.8, color="rgba(160,160,160,0.6)", dash="dot"),
                name="BB Upper", showlegend=True,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=bb_lo_s,
                line=dict(width=0.8, color="rgba(160,160,160,0.6)", dash="dot"),
                fill="tonexty", fillcolor="rgba(160,160,160,0.07)",
                name="BB Lower", showlegend=True,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=I.ema(df["Close"], C.INDICATOR_CFG["ema_fast"]),
                line=dict(width=1, dash="dot", color="green"), name=f"EMA{C.INDICATOR_CFG['ema_fast']}",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=I.sma(df["Close"], C.INDICATOR_CFG["sma_fast"]),
                line=dict(width=1, color="blue"), name=f"SMA{C.INDICATOR_CFG['sma_fast']}",
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=I.sma(df["Close"], C.INDICATOR_CFG["sma_slow"]),
                line=dict(width=1, color="red"), name=f"SMA{C.INDICATOR_CFG['sma_slow']}",
            ), row=1, col=1,
        )

        def add_level(fig, y, label, color, dash="dash"):
            fig.add_hline(
                y=y, line_color=color, line_width=1, line_dash=dash,
                annotation_text=f"{label}  ${y:.2f}",
                annotation_position="right",
                annotation_font_color=color,
                row=1, col=1,
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=0, y=y,
                text=f"<b>${y:.2f}</b>",
                showarrow=False,
                xanchor="right",
                font=dict(size=9, color=color),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor=color,
                borderwidth=1,
            )

        for label in ("38.2%", "50%", "61.8%"):
            add_level(fig, ind["fib"][label], f"fib {label}", "#888888", dash="dot")
        for label, price in sr_levels["support"]:
            add_level(fig, price, label, "#1d9e75")
        for label, price in sr_levels["resistance"]:
            add_level(fig, price, label, "#e24b4a")

        fig.add_trace(
            go.Bar(
                x=df.index, y=macd_hist, name="MACD Hist",
                marker_color=hist_colors, showlegend=False,
            ), row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd_line,
                line=dict(width=1, color="blue"), name="MACD",
            ), row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=signal_line,
                line=dict(width=1, color="orange"), name="Signal",
            ), row=2, col=1,
        )
        fig.update_layout(
            height=680, margin=dict(l=60, r=120, t=10, b=0),
            xaxis_rangeslider_visible=False, showlegend=True,
        )
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df["Close"])

    primary_sup = sr_levels["support"][0][1] if sr_levels["support"] else None
    primary_res = sr_levels["resistance"][0][1] if sr_levels["resistance"] else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${ind['price']:.2f}", f"{ind['pct_change']:.2f}%")
    c2.metric("RSI", f"{ind['rsi']:.1f}")
    c3.metric("Support", f"${primary_sup:.2f}" if primary_sup else "—")
    c4.metric("Resistance", f"${primary_res:.2f}" if primary_res else "—")

    _na = lambda v: f"{v:.2f}" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "—"
    adx_val = ind.get("adx")
    adx_str = f"{adx_val:.1f} {'▲ trend' if ind.get('adx_trending') else '▽ range'}" if adx_val and not pd.isna(adx_val) else "—"
    mom_rank = ind.get("mom_rank")
    mom_str  = f"{mom_rank}/{ind.get('mom_rank_total')}" if mom_rank else "—"
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("%B",       _na(ind.get("bb_pct_b")))
    q2.metric("Z-score",  _na(ind.get("zscore")))
    q3.metric("RVol %",   f"{ind['realized_vol']:.1f}" if pd.notna(ind.get("realized_vol", float("nan"))) else "—")
    q4.metric("ADX",      adx_str)
    q5.metric("Mom Rank", mom_str)

# --- alerts + paper account (below chart) --------------------------------
st.subheader("Alerts")
if st.session_state.alert_feed:
    feed = pd.DataFrame(st.session_state.alert_feed)[["time", "ticker", "name", "price"]]
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

# ── Trade Recommendations ──────────────────────────────────────────────────────
st.subheader("💡 Trade Recommendations")

# Confluence scores for every ticker
confluence = {tk: Adv.confluence_score(ind) for tk, ind in snapshots.items()}
eligible   = [(tk, cs) for tk, cs in confluence.items() if cs["score"] >= Adv.MIN_CONFLUENCE]

# Keys of positions already being tracked (avoid recommending duplicates)
_active_keys = {
    f"{p['ticker']}|{p.get('option_type','')}|{p.get('strike','')}|{p.get('expiration','')}"
    for p in P.active()
}
_active_swing    = {p["ticker"] for p in P.active() if p.get("trade_type") == "swing"}
_active_lt_swing = {p["ticker"] for p in P.active() if p.get("trade_type") == "lt_swing"}

# Long-term confluence (runs on all tickers, not just short-term eligible)
lt_confluence = {tk: Adv.lt_confluence_score(ind) for tk, ind in lt_snapshots.items()}
lt_eligible   = [(tk, cs) for tk, cs in lt_confluence.items()
                 if cs["score"] >= Adv.LT_MIN_CONFLUENCE]

if not eligible and not lt_eligible:
    st.info("No strong signals right now. Recommendations appear when ≥3 indicators agree.")
else:
    option_recs:  list[dict] = []
    swing_recs:   list[dict] = []
    lt_swing_recs: list[dict] = []

    with st.spinner("Fetching options chains for eligible tickers…"):
        for tk, cs in eligible:
            ind_tk    = snapshots[tk]
            direction = cs["direction"]
            signals   = "; ".join(cs["signals"])

            opt = _fetch_option(tk, direction, round(ind_tk["price"], 2))
            if opt:
                key = f"{tk}|{opt['option_type']}|{opt['strike']}|{opt['expiration']}"
                if key not in _active_keys:
                    option_recs.append({**opt, "signals": signals, "confluence": cs["score"]})

            if tk not in _active_swing:
                srec = Adv.swing_rec(tk, direction, ind_tk)
                swing_recs.append({**srec, "signals": signals, "confluence": cs["score"]})

    for tk, cs in lt_eligible:
        if tk not in _active_lt_swing:
            lrec = Adv.lt_swing_rec(tk, cs["direction"], lt_snapshots[tk])
            # de-duplicate signals list (double-counted signals are intentional for scoring
            # but look redundant displayed; show unique only)
            unique_signals = list(dict.fromkeys(cs["signals"]))
            lt_swing_recs.append({**lrec,
                                   "signals":   "; ".join(unique_signals),
                                   "confluence": cs["score"]})

    tab_opt, tab_sw, tab_lt = st.tabs([
        f"Options ({len(option_recs)})",
        f"Swing 1-2W ({len(swing_recs)})",
        f"Swing 1-3M ({len(lt_swing_recs)})",
    ])

    with tab_opt:
        if not option_recs:
            st.info("No suitable options contracts found for current signals.")
        else:
            opt_df = pd.DataFrame([{
                "#":          i + 1,
                "Ticker":     r["ticker"],
                "Action":     r["action"],
                "Type":       r["option_type"].title(),
                "Strike":     f"${r['strike']:.2f}",
                "Expiry":     r["expiration"],
                "DTE":        r["dte"],
                "Premium":    f"${r['premium']:.2f}",
                "IV %":       f"{r['iv']:.1f}",
                "Delta":      f"{r['delta']:.3f}",
                "Prob OTM":   f"{r['prob_success']:.1f}%",
                "Breakeven":  f"${r['breakeven']:.2f}",
                "Close @50%": f"${r['close_50pct']:.2f}",
                "Close @90%": f"${r['close_90pct']:.2f}",
                "Signals":    r["signals"],
            } for i, r in enumerate(option_recs)])
            st.dataframe(opt_df, use_container_width=True, hide_index=True)

            st.caption("Select trades to take:")
            chosen = st.multiselect(
                "Recommendations (by #)",
                options=[r["ticker"] + f" {r['option_type'].title()} ${r['strike']:.2f} {r['expiration']}"
                         for r in option_recs],
                label_visibility="collapsed",
            )
            if st.button("✅ Take selected option trades", disabled=not chosen):
                labels = [r["ticker"] + f" {r['option_type'].title()} ${r['strike']:.2f} {r['expiration']}"
                          for r in option_recs]
                for rec in [option_recs[labels.index(c)] for c in chosen]:
                    P.add(rec)
                st.success(f"Saved {len(chosen)} position(s).")
                st.rerun()

    with tab_sw:
        if not swing_recs:
            st.info("No short-term swing setups at this time.")
        else:
            sw_df = pd.DataFrame([{
                "#":         i + 1,
                "Ticker":    r["ticker"],
                "Direction": r["direction"],
                "Entry":     f"${r['entry_price']:.2f}",
                "Target":    f"${r['target_price']:.2f}",
                "Stop":      f"${r['stop_price']:.2f}",
                "R:R":       r["risk_reward"],
                "Period":    r["hold_period"],
                "Signals":   r["signals"],
            } for i, r in enumerate(swing_recs)])
            st.dataframe(sw_df, use_container_width=True, hide_index=True)

            chosen_sw = st.multiselect(
                "Swing recs",
                options=[f"{r['ticker']} {r['direction']}" for r in swing_recs],
                label_visibility="collapsed",
            )
            if st.button("✅ Take selected swing trades", disabled=not chosen_sw):
                labels = [f"{r['ticker']} {r['direction']}" for r in swing_recs]
                for rec in [swing_recs[labels.index(c)] for c in chosen_sw]:
                    P.add(rec)
                st.success(f"Saved {len(chosen_sw)} position(s).")
                st.rerun()

    with tab_lt:
        if not lt_swing_recs:
            st.info("No long-term swing setups at this time.")
        else:
            lt_df = pd.DataFrame([{
                "#":           i + 1,
                "Ticker":      r["ticker"],
                "Direction":   r["direction"],
                "Entry":       f"${r['entry_price']:.2f}",
                "Target":      f"${r['target_price']:.2f}",
                "Stop":        f"${r['stop_price']:.2f}",
                "Upside %":    f"{r['upside_pct']:.1f}",
                "R:R":         r["risk_reward"],
                "Period":      r["hold_period"],
                "Exp. Move":   r["expected_move"],
                "Fib anchor":  r["fib_anchor"],
                "Signals":     r["signals"],
            } for i, r in enumerate(lt_swing_recs)])
            st.dataframe(lt_df, use_container_width=True, hide_index=True)
            st.caption(
                "Targets are the nearest 52-week fib retracement level above (Long) "
                "or below (Reduce) the current price. Stops are 3× ATR."
            )

            chosen_lt = st.multiselect(
                "LT swing recs",
                options=[f"{r['ticker']} {r['direction']}" for r in lt_swing_recs],
                label_visibility="collapsed",
            )
            if st.button("✅ Take selected long-term swing trades", disabled=not chosen_lt):
                labels = [f"{r['ticker']} {r['direction']}" for r in lt_swing_recs]
                for rec in [lt_swing_recs[labels.index(c)] for c in chosen_lt]:
                    P.add(rec)
                st.success(f"Saved {len(chosen_lt)} position(s).")
                st.rerun()

# ── Active Positions Monitor ───────────────────────────────────────────────────
st.subheader("📍 Active Positions")
active_pos = P.active()

if not active_pos:
    st.caption("No active positions. Take a trade above to start monitoring.")
else:
    today_iso = dt.date.today().isoformat()
    to_close  = st.session_state.get("_close_queue", [])

    for pos in active_pos:
        pid   = pos["id"]
        label = (
            f"{pos['ticker']}  ·  "
            + (f"{pos['option_type'].title()} ${pos['strike']:.2f}  exp {pos['expiration']}"
               if pos["trade_type"] == "option"
               else f"{pos['direction']}  entry ${pos['entry_price']:.2f}")
            + f"  ·  opened {pos['opened']}"
        )

        with st.expander(label, expanded=False):
            if pos["trade_type"] == "option":
                expiry   = pos["expiration"]
                expired  = expiry < today_iso
                dte      = (dt.date.fromisoformat(expiry) - dt.date.today()).days
                premium  = pos["premium"]

                if not expired:
                    curr = _monitor_price(pos["ticker"], pos["option_type"],
                                          pos["strike"], expiry)
                else:
                    curr = 0.0  # assumed worthless if expired OTM

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Strike",      f"${pos['strike']:.2f}")
                mc2.metric("Expiry / DTE", f"{expiry} ({max(dte, 0)}d)")
                mc3.metric("Entry premium", f"${premium:.2f}")
                mc4.metric("Current price", f"${curr:.2f}" if curr is not None else "—")

                if curr is not None and premium > 0:
                    pnl_pct = (1 - curr / premium) * 100
                    mc5.metric("P&L", f"{pnl_pct:.1f}%",
                               delta=f"{pnl_pct:.1f}%",
                               delta_color="normal")

                    if pnl_pct >= 90:
                        st.success(
                            f"🎯 **90 % profit target hit** — "
                            f"Buy to Close at ~${pos['close_90pct']:.2f}"
                        )
                    elif pnl_pct >= 50:
                        st.warning(
                            f"⚡ **50 % profit target hit** — "
                            f"Buy to Close at ~${pos['close_50pct']:.2f}"
                        )
                    else:
                        st.caption(
                            f"Watching for 50 % close @ ${pos['close_50pct']:.2f}  |  "
                            f"90 % close @ ${pos['close_90pct']:.2f}"
                        )

                if expired:
                    st.info("Option has expired. Review and mark closed.")

                st.caption(f"Signals at entry: {pos.get('signals', '—')}")
                cl1, cl2, _ = st.columns([1, 1, 4])
                if cl1.button("Close position", key=f"close_{pid}"):
                    P.close(pid, "manual")
                    st.rerun()
                if expired and cl2.button("Mark expired", key=f"expire_{pid}"):
                    P.close(pid, "expired")
                    st.rerun()

            else:  # swing
                curr_price = (snapshots.get(pos["ticker"]) or {}).get("price")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Direction",  pos["direction"])
                sc2.metric("Entry",      f"${pos['entry_price']:.2f}")
                sc3.metric("Target",     f"${pos['target_price']:.2f}")
                sc4.metric("Stop",       f"${pos['stop_price']:.2f}")
                sc5.metric("Current",    f"${curr_price:.2f}" if curr_price else "—")

                if curr_price:
                    is_long = pos["direction"] == "Long"
                    pnl_pct = ((curr_price / pos["entry_price"] - 1) * 100
                               if is_long else
                               (pos["entry_price"] / curr_price - 1) * 100)
                    st.metric("P&L", f"{pnl_pct:.1f}%")

                    if is_long and curr_price >= pos["target_price"]:
                        st.success(f"🎯 Target hit — consider closing at ${curr_price:.2f}")
                    elif is_long and curr_price <= pos["stop_price"]:
                        st.error(f"⛔ Stop hit — consider closing at ${curr_price:.2f}")
                    elif not is_long and curr_price <= pos["target_price"]:
                        st.success(f"🎯 Target hit — consider closing at ${curr_price:.2f}")
                    elif not is_long and curr_price >= pos["stop_price"]:
                        st.error(f"⛔ Stop hit — consider closing at ${curr_price:.2f}")

                st.caption(f"Hold period: {pos['hold_period']}  ·  R:R {pos['risk_reward']}  ·  "
                           f"Signals: {pos.get('signals', '—')}")
                if st.button("Close position", key=f"close_{pid}"):
                    P.close(pid, "manual")
                    st.rerun()
