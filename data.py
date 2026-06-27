"""
Data access layer. Kept free of Streamlit so it can be reused in scripts,
notebooks, or a scheduler. Default source is yfinance (free, no key). Swap in
a real-time feed (Alpaca, Polygon, IEX) by reimplementing get_history().
"""
from __future__ import annotations

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def get_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Return an ascending OHLCV DataFrame for a single ticker, or an empty
    DataFrame on failure (never raises, so one bad symbol can't kill the UI).
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. `pip install yfinance`")
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        return pd.DataFrame(columns=_OHLCV)
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLCV)
    df = df.rename(columns=str.title)
    keep = [c for c in _OHLCV if c in df.columns]
    df = df[keep].dropna()
    return df


def get_watchlist(tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict:
    """Map of ticker -> OHLCV DataFrame. Skips symbols that return no data."""
    out = {}
    for t in tickers:
        df = get_history(t, period=period, interval=interval)
        if not df.empty and len(df) > 5:
            out[t] = df
    return out
