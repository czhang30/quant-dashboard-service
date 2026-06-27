"""
Technical indicators and signal evaluation.

Pure pandas/numpy. No network, no Streamlit, no broker code here, so this
module is trivially unit-testable on synthetic data. Every public function
takes a DataFrame with the standard OHLCV columns:
    ['Open', 'High', 'Low', 'Close', 'Volume']
indexed by datetime (ascending).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Core indicators
# --------------------------------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When there have been zero losses, RSI is 100 by definition.
    out[avg_loss == 0] = 100.0
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# --------------------------------------------------------------------------
# Level-based helpers
# --------------------------------------------------------------------------
def support_resistance(df: pd.DataFrame, lookback: int = 20):
    """Simple rolling support/resistance over the last `lookback` bars."""
    window = df.tail(lookback)
    return float(window["Low"].min()), float(window["High"].max())


def fib_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Fibonacci retracement levels of the swing low->high over `lookback`."""
    window = df.tail(lookback)
    lo = float(window["Low"].min())
    hi = float(window["High"].max())
    rng = hi - lo
    return {
        "low": lo,
        "high": hi,
        "23.6%": hi - 0.236 * rng,
        "38.2%": hi - 0.382 * rng,
        "50%": hi - 0.500 * rng,
        "61.8%": hi - 0.618 * rng,
        "78.6%": hi - 0.786 * rng,
    }


# --------------------------------------------------------------------------
# Snapshot: collapse a full history into the latest-state dict the UI needs
# --------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """
    Compute every indicator and return a flat dict of the *latest* values
    plus a couple of crossover flags that need the previous bar.
    """
    cfg = cfg or {}
    p = {
        "rsi": cfg.get("rsi_period", 14),
        "ema_fast": cfg.get("ema_fast", 20),
        "sma_fast": cfg.get("sma_fast", 50),
        "sma_slow": cfg.get("sma_slow", 200),
        "sr_lookback": cfg.get("sr_lookback", 20),
        "fib_lookback": cfg.get("fib_lookback", 60),
        "vol_lookback": cfg.get("vol_lookback", 20),
    }
    close = df["Close"]
    out: dict = {}

    out["price"] = float(close.iloc[-1])
    out["prev_close"] = float(close.iloc[-2]) if len(close) > 1 else np.nan
    out["pct_change"] = (
        (out["price"] / out["prev_close"] - 1) * 100 if out["prev_close"] else np.nan
    )

    rsi_s = rsi(close, p["rsi"])
    out["rsi"] = float(rsi_s.iloc[-1])

    macd_line, signal_line, hist = macd(close)
    out["macd"] = float(macd_line.iloc[-1])
    out["macd_signal"] = float(signal_line.iloc[-1])
    out["macd_hist"] = float(hist.iloc[-1])
    # Bullish/bearish MACD cross on the most recent bar.
    if len(hist) > 1:
        out["macd_cross_up"] = bool(hist.iloc[-2] <= 0 < hist.iloc[-1])
        out["macd_cross_down"] = bool(hist.iloc[-2] >= 0 > hist.iloc[-1])
    else:
        out["macd_cross_up"] = out["macd_cross_down"] = False

    ema_fast_s = ema(close, p["ema_fast"])
    sma_fast = sma(close, p["sma_fast"])
    sma_slow = sma(close, p["sma_slow"])
    out["ema_fast"] = float(ema_fast_s.iloc[-1]) if ema_fast_s.notna().any() else np.nan
    out["sma_fast"] = float(sma_fast.iloc[-1]) if sma_fast.notna().any() else np.nan
    out["sma_slow"] = float(sma_slow.iloc[-1]) if sma_slow.notna().any() else np.nan
    out["above_sma_fast"] = (
        out["price"] > out["sma_fast"] if not np.isnan(out["sma_fast"]) else None
    )

    atr_s = atr(df, p["rsi"])
    out["atr"] = float(atr_s.iloc[-1]) if atr_s.notna().any() else np.nan
    out["atr_pct"] = (out["atr"] / out["price"] * 100) if out["atr"] else np.nan

    sup, res = support_resistance(df, p["sr_lookback"])
    out["support"] = sup
    out["resistance"] = res
    out["dist_to_support_pct"] = (out["price"] / sup - 1) * 100 if sup else np.nan
    out["dist_to_resistance_pct"] = (out["price"] / res - 1) * 100 if res else np.nan

    out["fib"] = fib_levels(df, p["fib_lookback"])

    vol = df["Volume"]
    avg_vol = vol.tail(p["vol_lookback"]).mean()
    out["volume"] = float(vol.iloc[-1])
    out["avg_volume"] = float(avg_vol) if avg_vol else np.nan
    out["vol_ratio"] = (out["volume"] / avg_vol) if avg_vol else np.nan

    return out


def trend_label(ind: dict) -> str:
    """A coarse, human-readable trend read used for table colour-coding."""
    score = 0
    if ind.get("above_sma_fast"):
        score += 1
    elif ind.get("above_sma_fast") is False:
        score -= 1
    if ind.get("macd_hist", 0) > 0:
        score += 1
    else:
        score -= 1
    r = ind.get("rsi", 50)
    if r >= 55:
        score += 1
    elif r <= 45:
        score -= 1
    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"
