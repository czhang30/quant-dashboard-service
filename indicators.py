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
# Quant indicators: Bollinger Bands, Z-score, Realized Vol, ADX, CS Momentum
# --------------------------------------------------------------------------
def bollinger_bands(series: pd.Series, period: int = 20, n_std: float = 2.0):
    """Returns (upper, middle, lower, pct_b, bandwidth)."""
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=1)
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return upper, mid, lower, pct_b, bandwidth


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score of price around its mean."""
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=1)
    return (series - mean) / std


def realized_vol(series: pd.Series, period: int = 20, ann_factor: int = 252) -> pd.Series:
    """Annualized realized volatility (std of log returns), in %."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period, min_periods=period).std(ddof=1) * np.sqrt(ann_factor) * 100


def adx_dmi(df: pd.DataFrame, period: int = 14):
    """ADX trend strength + DMI. Returns (adx, plus_di, minus_di)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = pd.Series(
        np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0),
        index=df.index,
    )
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / period
    tr_s = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / tr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_s = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx_s, plus_di, minus_di


def cross_sectional_momentum(hist_dict: dict, lookback: int = 252, skip: int = 21) -> dict:
    """
    12-1 momentum: return from `lookback` bars ago to `skip` bars ago, ranked across
    the watchlist. Rank 1 = highest momentum.
    Returns {ticker: {"mom_12_1": float, "mom_rank": int, "mom_rank_total": int}}.
    """
    moms: dict[str, float] = {}
    for tk, df in hist_dict.items():
        close = df["Close"]
        lb = min(lookback, len(close) - skip - 1)
        if lb < 1:
            moms[tk] = np.nan
        else:
            moms[tk] = float(close.iloc[-skip] / close.iloc[-lb - skip] - 1) * 100

    valid = {tk: v for tk, v in moms.items() if not np.isnan(v)}
    ranked = sorted(valid, key=valid.__getitem__, reverse=True)
    rank_map = {tk: i + 1 for i, tk in enumerate(ranked)}
    total = len(valid)
    return {
        tk: {
            "mom_12_1": moms.get(tk, np.nan),
            "mom_rank": rank_map.get(tk),
            "mom_rank_total": total,
        }
        for tk in hist_dict
    }


# --------------------------------------------------------------------------
# Level-based helpers
# --------------------------------------------------------------------------
def support_resistance(df: pd.DataFrame, lookback: int = 20):
    """Simple rolling support/resistance over the last `lookback` bars."""
    window = df.tail(lookback)
    return float(window["Low"].min()), float(window["High"].max())


def sr_swing_clusters(df: pd.DataFrame, lookback: int = 100, n: int = 2, tolerance: float = 0.015):
    """Swing high/low clustering. Groups nearby pivot points within `tolerance` %."""
    window = df.tail(lookback)
    highs = window["High"].values
    lows = window["Low"].values
    length = len(window)

    points = []
    for i in range(n, length - n):
        if all(highs[i] >= highs[i - j] for j in range(1, n + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, n + 1)):
            points.append(float(highs[i]))
        if all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, n + 1)):
            points.append(float(lows[i]))

    if not points:
        return {"support": [float(window["Low"].min())], "resistance": [float(window["High"].max())]}

    points.sort()
    clusters = [[points[0]]]
    for lvl in points[1:]:
        if (lvl - clusters[-1][-1]) / clusters[-1][-1] <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    means = sorted([float(np.mean(c)) for c in sorted(clusters, key=len, reverse=True)])

    price = float(df["Close"].iloc[-1])
    return {
        "support": sorted([l for l in means if l < price], reverse=True)[:4],
        "resistance": sorted([l for l in means if l >= price])[:4],
    }


def sr_volume_profile(df: pd.DataFrame, lookback: int = 100, n_bins: int = 50, top_n: int = 4):
    """High-volume nodes from a price-vs-volume histogram."""
    window = df.tail(lookback)
    lo_min = float(window["Low"].min())
    hi_max = float(window["High"].max())
    bins = np.linspace(lo_min, hi_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vol_at_price = np.zeros(n_bins)

    for _, row in window.iterrows():
        lo, hi, vol = row["Low"], row["High"], row["Volume"]
        span = hi - lo
        if span == 0:
            idx = min(max(int(np.searchsorted(bins, lo, side="right")) - 1, 0), n_bins - 1)
            vol_at_price[idx] += vol
        else:
            overlaps = np.minimum(hi, bins[1:]) - np.maximum(lo, bins[:-1])
            mask = overlaps > 0
            vol_at_price[mask] += vol * overlaps[mask] / span

    peaks = [i for i in range(1, n_bins - 1)
             if vol_at_price[i] > vol_at_price[i - 1] and vol_at_price[i] > vol_at_price[i + 1]]
    peaks.sort(key=lambda i: vol_at_price[i], reverse=True)
    levels = [float(bin_centers[i]) for i in peaks[:top_n * 2]]

    price = float(df["Close"].iloc[-1])
    return {
        "support": sorted([l for l in levels if l < price], reverse=True)[:top_n],
        "resistance": sorted([l for l in levels if l >= price])[:top_n],
    }


def sr_fractals(df: pd.DataFrame, n: int = 2, lookback: int = 100):
    """Bill Williams fractals: confirmed swing highs/lows with n bars on each side."""
    window = df.tail(lookback + n)
    highs = window["High"].values
    lows = window["Low"].values
    length = len(window)

    fractal_highs, fractal_lows = [], []
    for i in range(n, length - n):
        if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, n + 1)):
            fractal_highs.append(float(highs[i]))
        if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, n + 1)):
            fractal_lows.append(float(lows[i]))

    price = float(df["Close"].iloc[-1])
    return {
        "support": sorted([l for l in fractal_lows if l < price], reverse=True)[:4],
        "resistance": sorted([l for l in fractal_highs if l >= price])[:4],
    }


def sr_moving_averages(df: pd.DataFrame, cfg: dict | None = None):
    """Current EMA/SMA values as dynamic S/R. Below price = support, above = resistance."""
    cfg = cfg or {}
    close = df["Close"]
    price = float(close.iloc[-1])
    specs = [
        (f"EMA{cfg.get('ema_fast', 20)}", ema(close, cfg.get("ema_fast", 20))),
        (f"SMA{cfg.get('sma_fast', 50)}", sma(close, cfg.get("sma_fast", 50))),
        (f"SMA{cfg.get('sma_slow', 200)}", sma(close, cfg.get("sma_slow", 200))),
    ]
    support, resistance = {}, {}
    for label, series in specs:
        v = series.iloc[-1]
        if pd.notna(v):
            (support if float(v) < price else resistance)[label] = float(v)
    return {"support": support, "resistance": resistance}


def sr_kde(df: pd.DataFrame, lookback: int = 100, n_levels: int = 4):
    """Gaussian KDE over recent High/Low prices; density peaks become S/R levels."""
    window = df.tail(lookback)
    prices = np.concatenate([window["High"].values, window["Low"].values])
    lo, hi = prices.min(), prices.max()
    bandwidth = (hi - lo) * 0.02
    if bandwidth == 0:
        return {"support": [], "resistance": []}

    grid = np.linspace(lo, hi, 500)
    density = np.exp(-0.5 * ((grid[:, None] - prices[None, :]) / bandwidth) ** 2).sum(axis=1)

    peaks = [i for i in range(1, len(density) - 1)
             if density[i] > density[i - 1] and density[i] > density[i + 1]]
    peaks.sort(key=lambda i: density[i], reverse=True)
    levels = [float(grid[i]) for i in peaks[:n_levels * 2]]

    price = float(df["Close"].iloc[-1])
    return {
        "support": sorted([l for l in levels if l < price], reverse=True)[:n_levels],
        "resistance": sorted([l for l in levels if l >= price])[:n_levels],
    }


def compute_sr(df: pd.DataFrame, method: str, cfg: dict | None = None) -> dict:
    """
    Dispatcher for all S/R methods.
    Returns {"support": [(label, price), ...], "resistance": [(label, price), ...]}
    """
    cfg = cfg or {}
    if method == "Price Extremes":
        sup, res = support_resistance(df, cfg.get("sr_lookback", 20))
        return {"support": [("S", sup)], "resistance": [("R", res)]}

    if method == "Swing Clusters":
        r = sr_swing_clusters(df)
        return {
            "support": [(f"S{i+1}", p) for i, p in enumerate(r["support"])],
            "resistance": [(f"R{i+1}", p) for i, p in enumerate(r["resistance"])],
        }

    if method == "Volume Profile":
        r = sr_volume_profile(df)
        return {
            "support": [(f"HVN-S{i+1}", p) for i, p in enumerate(r["support"])],
            "resistance": [(f"HVN-R{i+1}", p) for i, p in enumerate(r["resistance"])],
        }

    if method == "Fractals":
        r = sr_fractals(df)
        return {
            "support": [(f"FL{i+1}", p) for i, p in enumerate(r["support"])],
            "resistance": [(f"FH{i+1}", p) for i, p in enumerate(r["resistance"])],
        }

    if method == "Moving Averages":
        r = sr_moving_averages(df, cfg)
        return {
            "support": list(r["support"].items()),
            "resistance": list(r["resistance"].items()),
        }

    if method == "KDE":
        r = sr_kde(df)
        return {
            "support": [(f"KS{i+1}", p) for i, p in enumerate(r["support"])],
            "resistance": [(f"KR{i+1}", p) for i, p in enumerate(r["resistance"])],
        }

    return {"support": [], "resistance": []}


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

    # Bollinger Bands
    bb_up, bb_mid, bb_lo, pct_b, bb_bw = bollinger_bands(close)
    out["bb_upper"]    = float(bb_up.iloc[-1])  if bb_up.notna().any()  else np.nan
    out["bb_middle"]   = float(bb_mid.iloc[-1]) if bb_mid.notna().any() else np.nan
    out["bb_lower"]    = float(bb_lo.iloc[-1])  if bb_lo.notna().any()  else np.nan
    out["bb_pct_b"]    = float(pct_b.iloc[-1])  if pct_b.notna().any()  else np.nan
    out["bb_bandwidth"]= float(bb_bw.iloc[-1])  if bb_bw.notna().any()  else np.nan

    # Z-score
    z = zscore(close)
    out["zscore"] = float(z.iloc[-1]) if z.notna().any() else np.nan

    # Realized volatility
    rv = realized_vol(close)
    out["realized_vol"] = float(rv.iloc[-1]) if rv.notna().any() else np.nan

    # ADX / DMI
    adx_s, dmi_p, dmi_m = adx_dmi(df)
    out["adx"]          = float(adx_s.iloc[-1]) if adx_s.notna().any() else np.nan
    out["dmi_plus"]     = float(dmi_p.iloc[-1]) if dmi_p.notna().any() else np.nan
    out["dmi_minus"]    = float(dmi_m.iloc[-1]) if dmi_m.notna().any() else np.nan
    out["adx_trending"] = (out["adx"] >= 25) if not np.isnan(out["adx"]) else False

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
