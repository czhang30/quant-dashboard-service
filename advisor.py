"""
Trade recommendation engine.

Computes confluence scores from indicator snapshots, fetches live options
chains via yfinance, selects optimal short-premium strikes, and provides
current option prices for P&L monitoring.
"""
from __future__ import annotations
import math
import datetime as dt

import numpy as np
import yfinance as yf

# ── configuration ─────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.05   # approx 3-month T-bill; update as needed
MIN_CONFLUENCE = 3       # signals required to emit a recommendation
DELTA_MIN      = 0.15   # |delta| lower bound for strike selection
DELTA_MAX      = 0.40   # |delta| upper bound for strike selection
MIN_PREMIUM    = 0.05   # minimum bid price ($) to consider
MIN_DAYS       = 7      # minimum DTE
MAX_DAYS       = 35     # maximum DTE


# ── Black-Scholes helpers ──────────────────────────────────────────────────────
def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes delta.  T in years, sigma as decimal (e.g. 0.30 = 30%)."""
    if T < 1e-6 or sigma < 1e-6 or S <= 0 or K <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if option_type == "call" else _ncdf(d1) - 1.0


# ── Confluence scoring ─────────────────────────────────────────────────────────
def confluence_score(ind: dict) -> dict:
    """
    Evaluate how many bullish vs bearish signals are present in an indicator
    snapshot.  Returns:
        {"direction": "bullish"|"bearish", "score": int, "signals": list[str]}
    Only tickers whose score >= MIN_CONFLUENCE get a trade recommendation.
    """
    price = ind.get("price") or 0
    bull: list[str] = []
    bear: list[str] = []

    # RSI
    rsi = ind.get("rsi") or 50
    if rsi < 35:
        bull.append(f"RSI {rsi:.0f} (oversold)")
    elif rsi > 65:
        bear.append(f"RSI {rsi:.0f} (overbought)")

    # Bollinger %B
    pct_b = ind.get("bb_pct_b")
    if pct_b is not None and not math.isnan(float(pct_b)):
        if pct_b < 0.2:
            bull.append(f"%B {pct_b:.2f} (low)")
        elif pct_b > 0.8:
            bear.append(f"%B {pct_b:.2f} (high)")

    # Z-score
    z = ind.get("zscore")
    if z is not None and not math.isnan(float(z)):
        if z < -1.5:
            bull.append(f"Z-score {z:.2f}")
        elif z > 1.5:
            bear.append(f"Z-score {z:.2f}")

    # MACD
    if ind.get("macd_cross_up"):
        bull.append("MACD cross ↑")
    if ind.get("macd_cross_down"):
        bear.append("MACD cross ↓")
    hist = ind.get("macd_hist") or 0
    if hist > 0:
        bull.append("MACD hist +")
    else:
        bear.append("MACD hist −")

    # Moving averages
    sma_fast = ind.get("sma_fast")
    if sma_fast and not math.isnan(float(sma_fast)):
        (bull if price > sma_fast else bear).append(
            "Above SMA50" if price > sma_fast else "Below SMA50"
        )
    sma_slow = ind.get("sma_slow")
    if sma_slow and not math.isnan(float(sma_slow)):
        (bull if price > sma_slow else bear).append(
            "Above SMA200" if price > sma_slow else "Below SMA200"
        )

    # Support / resistance proximity
    d_sup = ind.get("dist_to_support_pct")
    if d_sup is not None and not math.isnan(float(d_sup)) and 0 <= d_sup <= 2:
        bull.append(f"Near support ({d_sup:.1f}%)")
    d_res = ind.get("dist_to_resistance_pct")
    if d_res is not None and not math.isnan(float(d_res)) and -2 <= d_res <= 0:
        bear.append(f"Near resistance ({abs(d_res):.1f}%)")

    # Cross-sectional momentum rank
    mom_rank  = ind.get("mom_rank")
    mom_total = ind.get("mom_rank_total") or 1
    if mom_rank:
        if mom_rank <= max(1, mom_total // 3):
            bull.append(f"Top momentum #{mom_rank}/{mom_total}")
        elif mom_rank >= mom_total - mom_total // 3:
            bear.append(f"Weak momentum #{mom_rank}/{mom_total}")

    if len(bull) >= len(bear):
        return {"direction": "bullish", "score": len(bull), "signals": bull}
    return {"direction": "bearish", "score": len(bear), "signals": bear}


# ── Options chain helpers ──────────────────────────────────────────────────────
def _valid_expiries(tk_obj) -> list[tuple[int, str]]:
    """Return (days_out, date_str) pairs within the MIN/MAX DTE window."""
    today = dt.date.today()
    out: list[tuple[int, str]] = []
    for s in tk_obj.options or []:
        try:
            d = dt.date.fromisoformat(s)
        except ValueError:
            continue
        days = (d - today).days
        if MIN_DAYS <= days <= MAX_DAYS:
            out.append((days, s))
    return sorted(out)


def best_option(ticker: str, direction: str, price: float) -> dict | None:
    """
    Fetch the live options chain and select the optimal short-premium trade.

    Strategy:
      bullish  → sell OTM put  (below current price)
      bearish  → sell OTM covered call (above current price)

    Strike selection: maximise expected value = mid_price × prob_OTM,
    restricted to |delta| in [DELTA_MIN, DELTA_MAX] when possible.

    Returns a dict ready to display / store, or None when no suitable
    contract exists.
    """
    if not price:
        return None
    option_type = "put" if direction == "bullish" else "call"

    try:
        tk_obj  = yf.Ticker(ticker)
        expiries = _valid_expiries(tk_obj)
        if not expiries:
            return None

        best_rec: dict | None = None
        best_ev  = -1.0

        for days_out, expiry_str in expiries:
            T = days_out / 365.0
            try:
                chain = tk_obj.option_chain(expiry_str)
            except Exception:
                continue

            opts = chain.puts if option_type == "put" else chain.calls
            if opts is None or opts.empty:
                continue

            # Keep only OTM contracts with positive IV and a tradeable bid
            otm_mask = opts["strike"] < price if option_type == "put" else opts["strike"] > price
            opts = opts[
                otm_mask
                & (opts["impliedVolatility"] > 0)
                & (opts["bid"] >= MIN_PREMIUM)
            ].copy()
            if opts.empty:
                continue

            opts["mid"]       = (opts["bid"] + opts["ask"]) / 2
            opts["delta"]     = opts.apply(
                lambda r: bs_delta(price, r["strike"], T, r["impliedVolatility"], option_type),
                axis=1,
            )
            opts["abs_delta"] = opts["delta"].abs()
            opts["prob_otm"]  = 1.0 - opts["abs_delta"]
            opts["ev"]        = opts["mid"] * opts["prob_otm"]

            pool = opts[(opts["abs_delta"] >= DELTA_MIN) & (opts["abs_delta"] <= DELTA_MAX)]
            if pool.empty:
                pool = opts  # fall back to full set

            row = pool.nlargest(1, "ev").iloc[0]
            if float(row["ev"]) > best_ev:
                best_ev = float(row["ev"])
                strike  = float(row["strike"])
                mid     = float(row["mid"])
                best_rec = {
                    "ticker":       ticker,
                    "trade_type":   "option",
                    "option_type":  option_type,
                    "action":       "Sell to Open",
                    "direction":    direction,
                    "strike":       strike,
                    "expiration":   expiry_str,
                    "dte":          days_out,
                    "premium":      round(mid, 2),
                    "bid":          round(float(row["bid"]), 2),
                    "ask":          round(float(row["ask"]), 2),
                    "iv":           round(float(row["impliedVolatility"]) * 100, 1),
                    "delta":        round(float(row["delta"]), 3),
                    "prob_success": round(float(row["prob_otm"]) * 100, 1),
                    "breakeven":    round(strike - mid, 2) if option_type == "put"
                                    else round(strike + mid, 2),
                    "close_50pct":  round(mid * 0.50, 2),
                    "close_90pct":  round(mid * 0.10, 2),
                }

        return best_rec

    except Exception as e:
        print(f"[advisor] {ticker} options error: {e}")
        return None


# ── Swing trade recommendation ─────────────────────────────────────────────────
def swing_rec(ticker: str, direction: str, ind: dict) -> dict:
    """
    Generate a swing trade recommendation using ATR-based targets and the
    nearest S/R levels as guardrails.
    """
    price   = ind.get("price")   or 0
    atr_val = ind.get("atr")     or 0
    sup     = ind.get("support") or (price - atr_val)
    res     = ind.get("resistance") or (price + atr_val)
    adx_val = ind.get("adx")    or 0
    hold    = "3-7 days" if adx_val < 20 else "7-15 days"

    if direction == "bullish":
        target = min(price + 2 * atr_val, res)
        stop   = max(price - atr_val, sup)
        rr     = round((target - price) / max(price - stop, 0.01), 1)
        return {
            "ticker":       ticker,
            "trade_type":   "swing",
            "direction":    "Long",
            "entry_price":  round(price, 2),
            "target_price": round(target, 2),
            "stop_price":   round(stop, 2),
            "risk_reward":  f"{rr}:1",
            "hold_period":  hold,
        }
    else:
        target = max(price - 2 * atr_val, sup)
        stop   = min(price + atr_val, res)
        rr     = round((price - target) / max(stop - price, 0.01), 1)
        return {
            "ticker":       ticker,
            "trade_type":   "swing",
            "direction":    "Reduce / Short",
            "entry_price":  round(price, 2),
            "target_price": round(target, 2),
            "stop_price":   round(stop, 2),
            "risk_reward":  f"{rr}:1",
            "hold_period":  hold,
        }


# ── Long-term swing (1-3 months) ──────────────────────────────────────────────
LT_MIN_CONFLUENCE = 3

def lt_confluence_score(ind: dict) -> dict:
    """
    Confluence score weighted toward multi-month signals:
    SMA200 and 12-1 momentum rank count double; RSI/Z/%B use
    wider thresholds (40/60, ±1.0, 0.25/0.75).
    """
    price = ind.get("price") or 0
    bull: list[str] = []
    bear: list[str] = []

    # SMA200 — double-weighted (primary multi-month filter)
    sma_slow = ind.get("sma_slow")
    if sma_slow and not math.isnan(float(sma_slow)):
        label = "Above SMA200" if price > sma_slow else "Below SMA200"
        (bull if price > sma_slow else bear).extend([label, label])

    # 12-1 momentum rank — double-weighted (robust multi-month factor)
    mom_rank  = ind.get("mom_rank")
    mom_total = ind.get("mom_rank_total") or 1
    if mom_rank:
        if mom_rank <= max(1, mom_total // 3):
            lbl = f"Top momentum #{mom_rank}/{mom_total}"
            bull.extend([lbl, lbl])
        elif mom_rank >= mom_total - mom_total // 3:
            lbl = f"Weak momentum #{mom_rank}/{mom_total}"
            bear.extend([lbl, lbl])

    # Position within 52-week fib range
    fib = ind.get("fib", {})
    mid = fib.get("50%")
    if mid:
        (bull if price < mid else bear).append(
            "Lower half of yearly range" if price < mid else "Upper half of yearly range"
        )

    # RSI (wider thresholds for multi-month)
    rsi = ind.get("rsi") or 50
    if rsi < 40:
        bull.append(f"RSI {rsi:.0f}")
    elif rsi > 60:
        bear.append(f"RSI {rsi:.0f}")

    # Z-score
    z = ind.get("zscore")
    if z is not None and not math.isnan(float(z)):
        if z < -1.0:
            bull.append(f"Z-score {z:.2f}")
        elif z > 1.0:
            bear.append(f"Z-score {z:.2f}")

    # Bollinger %B
    pct_b = ind.get("bb_pct_b")
    if pct_b is not None and not math.isnan(float(pct_b)):
        if pct_b < 0.25:
            bull.append(f"%B {pct_b:.2f}")
        elif pct_b > 0.75:
            bear.append(f"%B {pct_b:.2f}")

    # SMA50
    sma_fast = ind.get("sma_fast")
    if sma_fast and not math.isnan(float(sma_fast)):
        (bull if price > sma_fast else bear).append(
            "Above SMA50" if price > sma_fast else "Below SMA50"
        )

    # MACD direction
    hist = ind.get("macd_hist") or 0
    (bull if hist > 0 else bear).append("MACD hist +" if hist > 0 else "MACD hist −")

    if len(bull) >= len(bear):
        return {"direction": "bullish", "score": len(bull), "signals": bull}
    return {"direction": "bearish", "score": len(bear), "signals": bear}


def lt_swing_rec(ticker: str, direction: str, ind: dict) -> dict:
    """
    Long-term swing recommendation (4-12 weeks).

    Uses 52-week fib levels (computed with fib_lookback=252) as targets and
    3× ATR as the stop distance.  Targets are the nearest fib retracement
    levels above (bullish) or below (bearish) the current price.
    """
    price   = ind.get("price")   or 0
    atr_val = ind.get("atr")     or 0
    rvol    = ind.get("realized_vol") or 0
    fib     = ind.get("fib", {})

    fib_high = fib.get("high")  or price * 1.25
    fib_low  = fib.get("low")   or price * 0.75
    f236 = fib.get("23.6%") or price * 1.12
    f382 = fib.get("38.2%") or price * 1.06
    f500 = fib.get("50%")   or price
    f618 = fib.get("61.8%") or price * 0.94
    f786 = fib.get("78.6%") or price * 0.88

    # 3-month expected move from realized vol
    exp_move = f"±{(rvol / 100) * (3 / 12) ** 0.5 * 100:.1f}%" if rvol > 0 else "—"

    if direction == "bullish":
        candidates = sorted([l for l in [f236, f382, f500, fib_high] if l > price])
        target = candidates[0] if candidates else price * 1.15
        stop_atr = price - 3 * atr_val
        stop     = max(stop_atr, f786) if f786 < price else stop_atr
        rr     = round((target - price) / max(price - stop, 0.01), 1)
        return {
            "ticker":         ticker,
            "trade_type":     "lt_swing",
            "direction":      "Long",
            "entry_price":    round(price, 2),
            "target_price":   round(target, 2),
            "stop_price":     round(stop, 2),
            "upside_pct":     round((target / price - 1) * 100, 1),
            "risk_reward":    f"{rr}:1",
            "hold_period":    "4-12 weeks",
            "expected_move":  exp_move,
            "fib_anchor":     "52-week range",
        }
    else:
        candidates = sorted([l for l in [f618, f786, fib_low] if l < price], reverse=True)
        target = candidates[0] if candidates else price * 0.85
        stop_atr = price + 3 * atr_val
        stop     = min(stop_atr, f236) if f236 > price else stop_atr
        rr     = round((price - target) / max(stop - price, 0.01), 1)
        return {
            "ticker":         ticker,
            "trade_type":     "lt_swing",
            "direction":      "Reduce / Short",
            "entry_price":    round(price, 2),
            "target_price":   round(target, 2),
            "stop_price":     round(stop, 2),
            "upside_pct":     round((1 - target / price) * 100, 1),
            "risk_reward":    f"{rr}:1",
            "hold_period":    "4-12 weeks",
            "expected_move":  exp_move,
            "fib_anchor":     "52-week range",
        }


# ── Position monitoring ────────────────────────────────────────────────────────
def current_option_price(ticker: str, option_type: str, strike: float, expiry: str) -> float | None:
    """Re-fetch the mid-price of a specific option for P&L tracking."""
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        opts  = chain.puts if option_type == "put" else chain.calls
        row   = opts[opts["strike"] == strike]
        if row.empty:
            return None
        r = row.iloc[0]
        return float((r["bid"] + r["ask"]) / 2)
    except Exception:
        return None
